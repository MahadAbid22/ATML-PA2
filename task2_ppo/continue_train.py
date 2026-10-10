from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl_utils import (
    cooldown_if_hot,
    disable_dropout,
    filter_rows_by_prompt_length,
    peak_vram_gib,
    reference_response_logprobs,
    response_logprobs,
)
from task2_ppo.ppo import (
    clipping_active_fraction,
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def upcast_value_head(value_model) -> None:
    """Keep the trainable critic head in fp32.

    The released critic is fp16; AdamW on fp16 parameters is numerically unsafe (eps=1e-8
    underflows to 0 in fp16, so 0/0 can appear for parameters with zero gradient). The
    LoRA weights are already fp32 (PEFT autocasts adapters); here we upcast the trainable
    copy of the scalar head and cast its input accordingly. The forward value is unchanged
    up to fp16->fp32 rounding.
    """
    for name, module in value_model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("score" in name) and any(p.requires_grad for p in module.parameters()):
            module.float()
            module.register_forward_pre_hook(lambda m, args: (args[0].float(),))


def selected_prompt_rows(cfg, tokenizer, n: int):
    """Fixed, seeded prompt order shared by the standard run and every fork."""
    rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    kept, _ = filter_rows_by_prompt_length(rows, tokenizer, int(cfg["max_prompt_length"]), prompt_messages)
    order = np.random.default_rng(int(cfg["seed"])).permutation(len(kept))
    return [kept[i] for i in order[:n]]


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    upcast_value_head(value_model)
    # Activation checkpointing for the critic too (VRAM headroom on an 8 GB GPU); values are unchanged.
    value_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def response_values(value_model, sequences, attention_mask, prompt_width, width):
    # V(s_t) for response step t is read at the position BEFORE token t (TRL convention:
    # values[:, context_length-1 : -1]).
    v = token_values(value_model, sequences, attention_mask)
    return v[:, prompt_width - 1 : -1][:, :width].float()


def explained_variance(pred, target, mask) -> float:
    m = mask.bool()
    y, p = target[m], pred[m]
    var_y = y.var(unbiased=False)
    if var_y.item() == 0:
        return float("nan")
    return float(1.0 - (y - p).var(unbiased=False) / var_y)


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok = bundle["tokenizer"]
    policy, value_model = bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    disable_dropout(policy)
    disable_dropout(value_model)

    n_updates = int(cfg["updates"])
    ppu = int(cfg["prompts_per_update"])
    eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])
    seed = int(cfg["seed"])
    max_grad_norm = float(cfg["max_grad_norm"])
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_log_{run_name}.jsonl"
    if log_path.exists():
        log_path.unlink()

    prompt_rows = selected_prompt_rows(cfg, tok, n_updates * ppu)
    save_json(results_dir / f"train_prompts_{run_name}.json", [
        {"update": i // ppu, "prompt_id": r["prompt_id"], "source_index": str(r["source_index"])} for i, r in enumerate(prompt_rows)
    ])
    config_record = {k: cfg[k] for k in [
        "updates", "prompts_per_update", "ppo_epochs", "policy_learning_rate", "value_lora_learning_rate",
        "value_head_learning_rate", "value_train_mode", "clip_epsilon", "kl_beta", "gamma", "gae_lambda",
        "value_coef", "missing_eos_penalty", "max_prompt_length", "max_response_length", "reward_max_length",
        "max_grad_norm", "seed"]}
    config_record["generation"] = cfg["generation"]
    print(f"[{run_name}] PPO continuation: {json.dumps(config_record)}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_gen_tokens = 0
    thermal_wait = 0.0
    policy_params = trainable_parameters(policy)
    value_params = trainable_parameters(value_model)

    for u in range(n_updates):
        thermal_wait += cooldown_if_hot()
        t_update = time.time()
        rows = prompt_rows[u * ppu : (u + 1) * ppu]
        msgs = [prompt_messages(r) for r in rows]

        # ---------------- Rollout (old policy snapshot = current weights, no grad) -------------
        set_seed(seed + u)  # identical sampling stream for every fork
        gen = batch_generate(
            policy, tok, msgs,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_response_length"]),
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=True,
        )
        seq, attn, pw = gen["sequences"], gen["attention_mask"], gen["prompt_width"]
        width = gen["response_ids"].shape[1]
        mask = gen["response_mask"]
        total_gen_tokens += int(mask.sum().item())
        # Old log-probs are computed through the SAME (grad-mode) kernel path as the PPO epochs and
        # then detached, so the ratio is exactly 1 before the first optimizer step; a no-grad pass
        # can select different fp16 attention kernels and inject ~1% ratio noise.
        old_logp, ent = response_logprobs(policy, seq, attn, width, grad=True, entropy=True)
        old_logp = old_logp.detach()
        with torch.no_grad():
            ref_logp = reference_response_logprobs(policy, seq, attn, width)
            values = response_values(value_model, seq, attn, pw, width) * mask
            raw_reward = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=int(cfg["reward_max_length"])).to(seq.device)
            no_eos = torch.tensor([not t for t in gen["terminated_with_eos"]], device=seq.device, dtype=torch.float32)
            task_reward = raw_reward - float(cfg["missing_eos_penalty"]) * no_eos

            rewards = shaped_rewards(task_reward, old_logp, ref_logp, mask, beta_kl)
            adv, returns = compute_gae(rewards, values, mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
            adv_norm = normalize_advantages(adv, mask)
            ev_before = explained_variance(values, returns, mask)

        # ---------------- PPO epochs on this rollout ----------------
        ep_stats = []
        for epoch in range(int(cfg["ppo_epochs"])):
            new_logp, _ = response_logprobs(policy, seq, attn, width, grad=True)
            pl, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, adv_norm, mask, eps=eps)
            vpred = response_values(value_model, seq, attn, pw, width)
            vl = value_mse_loss(vpred, returns, mask)
            loss = pl + float(cfg["value_coef"]) * vl
            popt.zero_grad(set_to_none=True)
            vopt.zero_grad(set_to_none=True)
            loss.backward()
            gnp = torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
            gnv = torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm)
            popt.step()
            vopt.step()
            with torch.no_grad():
                lr_ = new_logp.detach() - old_logp
                ep_stats.append({
                    "policy_loss": float(pl.item()),
                    "value_loss": float(vl.item()),
                    "clip_fraction": float(clip_frac.item()),
                    "clip_active_fraction": float(clipping_active_fraction(ratio, adv_norm, mask, eps).item()),
                    "ratio_max": float(ratio[mask.bool()].max().item()),
                    "ratio_min": float(ratio[mask.bool()].min().item()),
                    "approx_kl_old_new": float(masked_mean(-lr_, mask).item()),
                    "grad_norm_policy": float(gnp.item()),
                    "grad_norm_value": float(gnv.item()),
                })
            del new_logp, vpred, loss

        # Realised size of this update: re-evaluate the rollout under the updated policy.
        post_logp = response_logprobs(policy, seq, attn, width, grad=True)[0].detach()
        with torch.no_grad():
            post_ratio = torch.exp(post_logp - old_logp)
            post = {
                "post_update_approx_kl": float(masked_mean(old_logp - post_logp, mask).item()),
                "post_update_k3_kl": float(masked_mean(post_ratio - 1 - (post_logp - old_logp), mask).item()),
                "post_update_clip_fraction": float(masked_mean(((post_ratio < 1 - eps) | (post_ratio > 1 + eps)).float(), mask).item()),
                "post_update_max_abs_log_ratio": float((post_logp - old_logp).abs()[mask.bool()].max().item()),
            }

        n_tok = mask.sum(-1)
        rec = {
            "run": run_name, "update": u + 1, "clip_epsilon": eps, "kl_beta": beta_kl,
            "prompt_id": rows[0]["prompt_id"],
            "reward_raw": float(raw_reward.mean().item()),
            "reward": float(task_reward.mean().item()),
            "kl_token": float(sampled_kl(old_logp, ref_logp, mask).item()),
            "kl_seq": float(((old_logp - ref_logp) * mask).sum(-1).mean().item()),
            "kl_penalty_total": float((beta_kl * (old_logp - ref_logp) * mask).sum(-1).mean().item()),
            "entropy": float(masked_mean(ent, mask).item()),
            "sample_entropy": float(sample_entropy(old_logp, mask).item()),
            "response_length": float(n_tok.float().mean().item()),
            "terminated_with_eos": float(np.mean(gen["terminated_with_eos"])),
            "truncated": float(np.mean(gen["truncated"])),
            "value_mean": float(masked_mean(values, mask).item()),
            "return_mean": float(masked_mean(returns, mask).item()),
            "value_at_first_token": float(values[:, 0].mean().item()),
            "explained_variance": ev_before,
            "advantage_raw_mean": float(masked_mean(adv, mask).item()),
            "advantage_raw_std": float(adv[mask.bool()].std().item()) if mask.sum() > 1 else 0.0,
            # Per-update means over PPO epochs, plus the last-epoch value for the ratio-based stats.
            **{k: float(np.mean([e[k] for e in ep_stats])) for k in ep_stats[0]},
            "clip_fraction_last_epoch": ep_stats[-1]["clip_fraction"],
            "clip_active_fraction_last_epoch": ep_stats[-1]["clip_active_fraction"],
            "grad_norm_policy_max": float(max(e["grad_norm_policy"] for e in ep_stats)),
            "per_epoch": ep_stats,
            **post,
            "update_time_s": time.time() - t_update,
            "elapsed_s": time.time() - t0,
            "peak_vram_alloc_gib": peak_vram_gib()["allocated_gib"],
        }
        append_jsonl(log_path, rec)
        print(
            f"[{run_name}] update {u + 1}/{n_updates} R={rec['reward']:.3f} KL={rec['kl_token']:.4f} "
            f"H={rec['entropy']:.3f} L={rec['response_length']:.0f} pl={rec['policy_loss']:.4f} vl={rec['value_loss']:.3f} "
            f"clip={rec['clip_fraction']:.3f} gn={rec['grad_norm_policy']:.3f} EV={ev_before:.2f} t={rec['update_time_s']:.0f}s",
            flush=True,
        )
        del gen, seq, attn, old_logp, ref_logp, values, ent, rewards, adv, returns, adv_norm, post_logp
        torch.cuda.empty_cache()

    policy.save_pretrained(str(out))
    value_model.save_pretrained(str(out) + "_value")
    wall = time.time() - t0
    summary = {
        "run": run_name,
        "config": config_record,
        "updates": n_updates,
        "generated_tokens_total": total_gen_tokens,
        "wall_clock_s": wall,
        "wall_clock_excluding_thermal_pauses_s": wall - thermal_wait,
        "thermal_pause_s": thermal_wait,
        "peak_vram": peak_vram_gib(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "output": str(out),
    }
    save_json(results_dir / f"train_summary_{run_name}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
