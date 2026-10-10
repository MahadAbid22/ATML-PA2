from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters
from common.rl_utils import (
    cooldown_if_hot,
    disable_dropout,
    filter_rows_by_prompt_length,
    peak_vram_gib,
    reference_response_logprobs,
    response_logprobs,
)
from task3_grpo.grpo import group_relative_advantages, group_reward_std, grpo_policy_loss, mask_truncated_sequences

ZERO_STD_TOL = 1e-6  # same tolerance as the eps in group_relative_advantages


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def selected_prompt_rows(cfg, tokenizer, n: int):
    rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    kept, _ = filter_rows_by_prompt_length(rows, tokenizer, int(cfg["max_prompt_length"]), prompt_messages)
    order = np.random.default_rng(int(cfg["seed"])).permutation(len(kept))
    return [kept[i] for i in order[:n]]


def k3_kl_sum(new_logp, ref_logp, mask):
    """Sum over tokens of the k3 estimator used inside grpo_policy_loss."""
    lr_ = ref_logp - new_logp
    return ((torch.exp(lr_) - lr_ - 1.0) * mask).sum()


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().float() for p in params])


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, policy, opt = bundle["tokenizer"], bundle["policy"], bundle["optimizer"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    disable_dropout(policy)
    params = trainable_parameters(policy)

    n_updates = int(cfg["updates"])
    ppu = int(cfg["prompts_per_update"])
    K = int(cfg["num_generations"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    lmax = int(cfg["max_completion_length"])
    seed = int(cfg["seed"])
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_log_{run_name}.jsonl"
    seq_path = results_dir / f"train_sequences_{run_name}.jsonl"
    for p in (log_path, seq_path):
        if p.exists():
            p.unlink()

    prompt_rows = selected_prompt_rows(cfg, tok, n_updates * ppu)
    save_json(results_dir / f"train_prompts_{run_name}.json", [
        {"update": i // ppu, "prompt_id": r["prompt_id"], "source_index": str(r["source_index"])} for i, r in enumerate(prompt_rows)
    ])
    config_record = {k: cfg[k] for k in [
        "updates", "prompts_per_update", "policy_epochs", "learning_rate", "clip_epsilon", "kl_beta",
        "num_generations", "max_prompt_length", "max_completion_length", "mask_truncated_completions",
        "max_grad_norm", "seed"]}
    config_record.update({"loss_type": loss_type, "generation": cfg["generation"]})
    print(f"[{run_name}] GRPO continuation: {json.dumps(config_record)}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    thermal_wait = 0.0
    total_gen_tokens = 0
    n_uninformative = 0

    for u in range(n_updates):
        thermal_wait += cooldown_if_hot()
        t_update = time.time()
        rows = prompt_rows[u * ppu : (u + 1) * ppu]
        msgs = [prompt_messages(r) for r in rows for _ in range(K)]
        group_ids = torch.tensor([g for g in range(len(rows)) for _ in range(K)])

        set_seed(seed + u)  # identical sampling stream across forks
        gen = batch_generate(
            policy, tok, msgs,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=lmax,
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=True,
        )
        seq, attn = gen["sequences"], gen["attention_mask"]
        width = gen["response_ids"].shape[1]
        resp_mask = gen["response_mask"]
        total_gen_tokens += int(resp_mask.sum().item())
        token_mask = mask_truncated_sequences(resp_mask, gen["truncated"]) if cfg.get("mask_truncated_completions", True) else resp_mask
        device = seq.device

        # Old log-probs via the same grad-mode kernel path as the loss (then detached), one sequence
        # at a time, so ratio == 1 exactly before the update (see task2_ppo.continue_train).
        olds, ents = [], []
        for i in range(seq.shape[0]):
            lp_i, ent_i = response_logprobs(policy, seq[i : i + 1], attn[i : i + 1], width, grad=True, entropy=True)
            olds.append(lp_i.detach()); ents.append(ent_i)
        old_logp, ent = torch.cat(olds), torch.cat(ents)
        with torch.no_grad():
            ref_logp = reference_response_logprobs(policy, seq, attn, width)
            rewards = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=int(cfg.get("reward_max_length", 1280))).float()
            adv = group_relative_advantages(rewards, group_ids.to(rewards.device)).to(device)
            g_std = group_reward_std(rewards, group_ids.to(rewards.device))
        uninformative = (g_std <= ZERO_STD_TOL).float()
        n_uninformative += int(uninformative.sum().item())

        # ---- Exact per-sequence decomposition of the batch GRPO loss (memory: one sequence at a time)
        # loss = -(1/B) sum_i per_seq_i + beta * sum_i sum_t kl_it m_it / N_total
        B = seq.shape[0]
        n_total = float(token_mask.sum().clamp_min(1.0))
        opt.zero_grad(set_to_none=True)
        prev = flat_grad(params)
        per_seq = []
        loss_total, policy_term_total, kl_total = 0.0, 0.0, 0.0
        clip_num, ratio_num = 0.0, 0.0
        for i in range(B):
            n_i = float(token_mask[i].sum())
            rec_i = {"update": u + 1, "i": i, "reward": float(rewards[i]), "advantage": float(adv[i]),
                     "length": int(resp_mask[i].sum()), "truncated": bool(gen["truncated"][i]), "masked": n_i == 0}
            if n_i == 0:
                rec_i.update({"grad_norm": 0.0, "token_weight": 0.0})
                per_seq.append(rec_i)
                continue
            new_i, _ = response_logprobs(policy, seq[i : i + 1], attn[i : i + 1], width, grad=True)
            m_i = token_mask[i : i + 1]
            pol_i, st_i = grpo_policy_loss(new_i, old_logp[i : i + 1], adv[i : i + 1], m_i, ref_logp[i : i + 1], eps, 0.0,
                                           loss_type=loss_type, max_completion_length=lmax)
            kl_i = k3_kl_sum(new_i, ref_logp[i : i + 1], m_i)
            l_i = pol_i / B + beta * kl_i / n_total
            l_i.backward()
            cur = flat_grad(params)
            gnorm_i = float((cur - prev).norm())
            prev = cur
            loss_total += float(l_i)
            policy_term_total += float(pol_i) / B
            kl_total += float(kl_i)
            clip_num += float(st_i["clip_fraction"]) * n_i
            ratio_num += float(st_i["ratio_mean"]) * n_i
            # Weight each token of this completion receives in the policy term.
            denom = n_i if loss_type == "grpo" else float(lmax)
            rec_i.update({"grad_norm": gnorm_i, "token_weight": abs(float(adv[i])) / (B * denom),
                          "sequence_weight": abs(float(adv[i])) * n_i / (B * denom)})
            per_seq.append(rec_i)
            del new_i, l_i

        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
        opt.step()
        opt.zero_grad(set_to_none=True)
        for r in per_seq:
            r["run"] = run_name
            append_jsonl(seq_path, r)

        valid = resp_mask
        rec = {
            "run": run_name, "update": u + 1, "loss_type": loss_type, "prompt_id": rows[0]["prompt_id"],
            "reward": float(rewards.mean()),
            "reward_group_std": float(g_std.mean()),
            "uninformative_group_fraction": float(uninformative.mean()),
            "running_uninformative_fraction": n_uninformative / ((u + 1) * len(rows)),
            "kl_token": float(sampled_kl(old_logp, ref_logp, valid)),
            "kl_k3_loss": kl_total / n_total,
            "kl_seq": float(((old_logp - ref_logp) * valid).sum(-1).mean()),
            "loss": loss_total,
            "policy_loss": policy_term_total,
            "grad_norm": float(grad_norm),
            "entropy": float(masked_mean(ent, valid)),
            "sample_entropy": float(sample_entropy(old_logp, valid)),
            "response_length": float(valid.sum(-1).float().mean()),
            "response_length_std": float(valid.sum(-1).float().std()),
            "n_truncated_masked": int(sum(gen["truncated"])),
            "clip_fraction": clip_num / n_total,
            "ratio_mean": ratio_num / n_total,
            "advantage_abs_mean": float(adv.abs().mean()),
            "update_time_s": time.time() - t_update,
            "elapsed_s": time.time() - t0,
            "peak_vram_alloc_gib": peak_vram_gib()["allocated_gib"],
        }
        append_jsonl(log_path, rec)
        print(
            f"[{run_name}] update {u + 1}/{n_updates} R={rec['reward']:.3f} std={rec['reward_group_std']:.3f} "
            f"KL={rec['kl_token']:.4f} H={rec['entropy']:.3f} L={rec['response_length']:.0f} trunc={rec['n_truncated_masked']} "
            f"loss={rec['loss']:.4f} gn={rec['grad_norm']:.3f} t={rec['update_time_s']:.0f}s",
            flush=True,
        )
        del gen, seq, attn, old_logp, ref_logp, ent, adv
        torch.cuda.empty_cache()

    policy.save_pretrained(str(out))
    wall = time.time() - t0
    summary = {
        "run": run_name, "config": config_record, "updates": n_updates,
        "generated_tokens_total": total_gen_tokens,
        "uninformative_group_fraction": n_uninformative / (n_updates * ppu),
        "wall_clock_s": wall, "thermal_pause_s": thermal_wait,
        "wall_clock_excluding_thermal_pauses_s": wall - thermal_wait,
        "peak_vram": peak_vram_gib(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "output": str(out),
    }
    save_json(results_dir / f"train_summary_{run_name}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
