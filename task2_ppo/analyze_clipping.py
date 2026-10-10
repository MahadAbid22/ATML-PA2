from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean
from common.models import load_policy, load_tokenizer, trainable_parameters
from common.rl_utils import disable_dropout, response_logprobs
from task2_ppo.forks import ensure_fork, ensure_midpoint_eval, heldout_row, stability_stats
from task2_ppo.ppo import clipping_active_fraction, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards

MINIBATCH = 4  # rollouts per optimizer step in the cached-batch replay


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def reconstruct_batch(rows, cfg, tokenizer):
    """Rebuild token sequences for the cached batch.

    Prompts are tokenised exactly as the release `batch_generate` does (chat template, then
    `truncation=True, max_length=max_prompt_length`), which reproduces the cached log-probs.
    Each rollout is kept as its own unpadded sequence.
    """
    prompts = {str(r["source_index"]): r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    batch = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(prompt_messages(prompts[str(row["source_index"])]), tokenize=False, add_generation_prompt=True)
        p_ids = tokenizer(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]))["input_ids"]
        r_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        if row.get("terminated_with_eos"):
            r_ids = r_ids + [tokenizer.eos_token_id]
        n = int(row["old_logprobs"].shape[0])
        batch.append({
            "row": row,
            "prompt_len": len(p_ids),
            "prompt_truncated": len(tokenizer(rendered)["input_ids"]) > len(p_ids),
            "retokenized_len": len(r_ids),
            "ids": torch.tensor([p_ids + r_ids[:n]]),
            "n": n,
        })
    return batch


def padded(batch, key, fill=0.0):
    width = max(b["n"] for b in batch)
    out = torch.full((len(batch), width), fill, dtype=torch.float32)
    for i, b in enumerate(batch):
        out[i, : b["n"]] = b["row"][key].float()
    return out


def cached_advantages(batch, cfg, beta_kl):
    width = max(b["n"] for b in batch)
    mask = torch.zeros(len(batch), width)
    for i, b in enumerate(batch):
        mask[i, : b["n"]] = 1.0
    old, ref, values = padded(batch, "old_logprobs"), padded(batch, "ref_logprobs"), padded(batch, "values")
    task_reward = torch.tensor([float(b["row"]["effective_terminal_reward"]) for b in batch])
    rewards = shaped_rewards(task_reward, old, ref, mask, beta_kl)
    adv, returns = compute_gae(rewards, values * mask, mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
    return old, ref, mask, adv, normalize_advantages(adv, mask), returns


def current_logprobs(policy, batch, grad=False):
    device = next(policy.parameters()).device
    width = max(b["n"] for b in batch)
    out = torch.zeros(len(batch), width, device=device)
    for i, b in enumerate(batch):
        ids = b["ids"].to(device)
        lp, _ = response_logprobs(policy, ids, torch.ones_like(ids), b["n"], grad=grad)
        out[i, : b["n"]] = lp[0]
    return out


def geometry(new_logp, old, adv, mask, eps):
    ratio = torch.exp(new_logp - old)
    loss, _, clip_frac = ppo_policy_loss(new_logp, old, adv, mask, eps=eps)
    unclipped = masked_mean(ratio * adv, mask)
    pos, neg = mask * (adv > 0), mask * (adv < 0)
    lr_ = (new_logp - old)[mask.bool()]
    return {
        "eps": eps,
        "clip_fraction": float(clip_frac),
        "affected_token_fraction": float(clipping_active_fraction(ratio, adv, mask, eps)),
        "clip_fraction_adv_pos": float(masked_mean(((ratio < 1 - eps) | (ratio > 1 + eps)).float(), pos)),
        "clip_fraction_adv_neg": float(masked_mean(((ratio < 1 - eps) | (ratio > 1 + eps)).float(), neg)),
        "clipped_surrogate": float(-loss),
        "unclipped_surrogate": float(unclipped),
        "surrogate_gap": float(unclipped + loss),
        "ratio_p01": float(torch.quantile(torch.exp(lr_), 0.01)),
        "ratio_p99": float(torch.quantile(torch.exp(lr_), 0.99)),
        "mean_abs_log_ratio": float(lr_.abs().mean()),
        "approx_kl_old_new": float(masked_mean(old - new_logp, mask)),
        "n_valid_tokens": int(mask.sum()),
    }


def replay_updates(policy, batch, old, adv, mask, cfg, eps, init_state):
    """Run the release PPO policy update (ppo_epochs passes, minibatches of 4 rollouts) on the
    fixed cached batch from the identical midpoint weights, recording the clipping geometry
    before every optimizer step."""
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if n in init_state:
                p.copy_(init_state[n])
    params = trainable_parameters(policy)
    opt = AdamW(params, lr=float(cfg["policy_learning_rate"]))
    device = next(policy.parameters()).device
    old, adv, mask = old.to(device), adv.to(device), mask.to(device)
    log, step = [], 0
    for epoch in range(int(cfg["ppo_epochs"])):
        for start in range(0, len(batch), MINIBATCH):
            idx = list(range(start, min(start + MINIBATCH, len(batch))))
            n_mb = float(mask[idx].sum())
            opt.zero_grad(set_to_none=True)
            stats = []
            for i in idx:
                b = batch[i]
                ids = b["ids"].to(device)
                lp, _ = response_logprobs(policy, ids, torch.ones_like(ids), b["n"], grad=True)
                o, a, m = old[i : i + 1, : b["n"]], adv[i : i + 1, : b["n"]], mask[i : i + 1, : b["n"]]
                loss_i, ratio_i, _ = ppo_policy_loss(lp, o, a, m, eps=eps)
                # masked_mean over the minibatch == sum_i (n_i / N) * masked_mean_i
                (loss_i * (b["n"] / n_mb)).backward()
                stats.append((ratio_i, a, m, loss_i.detach() * (b["n"] / n_mb)))
            ratio = torch.cat([s[0].flatten() for s in stats]); a = torch.cat([s[1].flatten() for s in stats]); m = torch.cat([s[2].flatten() for s in stats])
            gn = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
            opt.step()
            step += 1
            log.append({
                "eps": eps, "epoch": epoch, "step": step,
                "clip_fraction": float(masked_mean(((ratio < 1 - eps) | (ratio > 1 + eps)).float(), m)),
                "affected_token_fraction": float(clipping_active_fraction(ratio, a, m, eps)),
                "clipped_surrogate": float(-sum(s[3] for s in stats)),
                "max_abs_log_ratio": float(ratio[m.bool()].log().abs().max()),
                "grad_norm": float(gn),
            })
    with torch.no_grad():
        final = geometry(current_logprobs(policy, batch), old, adv, mask, eps)
    return log, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-forks", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = repo_path(cfg["results_dir"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))

    static_path = out_dir / "clipping_cached_static.csv"
    if not static_path.exists():
        set_seed(int(cfg["seed"]))
        tok = load_tokenizer(cfg["base_model"])
        batch = reconstruct_batch(rows, cfg, tok)
        recon = [{"source_index": b["row"]["source_index"], "prompt_len": b["prompt_len"], "prompt_truncated": b["prompt_truncated"],
                  "cached_tokens": b["n"], "retokenized_tokens": b["retokenized_len"]} for b in batch]
        policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
        disable_dropout(policy)
        old, ref, mask, adv_raw, adv, returns = cached_advantages(batch, cfg, float(cfg["kl_beta"]))
        with torch.no_grad():
            new = current_logprobs(policy, batch).cpu()
        # Fidelity of the reconstruction: midpoint log-probs vs cached old log-probs, per rollout.
        for i, b in enumerate(batch):
            d = (new[i, : b["n"]] - old[i, : b["n"]]).abs()
            recon[i].update({"mean_abs_logp_diff": float(d.mean()), "max_abs_logp_diff": float(d.max())})
        write_jsonl(out_dir / "clipping_cached_reconstruction.jsonl", recon)

        # (a) Static geometry: ratios of the supplied midpoint vs the cached behaviour policy.
        static = [geometry(new, old, adv, mask, float(e)) for e in cfg["clip_values"]]
        pd.DataFrame(static).to_csv(static_path, index=False)
        print(pd.DataFrame(static).to_string(index=False))
        save_json(out_dir / "clipping_cached_batch_stats.json", {
            "n_rollouts": len(batch), "n_valid_tokens": int(mask.sum()),
            "n_prompts_truncated_by_release_tokenizer": int(sum(r["prompt_truncated"] for r in recon)),
            "mean_effective_reward": float(np.mean([float(r["effective_terminal_reward"]) for r in rows])),
            "advantage_raw_mean": float(masked_mean(adv_raw, mask)), "advantage_raw_std": float(adv_raw[mask.bool()].std()),
            "returns_mean": float(masked_mean(returns, mask)),
            "value_mean": float(masked_mean(padded(batch, "values"), mask)),
            "kl_beta_used_for_shaping": float(cfg["kl_beta"]),
        })

        # (b) Dynamic geometry: identical PPO policy updates on the fixed batch for each eps.
        init_state = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
        all_logs, finals = [], []
        for e in cfg["clip_values"]:
            log, final = replay_updates(policy, batch, old, adv, mask, cfg, float(e), init_state)
            all_logs += log
            finals.append(final)
            print(f"eps={e}: after {len(log)} steps -> {json.dumps(final)}", flush=True)
        write_jsonl(out_dir / "clipping_cached_replay_log.jsonl", all_logs)
        pd.DataFrame(finals).to_csv(out_dir / "clipping_cached_after_replay.csv", index=False)
        del policy
        torch.cuda.empty_cache()

    if args.skip_forks:
        return

    # (c) Matched short continuation forks from the same checkpoint (identical reward/KL settings).
    ensure_midpoint_eval(args.config, cfg)
    beta = float(cfg["kl_beta"])
    static = pd.read_csv(static_path).set_index("eps")
    after = pd.read_csv(out_dir / "clipping_cached_after_replay.csv").set_index("eps")
    table = []
    for e in cfg["clip_values"]:
        name = ensure_fork(args.config, cfg, float(e), beta)
        row = {"eps": float(e), "fork": name,
               "cached_clip_fraction": static.loc[float(e), "clip_fraction"],
               "cached_affected_token_fraction": static.loc[float(e), "affected_token_fraction"],
               "cached_clipped_surrogate": static.loc[float(e), "clipped_surrogate"],
               "replay_clip_fraction_final": after.loc[float(e), "clip_fraction"],
               "replay_affected_fraction_final": after.loc[float(e), "affected_token_fraction"],
               "replay_approx_kl_final": after.loc[float(e), "approx_kl_old_new"]}
        row.update(heldout_row(cfg, name))
        row.update(stability_stats(cfg, name))
        table.append(row)
    df = pd.DataFrame(table)
    df.to_csv(out_dir / "clipping_study_table.csv", index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
