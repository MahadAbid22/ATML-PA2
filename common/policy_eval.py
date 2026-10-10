"""Common held-out generation protocol shared by Tasks 1-3.

For every condition we:
  1. sample one response per fixed prompt with the release decoding config,
     re-seeding before each batch so every condition sees the same RNG stream;
  2. score it with the frozen course reward model;
  3. compute the sampled-response KL estimator against the frozen reference
     (LoRA disabled), using the course helper convention `sampled_kl` (token-pooled mean),
     and also report the per-sequence sum for completeness;
  4. record full-distribution token entropy and response length.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.rl_utils import (
    cooldown_if_hot,
    describe,
    reference_response_logprobs,
    rendered_prompt_length,
    response_logprobs,
)


@torch.no_grad()
def generate_score_records(
    policy,
    tokenizer,
    reward_model,
    reward_tokenizer,
    rows: list[dict],
    messages_fn,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    seed: int,
    batch_size: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    reward_max_length: int = 1280,
    missing_eos_penalty: float | None = None,
    has_adapter: bool = True,
    log_prefix: str = "",
):
    """Generate, score and measure drift for a list of prompt rows. Returns per-row records."""
    # Sort by prompt length so batches have little padding; the order is deterministic.
    order = sorted(range(len(rows)), key=lambda i: rendered_prompt_length(tokenizer, messages_fn(rows[i])))
    records = [None] * len(rows)
    t0 = time.time()
    for bi, start in enumerate(range(0, len(order), batch_size)):
        cooldown_if_hot()
        idx = order[start : start + batch_size]
        prompts = [messages_fn(rows[i]) for i in idx]
        set_seed(seed + bi)
        gen = batch_generate(
            policy, tokenizer, prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
        )
        width = gen["response_ids"].shape[1]
        mask = gen["response_mask"]
        logp, ent = response_logprobs(policy, gen["sequences"], gen["attention_mask"], width, grad=False, entropy=True)
        if has_adapter:
            ref = reference_response_logprobs(policy, gen["sequences"], gen["attention_mask"], width)
        else:
            ref = logp.clone()
        rewards = score_reward_pairs(reward_model, reward_tokenizer, prompts, gen["responses"], max_length=reward_max_length)
        diff = (logp - ref) * mask
        for j, i in enumerate(idx):
            n = int(mask[j].sum().item())
            raw = float(rewards[j].item())
            eff = raw
            if missing_eos_penalty is not None and not gen["terminated_with_eos"][j]:
                eff = raw - float(missing_eos_penalty)
            row = rows[i]
            records[i] = {
                "prompt_id": row.get("prompt_id"),
                "source_index": str(row.get("source_index")),
                "prompt": messages_fn(row)[-1]["content"],
                "response": gen["responses"][j],
                "response_tokens": n,
                "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                "truncated": bool(gen["truncated"][j]),
                "reward": raw,
                "effective_reward": eff,
                "kl_sum": float(diff[j].sum().item()),
                "logp_diff_sum": float(diff[j].sum().item()),
                "entropy_sum": float((ent[j] * mask[j]).sum().item()),
                "sampled_logp_sum": float((logp[j] * mask[j]).sum().item()),
            }
        done = min(start + batch_size, len(order))
        print(f"{log_prefix}generated {done}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
        del gen, logp, ent, ref, rewards, diff
        torch.cuda.empty_cache()
    return records


def aggregate_records(records: list[dict]) -> dict:
    """Aggregate per-row records with the conventions used everywhere in this repo."""
    n_tok = np.array([r["response_tokens"] for r in records], dtype=float)
    kl_sum = np.array([r["kl_sum"] for r in records], dtype=float)
    ent_sum = np.array([r["entropy_sum"] for r in records], dtype=float)
    slp_sum = np.array([r["sampled_logp_sum"] for r in records], dtype=float)
    total = max(n_tok.sum(), 1.0)
    # Token-pooled KL == common.metrics.sampled_kl applied to all evaluation tokens at once.
    kl_tok = torch.tensor(kl_sum.sum() / total).item()
    return {
        "n_prompts": len(records),
        "reward_mean": float(np.mean([r["reward"] for r in records])),
        "reward_std": float(np.std([r["reward"] for r in records], ddof=1)),
        "effective_reward_mean": float(np.mean([r["effective_reward"] for r in records])),
        "kl_token_mean": float(kl_tok),
        "kl_seq_mean": float(kl_sum.mean()),
        "entropy_token_mean": float(ent_sum.sum() / total),
        "sample_entropy_token_mean": float(-slp_sum.sum() / total),
        "length": describe(n_tok),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
    }


# Re-export the course helpers so callers can verify the conventions in one place.
__all__ = [
    "generate_score_records",
    "aggregate_records",
    "masked_mean",
    "sampled_kl",
    "sample_entropy",
]
