"""Shared, memory-efficient policy utilities used by Tasks 1-5.

The 8 GB VRAM budget does not allow materialising full [batch, length, 151936] fp32 logit
tensors (and their log-softmax copies) for several 768-token sequences at once. The helpers
here run the transformer backbone once, then apply the (frozen, tied) LM head chunk-by-chunk.
When gradients are needed each chunk is wrapped in activation checkpointing, so only the
hidden states are kept for backward and the logits are recomputed one chunk at a time.

Numerically this is identical to `common.generation.response_token_logprobs`
(log_softmax(logits.float()) gathered at the realised tokens); `scripts/smoke_tests.py`
checks the two against each other.
"""
from __future__ import annotations

import subprocess
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.checkpoint import checkpoint

from common.models import reference_mode


# --------------------------------------------------------------------------------------
# Model plumbing
# --------------------------------------------------------------------------------------

def _backbone_and_head(model):
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    return base.model, base.lm_head


def disable_dropout(model) -> None:
    """Set every dropout probability to zero.

    Used for on-policy RL (PPO/GRPO): with dropout active, the log-probs recomputed for the
    loss would differ from the rollout log-probs even before any update, so the PPO ratio
    would not start at exactly 1. This mirrors standard RLHF practice (e.g. TRL's
    `disable_dropout=True`).
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Same position ids that `generate` uses for left-padded batches."""
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 1)


def _chunk_logp(hidden, weight, labels, temperature: float):
    logits = F.linear(hidden, weight).float()
    if temperature != 1.0:
        logits = logits / temperature
    lse = torch.logsumexp(logits, dim=-1)
    return logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - lse


@torch.no_grad()
def _chunk_entropy(hidden, weight, temperature: float):
    logits = F.linear(hidden, weight).float()
    if temperature != 1.0:
        logits = logits / temperature
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1)


def tail_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_tail: int,
    *,
    grad: bool,
    entropy: bool = False,
    temperature: float = 1.0,
    chunk_size: int = 128,
):
    """log p(x_t | x_<t) for the last `n_tail` positions of every row.

    Returns (logp [B, n_tail], entropy [B, n_tail] or None). With left-padded prompts and
    right-padded responses (the layout produced by `batch_generate` and by `pad_batch`),
    the last `n_tail` positions are exactly the response slots.
    """
    backbone, head = _backbone_and_head(model)
    if grad:
        # Tensors produced under torch.inference_mode (e.g. by batch_generate) cannot be saved
        # for backward; a clone made outside inference mode is an ordinary tensor.
        input_ids, attention_mask = input_ids.clone(), attention_mask.clone()
    pos = position_ids_from_mask(attention_mask)
    length = input_ids.shape[1]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        hidden = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        h = hidden[:, length - n_tail - 1 : length - 1]
        labels = input_ids[:, length - n_tail :]
        logps, ents = [], []
        for s in range(0, n_tail, chunk_size):
            hc, lc = h[:, s : s + chunk_size], labels[:, s : s + chunk_size]
            if grad:
                lp = checkpoint(_chunk_logp, hc, head.weight, lc, temperature, use_reentrant=False)
            else:
                lp = _chunk_logp(hc, head.weight, lc, temperature)
            logps.append(lp)
            if entropy:
                ents.append(_chunk_entropy(hc.detach(), head.weight, temperature))
    logp = torch.cat(logps, dim=1)
    ent = torch.cat(ents, dim=1) if entropy else None
    return logp, ent


def response_logprobs(model, sequences, attention_mask, response_width: int, *, grad: bool, entropy: bool = False, temperature: float = 1.0):
    """Log-probs of generated response tokens for `batch_generate` outputs."""
    return tail_token_logprobs(
        model, sequences, attention_mask, response_width, grad=grad, entropy=entropy, temperature=temperature
    )


def reference_response_logprobs(model, sequences, attention_mask, response_width: int, temperature: float = 1.0):
    """Same as `response_logprobs` but with the LoRA adapter disabled (frozen reference)."""
    with reference_mode(model):
        logp, _ = tail_token_logprobs(
            model, sequences, attention_mask, response_width, grad=False, temperature=temperature
        )
    return logp


def pair_batch_logprobs(model, batch: dict, *, grad: bool):
    """Summed response log-probs for a left-padded DPO batch from `common.data.pad_batch`.

    Returns (sequence_logp [B], token_logp [B, R], mask [B, R]).
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    rmask = batch["response_mask"].to(device)
    # Responses sit at the right edge of each left-padded row.
    n_tail = int(rmask.sum(-1).max().item())
    tok, _ = tail_token_logprobs(model, input_ids, attn, n_tail, grad=grad)
    mask = rmask[:, -n_tail:]
    return (tok * mask).sum(-1), tok, mask


def left_pad_to(batch: dict, width: int, pad_id: int) -> dict:
    """Left-pad a `pad_batch` output to a common width so chosen/rejected share one forward."""
    cur = batch["input_ids"].shape[1]
    if cur == width:
        return batch
    extra = width - cur
    b = batch["input_ids"].shape[0]
    return {
        "input_ids": torch.cat([torch.full((b, extra), pad_id, dtype=torch.long), batch["input_ids"]], 1),
        "attention_mask": torch.cat([torch.zeros(b, extra, dtype=torch.long), batch["attention_mask"]], 1),
        "response_mask": torch.cat([torch.zeros(b, extra), batch["response_mask"]], 1),
    }


def concat_pair_batches(chosen: dict, rejected: dict, pad_id: int) -> dict:
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
    c, r = left_pad_to(chosen, width, pad_id), left_pad_to(rejected, width, pad_id)
    return {k: torch.cat([c[k], r[k]], 0) for k in c}


# --------------------------------------------------------------------------------------
# Prompt filtering
# --------------------------------------------------------------------------------------

def rendered_prompt_length(tokenizer, messages) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def filter_rows_by_prompt_length(rows, tokenizer, max_prompt_length: int, messages_fn):
    """Keep rows whose rendered chat prompt fits `max_prompt_length`.

    `batch_generate` truncates over-long prompts from the right, which would delete the
    assistant header; we therefore exclude such prompts instead and record the IDs.
    """
    kept, dropped = [], []
    for row in rows:
        n = rendered_prompt_length(tokenizer, messages_fn(row))
        (kept if n <= max_prompt_length else dropped).append(row)
    return kept, dropped


# --------------------------------------------------------------------------------------
# Hardware safety
# --------------------------------------------------------------------------------------

def gpu_temperature():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def cooldown_if_hot(max_temp: int | None = None, resume_temp: int | None = None, poll_s: int = 15) -> float:
    """Pause work while the GPU is hot (laptop default 84C; override with PA2_THERMAL_MAX). Returns seconds waited."""
    max_temp = int(os.environ.get("PA2_THERMAL_MAX", 84)) if max_temp is None else max_temp
    resume_temp = max_temp - 10 if resume_temp is None else resume_temp
    t = gpu_temperature()
    if t is None or t < max_temp:
        return 0.0
    start = time.time()
    print(f"[thermal] GPU at {t}C >= {max_temp}C, pausing until <= {resume_temp}C", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    while t is not None and t > resume_temp:
        time.sleep(poll_s)
        t = gpu_temperature()
    waited = time.time() - start
    print(f"[thermal] resumed at {t}C after {waited:.0f}s", flush=True)
    return waited


# --------------------------------------------------------------------------------------
# Small statistics helpers
# --------------------------------------------------------------------------------------

def describe(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def peak_vram_gib() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0}
    return {
        "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
