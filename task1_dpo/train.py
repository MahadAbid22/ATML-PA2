from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.rl_utils import concat_pair_batches, cooldown_if_hot, pair_batch_logprobs, peak_vram_gib
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def row_key(row: dict) -> str:
    return f"{row.get('source_split')}:{row.get('source_index')}:{row.get('prompt_id')}"


def pair_token_lengths(row, tokenizer):
    """(prompt, chosen, rejected) token counts exactly as `encode_prompt_response` builds them
    (chat-templated prompt with generation header; response tokens + 1 EOS)."""
    prompt_n = len(tokenizer.apply_chat_template(prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
    yc, yr = preference_responses(row)
    eos = 1 if tokenizer.eos_token_id is not None else 0
    c = len(tokenizer(yc, add_special_tokens=False)["input_ids"]) + eos
    r = len(tokenizer(yr, add_special_tokens=False)["input_ids"]) + eos
    return prompt_n, c, r


def filter_fitting_rows(rows, tokenizer, max_length):
    """Overlength filter (TA-approved option): keep a pair only if BOTH prompt+chosen and
    prompt+rejected (incl. EOS) fit in `max_length` tokens, so no prompt or response token is
    ever truncated. Applied identically to every DPO training and evaluation file; dropped keys
    are logged so the length-related experiments can account for the filter.
    """
    kept, dropped = [], []
    for row in rows:
        p, c, r = pair_token_lengths(row, tokenizer)
        (kept if p + max(c, r) <= max_length else dropped).append(row)
    return kept, dropped


def drop_breakdown(dropped) -> dict:
    """Count dropped pairs per length stratum (only the length-controlled files have strata)."""
    out = {}
    for row in dropped:
        k = str(row.get("length_stratum"))
        out[k] = out.get(k, 0) + 1
    return out


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped = filter_fitting_rows(rows, tokenizer, int(cfg["max_sequence_length"]))
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped": dropped,
        "dataset_path": path,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, loader, optimizer = bundle["model"], bundle["tokenizer"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_grad_norm = float(cfg["max_grad_norm"])
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_log_{run_name}.jsonl"
    if log_path.exists():
        log_path.unlink()

    # Record the exact data indices used (requirement: preserve fixed indices).
    save_json(results_dir / f"train_ids_{run_name}.json", {
        "dataset": str(bundle["dataset_path"]),
        "n_used": len(bundle["rows"]),
        "used_keys": [row_key(r) for r in bundle["rows"]],
        "filter_rule": "keep iff prompt + max(chosen, rejected) + EOS <= max_sequence_length",
        "n_dropped_overlength": len(bundle["dropped"]),
        "dropped_by_stratum": drop_breakdown(bundle["dropped"]),
        "dropped_overlength": [row_key(r) for r in bundle["dropped"]],
    })

    pad_id = tokenizer.pad_token_id
    n_micro = len(loader) * epochs
    n_steps = math.ceil(len(loader) / accum) * epochs
    print(f"[{run_name}] beta={beta} pairs={len(bundle['rows'])} micro-batches={n_micro} optimizer steps={n_steps}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window = []  # micro-batch diagnostics inside the current accumulation window

    for epoch in range(epochs):
        for mi, (chosen, rejected) in enumerate(loader):
            b = chosen["input_ids"].shape[0]
            batch = concat_pair_batches(chosen, rejected, pad_id)
            # Frozen reference: same weights with the LoRA adapter disabled.
            with torch.no_grad(), reference_mode(model):
                ref_logp, _, _ = pair_batch_logprobs(model, batch, grad=False)
            pol_logp, _, _ = pair_batch_logprobs(model, batch, grad=True)
            loss, stats = dpo_loss(pol_logp[:b], pol_logp[b:], ref_logp[:b], ref_logp[b:], beta)

            # Size of the accumulation window this micro-batch belongs to (the last one may be short).
            window_start = (mi // accum) * accum
            window_size = min(accum, len(loader) - window_start)
            (loss / window_size).backward()

            chosen_reward = beta * (pol_logp[:b] - ref_logp[:b]).detach()
            rejected_reward = beta * (pol_logp[b:] - ref_logp[b:]).detach()
            window.append({
                "loss": float(loss.item()),
                "acc": float(stats["preference_accuracy"].item()),
                "margin": float((chosen_reward - rejected_reward).mean().item()),
                "chosen_reward": float(chosen_reward.mean().item()),
                "rejected_reward": float(rejected_reward.mean().item()),
                "policy_chosen_logp": float(pol_logp[:b].mean().item()),
                "policy_rejected_logp": float(pol_logp[b:].mean().item()),
                "n": b,
            })

            if len(window) == window_size:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                tot = sum(w["n"] for w in window)
                rec = {k: sum(w[k] * w["n"] for w in window) / tot for k in window[0] if k != "n"}
                rec.update({
                    "run": run_name, "beta": beta, "epoch": epoch, "step": step,
                    "pairs_seen": (mi + 1) * int(cfg["batch_size"]) if epoch == 0 else None,
                    "grad_norm": float(grad_norm.item()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_s": time.time() - t0,
                })
                append_jsonl(log_path, rec)
                if step % 5 == 0 or step == n_steps:
                    print(
                        f"[{run_name}] step {step}/{n_steps} loss={rec['loss']:.4f} acc={rec['acc']:.3f} "
                        f"margin={rec['margin']:.3f} gnorm={rec['grad_norm']:.3f} t={rec['elapsed_s']:.0f}s",
                        flush=True,
                    )
                window = []
                cooldown_if_hot()
            del batch, ref_logp, pol_logp, loss

    model.save_pretrained(str(output))
    summary = {
        "run": run_name,
        "beta": beta,
        "dataset": str(bundle["dataset_path"]),
        "n_pairs": len(bundle["rows"]),
        "epochs": epochs,
        "optimizer_steps": step,
        "effective_batch_pairs": int(cfg["batch_size"]) * accum,
        "learning_rate": float(cfg["learning_rate"]),
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "lora": cfg["lora"],
        "seed": int(cfg["seed"]),
        "wall_clock_s": time.time() - t0,
        "peak_vram": peak_vram_gib(),
        "output": str(output),
    }
    save_json(results_dir / f"train_summary_{run_name}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
