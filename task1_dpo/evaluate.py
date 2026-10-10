from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from common.policy_eval import aggregate_records, generate_score_records
from common.rl_utils import concat_pair_batches, describe, filter_rows_by_prompt_length, pair_batch_logprobs
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import drop_breakdown, filter_fitting_rows, row_key

# Generation protocol for Task 1 (identical for every condition).
GEN_MAX_PROMPT_TOKENS = 512
WORD_LIMIT_SAMPLES = 5


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def pair_logprob_records(model, tokenizer, rows, max_length: int, has_adapter: bool, batch_pairs: int = 4):
    """Summed response log-probs of chosen/rejected under the policy and the frozen reference."""
    out = []
    for start in range(0, len(rows), batch_pairs):
        chunk = rows[start : start + batch_pairs]
        ch, rj = [], []
        for row in chunk:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            ch.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rj.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        b = len(chunk)
        batch = concat_pair_batches(pad_batch(tokenizer, ch), pad_batch(tokenizer, rj), tokenizer.pad_token_id)
        pol, _, mask = pair_batch_logprobs(model, batch, grad=False)
        if has_adapter:
            with reference_mode(model):
                ref, _, _ = pair_batch_logprobs(model, batch, grad=False)
        else:
            ref = pol.clone()
        ntok = mask.sum(-1)
        for i, row in enumerate(chunk):
            out.append({
                "key": row_key(row),
                "prompt_id": row.get("prompt_id"),
                "length_stratum": row.get("length_stratum"),
                "chosen_tokens_trunc": int(ntok[i].item()),
                "rejected_tokens_trunc": int(ntok[b + i].item()),
                "policy_chosen": float(pol[i]), "policy_rejected": float(pol[b + i]),
                "ref_chosen": float(ref[i]), "ref_rejected": float(ref[b + i]),
            })
        del batch, pol, ref
    for r in out:
        r["margin"] = (r["policy_chosen"] - r["ref_chosen"]) - (r["policy_rejected"] - r["ref_rejected"])
    return out


def pair_summary(records, beta: float) -> dict:
    m = torch.tensor([r["margin"] for r in records], dtype=torch.float64)
    pc = torch.tensor([r["policy_chosen"] for r in records]); pr = torch.tensor([r["policy_rejected"] for r in records])
    rc = torch.tensor([r["ref_chosen"] for r in records]); rr = torch.tensor([r["ref_rejected"] for r in records])
    loss, stats = dpo_loss(pc, pr, rc, rr, beta)
    loss_b01, _ = dpo_loss(pc, pr, rc, rr, 0.10)
    return {
        "n_pairs": len(records),
        "preference_accuracy": float((m > 0).double().mean()),
        "tie_fraction": float((m == 0).double().mean()),
        "dpo_loss_own_beta": float(loss),
        "dpo_loss_beta_0.10": float(loss_b01),
        "margin_mean": float(m.mean()),
        "margin_median": float(m.median()),
        "chosen_logratio_mean": float((pc - rc).mean()),
        "rejected_logratio_mean": float((pr - rr).mean()),
        "check_preference_accuracy_from_dpo_loss": float(stats["preference_accuracy"]),
    }


@torch.no_grad()
def word_limit_records(model, tokenizer, rows, cfg, reward_model=None, reward_tokenizer=None):
    recs = []
    prompts = [prompt_messages(r) for r in rows]
    for s in range(WORD_LIMIT_SAMPLES):
        set_seed(int(cfg["seed"]) + 1000 + s)
        gen = batch_generate(
            model, tokenizer, prompts,
            max_prompt_length=GEN_MAX_PROMPT_TOKENS,
            max_new_tokens=int(cfg["max_generation_tokens"]),
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=True,
        )
        rm_scores = [None] * len(rows)
        if reward_model is not None:
            rm_scores = score_reward_pairs(reward_model, reward_tokenizer, prompts, gen["responses"]).tolist()
        for row, msg, text, n, rm in zip(rows, prompts, gen["responses"], gen["response_lengths"], rm_scores):
            prompt = msg[-1]["content"]
            recs.append({
                "prompt_id": row["prompt_id"], "sample": s, "prompt": prompt, "response": text,
                "response_tokens": int(n), "words": word_count(text),
                "compliant": word_limit_compliance(prompt, text), "reward": rm,
            })
    return recs


def evaluate_adapter(config_path: str, adapter: str | None, name: str, beta: float, do_generation: bool = True, limit: int | None = None):
    t0 = time.time()
    cfg = load_yaml(config_path)
    results_dir = repo_path(cfg["results_dir"])
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    has_adapter = adapter is not None
    max_len = int(cfg["max_sequence_length"])
    summary = {"name": name, "adapter": adapter, "beta": beta}

    # 1) Held-out preference pairs.
    eval_rows, dropped = filter_fitting_rows(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tokenizer, max_len)
    eval_rows = eval_rows[:limit] if limit else eval_rows  # --limit is for smoke tests only
    recs = pair_logprob_records(policy, tokenizer, eval_rows, max_len, has_adapter)
    write_jsonl(results_dir / f"eval_{name}_pairs.jsonl", recs)
    summary["heldout"] = pair_summary(recs, beta)
    summary["heldout"]["n_dropped_overlength"] = len(dropped)
    print(f"[{name}] held-out pairs: {json.dumps(summary['heldout'])}", flush=True)

    # 2) Length-stratified held-out pairs.
    strat_rows, sdrop = filter_fitting_rows(read_jsonl(cfg["paths"]["dpo_length_eval"]), tokenizer, max_len)
    strat_rows = strat_rows[:limit] if limit else strat_rows
    srecs = pair_logprob_records(policy, tokenizer, strat_rows, max_len, has_adapter)
    write_jsonl(results_dir / f"eval_{name}_stratified.jsonl", srecs)
    summary["stratified"] = {"all": pair_summary(srecs, beta), "n_dropped_overlength": len(sdrop),
                             "dropped_by_stratum": drop_breakdown(sdrop)}
    for stratum in ["preferred_longer", "length_matched", "rejected_longer"]:
        summary["stratified"][stratum] = pair_summary([r for r in srecs if r["length_stratum"] == stratum], beta)
    print(f"[{name}] stratified acc: " + ", ".join(
        f"{s}={summary['stratified'][s]['preference_accuracy']:.3f}" for s in ["preferred_longer", "length_matched", "rejected_longer"]), flush=True)

    if do_generation:
        reward_model, reward_tok = load_reward_model(cfg)
        # 3) Generations on the held-out prompts (fixed set, fixed decoding, fixed seed).
        gen_rows, gdrop = filter_rows_by_prompt_length(eval_rows, tokenizer, GEN_MAX_PROMPT_TOKENS, prompt_messages_from_preference)
        grecs = generate_score_records(
            policy, tokenizer, reward_model, reward_tok, gen_rows, prompt_messages_from_preference,
            max_prompt_length=GEN_MAX_PROMPT_TOKENS,
            max_new_tokens=int(cfg["max_generation_tokens"]),
            seed=int(cfg["seed"]),
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            has_adapter=has_adapter,
            log_prefix=f"[{name}] ",
        )
        write_jsonl(results_dir / f"eval_{name}_generations.jsonl", grecs)
        summary["generation"] = aggregate_records(grecs)
        summary["generation"]["n_dropped_prompt_too_long"] = len(gdrop)
        summary["generation"]["max_new_tokens"] = int(cfg["max_generation_tokens"])
        print(f"[{name}] generation: reward={summary['generation']['reward_mean']:.3f} "
              f"KL_tok={summary['generation']['kl_token_mean']:.4f} len={summary['generation']['length']['mean']:.1f}", flush=True)

        # 4) Explicit word-limit compliance on the common prompt set.
        wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
        wrecs = word_limit_records(policy, tokenizer, wl_rows, cfg, reward_model, reward_tok)
        write_jsonl(results_dir / f"eval_{name}_wordlimit.jsonl", wrecs)
        comp = [r["compliant"] for r in wrecs if r["compliant"] is not None]
        summary["word_limit"] = {
            "n_prompts": len(wl_rows), "samples_per_prompt": WORD_LIMIT_SAMPLES,
            "compliance_rate": float(np.mean(comp)),
            "words": describe([r["words"] for r in wrecs]),
            "tokens": describe([r["response_tokens"] for r in wrecs]),
            "reward_mean": float(np.mean([r["reward"] for r in wrecs])),
            "reward_compliant_mean": float(np.mean([r["reward"] for r in wrecs if r["compliant"] == 1.0] or [np.nan])),
            "reward_noncompliant_mean": float(np.mean([r["reward"] for r in wrecs if r["compliant"] == 0.0] or [np.nan])),
        }
        print(f"[{name}] word-limit compliance={summary['word_limit']['compliance_rate']:.3f}", flush=True)
        clear_gpu(reward_model)

    summary["eval_wall_clock_s"] = time.time() - t0
    save_json(results_dir / f"eval_{name}.json", summary)
    clear_gpu(policy)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the untouched SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used to train this adapter (for the held-out DPO loss)")
    ap.add_argument("--no-generation", action="store_true")
    ap.add_argument("--limit", type=int, help="smoke tests only: evaluate the first N rows")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    adapter = None if args.adapter.lower() == "none" else args.adapter
    beta = float(cfg["beta"] if args.beta is None else args.beta)
    evaluate_adapter(args.config, adapter, args.name, beta, do_generation=not args.no_generation, limit=args.limit)


if __name__ == "__main__":
    main()
