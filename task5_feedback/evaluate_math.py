from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import load_policy, load_tokenizer
from common.orchestration import run_module
from common.rl_utils import cooldown_if_hot, describe
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

GEN_BATCH_SIZE = 16
JUDGE_CACHE = "results/task5_feedback/judge_cache.json"


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def generate_policy(config_path: str, dataset: str, name: str, out_path):
    """Deterministic (greedy) responses, common 512-token cap, fixed prompt order."""
    cfg, rows, tok = load_math_evaluation(config_path, dataset)
    model = load_frozen_policy(cfg, name)
    recs, t0 = [], time.time()
    for start in range(0, len(rows), GEN_BATCH_SIZE):
        cooldown_if_hot()
        chunk = rows[start : start + GEN_BATCH_SIZE]
        gen = batch_generate(model, tok, [prompt_messages(r) for r in chunk], max_prompt_length=512,
                             max_new_tokens=int(cfg["math_max_new_tokens"]), do_sample=False)
        for r, text, n, trunc, eos in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"], gen["terminated_with_eos"]):
            pred = extract_designated_final(text)
            recs.append({
                "source_index": str(r["source_index"]), "policy": name, "dataset": dataset,
                "question": r["question"], "gold_final": str(r["gold_final"]), "response": text,
                "pred_final": pred, "format_ok": pred is not None,
                "exact": exact_reward(text, str(r["gold_final"])),
                "response_tokens": int(n), "truncated": bool(trunc), "terminated_with_eos": bool(eos),
                **({"svamp_type": r.get("svamp_type")} if "svamp_type" in r else {}),
            })
        print(f"[{dataset}/{name}] {min(start + GEN_BATCH_SIZE, len(rows))}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    write_jsonl(out_path, recs)


def failure_type(r) -> str:
    if r["exact"] == 1.0:
        return "correct"
    if not r["format_ok"]:
        return "truncated_no_final" if r["truncated"] else "missing_final_format"
    return "wrong_final"


def judge_pairs(cfg, dataset: str, gens: dict):
    """AI pairwise evaluation of each RL policy against SFT on identical problems."""
    judge = PairwiseAIJudge(cfg, JUDGE_CACHE)
    sft = {r["source_index"]: r for r in gens["sft"]}
    out = []
    t0 = time.time()
    new_calls, new_time = 0, 0.0
    for name in ["rlvr", "rlaif"]:
        for i, r in enumerate(gens[name]):
            if i % 25 == 0:
                cooldown_if_hot()
            s = sft[r["source_index"]]
            fresh = judge._key(r["question"], r["response"], s["response"]) not in judge.cache
            tc = time.perf_counter()
            pref = judge.compare(r["question"], r["response"], s["response"])  # A = policy, B = SFT
            if fresh:
                new_calls += 1
                new_time += time.perf_counter() - tc
            judge_score = {"A": 1.0, "TIE": 0.5, "B": 0.0}[pref]
            vdiff = r["exact"] - s["exact"]
            verifier_pref = "A" if vdiff > 0 else ("B" if vdiff < 0 else "TIE")
            out.append({"dataset": dataset, "policy": name, "source_index": r["source_index"], "judge_pref": pref,
                        "judge_score": judge_score, "verifier_pref": verifier_pref,
                        "policy_exact": r["exact"], "sft_exact": s["exact"]})
            if (i + 1) % 50 == 0:
                print(f"[judge {dataset}/{name}] {i + 1}/{len(gens[name])} ({time.time() - t0:.0f}s)", flush=True)
    if new_calls:
        save_json(repo_path(cfg["results_dir"]) / "task5_feedback" / f"judge_cost_{dataset}.json",
                  {"new_judge_calls": new_calls, "total_seconds": new_time, "seconds_per_call": new_time / new_calls})
    return out


def summarise(gens: dict, pairs: list[dict]) -> pd.DataFrame:
    rows = []
    pdf = pd.DataFrame(pairs)
    for name, recs in gens.items():
        d = pd.DataFrame(recs)
        row = {
            "policy": name, "n": len(d),
            "exact_accuracy": d.exact.mean(),
            "format_compliance": d.format_ok.mean(),
            "accuracy_given_format": d[d.format_ok].exact.mean() if d.format_ok.any() else np.nan,
            "truncation_rate": d.truncated.mean(),
            "len_mean": d.response_tokens.mean(), "len_std": d.response_tokens.std(),
            "len_median": d.response_tokens.median(),
            "len_iqr": describe(d.response_tokens)["iqr"],
        }
        for ft, c in d.apply(failure_type, axis=1).value_counts(normalize=True).items():
            row[f"frac_{ft}"] = c
        if name == "sft":
            row.update({"ai_win_rate_vs_sft": 0.5, "note": "reference (win rate vs itself = 0.5 by definition)"})
        else:
            p = pdf[pdf.policy == name]
            decisive = p[p.verifier_pref != "TIE"]
            row.update({
                "ai_win_rate_vs_sft": p.judge_score.mean(),
                "ai_wins": int((p.judge_pref == "A").sum()), "ai_ties": int((p.judge_pref == "TIE").sum()),
                "ai_losses": int((p.judge_pref == "B").sum()),
                # Same win/tie/loss convention applied to the exact verifier.
                "verifier_win_rate_vs_sft": (p.verifier_pref.map({"A": 1.0, "TIE": 0.5, "B": 0.0})).mean(),
                "verifier_judge_agreement_3way": (p.judge_pref == p.verifier_pref).mean(),
                "verifier_judge_agreement_decisive": (decisive.judge_pref == decisive.verifier_pref).mean() if len(decisive) else np.nan,
                "n_verifier_decisive": int(len(decisive)),
                "judge_decisive_when_verifier_tie": (p[p.verifier_pref == "TIE"].judge_pref != "TIE").mean(),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--policy", help="internal: generate for one policy in its own process")
    args = ap.parse_args()
    cfg, rows, _ = load_math_evaluation(args.config, args.dataset)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    if args.policy:
        generate_policy(args.config, args.dataset, args.policy, outdir / f"generated_{args.dataset}_{args.policy}.jsonl")
        return
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    print("Exact verifier available as task5_feedback.rlvr.exact_reward")
    print("Pairwise judge available as task5_feedback.rlaif.PairwiseAIJudge")
    for name in policy_specs(cfg):
        run_module("task5_feedback.evaluate_math", "--config", args.config, "--dataset", args.dataset, "--policy", name,
                   skip_if_exists=outdir / f"generated_{args.dataset}_{name}.jsonl")
    gens = {n: read_jsonl(outdir / f"generated_{args.dataset}_{n}.jsonl") for n in policy_specs(cfg)}
    pairs = judge_pairs(cfg, args.dataset, gens)
    write_jsonl(outdir / f"pairwise_{args.dataset}.jsonl", pairs)
    summ = summarise(gens, pairs)
    summ.to_csv(outdir / f"summary_{args.dataset}.csv", index=False)
    # Verifier-vs-judge confusion per policy.
    pdf = pd.DataFrame(pairs)
    conf = pdf.groupby(["policy", "verifier_pref", "judge_pref"]).size().unstack(fill_value=0)
    conf.to_csv(outdir / f"verifier_judge_confusion_{args.dataset}.csv")
    save_json(outdir / f"summary_{args.dataset}.json", {"summary": summ.to_dict("records"),
              "decoding": {"do_sample": False, "max_new_tokens": int(cfg["math_max_new_tokens"])}})
    pd.set_option("display.width", 220)
    print(summ.T.to_string())
    print(conf.to_string())


if __name__ == "__main__":
    main()
