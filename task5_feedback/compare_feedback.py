from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task5_feedback.evaluate_math import failure_type
from task5_feedback.rlvr import exact_reward

POLICIES = ["sft", "rlvr", "rlaif"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm = pd.read_csv(outdir / "summary_gsm.csv").set_index("policy")
    tr = pd.read_csv(outdir / "summary_transfer.csv").set_index("policy")

    # In-domain vs out-of-domain with drops (OOD - in-domain).
    rows = []
    for p in POLICIES:
        r = {"policy": p}
        for metric in ["exact_accuracy", "format_compliance", "ai_win_rate_vs_sft", "len_mean", "len_std", "truncation_rate"]:
            r[f"gsm_{metric}"] = gsm.loc[p, metric]
            r[f"svamp_{metric}"] = tr.loc[p, metric]
            r[f"delta_{metric}"] = tr.loc[p, metric] - gsm.loc[p, metric]
        rows.append(r)
    io = pd.DataFrame(rows)
    io.to_csv(outdir / "in_vs_out_of_domain.csv", index=False)

    # Accuracy change relative to SFT on each dataset (what the feedback source bought).
    gain = pd.DataFrame([{"policy": p,
                          "gsm_acc_gain_vs_sft": gsm.loc[p, "exact_accuracy"] - gsm.loc["sft", "exact_accuracy"],
                          "svamp_acc_gain_vs_sft": tr.loc[p, "exact_accuracy"] - tr.loc["sft", "exact_accuracy"]} for p in POLICIES])
    gain.to_csv(outdir / "accuracy_gain_vs_sft.csv", index=False)

    # Failure types per dataset and policy.
    ft_rows, svamp_type_rows = [], []
    for ds in ["gsm", "transfer"]:
        for p in POLICIES:
            d = pd.DataFrame(read_jsonl(outdir / f"generated_{ds}_{p}.jsonl"))
            ft = d.apply(failure_type, axis=1).value_counts(normalize=True)
            ft_rows.append({"dataset": ds, "policy": p, **ft.to_dict()})
            if ds == "transfer" and "svamp_type" in d:
                for t, g in d.groupby("svamp_type"):
                    svamp_type_rows.append({"policy": p, "svamp_type": t, "n": len(g), "exact_accuracy": g.exact.mean()})
    pd.DataFrame(ft_rows).fillna(0.0).to_csv(outdir / "failure_types.csv", index=False)
    pd.DataFrame(svamp_type_rows).to_csv(outdir / "svamp_accuracy_by_type.csv", index=False)

    # Paired agreement between policies on the same problems (where do RLVR and RLAIF differ?).
    paired = []
    for ds in ["gsm", "transfer"]:
        g = {p: pd.DataFrame(read_jsonl(outdir / f"generated_{ds}_{p}.jsonl")).set_index("source_index") for p in POLICIES}
        for a, b in [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]:
            ea, eb = g[a].exact, g[b].exact.reindex(g[a].index)
            paired.append({"dataset": ds, "a": a, "b": b, "a_right_b_wrong": int(((ea == 1) & (eb == 0)).sum()),
                           "a_wrong_b_right": int(((ea == 0) & (eb == 1)).sum()), "both_right": int(((ea == 1) & (eb == 1)).sum()),
                           "both_wrong": int(((ea == 0) & (eb == 0)).sum())})
    pd.DataFrame(paired).to_csv(outdir / "paired_correctness.csv", index=False)

    # Inference cost of each feedback source.
    t0 = time.perf_counter()
    sample = read_jsonl(outdir / "generated_gsm_sft.jsonl")
    for r in sample:
        exact_reward(r["response"], r["gold_final"])
    verifier_s = (time.perf_counter() - t0) / max(len(sample), 1)
    judge_cost = {}
    for ds in ["gsm", "transfer"]:
        p = outdir / f"judge_cost_{ds}.json"
        if p.exists():
            judge_cost[ds] = load_json(p)
    per_call = np.mean([v["seconds_per_call"] for v in judge_cost.values()]) if judge_cost else float("nan")
    K = 4
    cost = {
        "verifier_seconds_per_response": verifier_s,
        "judge_seconds_per_pairwise_call": float(per_call),
        "judge_calls_per_prompt_group_K4": K * (K - 1) // 2,
        "judge_seconds_per_prompt_group_K4": float(per_call * K * (K - 1) / 2),
        "verifier_seconds_per_prompt_group_K4": verifier_s * K,
        "judge_model": cfg["ai_judge_model"],
        "judge_detail": judge_cost,
    }
    diag = load_json(outdir / "diagnostic_summary.json")
    rlvr_meta = load_json(repo_path("checkpoints/rlvr_metadata.json"))
    rlaif_meta = load_json(repo_path("checkpoints/rlaif_metadata.json"))
    save_json(outdir / "feedback_comparison.json", {
        "in_vs_out_of_domain": io.to_dict("records"),
        "accuracy_gain_vs_sft": gain.to_dict("records"),
        "diagnostics": diag,
        "inference_cost": cost,
        "training_metadata": {"rlvr": rlvr_meta, "rlaif": rlaif_meta},
    })
    pd.set_option("display.width", 220)
    print(io.T.to_string())
    print(gain.to_string(index=False))
    print(pd.DataFrame(ft_rows).fillna(0).to_string(index=False))
    print(cost)


if __name__ == "__main__":
    main()
