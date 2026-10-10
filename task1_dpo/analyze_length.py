from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.models import load_tokenizer
from common.orchestration import run_module
from task1_dpo.train import drop_breakdown, filter_fitting_rows

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def dataset_length_stats(rows, tokenizer, name: str) -> dict:
    """Length structure of a preference file (a property of the DATA, not of any policy)."""
    lc, lr, sc, sr = [], [], [], []
    for row in rows:
        yc, yr = preference_responses(row)
        lc.append(len(tokenizer(yc, add_special_tokens=False)["input_ids"]))
        lr.append(len(tokenizer(yr, add_special_tokens=False)["input_ids"]))
        sc.append(float(row.get("score_chosen", np.nan)))
        sr.append(float(row.get("score_rejected", np.nan)))
    lc, lr = np.array(lc, float), np.array(lr, float)
    d = lc - lr
    rel = d / np.maximum(lc, lr).clip(min=1)
    sdiff = np.array(sc) - np.array(sr)
    ok = ~np.isnan(sdiff)
    out = {
        "dataset": name,
        "n_pairs": len(rows),
        "chosen_tokens_mean": float(lc.mean()),
        "rejected_tokens_mean": float(lr.mean()),
        "frac_chosen_longer": float((d > 0).mean()),
        "frac_rejected_longer": float((d < 0).mean()),
        "mean_len_diff_chosen_minus_rejected": float(d.mean()),
        "median_len_diff": float(np.median(d)),
        "frac_relative_diff_gt_20pct_chosen_longer": float((rel > 0.2).mean()),
        "frac_relative_diff_gt_20pct_rejected_longer": float((rel < -0.2).mean()),
        "spearman_len_diff_vs_score_diff": float(pd.Series(d[ok]).corr(pd.Series(sdiff[ok]), method="spearman")) if ok.sum() > 2 else float("nan"),
    }
    if rows and "length_stratum" in rows[0]:
        out["strata_counts"] = pd.Series([r["length_stratum"] for r in rows]).value_counts().to_dict()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    # Train the length-balanced condition from the ORIGINAL initialisation (same budget as standard: 1 epoch).
    out = cfg["length_output"]
    run_module("task1_dpo.train", "--config", args.config, "--run-name", "length_balanced",
               "--dataset", cfg["paths"]["dpo_length_train"], "--output", out,
               skip_if_exists=f"{out}/adapter_model.safetensors")
    run_module("task1_dpo.evaluate", "--config", args.config, "--adapter", out, "--name", "length_balanced",
               skip_if_exists=f"{cfg['results_dir']}/eval_length_balanced.json")
    run_module("task1_dpo.evaluate", "--config", args.config, "--adapter", cfg["standard_output"], "--name", "standard",
               skip_if_exists=f"{cfg['results_dir']}/eval_standard.json")

    # Dataset-side length structure, before and after the overlength filter (so the effect of
    # dropping pairs with prompt + response > max_sequence_length on the length structure is visible).
    tok = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    ds = []
    for key, name in [("dpo_standard_train", "standard_train"), ("dpo_length_train", "length_balanced_train"),
                      ("dpo_standard_eval", "standard_eval"), ("dpo_length_eval", "length_stratified_eval")]:
        rows = read_jsonl(cfg["paths"][key])
        kept, dropped = filter_fitting_rows(rows, tok, max_len)
        full = dataset_length_stats(rows, tok, name + " (all)")
        filt = dataset_length_stats(kept, tok, name + " (after filter)")
        filt["n_dropped_overlength"] = len(dropped)
        filt["dropped_by_stratum"] = drop_breakdown(dropped)
        ds += [full, filt]
    save_json(results_dir / "length_dataset_stats.json", ds)
    print(pd.DataFrame(ds).drop(columns=["strata_counts", "dropped_by_stratum"], errors="ignore").to_string(index=False))

    # Policy-side: per-stratum held-out accuracy, generated length, word-limit compliance.
    rows = []
    for name in ["sft", "standard", "length_balanced", "beta_0.03", "beta_0.10", "beta_0.30"]:
        p = results_dir / f"eval_{name}.json"
        if not p.exists():
            continue
        s = load_json(p)
        row = {"condition": name}
        for stratum in STRATA:
            row[f"acc_{stratum}"] = s["stratified"][stratum]["preference_accuracy"]
            row[f"margin_{stratum}"] = s["stratified"][stratum]["margin_mean"]
        row["acc_all_strata"] = s["stratified"]["all"]["preference_accuracy"]
        row["acc_standard_heldout"] = s["heldout"]["preference_accuracy"]
        if "generation" in s:
            row["gen_len_mean"] = s["generation"]["length"]["mean"]
            row["gen_len_std"] = s["generation"]["length"]["std"]
            row["gen_len_iqr"] = s["generation"]["length"]["iqr"]
            row["rm_score_mean"] = s["generation"]["reward_mean"]
        if "word_limit" in s:
            row["word_limit_compliance"] = s["word_limit"]["compliance_rate"]
            row["word_limit_words_mean"] = s["word_limit"]["words"]["mean"]
            row["word_limit_words_std"] = s["word_limit"]["words"]["std"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "length_confounding_table.csv", index=False)
    print(df.to_string(index=False))

    # Per-prompt word-limit table (standard vs length-balanced) for qualitative inspection.
    wl = []
    for name in ["sft", "standard", "length_balanced"]:
        p = results_dir / f"eval_{name}_wordlimit.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                r["condition"] = name
                wl.append(r)
    if wl:
        wdf = pd.DataFrame(wl)
        piv = wdf.groupby(["prompt_id", "condition"]).agg(words=("words", "mean"), compliance=("compliant", "mean")).unstack("condition")
        piv.to_csv(results_dir / "word_limit_per_prompt.csv")
        print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
