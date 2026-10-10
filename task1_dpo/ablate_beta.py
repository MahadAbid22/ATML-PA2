from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from common.orchestration import run_module


def beta_tag(beta: float) -> str:
    return f"beta_{beta:.2f}"


def summary_row(name: str, budget: str, cfg: dict) -> dict:
    s = load_json(repo_path(cfg["results_dir"]) / f"eval_{name}.json")
    h, g = s["heldout"], s.get("generation", {})
    row = {
        "condition": name,
        "budget": budget,
        "beta": s["beta"],
        "heldout_dpo_loss_own_beta": h["dpo_loss_own_beta"],
        "heldout_dpo_loss_beta0.10": h["dpo_loss_beta_0.10"],
        "heldout_pref_acc": h["preference_accuracy"],
        "heldout_margin_mean": h["margin_mean"],
    }
    if g:
        row.update({
            "kl_token_mean": g["kl_token_mean"],
            "kl_seq_mean": g["kl_seq_mean"],
            "rm_score_mean": g["reward_mean"],
            "rm_score_std": g["reward_std"],
            "len_mean": g["length"]["mean"],
            "len_std": g["length"]["std"],
            "len_iqr": g["length"]["iqr"],
            "truncation_rate": g["truncation_rate"],
            "entropy": g["entropy_token_mean"],
        })
    if "word_limit" in s:
        row["word_limit_compliance"] = s["word_limit"]["compliance_rate"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    n = int(cfg["short_ablation_examples"])
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", n)

    # Matched short forks: same first-n data subset, optimizer, seed, LoRA init; only beta changes.
    for beta in cfg["betas"]:
        tag = beta_tag(float(beta))
        out = f"outputs/task1_dpo/{tag}"
        run_module("task1_dpo.train", "--config", args.config, "--run-name", tag, "--output", out,
                   "--beta", str(beta), "--max-examples", str(n), skip_if_exists=f"{out}/adapter_model.safetensors")
        run_module("task1_dpo.evaluate", "--config", args.config, "--adapter", out, "--name", tag,
                   "--beta", str(beta), skip_if_exists=f"{cfg['results_dir']}/eval_{tag}.json")

    rows = []
    for name, budget in [("sft", "none (reference policy)"), ("standard", f"1 epoch, all pairs")]:
        if (results_dir / f"eval_{name}.json").exists():
            rows.append(summary_row(name, budget, cfg))
    for beta in cfg["betas"]:
        rows.append(summary_row(beta_tag(float(beta)), f"short run, first {n} pairs", cfg))
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "dpo_summary_table.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
