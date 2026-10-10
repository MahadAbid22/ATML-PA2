from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from common.orchestration import run_module

LOSS_TYPES = ["grpo", "dr_grpo"]


def fork_name(loss_type: str) -> str:
    return f"fork_{loss_type}"


def length_gradient_stats(seq_rows: list[dict]) -> dict:
    """Length-conditioned gradient allocation for one run.

    For every completion that entered the loss we logged its realised length, its advantage and
    the L2 norm of the gradient it contributed (measured by differencing accumulated LoRA grads).
    Short/long split = the run's median completion length.
    """
    df = pd.DataFrame([r for r in seq_rows if not r["masked"]])
    if df.empty:
        return {}
    df["grad_per_abs_adv"] = df["grad_norm"] / df["advantage"].abs().clip(lower=1e-6)
    med = df["length"].median()
    short, long_ = df[df["length"] <= med], df[df["length"] > med]
    return {
        "n_completions_in_loss": int(len(df)),
        "median_length_split": float(med),
        "spearman_len_vs_gradnorm": float(df["length"].corr(df["grad_norm"], method="spearman")),
        "spearman_len_vs_grad_per_abs_adv": float(df["length"].corr(df["grad_per_abs_adv"], method="spearman")),
        "gradnorm_short_mean": float(short["grad_norm"].mean()),
        "gradnorm_long_mean": float(long_["grad_norm"].mean()),
        "gradnorm_long_over_short": float(long_["grad_norm"].mean() / max(short["grad_norm"].mean(), 1e-12)),
        "grad_per_abs_adv_short": float(short["grad_per_abs_adv"].mean()),
        "grad_per_abs_adv_long": float(long_["grad_per_abs_adv"].mean()),
        "token_weight_short_mean": float(short["token_weight"].mean()),
        "token_weight_long_mean": float(long_["token_weight"].mean()),
        "sequence_weight_short_mean": float(short["sequence_weight"].mean()),
        "sequence_weight_long_mean": float(long_["sequence_weight"].mean()),
        # Does the update favour long or short completions among positively-rewarded ones?
        "mean_length_pos_adv": float(df.loc[df["advantage"] > 0, "length"].mean()),
        "mean_length_neg_adv": float(df.loc[df["advantage"] < 0, "length"].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"])
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    run_module("task3_grpo.evaluate", "--config", args.config, "--adapter", cfg["paths"]["grpo_midpoint_policy"],
               "--name", "midpoint", skip_if_exists=f"{cfg['results_dir']}/heldout_midpoint.json")
    rows = []
    for lt in LOSS_TYPES:
        name = fork_name(lt)
        adapter = f"outputs/task3_grpo/{name}"
        run_module("task3_grpo.continue_train", "--config", args.config, "--output", adapter,
                   "--updates", str(cfg["fork_updates"]), "--loss-type", lt, "--run-name", name,
                   skip_if_exists=f"{adapter}/adapter_model.safetensors")
        run_module("task3_grpo.evaluate", "--config", args.config, "--adapter", adapter, "--name", name,
                   skip_if_exists=f"{cfg['results_dir']}/heldout_{name}.json")
        h = load_json(out / f"heldout_{name}.json")
        log = read_jsonl(out / f"train_log_{name}.jsonl")
        tr = load_json(out / f"train_summary_{name}.json")
        row = {
            "loss_type": lt, "fork": name,
            "heldout_reward": h["reward_mean"], "heldout_kl_token": h["kl_token_mean"], "heldout_kl_seq": h["kl_seq_mean"],
            "heldout_entropy": h["entropy_token_mean"], "heldout_len_mean": h["length"]["mean"],
            "heldout_len_std": h["length"]["std"], "heldout_len_iqr": h["length"]["iqr"],
            "heldout_truncation_rate": h["truncation_rate"],
            "train_reward_mean": float(np.mean([r["reward"] for r in log])),
            "train_len_mean": float(np.mean([r["response_length"] for r in log])),
            "train_grad_norm_mean": float(np.mean([r["grad_norm"] for r in log])),
            "train_generated_tokens": tr["generated_tokens_total"],
        }
        row.update(length_gradient_stats(read_jsonl(out / f"train_sequences_{name}.jsonl")))
        rows.append(row)
    mid = load_json(out / "heldout_midpoint.json")
    rows.append({"loss_type": "midpoint (no update)", "fork": "midpoint", "heldout_reward": mid["reward_mean"],
                 "heldout_kl_token": mid["kl_token_mean"], "heldout_kl_seq": mid["kl_seq_mean"],
                 "heldout_entropy": mid["entropy_token_mean"], "heldout_len_mean": mid["length"]["mean"],
                 "heldout_len_std": mid["length"]["std"], "heldout_len_iqr": mid["length"]["iqr"],
                 "heldout_truncation_rate": mid["truncation_rate"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "normalization_table.csv", index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
