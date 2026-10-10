from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from task2_ppo.forks import ensure_fork, ensure_midpoint_eval, heldout_row, stability_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    ensure_midpoint_eval(args.config, cfg)
    eps = float(cfg["clip_epsilon"])
    table = []
    for beta in cfg["kl_values"]:
        name = ensure_fork(args.config, cfg, eps, float(beta))
        row = {"kl_beta": float(beta), "fork": name}
        row.update(heldout_row(cfg, name))
        row.update(stability_stats(cfg, name))
        table.append(row)
    df = pd.DataFrame(table)
    df.to_csv(repo_path(cfg["results_dir"]) / "kl_study_table.csv", index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
