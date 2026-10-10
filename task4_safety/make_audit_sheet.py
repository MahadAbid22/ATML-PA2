from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")
    print("Join these IDs to each policy's generated responses and label without viewing AI labels first.")

    # Blind labelling sheet: the 60 fixed IDs joined to every policy's response, in a seeded random
    # order, with the policy name hidden (stored in a separate key file) and NO AI labels.
    rows = []
    for policy in ["sft", "dpo", "ppo", "grpo"]:
        p = outdir / f"generated_{policy}.jsonl"
        if not p.exists():
            print("missing", p)
            continue
        by_id = {int(r["xstest_id"]): r for r in read_jsonl(p)}
        for i in ids:
            r = by_id[i]
            rows.append({"xstest_id": i, "policy": policy, "benchmark_class": r["benchmark_class"], "type": r["type"],
                         "prompt": r["prompt"], "response": r["response"]})
    sheet = pd.DataFrame(rows).sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
    sheet.insert(0, "audit_row", range(len(sheet)))
    sheet[["audit_row", "xstest_id", "policy"]].to_csv(outdir / "manual_audit_key.csv", index=False)
    blind = sheet.drop(columns=["policy"])
    blind["manual_label"] = ""
    blind["notes"] = ""
    blind_path = outdir / "manual_audit_sheet.csv"
    if blind_path.exists():
        print("Audit sheet already exists (not overwritten, it may contain labels):", blind_path)
    else:
        blind.to_csv(blind_path, index=False)
        print("Wrote blind audit sheet:", blind_path, len(blind), "rows")


if __name__ == "__main__":
    main()
