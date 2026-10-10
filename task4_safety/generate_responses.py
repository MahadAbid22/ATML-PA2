from __future__ import annotations

import argparse
import time

import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer
from common.orchestration import run_module
from common.rl_utils import cooldown_if_hot

GEN_BATCH_SIZE = 16


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    t0 = time.time()
    for start in range(0, len(df), batch_size):
        cooldown_if_hot()
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok, trunc in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"], gen["truncated"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
                "truncated": bool(trunc),
            })
        print(f"[{policy_name}] {min(start + batch_size, len(df))}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", help="generate for a single policy (used internally, one process per policy)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    if args.policy:
        recs = generate_for_policy(cfg, args.policy, batch_size=GEN_BATCH_SIZE)
        write_jsonl(outdir / f"generated_{args.policy}.jsonl", recs)  # fixed XSTest order preserved
        return
    print("Policies:", list(policy_specs(cfg)))
    print("XSTest rows:", len(load_xstest(cfg)))
    for name, adapter in policy_specs(cfg).items():
        if adapter is not None and not (repo_path(adapter) / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"{name} adapter missing: {adapter}. Run Tasks 1-3 standard runs first.")
        run_module("task4_safety.generate_responses", "--config", args.config, "--policy", name,
                   skip_if_exists=outdir / f"generated_{name}.jsonl")


if __name__ == "__main__":
    main()
