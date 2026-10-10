from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer
from task2_ppo.evaluate import evaluate_policy

# Same held-out protocol as Task 2 (identical prompts, decoding, seed and 768-token evaluation cap),
# so PPO and GRPO policies are directly comparable. GRPO uses the raw learned reward (no EOS penalty).
EVAL_MAX_NEW_TOKENS = 768


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    adapter = None if args.adapter.lower() == "none" else args.adapter
    evaluate_policy(args.config, adapter, args.name, max_new_tokens=EVAL_MAX_NEW_TOKENS, missing_eos_penalty=None)


if __name__ == "__main__":
    main()
