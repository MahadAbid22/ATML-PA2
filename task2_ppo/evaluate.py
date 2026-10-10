from __future__ import annotations

import argparse
import json

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from common.models import load_policy, load_reward_model, load_tokenizer
from common.policy_eval import aggregate_records, generate_score_records
from common.rl_utils import filter_rows_by_prompt_length


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def heldout_rows(cfg, tokenizer):
    """Fixed held-out prompt set: the eval pool filtered by the release max_prompt_length."""
    kept, dropped = filter_rows_by_prompt_length(
        read_jsonl(cfg["paths"]["rl_prompt_eval"]), tokenizer, int(cfg["max_prompt_length"]), prompt_messages
    )
    return kept, dropped


def evaluate_policy(config_path: str, adapter: str | None, name: str, results_dir: str | None = None, max_new_tokens: int | None = None, missing_eos_penalty: float | None = None):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    out_dir = repo_path(results_dir or cfg["results_dir"])
    tok = bundle["tokenizer"]
    rm, rm_tok = bundle["reward"]
    rows, dropped = heldout_rows(cfg, tok)
    max_new = int(max_new_tokens or cfg.get("eval_max_response_length", cfg.get("max_completion_length", 512)))
    penalty = missing_eos_penalty if missing_eos_penalty is not None else cfg.get("missing_eos_penalty")
    records = generate_score_records(
        bundle["policy"], tok, rm, rm_tok, rows, prompt_messages,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=max_new,
        seed=int(cfg["seed"]),
        temperature=float(cfg["generation"]["temperature"]),
        top_p=float(cfg["generation"]["top_p"]),
        reward_max_length=int(cfg.get("reward_max_length", 1280)),
        missing_eos_penalty=penalty,
        has_adapter=adapter is not None,
        log_prefix=f"[eval {name}] ",
    )
    write_jsonl(out_dir / f"heldout_{name}.jsonl", records)
    summary = aggregate_records(records)
    summary.update({
        "name": name, "adapter": adapter, "max_new_tokens": max_new, "missing_eos_penalty": penalty,
        "n_dropped_prompt_too_long": len(dropped), "seed": int(cfg["seed"]), "decoding": cfg["generation"],
    })
    save_json(out_dir / f"heldout_{name}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the SFT reference policy")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    adapter = None if args.adapter.lower() == "none" else args.adapter
    evaluate_policy(args.config, adapter, args.name)


if __name__ == "__main__":
    main()
