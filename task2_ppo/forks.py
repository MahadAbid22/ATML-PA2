"""Matched short PPO forks shared by the clipping and KL-pressure studies.

Every fork restarts from the identical supplied midpoint policy/value state, uses the same
seeded prompt order and per-update sampling seeds, and runs `fork_updates` updates. Only the
studied hyper-parameter changes. The (eps=0.20, beta_KL=0.10) fork is the reference
configuration and is shared by both studies.
"""
from __future__ import annotations

import numpy as np

from common.data import read_jsonl, repo_path
from common.logging_utils import load_json
from common.orchestration import run_module


def fork_name(eps: float, beta: float) -> str:
    return f"fork_eps{eps:.2f}_kl{beta:.2f}"


def ensure_fork(config_path: str, cfg: dict, eps: float, beta: float) -> str:
    name = fork_name(eps, beta)
    out = f"outputs/task2_ppo/{name}"
    run_module("task2_ppo.continue_train", "--config", config_path, "--output", out,
               "--updates", str(cfg["fork_updates"]), "--clip-epsilon", str(eps), "--kl-beta", str(beta),
               "--run-name", name, skip_if_exists=f"{out}/adapter_model.safetensors")
    run_module("task2_ppo.evaluate", "--config", config_path, "--adapter", out, "--name", name,
               skip_if_exists=f"{cfg['results_dir']}/heldout_{name}.json")
    return name


def ensure_midpoint_eval(config_path: str, cfg: dict) -> str:
    run_module("task2_ppo.evaluate", "--config", config_path, "--adapter", cfg["paths"]["ppo_midpoint_policy"],
               "--name", "midpoint", skip_if_exists=f"{cfg['results_dir']}/heldout_midpoint.json")
    return "midpoint"


def stability_stats(cfg: dict, name: str) -> dict:
    """Clearly defined stability statistics from a fork's per-update training log."""
    log = read_jsonl(repo_path(cfg["results_dir"]) / f"train_log_{name}.jsonl")
    g = lambda k: np.array([r[k] for r in log], dtype=float)
    return {
        # Realised policy movement per update on its own rollout: KL(old || new) after the PPO epochs.
        "stab_mean_post_update_kl": float(g("post_update_approx_kl").mean()),
        "stab_max_post_update_kl": float(g("post_update_approx_kl").max()),
        "stab_max_abs_log_ratio": float(g("post_update_max_abs_log_ratio").max()),
        "stab_policy_loss_std": float(g("policy_loss").std(ddof=1)),
        "stab_grad_norm_max": float(g("grad_norm_policy_max").max()),
        "train_clip_fraction_mean": float(g("clip_fraction").mean()),
        "train_clip_fraction_last_epoch_mean": float(g("clip_fraction_last_epoch").mean()),
        "train_reward_mean": float(g("reward").mean()),
        "train_kl_token_last": float(g("kl_token")[-1]),
        "train_generated_tokens": float(g("response_length").sum()),
    }


def heldout_row(cfg: dict, name: str) -> dict:
    s = load_json(repo_path(cfg["results_dir"]) / f"heldout_{name}.json")
    return {
        "heldout_reward": s["reward_mean"],
        "heldout_reward_effective": s["effective_reward_mean"],
        "heldout_kl_token": s["kl_token_mean"],
        "heldout_kl_seq": s["kl_seq_mean"],
        "heldout_entropy": s["entropy_token_mean"],
        "heldout_len_mean": s["length"]["mean"],
        "heldout_len_std": s["length"]["std"],
        "heldout_len_iqr": s["length"]["iqr"],
        "heldout_truncation_rate": s["truncation_rate"],
    }
