from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages

ZERO_STD_TOL = 1e-6           # same tolerance as group_relative_advantages' eps
LOW_SIGNAL_STD = 0.25         # reward units; "weak" relative signal threshold (reported separately)
DIFFICULTY_BINS = ["hard", "medium", "easy"]


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Partition rule (documented once): for every prompt, its 8 cached completions, in
    generation_index order, are split into 8/K disjoint consecutive groups. Every K therefore
    uses exactly the same 24 x 8 = 192 generations: K=8 -> 24 groups (1/prompt),
    K=4 -> 48 groups (2/prompt), K=2 -> 96 groups (4/prompt). Equal total generations,
    different number of prompt groups, exactly as in a fixed-budget GRPO batch.
    """
    if 8 % k:
        raise ValueError("K must divide 8")
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:8]
        for g in range(8 // k):
            part = comps[g * k : (g + 1) * k]
            groups.append({"source_index": pid, "subgroup": g, "rewards": [float(c["reward"]) for c in part],
                           "lengths": [int(c.get("completion_tokens", 0)) for c in part],
                           "clipped": [str(c.get("clipped_at_max", "False")) == "True" for c in part]})
    return groups


def difficulty_bins(by_prompt):
    """Prompt difficulty = mean learned reward over all 8 cached completions; tertiles -> hard/medium/easy."""
    means = {pid: float(np.mean([float(c["reward"]) for c in comps[:8]])) for pid, comps in by_prompt.items()}
    order = sorted(means, key=means.get)
    n = len(order)
    cuts = [order[: n // 3], order[n // 3 : 2 * n // 3], order[2 * n // 3 :]]
    return {pid: name for name, ids in zip(DIFFICULTY_BINS, cuts) for pid in ids}, means


def group_metrics(groups, binarize_at: float | None = None) -> dict:
    if not groups:
        return {}
    r_all, gid, prompt_mean8 = [], [], []
    stds, means = [], []
    for j, g in enumerate(groups):
        r = np.array(g["rewards"], dtype=float)
        if binarize_at is not None:
            r = (r > binarize_at).astype(float)
        r_all.extend(r.tolist()); gid.extend([j] * len(r))
        stds.append(r.std()); means.append(r.mean())
    r_all, stds, means = np.array(r_all), np.array(stds), np.array(means)
    centred = np.concatenate([np.array(g["rewards"] if binarize_at is None else (np.array(g["rewards"]) > binarize_at).astype(float)) - m
                              for g, m in zip(groups, means)])
    adv = group_relative_advantages(torch.tensor(r_all), torch.tensor(gid)).numpy()
    return {
        "n_groups": len(groups),
        "n_generations": int(len(r_all)),
        "informative_group_rate": float((stds > ZERO_STD_TOL).mean()),
        "uninformative_group_fraction": float((stds <= ZERO_STD_TOL).mean()),
        "weak_signal_group_fraction": float((stds < LOW_SIGNAL_STD).mean()) if binarize_at is None else float("nan"),
        "mean_within_group_std": float(stds.mean()),
        # Variance of the raw relative signal r_k - mu_g pooled over all completions.
        "relative_signal_variance": float(centred.var()),
        # Variance of the normalised group-relative advantage actually used by GRPO.
        "normalized_advantage_variance": float(adv.var()),
        "frac_completions_nonzero_advantage": float((np.abs(adv) > 1e-8).mean()),
        "mean_abs_advantage": float(np.abs(adv).mean()),
    }


def all_subsets_expectation(by_prompt, k: int, binarize_at: float | None = None) -> dict:
    """Partition-free check: average over ALL C(8,K) subsets of each prompt's completions."""
    inf, std = [], []
    for comps in by_prompt.values():
        r = np.array([float(c["reward"]) for c in comps[:8]])
        if binarize_at is not None:
            r = (r > binarize_at).astype(float)
        s = np.array([r[list(idx)].std() for idx in itertools.combinations(range(8), k)])
        inf.append((s > ZERO_STD_TOL).mean()); std.append(s.mean())
    return {"exp_informative_rate_all_subsets": float(np.mean(inf)), "exp_within_group_std_all_subsets": float(np.mean(std))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"])
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    bins, prompt_means = difficulty_bins(by_prompt)
    all_rewards = np.array([float(c["reward"]) for comps in by_prompt.values() for c in comps[:8]])
    median_reward = float(np.median(all_rewards))

    rows, binned, binary = [], [], []
    for k in cfg["group_sizes"]:
        groups = regroup_equal_generation_budget(by_prompt, int(k))
        rows.append({"K": int(k), **group_metrics(groups), **all_subsets_expectation(by_prompt, int(k))})
        binary.append({"K": int(k), "reward": f"binary: 1[r > median={median_reward:.3f}]",
                       **group_metrics(groups, binarize_at=median_reward),
                       **all_subsets_expectation(by_prompt, int(k), binarize_at=median_reward)})
        for b in DIFFICULTY_BINS:
            sub = [g for g in groups if bins[g["source_index"]] == b]
            binned.append({"K": int(k), "difficulty": b, "reward": "continuous", **group_metrics(sub)})
            binned.append({"K": int(k), "difficulty": b, "reward": "binary", **group_metrics(sub, binarize_at=median_reward)})

    df, dfb, dfbin = pd.DataFrame(rows), pd.DataFrame(binned), pd.DataFrame(binary)
    df.to_csv(out / "group_size_table.csv", index=False)
    dfb.to_csv(out / "group_size_by_difficulty.csv", index=False)
    dfbin.to_csv(out / "group_size_binary_reward.csv", index=False)
    save_json(out / "group_size_prompt_stats.json", {
        "binning_rule": "difficulty = mean learned reward of the prompt's 8 cached completions; tertiles (8 prompts each): hard < medium < easy",
        "partition_rule": "each prompt's 8 completions (generation_index order) split into 8/K disjoint consecutive groups; total generations fixed at 192",
        "zero_std_tolerance": ZERO_STD_TOL,
        "binary_threshold_median_reward": median_reward,
        "prompts": [{"source_index": pid, "mean_reward_8": prompt_means[pid], "difficulty": bins[pid],
                     "reward_std_8": float(np.std([float(c["reward"]) for c in comps[:8]])),
                     "n_clipped": int(sum(str(c.get("clipped_at_max")) == "True" for c in comps[:8]))}
                    for pid, comps in by_prompt.items()],
    })
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(dfbin.to_string(index=False))
    print(dfb.to_string(index=False))


if __name__ == "__main__":
    main()
