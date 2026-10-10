from _nb import SETUP, build, code, md

cells = [
md(r"""
# Task 3 — Group Relative Policy Optimization (GRPO) continuation

**What this notebook does.** (1) validates/fixes the group-advantage defect, (2) runs the required
20-update GRPO continuation from the supplied midpoint (K = 4 completions per prompt, max-length
completions masked), logging reward, KL, within-group reward std, uninformative-group fraction, policy
loss, gradient norm, entropy, length, peak VRAM and wall-clock, (3) the equal-generation group-size study
on the supplied K = 8 cache, (4) canonical GRPO vs Dr.-GRPO length normalisation (matched 8-update forks),
and (5) qualitative examples.

| Item | Setting (`configs/grpo.yaml`) |
|---|---|
| Policy | Qwen2.5-1.5B-Instruct + supplied GRPO LoRA; reference = LoRA disabled; no critic |
| Reward | same frozen RM as PPO (raw score, no EOS penalty) |
| Rollouts | 1 prompt/update × K = 4 samples, T = 0.7, top-p 0.9, ≤ 512 new tokens; truncated completions masked from the loss |
| Loss | clipped ratio objective with group-relative advantage, ε = 0.2, k3 KL penalty β = 0.1, 1 policy epoch, lr 5e-6, grad-clip 1.0 |
| Normalisation | canonical: Σ_t(·)/T_k per sequence then mean; Dr.-GRPO: Σ_t(·)/L_max (L_max = 512) |
| Held-out protocol | identical to Task 2 (same prompts, T = 0.7, ≤ 768 new tokens, seed 6304) |
"""),
code(SETUP + r'''
cfg = load_yaml("configs/grpo.yaml")
R = Path(cfg["results_dir"])
set_log_file("results/logs/t3_notebook.log")'''),

md(r"""
## 1. Validating the GRPO advantage (deliberate starter defect)
$A_k = (r_k-\mu_r)/(\sigma_r+\varepsilon)$ where μ and σ are computed **over the K completions of the same
prompt**. The starter ignored `group_ids` and normalised over the whole batch, so with several prompts per
batch an easy prompt's completions would all receive positive advantages and a hard prompt's all negative
ones — the update would learn "which prompt was easy" rather than "which answer was better". The fix
normalises inside each group. Tests also confirm the exact per-sequence decomposition used to fit the
loss in 8 GB of VRAM equals the batch loss of the supplied `grpo_policy_loss` for both normalisations.
"""),
code(r'''
import inspect, task3_grpo.grpo as grpo_mod
src = inspect.getsource(grpo_mod.group_relative_advantages)
print(src[src.index("# FIX"):])
from scripts.smoke_tests import test_grpo, RESULTS
test_grpo()
assert all(ok for _, ok, _ in RESULTS)'''),
code(r'''
display(pd.Series(js("checkpoints/grpo_midpoint_metadata.json")).to_frame("supplied GRPO midpoint (staff training metadata)"))'''),

md(r"""
## 2. Standard GRPO continuation (20 updates)
"""),
code(r'''
run_module("task3_grpo.continue_train", "--config", "configs/grpo.yaml", "--run-name", "standard",
           skip_if_exists="outputs/task3_grpo/standard/adapter_model.safetensors")
grep_log("results/logs/t3.log", "[standard] update")
summ = js(R / "train_summary_standard.json")
display(pd.DataFrame({"wall-clock (min)": [summ["wall_clock_s"] / 60], "wall-clock excl. thermal pauses (min)": [summ["wall_clock_excluding_thermal_pauses_s"] / 60],
                      "peak VRAM allocated (GiB)": [summ["peak_vram"]["allocated_gib"]], "peak VRAM reserved (GiB)": [summ["peak_vram"]["reserved_gib"]],
                      "generated tokens": [summ["generated_tokens_total"]], "uninformative group fraction": [summ["uninformative_group_fraction"]]},
                     index=["standard GRPO (20 updates)"]).T)'''),
code(r'''
log = jl(R / "train_log_standard.jsonl")
seqs = jl(R / "train_sequences_standard.jsonl")
panels = [("reward", "mean reward of the K=4 group"), ("kl_token", "KL to ref (token mean)"), ("reward_group_std", "within-group reward std"),
          ("running_uninformative_fraction", "running uninformative-group fraction"), ("policy_loss", "policy loss term"),
          ("grad_norm", "grad-norm (pre-clip)"), ("entropy", "entropy (nats/token)"), ("response_length", "mean completion length"),
          ("n_truncated_masked", "# completions masked (hit 512)"), ("clip_fraction", "clip fraction")]
fig, axes = plt.subplots(2, 5, figsize=(16, 5.2))
for ax, (c, t) in zip(axes.flat, panels):
    ax.plot(log["update"], log[c], "o-", ms=3, color=PALETTE[2])
    ax.set_title(t); ax.set_xlabel("update")
fig.suptitle("Standard GRPO continuation from the supplied midpoint", y=1.01)
fig.tight_layout(); savefig(fig, "t3_standard_trajectories"); plt.show()
display(log[["update", "reward", "reward_group_std", "uninformative_group_fraction", "kl_token", "kl_k3_loss", "policy_loss", "grad_norm", "entropy",
             "response_length", "n_truncated_masked", "advantage_abs_mean"]].round(4))
print(f"completions entering the loss: {(~seqs.masked).sum()}/{len(seqs)} ({(~seqs.masked).mean():.1%}); "
      f"updates with zero policy gradient because all 4 completions were masked: {(seqs.groupby('update').masked.all()).sum()}")'''),
code(r'''
run_module("task3_grpo.evaluate", "--config", "configs/grpo.yaml", "--adapter", "outputs/task3_grpo/standard", "--name", "standard",
           skip_if_exists="results/task3_grpo/heldout_standard.json")
run_module("task3_grpo.evaluate", "--config", "configs/grpo.yaml", "--adapter", cfg["paths"]["grpo_midpoint_policy"], "--name", "midpoint",
           skip_if_exists="results/task3_grpo/heldout_midpoint.json")
def hrow(path, name):
    s = js(path)
    return {"policy": name, "RM reward": s["reward_mean"], "KL token-mean": s["kl_token_mean"], "KL per-seq": s["kl_seq_mean"],
            "entropy": s["entropy_token_mean"], "len mean": s["length"]["mean"], "len std": s["length"]["std"], "len IQR": s["length"]["iqr"],
            "truncation rate (768)": s["truncation_rate"], "n prompts": s["n_prompts"]}
rows = [hrow("results/task2_ppo/heldout_sft.json", "SFT (from Task 2 protocol)")] if Path("results/task2_ppo/heldout_sft.json").exists() else []
rows += [hrow(R / "heldout_midpoint.json", "GRPO midpoint"), hrow(R / "heldout_standard.json", "GRPO standard (20 updates)")]
held = pd.DataFrame(rows).set_index("policy"); display(held)'''),

md(r"""
## 3. Equal-generation group-size study (K ∈ {2, 4, 8}) — no training
The cache holds 8 completions with RM rewards for each of 24 held-out prompts (192 generations).
**Partition rule:** each prompt's 8 completions (in `generation_index` order) are split into 8/K disjoint
groups, so every K uses the same 192 generations (K=8 → 24 groups, K=4 → 48, K=2 → 96).
**Difficulty bins (defined once):** a prompt's difficulty is the mean reward of its 8 cached completions;
tertiles → hard / medium / easy (8 prompts each). Because RM rewards are continuous, exact zero-std groups
are rare, so we also report (i) a "weak-signal" fraction (σ < 0.25 reward units) and (ii) the same
analysis with a **binary** reward 1[r > median] to show the low-resolution regime the manual warns about.
A partition-free check averages over *all* C(8, K) subsets.
"""),
code(r'''
run_module("task3_grpo.analyze_group_size", "--config", "configs/grpo.yaml", skip_if_exists="results/task3_grpo/group_size_table.csv")
gt, gb, gbin = pd.read_csv(R / "group_size_table.csv"), pd.read_csv(R / "group_size_by_difficulty.csv"), pd.read_csv(R / "group_size_binary_reward.csv")
display(Markdown("**Continuous learned reward**")); display(gt.set_index("K").T.round(4))
display(Markdown("**Binary reward 1[r > median]**")); display(gbin.set_index("K").T.round(4))
display(Markdown("**By prompt difficulty**")); display(gb.pivot_table(index=["reward", "difficulty"], columns="K",
        values=["informative_group_rate", "mean_within_group_std", "relative_signal_variance", "normalized_advantage_variance"]).round(4))'''),
code(r'''
ps = pd.DataFrame(js(R / "group_size_prompt_stats.json")["prompts"]).sort_values("mean_reward_8")
fig, axes = plt.subplots(1, 4, figsize=(16, 2.9))
for i, (rw, lab) in enumerate([("continuous", "continuous RM reward"), ("binary", "binary reward")]):
    sub = gb[gb.reward == rw]
    for j, dname in enumerate(["hard", "medium", "easy"]):
        d = sub[sub.difficulty == dname]
        axes[i].plot(d.K, d.informative_group_rate, "o-", label=dname, color=PALETTE[j])
    axes[i].set_title(f"informative-group rate ({lab})"); axes[i].set_xlabel("K"); axes[i].set_xticks([2, 4, 8])
axes[0].legend()
for j, dname in enumerate(["hard", "medium", "easy"]):
    d = gb[(gb.reward == "continuous") & (gb.difficulty == dname)]
    axes[2].plot(d.K, d.mean_within_group_std, "o-", label=dname, color=PALETTE[j])
axes[2].set_title("mean within-group reward std"); axes[2].set_xlabel("K"); axes[2].set_xticks([2, 4, 8])
axes[3].errorbar(range(len(ps)), ps.mean_reward_8, yerr=ps.reward_std_8, fmt="o", ms=3, color=PALETTE[0])
axes[3].set_title("24 cached prompts: mean ± std of 8 rewards"); axes[3].set_xlabel("prompt (sorted by mean reward)")
savefig(fig, "t3_group_size"); plt.show()
display(ps.round(3))'''),

md(r"""
## 4. Canonical GRPO vs Dr.-GRPO normalisation (matched 8-update forks)
Both forks start from the same midpoint with identical prompts, sampling seeds, K, ε, β, reward and
512-token cap; only the per-sequence normaliser changes (1/T_k vs 1/L_max). For every completion that
enters the loss we log its length, advantage, the per-token loss weight |A|/(B·denominator) and the L2 norm of
the gradient it contributes (measured by differencing accumulated LoRA gradients).
"""),
code(r'''
run_module("task3_grpo.compare_normalization", "--config", "configs/grpo.yaml")
nt = pd.read_csv(R / "normalization_table.csv").set_index("fork")
display(nt.T)'''),
code(r'''
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for i, lt in enumerate(["grpo", "dr_grpo"]):
    s = jl(R / f"train_sequences_fork_{lt}.jsonl"); s = s[~s.masked]
    axes[0].scatter(s.length, s.grad_norm, s=12, alpha=0.7, label=lt, color=PALETTE[i])
    axes[1].scatter(s.length, s.grad_norm / s.advantage.abs().clip(lower=1e-6), s=12, alpha=0.7, label=lt, color=PALETTE[i])
    axes[2].scatter(s.length, s.token_weight, s=12, alpha=0.7, label=lt, color=PALETTE[i])
    h = jl(R / f"heldout_fork_{lt}.jsonl")
    axes[3].hist(h.response_tokens, bins=30, alpha=0.5, label=lt, color=PALETTE[i])
for ax, t in zip(axes, ["per-completion gradient norm", "gradient norm / |advantage|", "per-token loss weight |A|/(B·denom)", "held-out length distribution"]):
    ax.set_title(t); ax.set_xlabel("completion length (tokens)"); ax.legend(fontsize=7)
axes[2].set_yscale("log")
savefig(fig, "t3_normalization"); plt.show()
for lt in ["grpo", "dr_grpo"]:
    d = jl(R / f"train_log_fork_{lt}.jsonl")
    print(lt, "per-update mean length:", d.response_length.round(0).tolist())'''),

md(r"""
## 5. Qualitative evidence
(a) The most and least informative groups in the standard run; (b) a long vs a short completion from the
same group under the two normalisations.
"""),
code(r'''
by = seqs.groupby("update")
stds = by.reward.std(ddof=0)
prompts = {p["update"]: p for p in js(R / "train_prompts_standard.json")}
from common.data import prompt_messages
pool = {str(r["source_index"]): r for r in read_jsonl(cfg["paths"]["rl_prompt_train"])}
for label, u in [("most informative group", stds.idxmax()), ("least informative group", stds.idxmin())]:
    p = pool[prompts[u - 1]["source_index"]]
    g = seqs[seqs["update"] == u]
    print(f"=== {label}: update {u}, reward std {stds[u]:.3f}\nPROMPT: {excerpt(prompt_messages(p)[-1]['content'], 200)}")
    display(g[["i", "reward", "advantage", "length", "truncated", "masked", "grad_norm"]].round(3))'''),
code(r'''
for lt in ["grpo", "dr_grpo"]:
    s = jl(R / f"train_sequences_fork_{lt}.jsonl"); s = s[~s.masked]
    g = s.groupby("update").filter(lambda d: len(d) >= 2 and d.length.max() > 2 * d.length.min())
    if g.empty:
        print(lt, ": no group with a 2x length spread among unmasked completions"); continue
    u = g["update"].iloc[0]; d = g[g["update"] == u]
    print(f"=== {lt}, update {u}: shortest vs longest unmasked completion")
    display(d.loc[[d.length.idxmin(), d.length.idxmax()], ["length", "reward", "advantage", "token_weight", "sequence_weight", "grad_norm"]].round(5))'''),

md(r"""
## 6. Summary tables for the report
"""),
code(r'''
display(Markdown("**Standard continuation (20 updates)**"))
display(pd.DataFrame({"mean over 20 updates": log[["reward", "kl_token", "reward_group_std", "uninformative_group_fraction", "policy_loss", "grad_norm", "entropy", "response_length"]].mean(),
                      "last update": log[["reward", "kl_token", "reward_group_std", "uninformative_group_fraction", "policy_loss", "grad_norm", "entropy", "response_length"]].iloc[-1]}).round(4))
display(Markdown("**Group-size study (continuous / binary)**"))
display(gt.set_index("K")[["n_groups", "informative_group_rate", "weak_signal_group_fraction", "mean_within_group_std", "relative_signal_variance", "normalized_advantage_variance"]].round(4))
display(gbin.set_index("K")[["n_groups", "informative_group_rate", "mean_within_group_std", "relative_signal_variance", "exp_informative_rate_all_subsets"]].round(4))
display(Markdown("**Normalisation study**"))
cols = [c for c in ["heldout_reward", "heldout_kl_token", "heldout_len_mean", "heldout_len_std", "spearman_len_vs_gradnorm", "gradnorm_long_over_short",
                    "token_weight_short_mean", "token_weight_long_mean", "train_generated_tokens"] if c in nt.columns]
display(nt[cols].round(5))
held.round(4).to_csv(R / "report_table_task3_heldout.csv"); nt.round(5).to_csv(R / "report_table_task3_norm.csv")'''),
]

build("Task3_GRPO", cells)
