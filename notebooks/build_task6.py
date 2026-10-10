from _nb import SETUP, build, code, md

cells = [
md(r"""
# Task 6 — Cross-task synthesis

No new training. This notebook gathers the saved results of Tasks 1–5 into cross-task tables and figures:
preference strength vs policy drift, offline vs online feedback, optimisation biases, safety calibration,
reward-source design, and compute. The only extra computation is scoring the *standard DPO adapter* under
the common Task 2/3 held-out protocol (same 165 prompts, T = 0.7, ≤ 768 tokens, same RM), so that SFT, DPO,
PPO and GRPO can be placed on one reward-vs-KL plane.
"""),
code(SETUP + r'''
T1, T2, T3, T4, T5 = (Path(p) for p in ["results/task1_dpo", "results/task2_ppo", "results/task3_grpo", "results/task4_safety", "results/task5_feedback"])
set_log_file("results/logs/t6_notebook.log")
run_module("task2_ppo.evaluate", "--config", "configs/ppo.yaml", "--adapter", "outputs/task1_dpo/standard", "--name", "dpo_standard",
           skip_if_exists="results/task2_ppo/heldout_dpo_standard.json")'''),

md(r"""
## 1. Preference strength vs policy drift (common RL held-out protocol)
"""),
code(r'''
def h(path, label, family):
    s = js(path)
    return {"condition": label, "family": family, "reward": s["reward_mean"], "kl_token": s["kl_token_mean"], "kl_seq": s["kl_seq_mean"],
            "entropy": s["entropy_token_mean"], "len_mean": s["length"]["mean"], "len_std": s["length"]["std"], "trunc": s["truncation_rate"]}
rows = [h(T2 / "heldout_sft.json", "SFT", "SFT"), h(T2 / "heldout_dpo_standard.json", "DPO standard", "DPO")]
rows += [h(T2 / f"heldout_{n}.json", lab, "PPO") for n, lab in [("midpoint", "PPO midpoint"), ("standard", "PPO standard (20)")]]
ct, kt = pd.read_csv(T2 / "clipping_study_table.csv"), pd.read_csv(T2 / "kl_study_table.csv")
rows += [h(T2 / f"heldout_{r.fork}.json", f"PPO ε={r.eps}", "PPO ε-fork") for r in ct.itertuples() if r.eps != 0.2]
rows += [h(T2 / f"heldout_{r.fork}.json", f"PPO β_KL={r.kl_beta}", "PPO β-fork") for r in kt.itertuples()]
rows += [h(T3 / f"heldout_{n}.json", lab, "GRPO") for n, lab in [("midpoint", "GRPO midpoint"), ("standard", "GRPO standard (20)"),
                                                                  ("fork_grpo", "GRPO fork (1/T)"), ("fork_dr_grpo", "Dr.GRPO fork (1/L)")]]
cross = pd.DataFrame(rows).set_index("condition")
display(cross.round(4))
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
fams = list(dict.fromkeys(cross.family))
for i, f in enumerate(fams):
    d = cross[cross.family == f]
    for ax, (x, y) in zip(axes, [("kl_token", "reward"), ("len_mean", "reward"), ("kl_token", "entropy")]):
        ax.scatter(d[x], d[y], label=f, color=PALETTE[i], s=28)
        for n, r in d.iterrows():
            ax.annotate(n.replace("PPO ", "").replace("GRPO ", "G:"), (r[x], r[y]), fontsize=6, xytext=(3, 2), textcoords="offset points")
for ax, (x, y) in zip(axes, [("KL to SFT (token mean)", "held-out RM reward"), ("mean length (tokens)", "held-out RM reward"), ("KL to SFT (token mean)", "entropy")]):
    ax.set_xlabel(x); ax.set_ylabel(y)
axes[0].legend(fontsize=7)
savefig(fig, "t6_reward_kl_length"); plt.show()
print("Spearman across all conditions: reward~length =", round(cross.reward.corr(cross.len_mean, method="spearman"), 3),
      "| reward~KL =", round(cross.reward.corr(cross.kl_token, method="spearman"), 3))'''),

md(r"""
### 1b. Are the reward differences larger than sampling noise? (paired bootstrap over prompts)
Every condition is scored on the same 165 held-out prompts with one sample each, so differences to SFT can be
paired by prompt. The table gives the mean reward difference to SFT and a 95 % percentile bootstrap interval
(10,000 resamples of prompts, seed 6304). An interval that contains 0 means the difference is not
distinguishable from sampling noise at this sample size.
"""),
code(r'''
rng = np.random.default_rng(6304)
def per_prompt(path):
    d = jl(path)
    return d.set_index("prompt_id")["reward"]
files = {"SFT": T2 / "heldout_sft.jsonl", "DPO standard": T2 / "heldout_dpo_standard.jsonl",
         "PPO midpoint": T2 / "heldout_midpoint.jsonl", "PPO standard (20)": T2 / "heldout_standard.jsonl",
         "GRPO midpoint": T3 / "heldout_midpoint.jsonl", "GRPO standard (20)": T3 / "heldout_standard.jsonl",
         "GRPO fork (1/T)": T3 / "heldout_fork_grpo.jsonl", "Dr.GRPO fork (1/L)": T3 / "heldout_fork_dr_grpo.jsonl"}
files.update({f"PPO ε={r.eps}": T2 / f"heldout_{r.fork}.jsonl" for r in ct.itertuples()})
files.update({f"PPO β_KL={r.kl_beta}": T2 / f"heldout_{r.fork}.jsonl" for r in kt.itertuples()})
base = per_prompt(files["SFT"])
ci = []
for name, path in files.items():
    if name == "SFT" or not Path(path).exists():
        continue
    x = per_prompt(path).reindex(base.index)
    diff = (x - base).dropna().to_numpy()
    boots = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ci.append({"condition": name, "n_prompts": len(diff), "mean_reward_diff_vs_SFT": diff.mean(), "ci95_low": lo, "ci95_high": hi,
               "excludes_zero": bool(lo > 0 or hi < 0)})
ci = pd.DataFrame(ci).set_index("condition")
display(ci.round(4))
ci.round(4).to_csv("results/report_table_task6_reward_ci.csv")'''),
md(r"""
## 2. DPO β, PPO β_KL and GRPO normalisation side by side (each on its own task protocol)
"""),
code(r'''
dpo = pd.read_csv(T1 / "dpo_summary_table.csv").set_index("condition")
display(Markdown("**DPO (Task 1 protocol: DPO held-out prompts, ≤ 256 tokens)**"))
display(dpo[["budget", "beta", "heldout_pref_acc", "kl_token_mean", "rm_score_mean", "len_mean", "entropy"]].round(4))
display(Markdown("**PPO KL pressure (Task 2 protocol)**"))
display(kt.set_index("kl_beta")[["heldout_reward", "heldout_kl_token", "heldout_entropy", "heldout_len_mean"]].round(4))
display(Markdown("**GRPO normalisation (Task 3 protocol)**"))
nt = pd.read_csv(T3 / "normalization_table.csv").set_index("fork")
display(nt[[c for c in ["heldout_reward", "heldout_kl_token", "heldout_entropy", "heldout_len_mean", "gradnorm_long_over_short"] if c in nt.columns]].round(4))'''),

md(r"""
## 3. Optimisation biases in one table
"""),
code(r'''
lt = pd.read_csv(T1 / "length_confounding_table.csv").set_index("condition")
gsz, gbin = pd.read_csv(T3 / "group_size_table.csv").set_index("K"), pd.read_csv(T3 / "group_size_binary_reward.csv").set_index("K")
bias = {
    "DPO standard: acc preferred-longer − acc rejected-longer": lt.loc["standard", "acc_preferred_longer"] - lt.loc["standard", "acc_rejected_longer"],
    "DPO length-balanced: same gap": lt.loc["length_balanced", "acc_preferred_longer"] - lt.loc["length_balanced", "acc_rejected_longer"],
    "DPO standard − SFT generated length (tokens)": lt.loc["standard", "gen_len_mean"] - lt.loc["sft", "gen_len_mean"],
    "DPO standard word-limit compliance − SFT": lt.loc["standard", "word_limit_compliance"] - lt.loc["sft", "word_limit_compliance"],
    "PPO cached clip fraction ε=0.05 / 0.20 / 0.50": " / ".join(f"{v:.3f}" for v in ct.sort_values("eps").cached_clip_fraction),
    "PPO β_KL=0 − β_KL=0.2 held-out reward": kt.set_index("kl_beta").heldout_reward[0.0] - kt.set_index("kl_beta").heldout_reward[0.2],
    "PPO β_KL=0 − β_KL=0.2 held-out KL": kt.set_index("kl_beta").heldout_kl_token[0.0] - kt.set_index("kl_beta").heldout_kl_token[0.2],
    "GRPO informative rate K=2/4/8 (continuous)": " / ".join(f"{v:.3f}" for v in gsz.informative_group_rate),
    "GRPO informative rate K=2/4/8 (binary)": " / ".join(f"{v:.3f}" for v in gbin.informative_group_rate),
}
if "gradnorm_long_over_short" in nt.columns:
    bias["GRPO long/short grad-norm ratio (1/T vs 1/L)"] = f"{nt.loc['fork_grpo', 'gradnorm_long_over_short']:.3f} vs {nt.loc['fork_dr_grpo', 'gradnorm_long_over_short']:.3f}"
display(pd.Series(bias, dtype=object).to_frame("value"))'''),

md(r"""
## 4. Safety calibration vs preference/reward metrics
"""),
code(r'''
cal = pd.read_csv(T4 / "safety_calibration_table.csv").set_index("policy")
pol_map = {"sft": "SFT", "dpo": "DPO standard", "ppo": "PPO standard (20)", "grpo": "GRPO standard (20)"}
saf = cal[["safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate", "unsafe_justified_refusal_rate", "ambiguous_rate", "mean_response_tokens"]].copy()
saf["held-out RM reward (common protocol)"] = [cross.loc[pol_map[p], "reward"] for p in saf.index]
saf["KL to SFT (common protocol)"] = [cross.loc[pol_map[p], "kl_token"] for p in saf.index]
display(saf.round(4))
fig, ax = plt.subplots(figsize=(5, 3.4))
for i, p in enumerate(saf.index):
    ax.scatter(saf.loc[p, "safe_over_refusal_rate"], saf.loc[p, "unsafe_compliance_rate"], s=60, color=PALETTE[i], label=f"{p} (RM {saf.loc[p, 'held-out RM reward (common protocol)']:.2f})")
ax.set_xlabel("over-refusal rate on safe prompts"); ax.set_ylabel("unsafe-compliance rate"); ax.legend(fontsize=7)
ax.set_title("calibration plane (lower-left is better)")
savefig(fig, "t6_safety_plane"); plt.show()'''),

md(r"""
## 5. Reward-source design and compute
"""),
code(r'''
fc = js(T5 / "feedback_comparison.json")
display(pd.read_csv(T5 / "report_table_task5_feedback_source.csv", index_col=0))
comp = []
for task, p in [("DPO standard (1 epoch)", T1 / "train_summary_standard.json"), ("PPO standard (20 updates)", T2 / "train_summary_standard.json"),
                ("GRPO standard (20 updates)", T3 / "train_summary_standard.json")]:
    s = js(p)
    comp.append({"run": task, "wall-clock (min)": s["wall_clock_s"] / 60, "peak VRAM alloc (GiB)": s["peak_vram"]["allocated_gib"],
                 "generated tokens": s.get("generated_tokens_total", 0), "models resident on GPU":
                 {"DPO": "policy (+LoRA-off reference)", "PPO": "policy, critic, RM", "GRPO": "policy, RM"}[task.split()[0]],
                 # Scheduling on the RTX 4090: the standard PPO/GRPO continuations ran alone on the GPU; the
                 # DPO run shared the GPU with other evaluation jobs, so its wall-clock is an upper bound.
                 "GPU exclusive during run": task.split()[0] in ("PPO", "GRPO")})
comp = pd.DataFrame(comp).set_index("run")
display(comp.round(2))
comp.round(3).to_csv("results/report_table_task6_compute.csv"); cross.round(4).to_csv("results/report_table_task6_cross.csv"); saf.round(4).to_csv("results/report_table_task6_safety.csv")'''),
]

build("Task6_Synthesis", cells)
