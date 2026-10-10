from _nb import SETUP, build, code, md

cells = [
md(r"""
# Task 5 — Feedback source: RLVR versus RLAIF

**What this notebook does.** Evaluates three frozen policies — SFT (no adapter), the supplied RLVR adapter
(trained with the exact GSM8K final-answer verifier) and the supplied RLAIF adapter (trained with pairwise
preferences from the frozen Qwen2.5-3B judge, group reward = (wins + 0.5·ties)/(K−1)) — on
(1) the fixed 300-problem GSM8K evaluation subset (in-domain), (2) the 100-response controlled reward
diagnostic set, scored by both feedback mechanisms, and (3) the fixed 100-problem SVAMP transfer subset
(out-of-domain). Decoding is greedy with a 512-token cap for every policy.

* Exact-answer accuracy: the released parser extracts the **last** `#### <number>` and compares numerically.
* Format compliance: a designated final answer was parsed (irrespective of correctness).
* AI pairwise win rate vs SFT: win = 1, tie = 0.5, loss = 0 (supplied judge, hash-balanced A/B order, cached).
* Verifier–judge agreement: the verifier's preference on the same (policy, SFT) pair is derived from the
  exact rewards (higher reward wins, equal = tie) and compared with the judge's A/B/TIE output.
"""),
code(SETUP + r'''
cfg = load_yaml("configs/feedback.yaml")
R = Path(cfg["results_dir"]) / "task5_feedback"
set_log_file("results/logs/t5_notebook.log")
POL = ["sft", "rlvr", "rlaif"]'''),

md(r"""
## 0. Supplied components: verifier behaviour and training metadata
"""),
code(r'''
from task5_feedback.rlvr import extract_designated_final, exact_reward
tests = ["... so the answer is 42.\n#### 42", "First 12, then 7. #### 7\nWait, recompute: #### 9", "The answer is 18", "#### 1,250", "A tempting candidate is 9, but rejected.\n#### 10"]
display(pd.DataFrame([{"response": t.replace("\n", " | "), "extracted": extract_designated_final(t)} for t in tests]))
display(pd.DataFrame({"RLVR": js("checkpoints/rlvr_metadata.json"), "RLAIF": js("checkpoints/rlaif_metadata.json")}))'''),

md(r"""
## 1. In-domain comparison (GSM8K, 300 problems)
"""),
code(r'''
run_module("task5_feedback.evaluate_math", "--config", "configs/feedback.yaml", "--dataset", "gsm", skip_if_exists="results/task5_feedback/summary_gsm.csv")
gsm = pd.read_csv(R / "summary_gsm.csv").set_index("policy")
show_cols = ["exact_accuracy", "format_compliance", "accuracy_given_format", "ai_win_rate_vs_sft", "ai_wins", "ai_ties", "ai_losses",
             "verifier_win_rate_vs_sft", "verifier_judge_agreement_3way", "verifier_judge_agreement_decisive", "n_verifier_decisive",
             "judge_decisive_when_verifier_tie", "len_mean", "len_std", "len_iqr", "truncation_rate"]
display(gsm[[c for c in show_cols if c in gsm.columns]].T.round(4))
display(Markdown("**Verifier preference (rows) vs judge preference (columns); A = RL policy, B = SFT**"))
display(pd.read_csv(R / "verifier_judge_confusion_gsm.csv", index_col=[0, 1]))'''),

md(r"""
## 2. Reward-sensitivity and robustness (controlled diagnostic set)
Every perturbation is contrasted with the clean response of the same problem (20 problems):

| pair (better ≻ worse) | what it probes |
|---|---|
| clean ≻ corrupt-reasoning (same correct final) | reasoning sensitivity, $S_{reason}$ |
| clean ≻ good-reasoning-wrong-final | outcome sensitivity, $S_{outcome}$ (reasoning held ≈ fixed) |
| clean ≻ persuasive filler (both correct) | style/verbosity susceptibility |
| clean ≻ gold-distractor-wrong-final | outcome robustness when the gold number appears as a discarded candidate |

The exact verifier "prefers" the response with the higher binary reward (equal → tie). The judge is called
with the supplied deterministic hash-based A/B orientation; we also call it with the arguments reversed as an
order-consistency check (reported separately, not mixed into the primary rates).
"""),
code(r'''
run_module("task5_feedback.score_perturbations", "--config", "configs/feedback.yaml", skip_if_exists="results/task5_feedback/diagnostic_summary.json")
cp = pd.read_csv(R / "diagnostic_controlled_pairs.csv")
display(cp.round(3))
ds = js(R / "diagnostic_summary.json")
display(pd.DataFrame({k: ds[k] for k in ["S_reason", "S_outcome_wrong_final", "S_outcome_gold_distractor", "S_outcome_pooled"]}).T.round(3))
print({k: ds[k] for k in ["filler_preference", "judge_order_consistency_all_pairs", "judge_tie_rate_all_pairs", "verifier_tie_rate_all_pairs",
                          "verifier_judge_agreement_all_pairs", "verifier_expected_reward_mismatches"]})'''),
code(r'''
prim = cp[cp.mechanism != "RLAIF judge (reversed call)"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
for ax, mech in zip(axes, ["RLVR exact verifier", "RLAIF pairwise judge"]):
    d = prim[prim.mechanism == mech].set_index("perturbation")
    left = np.zeros(len(d))
    for lab, col in [("better_rate", PALETTE[2]), ("tie_rate", "#999999"), ("wrong_rate", PALETTE[3])]:
        ax.barh(d.index, d[lab], left=left, color=col, label=lab.replace("_rate", "")); left += d[lab].values
    ax.set_title(mech); ax.set_xlim(0, 1)
axes[0].invert_yaxis(); axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
savefig(fig, "t5_diagnostics"); plt.show()
cat = pd.read_csv(R / "diagnostic_per_category.csv", index_col=0)
display(Markdown("**All five categories: mean RLVR reward and mean RLAIF round-robin group reward (K = 5 variants/problem)**")); display(cat.round(3))'''),

md(r"""
## 3. Out-of-domain comparison (SVAMP, 100 problems, no retraining)
"""),
code(r'''
run_module("task5_feedback.evaluate_math", "--config", "configs/feedback.yaml", "--dataset", "transfer", skip_if_exists="results/task5_feedback/summary_transfer.csv")
run_module("task5_feedback.compare_feedback", "--config", "configs/feedback.yaml")
io = pd.read_csv(R / "in_vs_out_of_domain.csv").set_index("policy")
display(io.T.round(4))
display(pd.read_csv(R / "accuracy_gain_vs_sft.csv").set_index("policy").round(4))
ft = pd.read_csv(R / "failure_types.csv")
display(Markdown("**Failure types (share of problems)**")); display(ft.set_index(["dataset", "policy"]).round(3))
display(Markdown("**SVAMP accuracy by problem type**")); display(pd.read_csv(R / "svamp_accuracy_by_type.csv").pivot(index="svamp_type", columns="policy", values="exact_accuracy").round(3))
display(Markdown("**Paired correctness on identical problems**")); display(pd.read_csv(R / "paired_correctness.csv"))'''),
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(13, 2.8))
x = np.arange(len(POL)); w = 0.38
for ax, m, t in zip(axes, ["exact_accuracy", "ai_win_rate_vs_sft", "len_mean"], ["exact-answer accuracy", "AI pairwise win rate vs SFT", "mean response tokens"]):
    ax.bar(x - w / 2, io[f"gsm_{m}"], w, label="GSM8K (in-domain)", color=PALETTE[0])
    ax.bar(x + w / 2, io[f"svamp_{m}"], w, label="SVAMP (transfer)", color=PALETTE[1])
    ax.set_xticks(x); ax.set_xticklabels(POL); ax.set_title(t)
axes[1].axhline(0.5, color="grey", ls=":", lw=0.8); axes[0].legend(fontsize=7)
savefig(fig, "t5_in_vs_ood"); plt.show()'''),

md(r"""
## 4. Qualitative diagnostic evidence
"""),
code(r'''
detail = pd.read_csv(R / "diagnostic_controlled_pairs_detail.csv")
diag = {(str(r["problem_id"]), r["variant_type"]): r for r in read_jsonl(cfg["paths"]["task5_diagnostics"])}
def show_pair(worse, cond, title):
    d = detail[(detail.b == worse) & cond(detail)]
    print(f"######## {title}: {len(d)} of 20 problems")
    for _, r in d.head(1).iterrows():
        pid = str(r.problem_id)
        print("PROBLEM:", excerpt(diag[(pid, 'clean_correct')]['question'], 200), "| gold =", diag[(pid, 'clean_correct')]['gold_final'])
        print(f"verifier={r.verifier}  judge={r.judge}  judge(reversed call)={r.judge_reversed_call}   [A = clean, B = {worse}]")
        print("  CLEAN :", excerpt(diag[(pid, 'clean_correct')]['response'], 300))
        print(f"  {worse}:", excerpt(diag[(pid, worse)]['response'], 300)); print()
show_pair("corrupt_reasoning_correct_final", lambda d: d.verifier == "TIE", "reasoning-only corruption: verifier ties (outcome-blind)")
show_pair("corrupt_reasoning_correct_final", lambda d: d.judge != "A", "reasoning-only corruption the judge misses")
show_pair("persuasive_filler_correct", lambda d: d.judge == "B", "judge prefers the persuasive filler")
show_pair("gold_distractor_wrong_final", lambda d: d.judge != "A", "gold distractor fools the judge")
show_pair("good_reasoning_wrong_final", lambda d: d.judge != "A", "wrong final answer the judge does not penalise")'''),
code(r'''
pw = jl(R / "pairwise_gsm.jsonl")
g = {p: jl(R / f"generated_gsm_{p}.jsonl").set_index("source_index") for p in POL}
dis = pw[(pw.policy == "rlaif") & (pw.verifier_pref == "B") & (pw.judge_pref == "A")]
print(f"RLAIF vs SFT on GSM8K: judge prefers RLAIF although only SFT is correct: {len(dis)} cases")
for _, r in dis.head(1).iterrows():
    sid = r.source_index
    print("PROBLEM:", excerpt(g['sft'].loc[sid, 'question'], 200), "| gold =", g['sft'].loc[sid, 'gold_final'])
    print("  RLAIF (wrong):", excerpt(g['rlaif'].loc[sid, 'response'], 300))
    print("  SFT (right)  :", excerpt(g['sft'].loc[sid, 'response'], 300))'''),

md(r"""
## 5. Feedback-source comparison: coverage, noise, exploitability, cost
"""),
code(r'''
fc = js(R / "feedback_comparison.json")
display(pd.Series(fc["inference_cost"]).drop("judge_detail").to_frame("cost"))
summary = pd.DataFrame({
    "coverage: fraction of controlled pairs decided (not tie)": {
        "RLVR": 1 - prim[prim.mechanism == "RLVR exact verifier"].tie_rate.mean(), "RLAIF": 1 - prim[prim.mechanism == "RLAIF pairwise judge"].tie_rate.mean()},
    "noise: wrong-preference rate on controlled pairs": {
        "RLVR": prim[prim.mechanism == "RLVR exact verifier"].wrong_rate.mean(), "RLAIF": prim[prim.mechanism == "RLAIF pairwise judge"].wrong_rate.mean()},
    "S_reason": ds["S_reason"], "S_outcome (pooled)": ds["S_outcome_pooled"],
    "exploitability: prefers persuasive filler over clean": {"RLVR": 0.0, "RLAIF": ds["filler_preference"]["RLAIF_prefers_filler"]},
    "seconds per prompt group (K=4)": {"RLVR": fc["inference_cost"]["verifier_seconds_per_prompt_group_K4"], "RLAIF": fc["inference_cost"]["judge_seconds_per_prompt_group_K4"]},
}).T
display(summary.round(4))
summary.round(4).to_csv(R / "report_table_task5_feedback_source.csv")
io.round(4).to_csv(R / "report_table_task5_domains.csv"); cp.round(4).to_csv(R / "report_table_task5_diagnostics.csv")'''),
]

build("Task5_RLVR_vs_RLAIF", cells)
