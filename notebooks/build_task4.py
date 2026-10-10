from _nb import SETUP, build, code, md

cells = [
md(r"""
# Task 4 — Safety calibration of the aligned policies (XSTest)

**What this notebook does.** Generates one deterministic (greedy, ≤ 256 new tokens) response per XSTest
prompt (250 safe + 200 unsafe) for the four frozen policies — SFT (no adapter), standard DPO (Task 1 step 1),
standard PPO (Task 2 step 1, 20 updates), standard GRPO (Task 3 step 1, 20 updates) — labels every response
with the fixed categorical AI judge (Qwen2.5-3B-Instruct, 4-bit, greedy, supplied prompt and parser), and
compares the policies on both sides of the safety trade-off. A fixed 60-prompt subset (30 safe + 30 unsafe,
seeded) is labelled manually and blind (policy hidden, no AI labels visible) for all four policies to
measure judge reliability.

Labels: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS.
Rates: safe-answer = P(SAFE_ANSWER | safe), over-refusal = P(OVER_REFUSAL | safe), unsafe-compliance =
P(UNSAFE_COMPLIANCE | unsafe), justified-refusal = P(JUSTIFIED_REFUSAL | unsafe).
"""),
code(SETUP + r'''
cfg = load_yaml("configs/feedback.yaml")
R = Path(cfg["results_dir"]) / "task4_safety"
set_log_file("results/logs/t4_notebook.log")
POL = ["sft", "dpo", "ppo", "grpo"]
LABELS = ["SAFE_ANSWER", "OVER_REFUSAL", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "AMBIGUOUS"]'''),

md(r"""
## 1. Generation (fixed policies, identical deterministic decoding)
"""),
code(r'''
run_module("task4_safety.generate_responses", "--config", "configs/feedback.yaml")
run_module("task4_safety.make_audit_sheet", "--config", "configs/feedback.yaml", skip_if_exists="results/task4_safety/manual_audit_sheet.csv")
for p in POL:
    d = jl(R / f"generated_{p}.jsonl")
    print(f"{p:5s} n={len(d)}  mean tokens={d.response_tokens.mean():.1f}  truncated at 256={d.truncated.mean():.3f}")
print("policy adapters:", {k: v for k, v in cfg["policies"].items() if k in POL})'''),

md(r"""
## 2. AI judge labels and calibration metrics
"""),
code(r'''
# All 4 x 450 AI labels are cached in judged_*.jsonl; re-judging is skipped (the judge is resumable anyway).
run_module("task4_safety.judge_responses", "--config", "configs/feedback.yaml", skip_if_exists="results/task4_safety/judged_grpo.jsonl")
run_module("task4_safety.evaluate_safety", "--config", "configs/feedback.yaml")
cal = pd.read_csv(R / "safety_calibration_table.csv").set_index("policy")
main_cols = ["safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate", "unsafe_justified_refusal_rate", "ambiguous_rate",
             "mean_response_tokens", "std_response_tokens"]
display(cal[main_cols].round(4))
display(Markdown("**Auxiliary columns** (off-class judge labels, keyword-refusal cross-check, parse failures)"))
display(cal.drop(columns=main_cols).round(4))'''),
code(r'''
counts = pd.read_csv(R / "label_counts_by_policy_class.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
colors = {"SAFE_ANSWER": PALETTE[2], "OVER_REFUSAL": PALETTE[1], "JUSTIFIED_REFUSAL": PALETTE[0], "UNSAFE_COMPLIANCE": PALETTE[3], "AMBIGUOUS": "#999999"}
for ax, cls in zip(axes, ["SAFE", "UNSAFE"]):
    d = counts[counts.benchmark_class == cls].set_index("policy").reindex(POL)[LABELS]
    d = d.div(d.sum(1), axis=0)
    left = np.zeros(len(d))
    for lab in LABELS:
        ax.barh(d.index, d[lab], left=left, color=colors[lab], label=lab); left += d[lab].values
    ax.set_title(f"{cls} prompts (n={250 if cls == 'SAFE' else 200})"); ax.set_xlim(0, 1); ax.invert_yaxis()
axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
savefig(fig, "t4_label_distribution"); plt.show()'''),

md(r"""
## 3. Category-level behaviour (all four policies)
Rows: XSTest category (safe categories first). Values: share of each judge label.
"""),
code(r'''
cat = pd.read_csv(R / "category_label_distribution.csv")
for cls in ["SAFE", "UNSAFE"]:
    key = "OVER_REFUSAL" if cls == "SAFE" else "UNSAFE_COMPLIANCE"
    good = "SAFE_ANSWER" if cls == "SAFE" else "JUSTIFIED_REFUSAL"
    d = cat[cat.benchmark_class == cls]
    display(Markdown(f"**{cls} categories — {good} / {key} / AMBIGUOUS rate per policy**"))
    display(d.pivot_table(index="type", columns="policy", values=[good, key, "AMBIGUOUS"]).reindex(columns=POL, level=1).round(2))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, cls, key in zip(axes, ["SAFE", "UNSAFE"], ["OVER_REFUSAL", "UNSAFE_COMPLIANCE"]):
    d = cat[cat.benchmark_class == cls].pivot_table(index="type", columns="policy", values=key).reindex(columns=POL)
    im = ax.imshow(d.values, cmap="Reds", vmin=0, vmax=max(0.05, d.values.max()))
    ax.set_xticks(range(len(POL))); ax.set_xticklabels(POL); ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=7)
    for (i, j), v in np.ndenumerate(d.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"{key} rate by {cls.lower()} category"); ax.grid(False)
fig.tight_layout(); savefig(fig, "t4_category_heatmap"); plt.show()'''),

md(r"""
## 4. Manual audit (blind) and judge reliability
The sheet `results/task4_safety/manual_audit_sheet.csv` (60 fixed XSTest IDs × 4 policies = 240 rows,
shuffled, policy hidden in `manual_audit_key.csv`) was labelled **before** any AI label existed.
"""),
code(r'''
sheet = pd.read_csv(R / "manual_audit_sheet.csv")
print("audit rows:", len(sheet), "| unique prompts:", sheet.xstest_id.nunique(), "| class balance:", sheet.drop_duplicates("xstest_id").benchmark_class.value_counts().to_dict())
print("manual label distribution:", sheet.manual_label.value_counts().to_dict())
summ = js(R / "safety_summary.json")["audit"]
display(pd.Series({k: v for k, v in summ.items() if k not in ("disagreement_pairs", "agreement_by_class")}).to_frame("judge vs manual"))
print("agreement by XSTest class:", summ.get("agreement_by_class"))
conf = pd.read_csv(R / "manual_audit_confusion.csv", index_col=0)
display(Markdown("**Confusion matrix (rows = manual label, columns = AI label)**")); display(conf)
fig, ax = plt.subplots(figsize=(5.2, 4))
ax.imshow(conf.values, cmap="Blues"); ax.grid(False)
ax.set_xticks(range(5)); ax.set_xticklabels([c.replace("ai_", "") for c in conf.columns], rotation=35, ha="right", fontsize=7)
ax.set_yticks(range(5)); ax.set_yticklabels([c.replace("manual_", "") for c in conf.index], fontsize=7)
for (i, j), v in np.ndenumerate(conf.values):
    ax.text(j, i, v, ha="center", va="center", fontsize=8)
ax.set_xlabel("AI judge"); ax.set_ylabel("manual"); savefig(fig, "t4_audit_confusion"); plt.show()
display(Markdown("**Policy comparison on the audit subset under each labeller**"))
display(pd.read_csv(R / "manual_audit_per_policy.csv").set_index("policy").round(3))'''),

md(r"""
## 5. Qualitative disagreement evidence
For each behaviour type we show prompts where the policies (or the judge and the manual label) disagree.
"""),
code(r'''
allr = pd.read_csv(R / "all_responses_with_labels.csv")
merged = pd.read_csv(R / "manual_audit_merged.csv") if (R / "manual_audit_merged.csv").exists() else pd.DataFrame()
def show(xid, note):
    d = allr[allr.xstest_id == xid].set_index("policy").reindex(POL)
    print(f"=== [{note}] xstest_id={xid} ({d.benchmark_class.iloc[0]}, {d.type.iloc[0]}): {d.prompt.iloc[0]}")
    for p, r in d.iterrows():
        man = ""
        if len(merged):
            m = merged[(merged.xstest_id == xid) & (merged.policy == p)]
            man = f" | manual={m.manual_label.iloc[0]}" if len(m) else ""
        print(f"  {p:4s} AI={r.ai_label:18s}{man}: {excerpt(r.response, 200)}")
    print()
piv = pd.read_csv(R / "per_prompt_policy_labels.csv")
# (i) harmful compliance by at least one policy
uc = piv[(piv[POL] == "UNSAFE_COMPLIANCE").any(axis=1)]
for x in uc.xstest_id.head(2): show(x, "harmful compliance")
# (ii) exaggerated refusal of a safe prompt by at least one policy while another answers
orf = piv[(piv[POL] == "OVER_REFUSAL").any(axis=1) & (piv[POL] == "SAFE_ANSWER").any(axis=1)]
for x in orf.xstest_id.head(2): show(x, "over-refusal vs answer")
# (iii) justified refusal agreed by all
jr = piv[(piv[POL] == "JUSTIFIED_REFUSAL").all(axis=1)]
for x in jr.xstest_id.head(1): show(x, "justified refusal (all policies)")'''),
code(r'''
if len(merged):
    dis = merged[merged.manual_label != merged.ai_label]
    print(f"judge errors on the audit subset: {len(dis)} / {len(merged)}")
    display(dis.groupby(["manual_label", "ai_label"]).size().rename("count").to_frame())
    for _, r in dis.head(6).iterrows():
        print(f"--- xstest {r.xstest_id} [{r.benchmark_class}/{r.type}] policy={r.policy}  manual={r.manual_label}  AI={r.ai_label} (conf {r.ai_confidence})")
        print("PROMPT:", r.prompt); print("RESPONSE:", excerpt(r.response, 280)); print("AUDIT NOTE:", r.get("notes", ""))'''),

md(r"""
## 6. Link to Tasks 1–3: do higher preference/reward scores mean better calibration?
"""),
code(r'''
rows = []
t1 = Path("results/task1_dpo/eval_standard.json"); t2 = Path("results/task2_ppo/heldout_standard.json"); t3 = Path("results/task3_grpo/heldout_standard.json"); t0 = Path("results/task2_ppo/heldout_sft.json")
rm = {"sft": js(t0)["reward_mean"] if t0.exists() else np.nan, "ppo": js(t2)["reward_mean"] if t2.exists() else np.nan,
      "grpo": js(t3)["reward_mean"] if t3.exists() else np.nan}
kl = {"sft": 0.0, "ppo": js(t2)["kl_token_mean"] if t2.exists() else np.nan, "grpo": js(t3)["kl_token_mean"] if t3.exists() else np.nan}
if t1.exists():
    kl["dpo"] = js(t1)["generation"]["kl_token_mean"]
link = cal[["safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate", "unsafe_justified_refusal_rate"]].copy()
link["held-out RM reward (Task 2/3 protocol)"] = pd.Series(rm)
link["KL to SFT (own task protocol)"] = pd.Series(kl)
display(link.round(4))
link.round(4).to_csv(R / "report_table_task4_link.csv"); cal.round(4).to_csv(R / "report_table_task4.csv")'''),
]

build("Task4_Safety", cells)
