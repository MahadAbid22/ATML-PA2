from _nb import SETUP, build, code, md

cells = [
md(r"""
# Task 1 — Direct Preference Optimization (DPO)

**What this notebook does.** It (1) validates and fixes the deliberate defect in the starter DPO objective,
(2) trains the standard one-epoch DPO adapter from Qwen2.5-1.5B-Instruct with LoRA, (3) runs the
β ∈ {0.03, 0.10, 0.30} short-run regularisation study, (4) trains the length-balanced DPO model and
evaluates length confounding per stratum, and (5) surfaces qualitative examples.

All heavy work is done by the Python modules in `task1_dpo/` (each in a fresh process to keep RAM/VRAM
safe). Every cell below that launches training **skips automatically if the output already exists**, so
re-running the notebook only re-does the analysis. Raw logs: `results/logs/t1.log`; machine-readable
results: `results/task1_dpo/`.

| Item | Setting (from `configs/dpo.yaml`) |
|---|---|
| Policy / reference | Qwen2.5-1.5B-Instruct + LoRA (r=8, α=16, dropout 0.05, q_proj/v_proj); reference = same weights with LoRA disabled |
| Optimiser | AdamW, lr 2e-5, constant, weight decay 0, grad-clip 1.0 |
| Batch | 2 pairs × 8 accumulation = 16 pairs / step |
| Max sequence length | 768 tokens. **Overlength filter** (TA-approved option): a pair is kept only if prompt + max(chosen, rejected) + EOS ≤ 768, so nothing is ever truncated; applied identically to every DPO training and evaluation file |
| Standard run | β = 0.10, 1 epoch over the standard training file |
| β study | β ∈ {0.03, 0.10, 0.30}, first 600 (fitting) training pairs, otherwise identical |
| Generation protocol | 1 sample / held-out prompt (prompts ≤ 512 tokens), T = 0.7, top-p 0.9, ≤ 256 new tokens, seed 6304 |
| Reward model | `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback` (8-bit, frozen) |
"""),
code(SETUP + r'''
cfg = load_yaml("configs/dpo.yaml")
R = Path(cfg["results_dir"])
set_log_file("results/logs/t1_notebook.log")'''),

md(r"""
## 1. Validating the DPO objective (deliberate starter defect)

The manual defines

$$\mathcal L_{\text{DPO}} = -\mathbb E\Big[\log\sigma\big(\beta\,[\underbrace{(\log\pi_\theta(y^+|x)-\log\pi_\theta(y^-|x))}_{\text{policy margin}} - \underbrace{(\log\pi_{\text{ref}}(y^+|x)-\log\pi_{\text{ref}}(y^-|x))}_{\text{reference margin}}]\big)\Big].$$

The starter computed `beta * (policy_margin + ref_margin)`. With the sign error the loss is **not** `log 2`
at initialisation (when π_θ = π_ref) and the policy is rewarded for agreeing with whatever the reference
already prefers, instead of for *changing* its relative preference. The fix is `policy_margin - ref_margin`.
"""),
code(r'''
import inspect, task1_dpo.dpo as dpo_mod
src = inspect.getsource(dpo_mod.dpo_loss)
print(src[src.index("# FIX"):src.index("loss = -F.logsigmoid")])
from scripts.smoke_tests import test_dpo, RESULTS
test_dpo()
assert all(ok for _, ok, _ in RESULTS), "objective validation failed"'''),

md(r"""
### 1b. Numerical demonstration on real held-out pairs (reference log-probs)
At initialisation π_θ = π_ref, so the correct loss is exactly log 2 ≈ 0.693 for *every* pair; the buggy
version instead depends on the reference model's own preference.
"""),
code(r'''
p = R / "eval_sft_pairs.jsonl"
if p.exists():
    import torch, torch.nn.functional as F
    d = jl(p)
    rc, rr = torch.tensor(d.ref_chosen.values), torch.tensor(d.ref_rejected.values)
    beta = 0.1
    correct = -F.logsigmoid(beta * ((rc - rr) - (rc - rr))).mean().item()
    buggy = -F.logsigmoid(beta * ((rc - rr) + (rc - rr))).mean().item()
    print(f"held-out pairs: {len(d)} | correct loss at init = {correct:.4f} (log2 = {np.log(2):.4f}) | buggy loss at init = {buggy:.4f}")
    print(f"reference model already prefers the chosen response on {(rc > rr).float().mean():.3f} of held-out pairs (sequence log-prob)")
else:
    print("run the SFT evaluation first")'''),

md(r"""
## 2. The preference data and its length structure (a property of the **dataset**)
"""),
code(r'''
from task1_dpo.analyze_length import dataset_length_stats
from common.models import load_tokenizer
p = R / "length_dataset_stats.json"
if p.exists():
    ds = pd.DataFrame(js(p))
else:
    tok = load_tokenizer(cfg["base_model"])
    ds = pd.DataFrame([dataset_length_stats(read_jsonl(cfg["paths"][k]), tok, k) for k in ["dpo_standard_train", "dpo_length_train", "dpo_standard_eval", "dpo_length_eval"]])
display(ds.drop(columns=["strata_counts", "dropped_by_stratum"], errors="ignore").set_index("dataset").T)
for name in ["dpo_standard_train", "dpo_standard_eval"]:
    ids = js(R / "train_ids_standard.json") if name == "dpo_standard_train" and (R / "train_ids_standard.json").exists() else None
    if ids:
        print(f"standard training pairs used: {ids['n_used']} (dropped {ids['n_dropped_overlength']} whose prompt + longer response > {cfg['max_sequence_length']} tokens)")'''),
code(r'''
from common.models import load_tokenizer
from common.data import preference_responses
tok = load_tokenizer(cfg["base_model"])
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=False)
for ax, key, title in [(axes[0], "dpo_standard_train", "standard train"), (axes[1], "dpo_length_train", "length-balanced train")]:
    rows = read_jsonl(cfg["paths"][key])
    diff = []
    for r in rows:
        c, j = preference_responses(r)
        diff.append(len(tok(c, add_special_tokens=False)["input_ids"]) - len(tok(j, add_special_tokens=False)["input_ids"]))
    diff = np.clip(diff, -600, 600)
    ax.hist(diff, bins=60, color=PALETTE[0])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{title}: P(chosen longer)={np.mean(np.array(diff) > 0):.2f}")
    ax.set_xlabel("tokens(chosen) - tokens(rejected)  [clipped to ±600]")
axes[0].set_ylabel("pairs")
savefig(fig, "t1_dataset_length_difference"); plt.show()'''),

md(r"""
## 3. Standard DPO (β = 0.10, one epoch)
Training and evaluation are launched below (skipped if already done). The loop:
for each micro-batch, chosen and rejected sequences share one forward pass; the frozen reference
log-probs are computed with the LoRA adapter disabled (`reference_mode`); the corrected `dpo_loss` is
divided by the accumulation window and back-propagated; every 8 micro-batches the LoRA gradients are
clipped to norm 1 and AdamW steps.
"""),
code(r'''
run_module("task1_dpo.train", "--config", "configs/dpo.yaml", "--run-name", "standard",
           skip_if_exists="outputs/task1_dpo/standard/adapter_model.safetensors")
run_module("task1_dpo.evaluate", "--config", "configs/dpo.yaml", "--adapter", "none", "--name", "sft",
           skip_if_exists="results/task1_dpo/eval_sft.json")
run_module("task1_dpo.evaluate", "--config", "configs/dpo.yaml", "--adapter", "outputs/task1_dpo/standard", "--name", "standard",
           skip_if_exists="results/task1_dpo/eval_standard.json")
print("\n--- training log (results/logs/t1.log) ---")
grep_log("results/logs/t1.log", "[standard] step")
display(pd.Series(js(R / "train_summary_standard.json")).to_frame("standard run"))'''),
code(r'''
def plot_train(names, title, fname, smooth=3):
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.8))
    for i, n in enumerate(names):
        p = R / f"train_log_{n}.jsonl"
        if not p.exists():
            continue
        d = jl(p)
        x = d.step
        for ax, col, lab in zip(axes, ["loss", "acc", "margin", "grad_norm", None], ["DPO loss (train)", "train pref. accuracy", "implicit reward margin  β·m", "grad-norm (pre-clip)", "implicit rewards"]):
            if col is None:
                ax.plot(x, d.chosen_reward.rolling(smooth, min_periods=1).mean(), color=PALETTE[i], label=f"{n}: chosen")
                ax.plot(x, d.rejected_reward.rolling(smooth, min_periods=1).mean(), color=PALETTE[i], ls="--", label=f"{n}: rejected")
            else:
                ax.plot(x, d[col].rolling(smooth, min_periods=1).mean(), color=PALETTE[i], label=n)
            ax.set_title(lab); ax.set_xlabel("optimizer step")
    axes[0].axhline(np.log(2), color="grey", lw=0.8, ls=":")
    axes[0].legend(); axes[4].legend(fontsize=6)
    fig.suptitle(title, y=1.04)
    savefig(fig, fname); plt.show()

plot_train(["standard"], "Standard DPO training curves (rolling mean over 3 steps)", "t1_standard_training")'''),
code(r'''
def eval_row(name):
    s = js(R / f"eval_{name}.json")
    h, g, w = s["heldout"], s.get("generation", {}), s.get("word_limit", {})
    return {"condition": name, "beta": s["beta"],
            "heldout DPO loss (own β)": h["dpo_loss_own_beta"], "heldout DPO loss (β=0.1)": h["dpo_loss_beta_0.10"],
            "pref. accuracy": h["preference_accuracy"], "ties": h["tie_fraction"], "mean margin m_θ": h["margin_mean"],
            "KL token-mean": g.get("kl_token_mean"), "KL per-seq": g.get("kl_seq_mean"),
            "RM score": g.get("reward_mean"), "RM std": g.get("reward_std"),
            "len mean": g.get("length", {}).get("mean"), "len std": g.get("length", {}).get("std"), "len IQR": g.get("length", {}).get("iqr"),
            "trunc. rate": g.get("truncation_rate"), "entropy": g.get("entropy_token_mean"),
            "word-limit compliance": w.get("compliance_rate")}
std_tab = pd.DataFrame([eval_row(n) for n in ["sft", "standard"] if (R / f"eval_{n}.json").exists()]).set_index("condition")
display(std_tab.T)'''),
md(r"""
*Notes on the conventions.* Preference accuracy counts pairs with m_θ > 0 (the SFT reference has m_θ = 0
for every pair by construction, i.e. 100 % ties). The KL column is the course's sampled-response estimator
`sampled_kl` = mean over all generated response tokens of log π_θ − log π_ref (token-pooled), and the
per-sequence column sums it over each response; both are reported for every condition. Responses are
sampled at T = 0.7 while log-probs are evaluated at T = 1, so this estimator can be slightly negative for
policies very close to the reference.
"""),

md(r"""
## 4. Regularisation-strength study: β ∈ {0.03, 0.10, 0.30}
Short runs from the **original** initialisation on the first 600 fitting training pairs (≈38 optimizer
steps), same seed/optimizer/LoRA. The standard run uses a different budget (1 epoch ≈ 91 steps) and is
shown only as a reference row.
"""),
code(r'''
run_module("task1_dpo.ablate_beta", "--config", "configs/dpo.yaml")
beta_tab = pd.read_csv(R / "dpo_summary_table.csv").set_index("condition")
display(beta_tab.T)'''),
code(r'''
plot_train(["beta_0.03", "beta_0.10", "beta_0.30"], "β-study training curves (short runs, 600 pairs)", "t1_beta_training")
b = beta_tab.loc[[c for c in beta_tab.index if c.startswith("beta_")]].sort_values("beta")
fig, axes = plt.subplots(1, 5, figsize=(15, 2.6))
cols = [("heldout_pref_acc", "held-out pref. accuracy"), ("heldout_margin_mean", "mean margin m_θ (nats)"),
        ("kl_token_mean", "KL to ref (token mean)"), ("rm_score_mean", "RM score of generations"), ("len_mean", "generated length (tokens)")]
for ax, (c, lab) in zip(axes, cols):
    ax.plot(b.beta, b[c], "o-", color=PALETTE[0], label="short runs")
    if "standard" in beta_tab.index:
        ax.axhline(beta_tab.loc["standard", c], color=PALETTE[3], ls="--", lw=1, label="standard (1 epoch, β=0.1)")
    if "sft" in beta_tab.index and c not in ("heldout_pref_acc", "heldout_margin_mean"):
        ax.axhline(beta_tab.loc["sft", c], color="grey", ls=":", lw=1, label="SFT reference")
    ax.set_xscale("log"); ax.set_xticks(b.beta); ax.set_xticklabels(b.beta); ax.set_xlabel("β"); ax.set_title(lab)
axes[0].legend(fontsize=6)
savefig(fig, "t1_beta_study"); plt.show()'''),
md(r"""
Monotonicity check over the tested range (sign of successive differences as β increases):
"""),
code(r'''
for c in ["heldout_pref_acc", "heldout_margin_mean", "kl_token_mean", "kl_seq_mean", "rm_score_mean", "len_mean"]:
    v = b[c].values
    diffs = np.sign(np.diff(v))
    print(f"{c:22s} values={np.round(v, 4)}  monotonic={'yes' if (diffs >= 0).all() or (diffs <= 0).all() else 'no'}")'''),

md(r"""
## 5. Length-confounding study
The length-balanced file has 500 pairs in each stratum (preferred-longer / length-matched /
rejected-longer). We train a second DPO model on it (one epoch, same settings as standard) and evaluate
**both** models on the supplied length-stratified held-out set (82 pairs per stratum), on generated
length, and on explicit word-limit compliance (10 fixed prompts × 5 samples).
"""),
code(r'''
run_module("task1_dpo.analyze_length", "--config", "configs/dpo.yaml")
lt = pd.read_csv(R / "length_confounding_table.csv").set_index("condition")
display(lt.T)'''),
code(r'''
conds = [c for c in ["sft", "standard", "length_balanced"] if c in lt.index]
strata = ["preferred_longer", "length_matched", "rejected_longer"]
fig, axes = plt.subplots(1, 3, figsize=(13, 2.9))
x = np.arange(len(strata)); w = 0.8 / max(len(conds), 1)
for i, c in enumerate(conds):
    axes[0].bar(x + i * w, [lt.loc[c, f"acc_{s}"] for s in strata], w, label=c, color=PALETTE[i])
    axes[1].bar(x + i * w, [lt.loc[c, f"margin_{s}"] for s in strata], w, label=c, color=PALETTE[i])
for ax, t in zip(axes[:2], ["held-out pref. accuracy by stratum", "mean margin m_θ by stratum"]):
    ax.set_xticks(x + w * (len(conds) - 1) / 2); ax.set_xticklabels(["pref. longer", "matched", "rej. longer"]); ax.set_title(t)
axes[0].axhline(0.5, color="grey", ls=":", lw=0.8); axes[0].legend(fontsize=7)
gl = {c: jl(R / f"eval_{c}_generations.jsonl").response_tokens for c in conds + [c for c in ["beta_0.03", "beta_0.10", "beta_0.30"] if (R / f"eval_{c}_generations.jsonl").exists()]}
axes[2].boxplot(list(gl.values()), tick_labels=list(gl.keys()), showfliers=False)
axes[2].set_title("generated response length (tokens)"); axes[2].tick_params(axis="x", rotation=30)
savefig(fig, "t1_length_confounding"); plt.show()'''),
code(r'''
wl = pd.concat([jl(R / f"eval_{c}_wordlimit.jsonl").assign(condition=c) for c in conds], ignore_index=True)
wl["limit"] = wl.prompt.str.extract(r"(\d+)\s+words").astype(float)
summary_wl = wl.groupby("condition").agg(compliance=("compliant", "mean"), words_mean=("words", "mean"), words_std=("words", "std"),
                                          over_limit_words=("words", lambda s: np.nan), reward_mean=("reward", "mean"))
summary_wl["mean words over limit (violations)"] = wl.assign(o=wl.words - wl.limit).query("o > 0").groupby("condition").o.mean()
display(summary_wl.drop(columns=["over_limit_words"]))
per_prompt = wl.groupby(["prompt_id", "limit", "condition"]).agg(words=("words", "mean"), compliance=("compliant", "mean")).unstack("condition")
display(per_prompt.round(2))
fig, ax = plt.subplots(figsize=(7, 2.8))
for i, c in enumerate(conds):
    d = wl[wl.condition == c]
    ax.scatter(d.limit + (i - 1) * 0.6, d.words, s=10, alpha=0.6, label=c, color=PALETTE[i])
lims = np.sort(wl.limit.unique()); ax.plot(lims, lims, "k--", lw=0.8, label="word limit")
ax.set_xlabel("requested word limit"); ax.set_ylabel("words in response"); ax.legend(fontsize=7)
savefig(fig, "t1_word_limit"); plt.show()'''),

md(r"""
## 6. Qualitative evidence
**(i) Preference/reward vs. quality disagreement.** Candidates are found automatically:
(a) word-limit responses whose reward-model score is *higher* than a compliant response to the same prompt
even though they violate the explicit limit; (b) held-out generations where DPO raised the RM score over
SFT the most, together with whether the DPO response ran into the 256-token cap.
"""),
code(r'''
best = []
for pid, g in wl[wl.condition.isin(["standard", "length_balanced", "sft"])].groupby("prompt_id"):
    comp, viol = g[g.compliant == 1.0], g[g.compliant == 0.0]
    if len(comp) and len(viol):
        v, c = viol.loc[viol.reward.idxmax()], comp.loc[comp.reward.idxmin()]
        if v.reward > c.reward:
            best.append((v.reward - c.reward, pid, v, c))
best.sort(key=lambda t: -t[0])
print(f"{len(best)} prompts where a limit-violating response out-scores a compliant one under the RM\n")
for gap, pid, v, c in best[:3]:
    print(f"=== {pid}: {v.prompt}")
    print(f"  VIOLATES ({v.condition}, {v.words} words, RM={v.reward:.2f}): {excerpt(v.response, 260)}")
    print(f"  COMPLIES ({c.condition}, {c.words} words, RM={c.reward:.2f}): {excerpt(c.response, 260)}\n")'''),
code(r'''
gs, gd = jl(R / "eval_sft_generations.jsonl").set_index("prompt_id"), jl(R / "eval_standard_generations.jsonl").set_index("prompt_id")
j = gd.join(gs, lsuffix="_dpo", rsuffix="_sft")
j["rm_gain"] = j.reward_dpo - j.reward_sft
j["len_gain"] = j.response_tokens_dpo - j.response_tokens_sft
print("correlation between RM gain and length gain (DPO - SFT, same prompt):", round(j[["rm_gain", "len_gain"]].corr().iloc[0, 1], 3))
print("share of prompts where DPO response hit the 256-token cap:", round(j.truncated_dpo.mean(), 3), "vs SFT", round(j.truncated_sft.mean(), 3))
for pid, r in j.sort_values("rm_gain", ascending=False).head(3).iterrows():
    print(f"\n=== RM gain {r.rm_gain:+.2f} | len {r.response_tokens_sft} -> {r.response_tokens_dpo} | DPO truncated={r.truncated_dpo}")
    print("PROMPT:", excerpt(r.prompt_dpo, 200))
    print("SFT :", excerpt(r.response_sft, 300))
    print("DPO :", excerpt(r.response_dpo, 300))'''),
md(r"""
**(ii) Length bias / instruction compliance** — the same word-limited prompt answered by the standard and
the length-balanced model (first sample of each):
"""),
code(r'''
for pid in wl.prompt_id.unique()[:10]:
    g = wl[(wl.prompt_id == pid) & (wl["sample"] == 0)].set_index("condition")
    if not {"standard", "length_balanced"}.issubset(g.index):
        continue
    s, lb = g.loc["standard"], g.loc["length_balanced"]
    if s.compliant != lb.compliant or abs(s.words - lb.words) > 15:
        print(f"=== {pid}: {s.prompt}")
        for c in ["sft", "standard", "length_balanced"]:
            if c in g.index:
                print(f"  {c:16s} {g.loc[c].words:4d} words, compliant={g.loc[c].compliant}: {excerpt(g.loc[c].response, 220)}")
        print()'''),
md(r"""
**(iii) Which held-out pairs flip?** Pairs from the rejected-longer stratum that the standard model gets
*wrong* (prefers the longer rejected answer) but the length-balanced model gets right:
"""),
code(r'''
ss = jl(R / "eval_standard_stratified.jsonl").set_index("key"); sb = jl(R / "eval_length_balanced_stratified.jsonl").set_index("key")
both = ss.join(sb, lsuffix="_std", rsuffix="_bal")
flip = both[(both.length_stratum_std == "rejected_longer") & (both.margin_std < 0) & (both.margin_bal > 0)]
print(f"rejected-longer pairs: standard wrong & balanced right = {len(flip)}; "
      f"standard right & balanced wrong = {len(both[(both.length_stratum_std == 'rejected_longer') & (both.margin_std > 0) & (both.margin_bal < 0)])}")
display(both.groupby("length_stratum_std").apply(lambda d: pd.Series({
    "std_acc": (d.margin_std > 0).mean(), "bal_acc": (d.margin_bal > 0).mean(),
    "agree": ((d.margin_std > 0) == (d.margin_bal > 0)).mean(), "n": len(d)})))'''),

md(r"""
## 7. Summary table for the report
Standard (1 epoch, all fitting pairs) and the three short β runs (600 pairs) use **different budgets**.
"""),
code(r'''
cols = ["budget", "beta", "heldout_dpo_loss_own_beta", "heldout_pref_acc", "heldout_margin_mean", "kl_token_mean", "kl_seq_mean",
        "rm_score_mean", "rm_score_std", "len_mean", "len_std", "len_iqr", "word_limit_compliance"]
final = beta_tab[[c for c in cols if c in beta_tab.columns]]
display(final.round(4))
final.round(4).to_csv(R / "report_table_task1.csv")
print("saved", R / "report_table_task1.csv")'''),
]

build("Task1_DPO", cells)
