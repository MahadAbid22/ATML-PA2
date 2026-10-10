from _nb import SETUP, build, code, md

cells = [
md(r"""
# Task 2 — Proximal Policy Optimization (PPO) continuation

**What this notebook does.** (1) validates/fixes the clipped-surrogate defect and checks GAE and KL shaping,
(2) runs the required 20-update continuation from the supplied PPO midpoint (policy LoRA + value model),
logging reward, KL, policy/value loss, entropy, gradient norm, clip fraction, length, peak VRAM and
wall-clock time, (3) the clipping study (cached-rollout geometry for ε ∈ {0.05, 0.20, 0.50} + matched
8-update forks), (4) the KL-pressure study (β_KL ∈ {0, 0.10, 0.20}, matched 8-update forks), and
(5) qualitative examples.

| Item | Setting (`configs/ppo.yaml`) |
|---|---|
| Policy | Qwen2.5-1.5B-Instruct + supplied PPO LoRA (trainable); reference = LoRA disabled |
| Critic | supplied 0.5B value model + fresh zero-init LoRA (q,v) + trainable score head (`lora_head`) |
| Reward | `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback` (8-bit), −1.0 penalty if the response has no EOS |
| Rollouts | 1 prompt/update, T = 0.7, top-p 0.9, ≤ 512 new tokens; prompts ≤ 256 tokens; seeded prompt order shared by all runs |
| Optimisation | 2 PPO epochs/rollout; policy lr 3e-6; critic LoRA lr 1e-4, head lr 3e-4; γ = 1, λ = 0.95; value coef 0.5; grad-clip 1.0 |
| Reference setting | ε = 0.20, β_KL = 0.10 |
| Held-out protocol | eval prompts ≤ 256 tokens, 1 sample each, T = 0.7, top-p 0.9, ≤ 768 new tokens, seed 6304 |
"""),
code(SETUP + r'''
cfg = load_yaml("configs/ppo.yaml")
R = Path(cfg["results_dir"])
set_log_file("results/logs/t2_notebook.log")'''),

md(r"""
## 1. Validating the PPO objective (deliberate starter defect)
PPO maximises the pessimistic bound $\min(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)$.
The starter used `torch.maximum`, which keeps the *optimistic* term: for A>0 and ρ>1+ε it keeps ρA (so the
ratio can grow without limit) and for A<0 and ρ<1−ε it keeps the clipped term, removing the trust region.
The tests below check the four sign/ratio cases by hand, that the gradient is zero exactly where clipping
is active, GAE against a naive loop, and the KL-shaped reward placement.
"""),
code(r'''
import inspect, task2_ppo.ppo as ppo_mod
src = inspect.getsource(ppo_mod.ppo_policy_loss)
print(src[src.index("# FIX"):src.index("loss = -masked_mean")])
from scripts.smoke_tests import test_ppo, test_gae, RESULTS
test_ppo(); test_gae()
assert all(ok for _, ok, _ in RESULTS)'''),

md(r"""
## 2. The supplied midpoint
The course notes that the released critic is weak on held-out Monte-Carlo returns (negative explained
variance); it is used as-is so that every fork starts from the identical policy/value state.
"""),
code(r'''
display(pd.Series(js("checkpoints/ppo_midpoint_acceptance.json")).to_frame("supplied PPO midpoint"))'''),

md(r"""
## 3. Standard PPO continuation (20 updates)
Per update: sample one response from the current policy (the "old" snapshot) → compute old log-probs,
reference log-probs (LoRA disabled), critic values V(s_t) and the terminal RM reward (−1 if no EOS) →
token rewards $r_t = r_{task}\mathbb 1[t=T] - \beta_{KL}(\log\pi_{old}-\log\pi_{ref})$ → GAE advantages and
returns → advantage whitening → 2 epochs of clipped policy loss + 0.5·value MSE → grad-clip → AdamW.
Dropout is disabled so the ratio is exactly 1 before the first step of each rollout.
"""),
code(r'''
run_module("task2_ppo.continue_train", "--config", "configs/ppo.yaml", "--run-name", "standard",
           skip_if_exists="outputs/task2_ppo/standard/adapter_model.safetensors")
grep_log("results/logs/t2.log", "[standard] update")
summ = js(R / "train_summary_standard.json")
display(pd.DataFrame({"wall-clock (min)": [summ["wall_clock_s"] / 60], "wall-clock excl. thermal pauses (min)": [summ["wall_clock_excluding_thermal_pauses_s"] / 60],
                      "peak VRAM allocated (GiB)": [summ["peak_vram"]["allocated_gib"]], "peak VRAM reserved (GiB)": [summ["peak_vram"]["reserved_gib"]],
                      "generated tokens": [summ["generated_tokens_total"]], "GPU": [summ["gpu"]]}, index=["standard PPO (20 updates)"]).T)'''),
code(r'''
log = jl(R / "train_log_standard.jsonl")
panels = [("reward", "learned reward (−1 if no EOS)"), ("kl_token", "KL to ref (token mean)"), ("policy_loss", "policy loss"),
          ("value_loss", "value loss (MSE)"), ("entropy", "entropy (nats/token)"), ("grad_norm_policy", "policy grad-norm (pre-clip)"),
          ("clip_fraction", "clip fraction (mean of 2 epochs)"), ("response_length", "response length (tokens)"),
          ("explained_variance", "critic explained variance"), ("post_update_approx_kl", "realised KL(old‖new) per update")]
fig, axes = plt.subplots(2, 5, figsize=(16, 5.2))
for ax, (c, t) in zip(axes.flat, panels):
    ax.plot(log["update"], log[c], "o-", ms=3, color=PALETTE[0])
    if c in ("reward", "kl_token", "response_length", "entropy"):
        ax.plot(log["update"], log[c].expanding().mean(), color=PALETTE[3], lw=1, label="running mean")
    ax.set_title(t); ax.set_xlabel("update")
axes[0, 0].legend(fontsize=7)
fig.suptitle("Standard PPO continuation from the supplied midpoint (1 rollout per update)", y=1.01)
fig.tight_layout(); savefig(fig, "t2_standard_trajectories"); plt.show()
display(log[["update", "reward", "reward_raw", "kl_token", "kl_seq", "policy_loss", "value_loss", "entropy", "grad_norm_policy",
             "clip_fraction", "clip_fraction_last_epoch", "response_length", "truncated", "explained_variance", "value_mean", "return_mean"]].round(4))'''),
code(r'''
first, last = log.head(10), log.tail(10)
print("first half vs second half (means):")
display(pd.DataFrame({"updates 1-10": first.mean(numeric_only=True), "updates 11-20": last.mean(numeric_only=True)}).loc[
    ["reward", "kl_token", "entropy", "response_length", "policy_loss", "value_loss", "explained_variance", "clip_fraction", "grad_norm_policy"]])
print("critic: Spearman(value at first token, realised return of the rollout) =",
      round(pd.Series(log.value_at_first_token).corr(pd.Series(log.reward), method="spearman"), 3))'''),

md(r"""
### Held-out evaluation: SFT reference vs supplied midpoint vs standard continuation
"""),
code(r'''
run_module("task2_ppo.evaluate", "--config", "configs/ppo.yaml", "--adapter", "outputs/task2_ppo/standard", "--name", "standard",
           skip_if_exists="results/task2_ppo/heldout_standard.json")
run_module("task2_ppo.evaluate", "--config", "configs/ppo.yaml", "--adapter", "none", "--name", "sft",
           skip_if_exists="results/task2_ppo/heldout_sft.json")
run_module("task2_ppo.evaluate", "--config", "configs/ppo.yaml", "--adapter", cfg["paths"]["ppo_midpoint_policy"], "--name", "midpoint",
           skip_if_exists="results/task2_ppo/heldout_midpoint.json")
def hrow(n):
    s = js(R / f"heldout_{n}.json")
    return {"policy": n, "RM reward": s["reward_mean"], "reward (−1 if no EOS)": s["effective_reward_mean"], "KL token-mean": s["kl_token_mean"],
            "KL per-seq": s["kl_seq_mean"], "entropy": s["entropy_token_mean"], "len mean": s["length"]["mean"], "len std": s["length"]["std"],
            "len IQR": s["length"]["iqr"], "truncation rate (768)": s["truncation_rate"], "n prompts": s["n_prompts"]}
held = pd.DataFrame([hrow(n) for n in ["sft", "midpoint", "standard"]]).set_index("policy")
display(held)'''),

md(r"""
## 4. Clipping study
**(a) Cached-rollout geometry.** The supplied cache holds 32 rollouts with the behaviour policy's
log-probs, reference log-probs and critic values. We rebuild the token sequences exactly as the release
`batch_generate` did (including its right-truncation of prompts longer than 256 tokens — 3 of the 32),
recompute advantages with KL-shaped rewards (β_KL = 0.1) and GAE, and compare the supplied midpoint's
log-probs with the cached ones. *Clip fraction* = share of valid tokens with ρ outside [1−ε, 1+ε];
*affected-token fraction* = share where the min actually selects the clipped constant (ρ>1+ε with A>0 or
ρ<1−ε with A<0), i.e. tokens whose gradient is zeroed.

**(b) Replay.** From the identical midpoint weights we run the release update (2 epochs × minibatches of 4
rollouts, lr 3e-6) on this fixed batch for each ε and track the geometry before every optimizer step.

**(c) Matched forks.** 8-update continuations from the same checkpoint with identical prompts, sampling
seeds, reward and β_KL; only ε changes. The ε = 0.20 fork is the reference setting (shared with the KL study).
"""),
code(r'''
run_module("task2_ppo.analyze_clipping", "--config", "configs/ppo.yaml")
rec = jl(R / "clipping_cached_reconstruction.jsonl")
print("reconstruction fidelity (|log π_midpoint − cached old log-prob| per rollout):")
display(rec.describe().T[["mean", "50%", "max"]].round(4))
static = pd.read_csv(R / "clipping_cached_static.csv")
after = pd.read_csv(R / "clipping_cached_after_replay.csv")
display(Markdown("**(a) static geometry: supplied midpoint vs cached behaviour policy**")); display(static.round(5))
display(Markdown("**(b) after replaying the PPO update on the cached batch (16 optimizer steps)**")); display(after.round(5))'''),
code(r'''
rl = jl(R / "clipping_cached_replay_log.jsonl")
fig, axes = plt.subplots(1, 4, figsize=(15, 2.8))
for i, (e, g) in enumerate(rl.groupby("eps")):
    for ax, c, t in zip(axes, ["clip_fraction", "affected_token_fraction", "max_abs_log_ratio", "clipped_surrogate"],
                        ["clip fraction (before step)", "affected-token fraction", "max |log ρ| in minibatch", "clipped surrogate"]):
        ax.plot(g.step, g[c], "o-", ms=3, label=f"ε={e}", color=PALETTE[i]); ax.set_title(t); ax.set_xlabel("optimizer step on cached batch")
axes[0].legend()
savefig(fig, "t2_clipping_replay"); plt.show()'''),
code(r'''
ct = pd.read_csv(R / "clipping_study_table.csv").set_index("eps")
display(ct.T)
fig, axes = plt.subplots(1, 5, figsize=(16, 2.7))
for i, e in enumerate(ct.index):
    d = jl(R / f"train_log_{ct.loc[e, 'fork']}.jsonl")
    for ax, c, t in zip(axes, ["reward", "kl_token", "clip_fraction_last_epoch", "post_update_approx_kl", "grad_norm_policy"],
                        ["reward", "KL to ref", "clip fraction (2nd epoch)", "realised KL(old‖new)", "grad-norm"]):
        ax.plot(d["update"], d[c], "o-", ms=3, label=f"ε={e}", color=PALETTE[i]); ax.set_title(t); ax.set_xlabel("update")
axes[0].legend(); savefig(fig, "t2_clipping_forks"); plt.show()'''),

md(r"""
## 5. KL-pressure (reward over-optimisation) study: β_KL ∈ {0, 0.10, 0.20}
"""),
code(r'''
run_module("task2_ppo.ablate_kl", "--config", "configs/ppo.yaml")
kt = pd.read_csv(R / "kl_study_table.csv").set_index("kl_beta")
display(kt.T)
fig, axes = plt.subplots(1, 5, figsize=(16, 2.7))
for i, b in enumerate(kt.index):
    d = jl(R / f"train_log_{kt.loc[b, 'fork']}.jsonl")
    for ax, c, t in zip(axes, ["reward", "kl_token", "entropy", "response_length", "kl_penalty_total"],
                        ["reward", "KL to ref (token mean)", "entropy", "response length", "KL penalty in return"]):
        ax.plot(d["update"], d[c], "o-", ms=3, label=f"β_KL={b}", color=PALETTE[i]); ax.set_title(t); ax.set_xlabel("update")
axes[0].legend(); savefig(fig, "t2_kl_forks"); plt.show()
fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
for ax, c, t in zip(axes, ["heldout_reward", "heldout_kl_token", "heldout_entropy", "heldout_len_mean"], ["held-out RM reward", "held-out KL", "held-out entropy", "held-out length"]):
    ax.plot(kt.index, kt[c], "o-", color=PALETTE[0]); ax.set_title(t); ax.set_xlabel("β_KL")
    mid = js(R / "heldout_midpoint.json")
    ref = {"heldout_reward": mid["reward_mean"], "heldout_kl_token": mid["kl_token_mean"], "heldout_entropy": mid["entropy_token_mean"], "heldout_len_mean": mid["length"]["mean"]}[c]
    ax.axhline(ref, color="grey", ls=":", lw=1, label="midpoint")
axes[0].legend(fontsize=7); savefig(fig, "t2_kl_heldout"); plt.show()'''),
md(r"""
Which observable moves first when KL pressure is removed? Per-update differences between the β_KL = 0
fork and the β_KL = 0.10 reference fork (same prompts and sampling seeds, so update k sees the same prompt):
"""),
code(r'''
d0, d1 = jl(R / f"train_log_{kt.loc[0.0, 'fork']}.jsonl").set_index("update"), jl(R / f"train_log_{kt.loc[0.1, 'fork']}.jsonl").set_index("update")
display((d0[["reward", "kl_token", "kl_seq", "entropy", "response_length", "post_update_approx_kl"]] -
         d1[["reward", "kl_token", "kl_seq", "entropy", "response_length", "post_update_approx_kl"]]).round(4).rename(columns=lambda c: f"Δ {c}"))'''),

md(r"""
## 6. Qualitative evidence: does reward track quality?
Same held-out prompt, responses from the supplied midpoint and from the β_KL = 0 fork (largest reward
changes in both directions), plus the standard continuation.
"""),
code(r'''
def pair(a, b):
    A, B = jl(R / f"heldout_{a}.jsonl").set_index("prompt_id"), jl(R / f"heldout_{b}.jsonl").set_index("prompt_id")
    j = A.join(B, lsuffix=f"_{a}", rsuffix=f"_{b}")
    j["dR"] = j[f"reward_{b}"] - j[f"reward_{a}"]
    return j
j = pair("midpoint", kt.loc[0.0, "fork"])
print("Spearman(Δreward, Δlength) across prompts:", round(j.dR.corr(j[f"response_tokens_{kt.loc[0.0, 'fork']}"] - j["response_tokens_midpoint"], method="spearman"), 3))
for title, sub in [("LARGEST REWARD INCREASE", j.sort_values("dR", ascending=False).head(2)), ("LARGEST REWARD DECREASE", j.sort_values("dR").head(2))]:
    print(f"\n######## {title}")
    for pid, r in sub.iterrows():
        b = kt.loc[0.0, "fork"]
        print(f"\nPROMPT: {excerpt(r.prompt_midpoint, 220)}")
        print(f"  midpoint  (R={r.reward_midpoint:.2f}, {r.response_tokens_midpoint} tok, eos={r.terminated_with_eos_midpoint}): {excerpt(r.response_midpoint, 300)}")
        print(f"  β_KL=0    (R={r[f'reward_{b}']:.2f}, {r[f'response_tokens_{b}']} tok, eos={r[f'terminated_with_eos_{b}']}): {excerpt(r[f'response_{b}'], 300)}")'''),

md(r"""
## 7. Summary tables for the report
"""),
code(r'''
display(Markdown("**Standard continuation (20 updates)**"))
display(pd.DataFrame({"mean over 20 updates": log[["reward", "kl_token", "policy_loss", "value_loss", "entropy", "grad_norm_policy", "clip_fraction", "response_length"]].mean(),
                      "last update": log[["reward", "kl_token", "policy_loss", "value_loss", "entropy", "grad_norm_policy", "clip_fraction", "response_length"]].iloc[-1]}).round(4))
display(Markdown("**Clipping study**"))
cols = ["cached_clip_fraction", "cached_affected_token_fraction", "replay_clip_fraction_final", "replay_affected_fraction_final", "heldout_reward", "heldout_kl_token",
        "heldout_len_mean", "heldout_len_std", "stab_mean_post_update_kl", "stab_max_abs_log_ratio", "stab_policy_loss_std", "train_generated_tokens"]
display(ct[cols].round(4))
display(Markdown("**KL-pressure study**"))
display(kt[["heldout_reward", "heldout_kl_token", "heldout_kl_seq", "heldout_entropy", "heldout_len_mean", "heldout_len_std", "train_reward_mean", "train_kl_token_last", "train_generated_tokens"]].round(4))
ct[cols].round(4).to_csv(R / "report_table_task2_clipping.csv"); kt.round(4).to_csv(R / "report_table_task2_kl.csv"); held.round(4).to_csv(R / "report_table_task2_heldout.csv")'''),
]

build("Task2_PPO", cells)
