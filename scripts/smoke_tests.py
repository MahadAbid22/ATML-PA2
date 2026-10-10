"""Validation of the corrected objectives (CPU) and of the memory-efficient log-prob path (GPU).

Run:  python -m scripts.smoke_tests            (CPU objective tests only)
      python -m scripts.smoke_tests --gpu      (also loads Qwen2.5-1.5B and checks log-prob/grad equality)
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import clipping_active_fraction, compute_gae, ppo_policy_loss, shaped_rewards
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def close(a, b, tol=1e-5):
    return abs(float(a) - float(b)) < tol


# ----------------------------------------------------------------------------- DPO
def test_dpo():
    pc, pr = torch.tensor([-10.0]), torch.tensor([-12.0])
    rc, rr = torch.tensor([-11.0]), torch.tensor([-11.5])
    beta = 0.1
    # Manual: [(-10)-(-11)] - [(-12)-(-11.5)] = 1 - (-0.5) = 1.5
    expected = -math.log(1 / (1 + math.exp(-beta * 1.5)))
    loss, st = dpo_loss(pc, pr, rc, rr, beta)
    check("DPO loss matches manual equation", close(loss, expected), f"{float(loss):.6f} vs {expected:.6f}")
    check("DPO logit = beta*(policy_margin - ref_margin)", close(st["logit_mean"], 0.15))
    # Policy == reference -> zero logit -> loss log 2, regardless of the reference's own preference.
    loss0, st0 = dpo_loss(rc, rr, rc, rr, beta)
    check("DPO at initialisation: loss = log 2", close(loss0, math.log(2)), f"{float(loss0):.6f}")
    # The starter bug (policy_margin + ref_margin) would give a non-log2 loss at initialisation.
    buggy = -F.logsigmoid(beta * ((rc - rr) + (rc - rr))).mean()
    check("Starter bug detectable at init (buggy loss != log 2)", not close(buggy, math.log(2)), f"buggy={float(buggy):.4f}")
    # Gradient direction: increasing chosen log-prob must decrease the loss.
    p = torch.tensor([-10.0], requires_grad=True)
    l, _ = dpo_loss(p, pr, rc, rr, beta)
    l.backward()
    check("DPO gradient pushes chosen log-prob up", p.grad.item() < 0)


# ----------------------------------------------------------------------------- PPO
def test_ppo():
    eps = 0.2
    old = torch.zeros(1, 4)
    ratio = torch.tensor([[1.5, 1.5, 0.5, 0.5]])
    new = ratio.log().requires_grad_(True)
    adv = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    mask = torch.ones(1, 4)
    loss, r, cf = ppo_policy_loss(new, old, adv, mask, eps)
    # min(rho*A, clip(rho)*A): [min(1.5,1.2)=1.2, min(-1.5,-1.2)=-1.5, min(0.5,0.8)=0.5, min(-0.5,-0.8)=-0.8]
    expected = -(1.2 - 1.5 + 0.5 - 0.8) / 4
    check("PPO clipped surrogate uses min()", close(loss, expected), f"{float(loss):.4f} vs {expected:.4f}")
    check("PPO clip fraction counts all out-of-range tokens", close(cf, 1.0))
    check("Affected (gradient-zeroed) fraction = 0.5", close(clipping_active_fraction(r, adv, mask, eps), 0.5))
    loss.backward()
    g = new.grad[0]
    check("Zero gradient exactly where clipping is active (A>0,rho>1+eps / A<0,rho<1-eps)",
          g[0].item() == 0 and g[3].item() == 0 and g[1].item() != 0 and g[2].item() != 0, str(g.tolist()))
    buggy = -torch.maximum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
    check("Starter max() objective differs (no trust region)", not close(buggy, expected), f"buggy={float(buggy):.4f}")


def naive_gae(r, v, m, gamma, lam):
    B, T = r.shape
    adv = torch.zeros_like(r)
    for b in range(B):
        n = int(m[b].sum())
        last = 0.0
        for t in reversed(range(n)):
            nv = v[b, t + 1] if t + 1 < n else 0.0
            delta = r[b, t] + gamma * nv - v[b, t]
            last = delta + gamma * lam * last
            adv[b, t] = last
    return adv, adv + v * m


def test_gae():
    torch.manual_seed(0)
    r, v = torch.randn(3, 7), torch.randn(3, 7)
    m = torch.tensor([[1] * 7, [1] * 4 + [0] * 3, [1] * 1 + [0] * 6], dtype=torch.float32)
    v = v * m
    a, ret = compute_gae(r * m, v, m, 1.0, 0.95)
    na, nret = naive_gae(r * m, v, m, 1.0, 0.95)
    check("GAE matches naive reference (gamma=1, lambda=0.95, ragged masks)", torch.allclose(a, na, atol=1e-5) and torch.allclose(ret * m, nret * m, atol=1e-5))
    tr = torch.tensor([2.0, -1.0])
    lp, rf = torch.zeros(2, 5), torch.full((2, 5), -0.5)
    mm = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    sr = shaped_rewards(tr, lp, rf, mm, 0.1)
    check("KL-shaped reward: -beta*(logp-ref) per token, task reward on last valid token",
          close(sr[0, 2], 2.0 - 0.05) and close(sr[0, 0], -0.05) and close(sr[0, 3], 0.0) and close(sr[1, 4], -1.0 - 0.05))


# ----------------------------------------------------------------------------- GRPO
def test_grpo():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0, 0.0, 1.0, 0.0, 1.0])
    gid = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    adv = group_relative_advantages(rewards, gid)
    ok_mean = all(abs(adv[gid == g].mean().item()) < 1e-6 for g in range(3))
    check("GRPO advantages are zero-mean within every prompt group", ok_mean)
    check("Uninformative (zero-std) group gets zero advantage", torch.allclose(adv[4:8], torch.zeros(4)))
    exp0 = (rewards[:4] - 2.5) / rewards[:4].std(unbiased=False)
    check("GRPO advantage = (r - mu_group)/(sigma_group + eps)", torch.allclose(adv[:4], exp0, atol=1e-4))
    buggy = (rewards - rewards.mean()) / rewards.std(unbiased=False)
    check("Starter batch-wide baseline leaks prompt difficulty (group 1 all positive)", bool((buggy[4:8] > 0).all()))

    # Exact per-sequence decomposition used in continue_train equals the batch helper loss.
    torch.manual_seed(1)
    B, T, lmax = 4, 6, 6
    old = torch.randn(B, T) * 0.1 - 2
    new = (old + torch.randn(B, T) * 0.05).requires_grad_(True)
    ref = old + torch.randn(B, T) * 0.05
    mask = torch.ones(B, T); mask[1, 4:] = 0; mask[2, 2:] = 0
    mask = mask_truncated_sequences(mask, [False, False, False, True])
    a = torch.tensor([0.5, -1.0, 1.5, -1.0])
    for lt in ["grpo", "dr_grpo"]:
        full, _ = grpo_policy_loss(new, old, a, mask, ref, 0.2, 0.1, loss_type=lt, max_completion_length=lmax)
        n_total = mask.sum()
        parts = 0.0
        for i in range(B):
            if mask[i].sum() == 0:
                continue
            p_i, _ = grpo_policy_loss(new[i:i + 1], old[i:i + 1], a[i:i + 1], mask[i:i + 1], ref[i:i + 1], 0.2, 0.0, loss_type=lt, max_completion_length=lmax)
            lr_ = ref[i:i + 1] - new[i:i + 1]
            kl_i = ((torch.exp(lr_) - lr_ - 1) * mask[i:i + 1]).sum()
            parts = parts + p_i / B + 0.1 * kl_i / n_total
        check(f"Per-sequence decomposition == batch GRPO loss ({lt})", close(full, parts, 1e-6), f"{float(full):.6f} vs {float(parts):.6f}")
    check("Truncated completion fully masked", mask[3].sum().item() == 0)


# ----------------------------------------------------------------------------- GPU path
def test_gpu():
    from common.data import load_yaml
    from common.generation import response_token_logprobs
    from common.models import load_policy, load_tokenizer
    from common.rl_utils import response_logprobs

    cfg = load_yaml("configs/ppo.yaml")
    tok = load_tokenizer(cfg["base_model"])
    pol = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    from common.rl_utils import disable_dropout
    disable_dropout(pol)  # dropout would make two forward passes differ
    msgs = [[{"role": "user", "content": "Explain PPO clipping in two sentences."}], [{"role": "user", "content": "Hi"}]]
    rendered = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(rendered, return_tensors="pt", padding=True).to("cuda")
    resp = tok(["PPO limits the policy ratio. It keeps updates small.<|im_end|>", "Hello there!<|im_end|>"], add_special_tokens=False, padding=True, padding_side="right", return_tensors="pt").input_ids.cuda()
    seq = torch.cat([enc.input_ids, resp], 1)
    attn = torch.cat([enc.attention_mask, torch.ones_like(resp)], 1)
    pw = enc.input_ids.shape[1]
    with torch.no_grad():
        ref_lp, _ = response_token_logprobs(pol, seq, attn, pw, resp)
        lp, ent = response_logprobs(pol, seq, attn, resp.shape[1], grad=False, entropy=True)
        # Ground truth: each row run alone without any padding.
        gts = []
        for i in range(seq.shape[0]):
            p = enc.input_ids[i][enc.attention_mask[i].bool()]
            s = torch.cat([p, resp[i]])[None]
            gt, _ = response_token_logprobs(pol, s, torch.ones_like(s), len(p), resp[i][None])
            gts.append(gt[0])
        gt = torch.stack(gts)
    d_mine, d_start = (lp - gt).abs(), (ref_lp - gt).abs()
    # fp16 batched-vs-unbatched kernels differ by ~1e-2 nats/token for ANY implementation, so the
    # criterion is: no worse than the starter helper on the same padded batch (within the same 1.5x
    # noise margin for mean and max; the per-token max differs slightly between GPU kernel choices).
    check("Chunked log-probs match unpadded ground truth as well as the starter helper does",
          d_mine.mean() <= 1.5 * d_start.mean() + 1e-3 and d_mine.max() <= 1.5 * d_start.max() + 1e-3,
          f"mine mean/max {float(d_mine.mean()):.1e}/{float(d_mine.max()):.1e}; starter mean/max {float(d_start.mean()):.1e}/{float(d_start.max()):.1e}")
    check("Entropy finite and non-negative", bool(torch.isfinite(ent).all() and (ent >= -1e-4).all()))
    params = [p for p in pol.parameters() if p.requires_grad]
    lp_g, _ = response_logprobs(pol, seq, attn, resp.shape[1], grad=True)
    lp_g.sum().backward()
    g1 = torch.cat([p.grad.flatten() for p in params]).clone()
    for p in params:
        p.grad = None
    ref_g, _ = response_token_logprobs(pol, seq, attn, pw, resp)
    ref_g.sum().backward()
    g2 = torch.cat([p.grad.flatten() for p in params])
    cos = F.cosine_similarity(g1, g2, dim=0).item()
    check("Chunked/checkpointed gradient == direct gradient", cos > 0.999, f"cosine={cos:.6f}, |g|={g1.norm():.3e}")
    check("LoRA trainable parameters are fp32", all(p.dtype == torch.float32 for p in params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    test_dpo(); test_ppo(); test_gae(); test_grpo()
    if args.gpu:
        test_gpu()
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
