"""Audit the raw outputs of the fixed pairwise AI judge (no new decisions are made).

The supplied `PairwiseAIJudge.compare` maps any output that does not contain A, B or TIE to TIE and
only caches the final decision. Because "judge ties or ambiguous outputs" must be reported
separately, this script replays exactly the same calls (same prompt, same hash-based A/B swap,
same greedy 4-token decoding) for every judged pair in Task 5, records the raw decoded text, and
classifies each decision as a single explicit A/B/TIE, an ambiguous output naming several labels (e.g.
"A, B", which the supplied parser resolves to the first label), or an unparseable output that defaulted to TIE.
It also checks that every replayed decision equals the cached decision used in the results.

    python -m task5_feedback.audit_judge_outputs --config configs/feedback.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import re

import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_feedback.score_perturbations import JUDGE_CACHE, VARIANT_ORDER, load_diagnostic_groups

SWAP = {"A": "B", "B": "A", "TIE": "TIE"}


@torch.no_grad()
def raw_compare(judge: PairwiseAIJudge, problem: str, a: str, b: str) -> dict:
    """Re-implementation of `PairwiseAIJudge.compare` that also returns the raw decoded text."""
    key = judge._key(problem, a, b)
    swap = int(key[:8], 16) % 2 == 1
    aa, bb = (b, a) if swap else (a, b)
    text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt",
                                              add_generation_prompt=True).to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=4, do_sample=False,
                               pad_token_id=judge.tokenizer.eos_token_id, eos_token_id=judge.tokenizer.eos_token_id)
    raw = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    found = re.findall(r"\b(A|B|TIE)\b", raw.strip().upper())
    parsed = found[0] if found else None  # the supplied parser keeps the FIRST label it finds
    result = parsed or "TIE"
    if swap:
        result = SWAP[result]
    return {"raw_output": raw, "explicit": parsed is not None, "multiple_labels": len(set(found)) > 1, "decision": result,
            "cached_decision": judge.cache.get(key), "swapped": swap, "prompt_tokens": int(ids.shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    judge = PairwiseAIJudge(cfg, JUDGE_CACHE)

    recs = []
    # (1) Controlled diagnostic set: all 10 pairs per problem, both presentation orders.
    for pid, vs in load_diagnostic_groups(cfg["paths"]["task5_diagnostics"]).items():
        q = vs["clean_correct"]["question"]
        for a, b in itertools.combinations(VARIANT_ORDER, 2):
            for x, y, order in [(a, b, "ab"), (b, a, "ba")]:
                r = raw_compare(judge, q, vs[x]["response"], vs[y]["response"])
                recs.append({"set": "diagnostic", "item": pid, "a": x, "b": y, "order": order,
                             "identical_responses": False, **r})
    # (2) Policy-vs-SFT comparisons exactly as evaluate_math makes them (A = policy, B = SFT), INCLUDING the
    #     pairs whose two responses are character-identical: a calibrated judge must return TIE there, so any
    #     A/B decision on an identical pair is pure position/noise and inflates the reported wins/losses.
    for ds in ["gsm", "transfer"]:
        sft = {r["source_index"]: r for r in read_jsonl(outdir / f"generated_{ds}_sft.jsonl")}
        for pol in ["rlvr", "rlaif"]:
            for r in read_jsonl(outdir / f"generated_{ds}_{pol}.jsonl"):
                s = sft[r["source_index"]]
                rr = raw_compare(judge, r["question"], r["response"], s["response"])
                recs.append({"set": ds, "item": r["source_index"], "a": pol, "b": "sft", "order": "ab",
                             "identical_responses": r["response"] == s["response"], **rr})
    write_jsonl(outdir / "judge_raw_outputs.jsonl", recs)

    df = pd.DataFrame(recs)
    df["identical_responses"] = df["identical_responses"].fillna(False).astype(bool)
    df["category"] = df.apply(lambda r: "UNPARSEABLE->TIE" if not r["explicit"]
                              else ("AMBIGUOUS_MULTI_LABEL" if r["multiple_labels"] else r["decision"]), axis=1)
    summary = {"description": "raw-output audit of the fixed pairwise judge; decisions are unchanged",
               "n_judged_calls": int(len(df)),
               "replay_matches_cached_decision": float((df.decision == df.cached_decision).mean()),
               "n_cached_missing": int(df.cached_decision.isna().sum())}

    def block(d):
        return {"n_calls": int(len(d)),
                "explicit_A": int((d.category == "A").sum()), "explicit_B": int((d.category == "B").sum()),
                "explicit_TIE": int((d.category == "TIE").sum()),
                "unparseable_defaulted_to_TIE": int((d.category == "UNPARSEABLE->TIE").sum()),
                # e.g. raw "A, B": the supplied parser silently keeps the first label
                "ambiguous_multi_label": int((d.category == "AMBIGUOUS_MULTI_LABEL").sum()),
                "ambiguous_multi_label_resolved_as": d[d.category == "AMBIGUOUS_MULTI_LABEL"].decision.value_counts().to_dict(),
                "final_decisions": d.decision.value_counts().to_dict(),
                "most_common_raw_outputs": d.raw_output.str.strip().value_counts().head(8).to_dict()}

    summary["diagnostic"] = block(df[df.set == "diagnostic"])
    for ds in ["gsm", "transfer"]:
        for pol in ["rlvr", "rlaif"]:
            d = df[(df.set == ds) & (df.a == pol)]
            summary[f"{ds}_{pol}_vs_sft"] = {
                "identical_response_pairs": block(d[d.identical_responses]),
                "different_response_pairs": block(d[~d.identical_responses]),
                # win rate vs SFT (win 1 / tie 0.5) recomputed with identical pairs forced to TIE
                "win_rate_as_reported": float(d.decision.map({"A": 1.0, "TIE": 0.5, "B": 0.0}).mean()),
                "win_rate_identical_pairs_forced_tie": float(d.apply(
                    lambda r: 0.5 if r["identical_responses"] else {"A": 1.0, "TIE": 0.5, "B": 0.0}[r["decision"]], axis=1).mean()),
            }
    save_json(outdir / "judge_raw_output_audit.json", summary)
    print(pd.Series({k: v for k, v in summary.items() if not isinstance(v, dict)}).to_string())
    for k, v in summary.items():
        if isinstance(v, dict):
            print(k, json.dumps(v, default=str))


if __name__ == "__main__":
    main()
