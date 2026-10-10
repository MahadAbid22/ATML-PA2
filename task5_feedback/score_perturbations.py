from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final"]

# Controlled pairs: (diagnostically better, worse, what is being probed).
# Every perturbation is contrasted with the clean anchor of the same problem.
CONTROLLED_PAIRS = [
    ("clean_correct", "corrupt_reasoning_correct_final", "reasoning sensitivity (final held correct)"),
    ("clean_correct", "good_reasoning_wrong_final", "outcome sensitivity (reasoning held ~fixed)"),
    ("clean_correct", "persuasive_filler_correct", "filler / style susceptibility (both correct)"),
    ("clean_correct", "gold_distractor_wrong_final", "distractor robustness (gold mentioned, wrong final)"),
]
JUDGE_CACHE = "results/task5_feedback/judge_cache.json"


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def verifier_pref(ra: float, rb: float) -> str:
    return "A" if ra > rb else ("B" if rb > ra else "TIE")


def outcome(pref: str) -> str:
    """Outcome when A is the diagnostically better response."""
    return {"A": "better", "TIE": "tie", "B": "wrong"}[pref]


def rates(df, col):
    return {f"{k}_rate": float((df[col] == k).mean()) for k in ["better", "tie", "wrong"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    print("Use exact_reward(...) for RLVR and PairwiseAIJudge(...) for RLAIF.")

    judge = PairwiseAIJudge(cfg, JUDGE_CACHE)

    # 1) Per-response exact rewards (RLVR) + check against the release's expected reward.
    resp = []
    for pid, vs in groups.items():
        for v in VARIANT_ORDER:
            r = vs[v]
            er = exact_reward(r["response"], str(r["gold_final"]))
            resp.append({"problem_id": pid, "variant_type": v, "exact_reward": er,
                         "expected_exact_reward": float(r["expected_exact_reward"]),
                         "response_words": len(r["response"].split())})
    rdf = pd.DataFrame(resp)
    mism = rdf[rdf.exact_reward != rdf.expected_exact_reward]
    print("Verifier vs release expected_exact_reward mismatches:", len(mism))

    # 2) All 10 unordered pairs per problem, judged in BOTH presentation orders
    #    (the supplied judge balances A/B orientation by hash, so the two calls are independent probes).
    pairs = []
    for pid, vs in groups.items():
        q = vs["clean_correct"]["question"]
        for a, b in itertools.combinations(VARIANT_ORDER, 2):
            ra = rdf[(rdf.problem_id == pid) & (rdf.variant_type == a)].exact_reward.item()
            rb = rdf[(rdf.problem_id == pid) & (rdf.variant_type == b)].exact_reward.item()
            j_ab = judge.compare(q, vs[a]["response"], vs[b]["response"])
            j_ba = judge.compare(q, vs[b]["response"], vs[a]["response"])
            j_ba_as_ab = {"A": "B", "B": "A", "TIE": "TIE"}[j_ba]
            pairs.append({"problem_id": pid, "a": a, "b": b, "verifier": verifier_pref(ra, rb),
                          "judge": j_ab, "judge_reversed_call": j_ba_as_ab, "judge_order_consistent": j_ab == j_ba_as_ab})
        print(f"judged problem {pid}", flush=True)
    pdf = pd.DataFrame(pairs)
    write_jsonl(outdir / "diagnostic_pairs.jsonl", pdf.to_dict("records"))

    # 3) Controlled-pair table: better / tie / wrong preference per perturbation and mechanism.
    table, detail = [], []
    for better, worse, probe in CONTROLLED_PAIRS:
        d = pdf[(pdf.a == better) & (pdf.b == worse)].copy()
        d["verifier_outcome"] = d.verifier.map(outcome)
        d["judge_outcome"] = d.judge.map(outcome)
        d["judge_reversed_outcome"] = d.judge_reversed_call.map(outcome)
        detail.append(d)
        for mech, col in [("RLVR exact verifier", "verifier_outcome"), ("RLAIF pairwise judge", "judge_outcome"),
                          ("RLAIF judge (reversed call)", "judge_reversed_outcome")]:
            table.append({"perturbation": worse, "probe": probe, "mechanism": mech, "n_pairs": len(d), **rates(d, col),
                          **({"order_consistency": float(d.judge_order_consistent.mean())} if "judge" in col else {})})
    tdf = pd.DataFrame(table)
    tdf.to_csv(outdir / "diagnostic_controlled_pairs.csv", index=False)
    pd.concat(detail).to_csv(outdir / "diagnostic_controlled_pairs_detail.csv", index=False)

    # 4) Per-category scores for all five variants: RLVR reward and RLAIF round-robin group reward
    #    r = (wins + 0.5 ties) / (K-1) with K = 5 variants per problem (the training-time aggregation).
    rr = []
    for pid, vs in groups.items():
        sub = pdf[pdf.problem_id == pid]
        wins = {v: 0.0 for v in VARIANT_ORDER}
        for _, p in sub.iterrows():
            if p.judge == "A":
                wins[p.a] += 1
            elif p.judge == "B":
                wins[p.b] += 1
            else:
                wins[p.a] += 0.5; wins[p.b] += 0.5
        for v in VARIANT_ORDER:
            rr.append({"problem_id": pid, "variant_type": v, "rlaif_group_reward": wins[v] / (len(VARIANT_ORDER) - 1)})
    rrdf = pd.DataFrame(rr).merge(rdf, on=["problem_id", "variant_type"])
    cat = rrdf.groupby("variant_type").agg(rlvr_reward_mean=("exact_reward", "mean"),
                                            rlaif_group_reward_mean=("rlaif_group_reward", "mean"),
                                            rlaif_group_reward_std=("rlaif_group_reward", "std"),
                                            response_words_mean=("response_words", "mean")).reindex(VARIANT_ORDER)
    cat.to_csv(outdir / "diagnostic_per_category.csv")
    rrdf.to_csv(outdir / "diagnostic_per_response.csv", index=False)

    # 5) Summary sensitivities.
    def s_rate(worse, col):
        d = pdf[(pdf.a == "clean_correct") & (pdf.b == worse)]
        return float((d[col] == "A").mean())
    summary = {
        "n_problems": len(groups),
        "verifier_expected_reward_mismatches": int(len(mism)),
        "S_reason": {"RLVR": s_rate("corrupt_reasoning_correct_final", "verifier"), "RLAIF": s_rate("corrupt_reasoning_correct_final", "judge")},
        "S_outcome_wrong_final": {"RLVR": s_rate("good_reasoning_wrong_final", "verifier"), "RLAIF": s_rate("good_reasoning_wrong_final", "judge")},
        "S_outcome_gold_distractor": {"RLVR": s_rate("gold_distractor_wrong_final", "verifier"), "RLAIF": s_rate("gold_distractor_wrong_final", "judge")},
        "S_outcome_pooled": {
            "RLVR": float(np.mean([s_rate("good_reasoning_wrong_final", "verifier"), s_rate("gold_distractor_wrong_final", "verifier")])),
            "RLAIF": float(np.mean([s_rate("good_reasoning_wrong_final", "judge"), s_rate("gold_distractor_wrong_final", "judge")]))},
        "filler_preference": {"RLAIF_prefers_clean": s_rate("persuasive_filler_correct", "judge"),
                              "RLAIF_prefers_filler": float((pdf[(pdf.a == "clean_correct") & (pdf.b == "persuasive_filler_correct")].judge == "B").mean())},
        "judge_order_consistency_all_pairs": float(pdf.judge_order_consistent.mean()),
        "judge_tie_rate_all_pairs": float((pdf.judge == "TIE").mean()),
        "verifier_tie_rate_all_pairs": float((pdf.verifier == "TIE").mean()),
        "verifier_judge_agreement_all_pairs": float((pdf.verifier == pdf.judge).mean()),
        "pair_definition": [{"better": b, "worse": w, "probe": p} for b, w, p in CONTROLLED_PAIRS],
    }
    save_json(outdir / "diagnostic_summary.json", summary)
    pd.set_option("display.width", 220)
    print(tdf.to_string(index=False))
    print(cat.round(3).to_string())
    print(summary)


if __name__ == "__main__":
    main()
