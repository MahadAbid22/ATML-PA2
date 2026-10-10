"""Run every required experiment in order (resumable: finished stages are skipped).

    python -m scripts.run_pipeline                 # everything
    python -m scripts.run_pipeline --stages t1 t2  # selected tasks

Each stage's stdout is appended to results/logs/<stage>.log. The per-task notebooks call the same
modules (with the same skip rules) and then load the saved results for tables and figures.
"""
from __future__ import annotations

import argparse
import time

from common.data import load_yaml
from common.orchestration import run_module, set_log_file


def t1():
    c = "configs/dpo.yaml"
    run_module("task1_dpo.train", "--config", c, "--run-name", "standard", skip_if_exists="outputs/task1_dpo/standard/adapter_model.safetensors")
    run_module("task1_dpo.evaluate", "--config", c, "--adapter", "none", "--name", "sft", skip_if_exists="results/task1_dpo/eval_sft.json")
    run_module("task1_dpo.evaluate", "--config", c, "--adapter", "outputs/task1_dpo/standard", "--name", "standard", skip_if_exists="results/task1_dpo/eval_standard.json")
    run_module("task1_dpo.ablate_beta", "--config", c)
    run_module("task1_dpo.analyze_length", "--config", c)


def t2():
    c = "configs/ppo.yaml"
    run_module("task2_ppo.continue_train", "--config", c, "--run-name", "standard", skip_if_exists="outputs/task2_ppo/standard/adapter_model.safetensors")
    run_module("task2_ppo.evaluate", "--config", c, "--adapter", "outputs/task2_ppo/standard", "--name", "standard", skip_if_exists="results/task2_ppo/heldout_standard.json")
    run_module("task2_ppo.evaluate", "--config", c, "--adapter", "none", "--name", "sft", skip_if_exists="results/task2_ppo/heldout_sft.json")
    run_module("task2_ppo.analyze_clipping", "--config", c)
    run_module("task2_ppo.ablate_kl", "--config", c)


def t3():
    c = "configs/grpo.yaml"
    run_module("task3_grpo.continue_train", "--config", c, "--run-name", "standard", skip_if_exists="outputs/task3_grpo/standard/adapter_model.safetensors")
    run_module("task3_grpo.evaluate", "--config", c, "--adapter", "outputs/task3_grpo/standard", "--name", "standard", skip_if_exists="results/task3_grpo/heldout_standard.json")
    run_module("task3_grpo.analyze_group_size", "--config", c, skip_if_exists="results/task3_grpo/group_size_table.csv")
    run_module("task3_grpo.compare_normalization", "--config", c)


def t4_generate():
    c = "configs/feedback.yaml"
    run_module("task4_safety.generate_responses", "--config", c)
    run_module("task4_safety.make_audit_sheet", "--config", c, skip_if_exists="results/task4_safety/manual_audit_sheet.csv")


def t4_judge():
    c = "configs/feedback.yaml"
    run_module("task4_safety.judge_responses", "--config", c)
    run_module("task4_safety.evaluate_safety", "--config", c)


def t5():
    c = "configs/feedback.yaml"
    run_module("task5_feedback.evaluate_math", "--config", c, "--dataset", "gsm", skip_if_exists="results/task5_feedback/summary_gsm.csv")
    run_module("task5_feedback.score_perturbations", "--config", c, skip_if_exists="results/task5_feedback/diagnostic_summary.json")
    run_module("task5_feedback.evaluate_math", "--config", c, "--dataset", "transfer", skip_if_exists="results/task5_feedback/summary_transfer.csv")
    run_module("task5_feedback.compare_feedback", "--config", c)


# Task 4 judging runs last so the blind manual audit can be completed before any AI label exists.
STAGES = {"t1": t1, "t2": t2, "t3": t3, "t4gen": t4_generate, "t5": t5, "t4judge": t4_judge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    args = ap.parse_args()
    for name in args.stages:
        set_log_file(f"results/logs/{name}.log")
        t0 = time.time()
        STAGES[name]()
        print(f"== stage {name} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
