#!/bin/bash
# Phase A (standard PPO/GRPO continuations alone on the GPU) is already done by driver.sh.
# Phase B: three concurrent tracks sized to fit 24 GB:
#   track 1 = Task 1 pipeline, then Task 4 generation + AI judging (needs the DPO standard adapter)
#   track 2 = Task 5 pipeline
#   track 3 = Task 2 then Task 3 pipelines, then RL-protocol evals of DPO and of the supplied midpoints
source /venv/main/bin/activate
cd /workspace/pa2
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PA2_THERMAL_MAX=88 PYTHONIOENCODING=utf-8
stage() { python -u -c "
from common.orchestration import run_module, set_log_file
set_log_file(\"results/logs/$1.log\")
run_module(*\"$2\".split(), skip_if_exists=\"$3\")"; }
date; echo PHASE_B
(
 python -u -m scripts.run_pipeline --stages t1 > results/logs/pipeline_t1.out 2>&1 || echo T1_FAILED
 date; echo T1_DONE
 python -u -m scripts.run_pipeline --stages t4gen > results/logs/pipeline_t4gen.out 2>&1 || echo T4GEN_FAILED
 stage t4judge "task4_safety.judge_responses --config configs/feedback.yaml" results/task4_safety/__never__ || echo T4JUDGE_FAILED
 python -u -m task4_safety.make_label_tool --config configs/feedback.yaml
 date; echo TRACK1_DONE
) &
(
 python -u -m scripts.run_pipeline --stages t5 > results/logs/pipeline_t5.out 2>&1 || echo T5_FAILED
 date; echo TRACK2_DONE
) &
(
 python -u -m scripts.run_pipeline --stages t2 > results/logs/pipeline_t2.out 2>&1 || echo T2_FAILED
 python -u -m scripts.run_pipeline --stages t3 > results/logs/pipeline_t3.out 2>&1 || echo T3_FAILED
 stage t2 "task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint" results/task2_ppo/heldout_midpoint.json || echo T2MID_FAILED
 stage t3 "task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint" results/task3_grpo/heldout_midpoint.json || echo T3MID_FAILED
 while [ ! -f outputs/task1_dpo/standard/adapter_model.safetensors ]; do sleep 30; done
 stage t6 "task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task1_dpo/standard --name dpo_standard" results/task2_ppo/heldout_dpo_standard.json || echo T6_FAILED
 date; echo TRACK3_DONE
) &
wait
date; echo ALL_DONE
