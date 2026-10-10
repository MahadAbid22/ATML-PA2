#!/bin/bash
# Phase A: standard PPO / GRPO continuations run ALONE on the GPU so their wall-clock and
# peak-VRAM numbers are not distorted by other jobs. Phase B: everything else in parallel.
source /venv/main/bin/activate
cd /workspace/pa2
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PA2_THERMAL_MAX=88 PYTHONIOENCODING=utf-8
mkdir -p results/logs
stage() { python -u -c "
from common.orchestration import run_module, set_log_file
set_log_file(\"results/logs/$1.log\")
run_module(*\"$2\".split(), skip_if_exists=\"$3\")"; }
date; echo PHASE_A
stage t2 "task2_ppo.continue_train --config configs/ppo.yaml --run-name standard" outputs/task2_ppo/standard/adapter_model.safetensors || exit 1
stage t3 "task3_grpo.continue_train --config configs/grpo.yaml --run-name standard" outputs/task3_grpo/standard/adapter_model.safetensors || exit 1
date; echo PHASE_B
for s in t1 t2 t3 t5; do python -u -m scripts.run_pipeline --stages $s > results/logs/pipeline_$s.out 2>&1 & done
wait
date; echo PHASE_C
python -u -m scripts.run_pipeline --stages t4gen > results/logs/pipeline_t4gen.out 2>&1 &
stage t6 "task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task1_dpo/standard --name dpo_standard" results/task2_ppo/heldout_dpo_standard.json &
wait
date; echo ALL_DONE
