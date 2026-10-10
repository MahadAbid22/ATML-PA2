#!/bin/bash
source /venv/main/bin/activate
cd /workspace/pa2
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PA2_THERMAL_MAX=88 PYTHONIOENCODING=utf-8
stage() { python -u -c "
from common.orchestration import run_module, set_log_file
set_log_file(\"results/logs/$1.log\")
run_module(*\"$2\".split(), skip_if_exists=\"$3\")"; }
date; echo T3_START
python -u -m scripts.run_pipeline --stages t3 > results/logs/pipeline_t3.out 2>&1 && echo T3_OK || echo T3_FAILED
while kill -0 3286 2>/dev/null; do sleep 20; done
date; echo T2_PIPELINE_EXITED
stage t6 "task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task1_dpo/standard --name dpo_standard" results/task2_ppo/heldout_dpo_standard.json && echo T6_OK || echo T6_FAILED
date; echo T3TRACK_DONE
