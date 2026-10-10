#!/bin/bash
source /venv/main/bin/activate
cd /workspace/pa2
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PA2_THERMAL_MAX=88 PYTHONIOENCODING=utf-8
date; echo T1_RERUN_START
python -u -m scripts.run_pipeline --stages t1 > results/logs/pipeline_t1.out 2>&1 && echo T1_OK || echo T1_FAILED
date
