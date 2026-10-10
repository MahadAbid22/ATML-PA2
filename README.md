# ATML PA2 — LLM Post-Training (DPO · PPO · GRPO · Safety · RLVR vs RLAIF)

Student implementation of EE-5102/CS-6304 Programming Assignment 2, built on the course starter
repository (https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining) and the pinned course assets
(https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets, revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`).

* Experiments are Python modules (`task*/`), one stage per process; every stage is resumable (skips if
  its output exists) and writes machine-readable results to `results/`.
* `notebooks/Task{1..6}_*.ipynb` call the same modules and build all tables, figures (saved to
  `report/figures/`) and qualitative examples from the saved results.
* Raw stdout of every stage: `results/logs/*.log`.

## 1. Setup

```bash
python -m pip install -r requirements.txt      # torch>=2.4 with CUDA; pinned transformers/trl/peft/tokenizers
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
python -m scripts.smoke_tests --gpu            # objective unit tests + log-prob/gradient equality checks
```

Reference environment used for the reported results: rented Vast.ai instance, Ubuntu container,
Python 3.12.14, PyTorch 2.14.0+cu132, Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, Tokenizers 0.22.1,
bitsandbytes 0.50.2, single NVIDIA RTX 4090 (24 GB). The code was developed and smoke-tested on an
RTX 4070 Laptop GPU (8 GB), which is why it is memory-frugal. If a stale Hugging Face token is cached, set
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (all public models are downloaded anonymously).

Scheduling used for the reported run (`driver.sh` logic): the standard PPO and standard GRPO continuations
were run **alone** on the GPU (so their reported wall-clock time / peak VRAM are not distorted), after which
the remaining stages of Tasks 1, 2, 3 and 5 ran as concurrent processes on the same GPU, followed by
Task 4 generation. Concurrency changes only wall-clock time, never data, seeds or decoding.

## 2. Reproduce everything

```bash
python -m scripts.run_pipeline                      # all tasks in order (~8-10 h on the GPU above)
python -m scripts.run_pipeline --stages t1          # or one task: t1 t2 t3 t4gen t5 t4judge
```

Then execute the notebooks top-to-bottom (they skip finished stages and only redo the analysis):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Task1_DPO.ipynb
```

### Per-task commands (what `run_pipeline` runs)

**Task 1 — DPO** (`configs/dpo.yaml`)
```bash
python -m task1_dpo.train     --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate  --config configs/dpo.yaml --adapter none --name sft
python -m task1_dpo.evaluate  --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta    --config configs/dpo.yaml   # beta in {0.03,0.10,0.30}: train + evaluate + table
python -m task1_dpo.analyze_length --config configs/dpo.yaml   # length-balanced train + per-stratum/word-limit analysis
```

**Task 2 — PPO** (`configs/ppo.yaml`)
```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard       # 20 updates
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter none --name sft
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml   # cached-batch geometry + replay + eps forks
python -m task2_ppo.ablate_kl        --config configs/ppo.yaml   # beta_KL in {0,0.10,0.20} forks
```

**Task 3 — GRPO** (`configs/grpo.yaml`)
```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard     # 20 updates
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size    --config configs/grpo.yaml   # K in {2,4,8}, equal generations
python -m task3_grpo.compare_normalization --config configs/grpo.yaml   # canonical vs Dr.GRPO forks
```

**Task 4 — Safety** (`configs/feedback.yaml`)
```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet   --config configs/feedback.yaml   # blind 60 x 4 audit sheet
python -m task4_safety.make_label_tool    --config configs/feedback.yaml   # optional HTML page for manual labelling
python -m task4_safety.judge_responses    --config configs/feedback.yaml
python -m task4_safety.evaluate_safety    --config configs/feedback.yaml
```

**Task 5 — RLVR vs RLAIF** (`configs/feedback.yaml`)
```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

**Task 6 — Synthesis**: `notebooks/Task6_Synthesis.ipynb` (also scores the standard DPO adapter under the
common Task 2/3 held-out protocol: `python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task1_dpo/standard --name dpo_standard`).

## 3. Corrected starter defects

| File | Starter | Fix |
|---|---|---|
| `task1_dpo/dpo.py` | `beta * (policy_margin + ref_margin)` | `beta * (policy_margin - ref_margin)` (difference of policy/reference log-ratios) |
| `task2_ppo/ppo.py` | `torch.maximum(surr1, surr2)` | `torch.minimum(...)` (pessimistic clipped surrogate) |
| `task3_grpo/grpo.py` | batch-wide mean/std | mean/std within each prompt group (`group_ids`) |

`scripts/smoke_tests.py` verifies each fix against hand-computed values and naive reference
implementations (GAE, KL shaping, per-sequence GRPO loss decomposition).

## 4. Implementation decisions (all applied identically to every compared condition)

* **Memory** (8 GB VRAM): LoRA policy with gradient checkpointing; reference = same weights with the
  adapter disabled; LM-head log-probs computed in 128-token chunks under activation checkpointing
  (`common/rl_utils.py`, numerically checked against the starter helper); GRPO back-propagates one
  completion at a time using an exact decomposition of the batch loss; 8-bit RM, 4-bit judge.
* **DPO overlength filter** (one of the three options approved by the course staff): a preference pair is
  kept only if *both* prompt + chosen and prompt + rejected (chat template + EOS) fit in
  `max_sequence_length = 768` tokens, so no prompt or response token is ever truncated. The same rule is
  applied to every DPO training file (standard, β forks, length-balanced) and every DPO evaluation file
  (standard held-out, length-stratified). Kept/dropped counts: standard train 1214/1500 (286 dropped),
  standard held-out 240/300, length-balanced train 1400/1500 (38 preferred-longer, 28 matched, 34
  rejected-longer dropped), length-stratified eval 230/246 (8/2/6 dropped). Dropped keys are saved in
  `results/task1_dpo/train_ids_*.json`; before/after-filter length statistics are in
  `results/task1_dpo/length_dataset_stats.json`. Implication: the filter removes the longest responses,
  so the retained data has a slightly weaker long-response tail than the released files. β forks use the
  first 600 retained pairs. Constant learning rate (no schedule in the config).
* **RL prompts**: prompts longer than `max_prompt_length = 256` are excluded from training and held-out
  sets, because the supplied `batch_generate` right-truncates and would delete the assistant header. The
  cached-rollout study reproduces the release tokenisation exactly (including that truncation) to match
  the supplied log-probs.
* **PPO/GRPO**: dropout disabled during RL; old log-probs computed through the same kernel path as the
  loss so the ratio is exactly 1 before each update; prompt order is a seeded permutation shared by the
  standard runs and all forks, and sampling is re-seeded per update (`seed + update`). PPO uses the
  missing-EOS penalty from the config; advantages are whitened; the critic head is kept in fp32.
* **Held-out protocol (Tasks 2/3/6)**: eval-pool prompts ≤ 256 tokens, one sample per prompt, T = 0.7,
  top-p 0.9, ≤ 768 new tokens, seed 6304. KL uses the course `sampled_kl` convention (token-pooled);
  per-sequence sums are reported too.
* **Thermal guard**: training/generation loops pause if the GPU reaches 84 °C (`PA2_THERMAL_MAX`, set to
  88 °C on the RTX 4090).

## 5. Attribution

* Starter code, configs, data, cached diagnostics, supplied checkpoints, the XSTest/AI-judge prompts
  and parsers, the exact verifier and the pairwise judge: course staff (starter repository above).
* Conventions borrowed (re-implemented, not copied): TRL's value-position convention
  (`values[:, context_length-1:-1]`), `disable_dropout` for RLHF, chunked selective log-softmax idea,
  and the Dr. GRPO normaliser (Liu et al., 2025).
* Coding assistance from an LLM (Claude) was used for the implementation; all code was reviewed and
  validated with the smoke tests above.

## 6. Repository hygiene

`outputs/` (trained adapters), `checkpoints/`, `cached/`, `data/` and model caches are git-ignored.
Small JSON/CSV/JSONL results, logs, notebooks and figures are committed.
