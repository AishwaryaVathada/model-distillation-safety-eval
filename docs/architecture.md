# Architecture

This repository implements a modular pipeline for distillation with safety preservation.

## Components

- Teacher adapters (`src/distill_safe/teacher/`)
  - `openai_compatible`: calls an OpenAI-style Chat Completions endpoint (`/v1/chat/completions`).
  - `dummy`: deterministic local generator for smoke tests.

- Synthetic data generation (`src/distill_safe/data/`)
  - Dataset adapters (starting with `toy`)
  - Prompt templates for distillation (final-answer-only by default)
  - JSONL outputs for training

- Distillation (`src/distill_safe/distill/`)
  - `sft.py`: minimal LoRA SFT using TRL + PEFT
  - `pipeline.py`: orchestration

- Safety specification transfer (`src/distill_safe/safety/`)
  - YAML-defined specs (refusal specs in scaffold)
  - Automated suite runner producing a JSON report

- Evaluation (`src/distill_safe/eval/`)
  - `toy_eval.py`: a minimal example
  - `harness.py`: scaffold for integrating lm-evaluation-harness and HF datasets
  - `compare.py`: report comparison utilities

## Extending to multi-stage compression

The motivating research question is whether constraints transfer across multiple compression stages:
teacher_671B -> intermediate_70B -> intermediate_14B -> student_7B.

To support this, implement:
1. A `ModelUnderTest` wrapper that loads a HF checkpoint and exposes `.generate(prompt)`.
2. A `stages.yaml` config and a stage runner that loops:
   - generate (teacher -> synthetic)
   - train (student <- synthetic)
   - validate (safety + benchmark gates)
   - promote (only if gates pass)

This scaffold already contains the necessary boundaries; only stage scheduling is missing.
