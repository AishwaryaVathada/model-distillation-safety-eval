# Distillation with Safety Preservation (Specification Transfer) and Pre/Post Evaluation

This repository is a rebuildable, modular research pipeline for:

1. Distilling a (very large) teacher into smaller open-source student models via synthetic data generation and supervised fine-tuning.
2. Verifying that safety specifications and behavioral constraints transfer across compression stages (specification transfer analysis).
3. Running pre- and post-distillation evaluation on common benchmarks (English, Math, Code, Mandarin) and producing comparison reports.

The project workflow and benchmark suite follow the course project plan: distill multiple open models and evaluate before and after distillation on MMLU, AGIEval, MATH, MGSM, HumanEval, APPS, C-Eval, and CLUEWSC. See `docs/benchmarks.md`.

## Why this repo exists

This repo is designed so you can reproduce the project end-to-end from scratch, including:

- Explicit, validated configs for each pipeline stage
- Pluggable teachers (OpenAI-compatible APIs like DeepSeek, or local vLLM)
- Pluggable students (any Hugging Face causal LM)
- Automated safety property tests (refusal / policy compliance / regression gates)
- Benchmark execution + structured reporting for pre/post comparison

## Hardware and deployment reality

The motivating case is an extreme compression pipeline (example: 671B -> 7B). Hosting or fine-tuning at that scale is not feasible on a laptop.

This repository is structured so that:
- The teacher is typically accessed through an OpenAI-compatible endpoint (recommended for 671B-class teachers).
- The student can be trained locally for small models, or on external GPU servers for 7B+.
- Evaluation can be done locally for small models, or on GPU servers for realistic throughput.

See `docs/hardware.md` for concrete options and a checklist.

## Repository layout

- `src/distill_safe/teacher/` teacher adapters (OpenAI-compatible API, local vLLM, dummy for smoke tests)
- `src/distill_safe/data/` dataset adapters + synthetic generation
- `src/distill_safe/distill/` training and multi-stage orchestration
- `src/distill_safe/safety/` safety specifications + verification suite
- `src/distill_safe/eval/` benchmark execution + reporting
- `configs/` YAML configs for distillation, evaluation, and safety
- `scripts/` runnable entrypoints used by the CLI
- `docs/` detailed technical notes (spec transfer, benchmarks, hardware)

## Quickstart (smoke test, no GPU, no external API)

This verifies the repo installs and the pipeline wiring works.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

# Run a tiny end-to-end pipeline with a dummy teacher and toy data
distill-safe pipeline run --config configs/pipelines/smoke_test.yaml
```

Expected outputs go under `runs/` (config snapshot, generated data, logs, and a report).

## Quickstart (real teacher via DeepSeek OpenAI-compatible API)

DeepSeek documents an OpenAI-compatible API, so you can use the same message schema and base URL configuration. Export credentials:

```bash
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_API_KEY="YOUR_KEY"
```

Then run synthetic data generation (example config):

```bash
distill-safe data generate --config configs/data/r1_synthetic_small.yaml
```

Notes:
- If your provider does not return chain-of-thought reasoning, this pipeline will still work by distilling final answers.
- If you do have reasoning traces, you can enable `store_rationale=true` in the config, but ensure you are allowed to store and use those traces.

## Quickstart (student fine-tuning)

Example LoRA SFT run (you still need a GPU for anything non-trivial):

```bash
distill-safe distill sft --config configs/distill/qwen25_0p5b_lora.yaml
```

## Quickstart (benchmark evaluation + pre/post comparison)

```bash
# Evaluate a model checkpoint (student or baseline)
distill-safe eval run --config configs/eval/suite_default.yaml --model_kind hf_local --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Compare two result JSONs (pre vs post)
distill-safe eval compare --pre runs/.../eval_results.json --post runs/.../eval_results.json
```

## Reproducing the course project

A suggested sequence (per model):

1. Baseline eval: run `eval` on the original model checkpoint.
2. Generate distillation dataset using DeepSeek R1 (or another teacher).
3. Train student (LoRA or full fine-tune).
4. Run safety specification transfer suite; fix regressions (data, prompts, training, filters).
5. Post-distillation eval; generate pre/post report and write discussion.

See `docs/spec_transfer.md` for how to define and test safety specs, and `docs/benchmarks.md` for dataset/task notes.

## License

MIT. See `LICENSE`.

## Benchmark-grade evaluation (lm-evaluation-harness)

```bash
# Example: run official lm-eval tasks (preferred for MMLU/HumanEval/etc.)
distill-safe eval lm-eval --model "Qwen/Qwen2.5-0.5B-Instruct" --tasks "mmlu,humaneval" --batch_size 1 --limit 50
```
