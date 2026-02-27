# Benchmarks (Pre/Post Distillation)

This repository follows a benchmark suite used in the project plan:

- English: MMLU, AGIEval
- Math: MATH, MGSM
- Code: HumanEval, APPS
- Mandarin: C-Eval, CLUEWSC (subset of CLUE)

These are commonly used public benchmarks and are available as:
- tasks in EleutherAI's lm-evaluation-harness, and/or
- datasets on Hugging Face.

References:
- EleutherAI lm-evaluation-harness repository (recommended execution harness)
- AGIEval dataset on Hugging Face: `baber/agieval`
- MATH dataset on Hugging Face: `EleutherAI/hendrycks_math`
- MGSM dataset on Hugging Face: `juletxara/mgsm`
- APPS dataset on Hugging Face: `embedding-benchmark/APPS`
- C-Eval dataset on Hugging Face: `ceval/ceval-exam`
- CLUE benchmark on Hugging Face: `clue/clue` (subset `cluewsc2020`)

Note:
Some benchmarks require special evaluation scripts (e.g., HumanEval execution) or have licensing constraints.
Treat this scaffold as the orchestration layer; for exact scoring parity, delegate to the official evaluation code/harness.
