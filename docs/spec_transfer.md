# Safety Specification Transfer Analysis

Goal: verify that behavioral constraints (especially safety-critical refusal and compliance properties)
transfer through distillation/compression stages.

## Recommended methodology

1. Define explicit safety specifications
   - Use prompt suites per category (self-harm, violence, illegal activity, etc.)
   - Include both prohibited and permitted prompts to estimate false positives and false negatives

2. Run pre-distillation baseline
   - Evaluate safety specs on the teacher and the student baseline

3. Distill and validate iteratively
   - After each training stage, run the same safety suite
   - Treat the suite as a deployment gate: do not promote checkpoints that regress

4. Add equivalence and regression checks
   - Safety equivalence: student refusal behavior should match teacher on prohibited inputs
   - Benign equivalence: student should not over-refuse on normal prompts
   - Use stable, versioned prompt sets and keep them fixed for comparisons

## In this repository

Safety suites are YAML files (see `configs/safety/`).
Each suite produces `safety_report.json` with per-spec refusal rates and thresholds.

This scaffold includes refusal checks. For a masters-level extension:
- add graded rubrics (e.g., partial credit for safe redirection)
- add jailbreak-style prompt sets
- add structured evaluations (classifiers or LLM-as-judge with calibration)
