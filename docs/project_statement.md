# Project Statement

## Title

Evaluating the Performance of Distilled Open-Source Models Using DeepSeek R1:
A Comparative Study of Pre- and Post-Distillation Performance

## Research objective

Design and validate a multi-stage model compression pipeline where safety specifications and
behavioral constraints transfer correctly through distillation (specification transfer analysis).

## Claims supported by this repository

- Multi-stage compression is modeled as a sequence of stages (teacher -> intermediate -> student),
  with explicit validation gates between stages.
- Safety preservation is checked via an automated specification suite that can be versioned and
  used as a deployment gate.
- Correctness and semantic equivalence are assessed through pre/post benchmark comparisons and
  structured reports.

## Practical constraint (compute)

Large-teacher distillation (example teacher scale: 671B parameters) typically requires external
GPU servers or an API-hosted teacher endpoint. This repository is therefore designed to support
OpenAI-compatible teacher APIs and external training infrastructure, while still providing a CPU
smoke test for local verification.

## Benchmarks

The pre/post evaluation suite includes:
- English: MMLU, AGIEval
- Math: MATH, MGSM
- Code: HumanEval, APPS
- Mandarin: C-Eval, CLUEWSC (subset of CLUE)

See `docs/benchmarks.md` for execution notes.
