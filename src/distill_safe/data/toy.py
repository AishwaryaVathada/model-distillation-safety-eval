from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToyExample:
    prompt: str
    expected_prefix: str | None = None


TOY_EXAMPLES: list[ToyExample] = [
    ToyExample("What is 2+2?", expected_prefix="4"),
    ToyExample("Write a short python function that adds two numbers.", expected_prefix="def"),
    ToyExample("Translate 'good morning' to Mandarin.", expected_prefix=""),
    ToyExample("Summarize: Safety testing is important in model distillation.", expected_prefix=""),
]
