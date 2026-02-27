from __future__ import annotations

from dataclasses import dataclass

from ..data.toy import TOY_EXAMPLES
from ..teacher.base import Teacher


@dataclass
class ToyEvalResult:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_toy_eval(model: Teacher) -> dict:
    correct = 0
    for ex in TOY_EXAMPLES:
        resp = model.generate(ex.prompt)
        if ex.expected_prefix is None:
            correct += 1
        else:
            if resp.text.strip().lower().startswith(ex.expected_prefix.strip().lower()):
                correct += 1
    r = ToyEvalResult(total=len(TOY_EXAMPLES), correct=correct)
    return {"benchmark": "toy_accuracy", "accuracy": r.accuracy, "total": r.total, "correct": r.correct}
