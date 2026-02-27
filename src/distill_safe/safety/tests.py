from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..teacher.base import Teacher
from .refusal import looks_like_refusal

log = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    spec_id: str
    total: int
    refused: int

    @property
    def refusal_rate(self) -> float:
        return self.refused / self.total if self.total else 0.0


def run_refusal_spec(teacher_or_model: Teacher, spec_id: str, prompts: list[str]) -> SafetyResult:
    refused = 0
    for p in prompts:
        resp = teacher_or_model.generate(p)
        if looks_like_refusal(resp.text):
            refused += 1
    return SafetyResult(spec_id=spec_id, total=len(prompts), refused=refused)


def save_safety_report(path: str | Path, report: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
