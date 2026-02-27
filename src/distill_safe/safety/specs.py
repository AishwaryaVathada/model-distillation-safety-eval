from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SpecKind = Literal["refusal"]


@dataclass
class SafetySpec:
    id: str
    kind: SpecKind
    category: str
    prompts: list[str]
