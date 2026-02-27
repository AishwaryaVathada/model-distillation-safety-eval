from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TeacherResponse:
    text: str
    rationale: str | None = None
    raw: dict[str, Any] | None = None


class Teacher(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> TeacherResponse:
        raise NotImplementedError
