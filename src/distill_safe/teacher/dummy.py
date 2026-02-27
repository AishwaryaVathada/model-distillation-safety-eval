from __future__ import annotations

from .base import Teacher, TeacherResponse


class DummyTeacher(Teacher):
    def __init__(self, model: str = "dummy-teacher", temperature: float = 0.0, max_tokens: int = 256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> TeacherResponse:
        # Deterministic, minimal answer for smoke tests.
        text = "Answer: " + prompt.strip()[:200]
        return TeacherResponse(text=text, rationale=None, raw={"model": self.model})
