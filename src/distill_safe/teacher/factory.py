from __future__ import annotations

from ..config import TeacherConfig
from .base import Teacher
from .dummy import DummyTeacher
from .hf_local import HFLocalTeacher
from .openai_compatible import OpenAICompatibleTeacher


def build_teacher(cfg: TeacherConfig) -> Teacher:
    kind = cfg.kind.lower()
    if kind == "dummy":
        return DummyTeacher(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    if kind == "openai_compatible":
        return OpenAICompatibleTeacher(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    if kind == "hf_local":
        return HFLocalTeacher(model=cfg.model, dtype=None, max_new_tokens=cfg.max_tokens, temperature=cfg.temperature)
    raise ValueError(f"Unknown teacher kind: {cfg.kind}")
