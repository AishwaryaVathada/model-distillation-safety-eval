from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from ..utils.io import write_jsonl
from .prompts import render
from .toy import TOY_EXAMPLES
from ..teacher.base import Teacher

log = logging.getLogger(__name__)


@dataclass
class SyntheticRow:
    prompt: str
    response: str
    rationale: str | None = None
    teacher_model: str | None = None


def _iter_prompts(dataset: str, num_samples: int) -> Iterable[str]:
    if dataset == "toy":
        base = [ex.prompt for ex in TOY_EXAMPLES]
        out: list[str] = []
        while len(out) < num_samples:
            out.extend(base)
        return out[:num_samples]
    raise ValueError(f"Unknown dataset: {dataset}")


def generate_synthetic(
    *,
    teacher: Teacher,
    dataset: str,
    num_samples: int,
    prompt_template: str,
    output_path: str,
    store_rationale: bool,
) -> str:
    prompts = list(_iter_prompts(dataset, num_samples))
    rows: list[dict] = []
    for p in prompts:
        rendered = render(prompt_template, p)
        resp = teacher.generate(rendered)
        rows.append(
            {
                "prompt": p,
                "input": rendered,
                "output": resp.text,
                "rationale": resp.rationale if store_rationale else None,
                "raw": resp.raw,
            }
        )
    write_jsonl(output_path, rows)
    log.info("Wrote synthetic data: %s (n=%d)", output_path, len(rows))
    return output_path
