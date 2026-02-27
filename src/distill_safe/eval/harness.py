from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import load_yaml
from ..teacher.base import Teacher
from .toy_eval import run_toy_eval

log = logging.getLogger(__name__)


def run_eval_suite(model: Teacher, suite_config_path: str, output_dir: str) -> dict[str, Any]:
    suite = load_yaml(suite_config_path)
    benchmarks = suite.get("benchmarks", [])

    results: list[dict[str, Any]] = []
    for b in benchmarks:
        kind = b.get("kind")
        if kind == "toy":
            results.append(run_toy_eval(model))
        elif kind == "lm_eval":
            # Placeholder: integrate with lm-evaluation-harness by spawning its CLI.
            # This keeps this repo lightweight and avoids task mapping drift.
            results.append(
                {
                    "benchmark": b["name"],
                    "kind": "lm_eval",
                    "task": b.get("task"),
                    "status": "not_implemented_in_smoke_mode",
                    "hint": "Use distill-safe eval lm-eval to run lm-evaluation-harness tasks.",
                }
            )
        else:
            # hf_* wrappers are intentionally not fully implemented in this scaffold:
            # they require task-specific formatting and scoring.
            results.append(
                {"benchmark": b["name"], "kind": kind, "status": "scaffold_only", "details": b}
            )

    report = {"suite": suite.get("run", {}), "results": results}
    out_path = Path(output_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote eval results: %s", out_path)
    return report
