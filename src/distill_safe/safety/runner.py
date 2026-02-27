from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import load_yaml
from ..teacher.base import Teacher
from .tests import run_refusal_spec, save_safety_report

log = logging.getLogger(__name__)


def run_safety_suite(model: Teacher, suite_config_path: str, output_dir: str) -> dict[str, Any]:
    suite = load_yaml(suite_config_path)
    specs = suite.get("specs", [])
    thresholds = suite.get("thresholds", {})

    results: list[dict[str, Any]] = []
    for s in specs:
        if s["kind"] == "refusal":
            r = run_refusal_spec(model, s["id"], s["prompts"])
            results.append(
                {"spec_id": r.spec_id, "total": r.total, "refused": r.refused, "refusal_rate": r.refusal_rate}
            )
        else:
            raise ValueError(f"Unknown spec kind: {s['kind']}")

    report = {"suite": suite.get("run", {}), "results": results, "thresholds": thresholds}
    out_path = Path(output_dir) / "safety_report.json"
    save_safety_report(out_path, report)
    log.info("Wrote safety report: %s", out_path)
    return report
