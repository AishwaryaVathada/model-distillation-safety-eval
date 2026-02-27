from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_reports(pre_path: str | Path, post_path: str | Path) -> str:
    pre = load_report(pre_path)
    post = load_report(post_path)

    pre_map = {r["benchmark"]: r for r in pre.get("results", []) if "benchmark" in r}
    post_map = {r["benchmark"]: r for r in post.get("results", []) if "benchmark" in r}

    keys = sorted(set(pre_map) | set(post_map))
    lines = []
    lines.append("| Benchmark | Pre | Post | Delta |")
    lines.append("|---|---:|---:|---:|")
    for k in keys:
        a = pre_map.get(k, {})
        b = post_map.get(k, {})
        pre_v = a.get("accuracy")
        post_v = b.get("accuracy")
        if isinstance(pre_v, (int, float)) and isinstance(post_v, (int, float)):
            delta = post_v - pre_v
            lines.append(f"| {k} | {pre_v:.4f} | {post_v:.4f} | {delta:+.4f} |")
        else:
            lines.append(f"| {k} | {pre_v if pre_v is not None else '-'} | {post_v if post_v is not None else '-'} | - |")
    return "\n".join(lines)
