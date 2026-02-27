from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def run_lm_eval(
    *,
    model_name: str,
    tasks: list[str],
    output_dir: str,
    batch_size: int = 1,
    limit: int | None = None,
    trust_remote_code: bool = False,
    device: str | None = None,
    extra_model_args: dict[str, Any] | None = None,
) -> Path:
    """Runs EleutherAI lm-evaluation-harness via its CLI.

    This is the recommended path for benchmark-grade evaluation because it
    maintains task definitions and scoring logic close to upstream.

    Notes:
    - Some tasks require additional packages or dataset access.
    - The harness CLI and model_args schema may evolve; check upstream docs if it errors.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_args = {"pretrained": model_name}
    if trust_remote_code:
        model_args["trust_remote_code"] = True
    if extra_model_args:
        model_args.update(extra_model_args)

    # lm-eval expects comma-separated key=value list
    model_args_str = ",".join(
        f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in model_args.items()
    )

    cmd = [
        "python",
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args_str,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(out_dir / "lm_eval_results.json"),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if device is not None:
        cmd += ["--device", device]

    log.info("Running lm-eval: %s", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)
    return out_dir / "lm_eval_results.json"


def load_lm_eval(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
