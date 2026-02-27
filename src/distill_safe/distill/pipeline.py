from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from ..data.generation import generate_synthetic
from ..eval.harness import run_eval_suite
from ..logging import setup_logging
from ..safety.runner import run_safety_suite
from ..teacher.factory import build_teacher
from ..utils.seeding import seed_everything
from .sft import run_sft_lora

log = logging.getLogger(__name__)


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    setup_logging(cfg.run.output_dir)
    seed_everything(cfg.run.seed)

    Path(cfg.run.output_dir).mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(cfg.run.output_dir) / "config_snapshot.json"
    snapshot_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    teacher = build_teacher(cfg.teacher) if cfg.teacher else None
    if teacher is None:
        raise ValueError("Pipeline requires a teacher (use dummy for smoke tests).")

    synthetic_path = None
    if cfg.data_generation:
        output_path = cfg.data_generation.output_path or str(Path(cfg.run.output_dir) / "synthetic.jsonl")
        synthetic_path = generate_synthetic(
            teacher=teacher,
            dataset=cfg.data_generation.dataset,
            num_samples=cfg.data_generation.num_samples,
            prompt_template=cfg.data_generation.prompt_template,
            output_path=output_path,
            store_rationale=cfg.data_generation.store_rationale,
        )

    # Distillation stage
    checkpoint_dir = None
    if cfg.distill and cfg.distill.method == "sft":
        if not (cfg.student and cfg.train and cfg.lora):
            raise ValueError("SFT requires student/train/lora configs.")
        checkpoint_dir = run_sft_lora(
            student_model=cfg.student.model,
            data_path=cfg.train.data_path or synthetic_path or "",
            output_dir=cfg.train.output_dir,
            max_seq_len=cfg.student.max_seq_len,
            per_device_train_batch_size=cfg.train.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            learning_rate=cfg.train.learning_rate,
            num_train_epochs=cfg.train.num_train_epochs,
            warmup_ratio=cfg.train.warmup_ratio,
            logging_steps=cfg.train.logging_steps,
            save_steps=cfg.train.save_steps,
            lora_r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            lora_target_modules=cfg.lora.target_modules,
        )
    elif cfg.distill and cfg.distill.method == "noop":
        checkpoint_dir = None

    # Safety and eval gates (for smoke mode, teacher acts as the "model under test")
    # In a real run, you would wrap the trained student as a Teacher-like generator.
    safety_report = None
    if cfg.safety and cfg.safety.get("suite"):
        suite_path = cfg.safety["suite"]
        safety_report = run_safety_suite(teacher, suite_path, cfg.run.output_dir)

    eval_report = None
    if cfg.eval and cfg.eval.get("enabled", False):
        suite_path = cfg.eval["suite"]
        eval_report = run_eval_suite(teacher, suite_path, cfg.run.output_dir)

    out = {
        "run": cfg.run.model_dump(),
        "synthetic_path": synthetic_path,
        "checkpoint_dir": checkpoint_dir,
        "safety_report": safety_report,
        "eval_report": eval_report,
    }
    (Path(cfg.run.output_dir) / "run_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Pipeline complete. Summary: %s", Path(cfg.run.output_dir) / "run_summary.json")
    return out
