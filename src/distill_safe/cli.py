from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from .config import load_pipeline_config, load_yaml
from .data.generation import generate_synthetic
from .distill.pipeline import run_pipeline
from .distill.sft import run_sft_lora
from .eval.compare import compare_reports
from .eval.harness import run_eval_suite
from .eval.lm_eval_runner import run_lm_eval
from .logging import setup_logging
from .safety.runner import run_safety_suite
from .teacher.factory import build_teacher

app = typer.Typer(add_completion=False, no_args_is_help=True)

data_app = typer.Typer(no_args_is_help=True)
distill_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
safety_app = typer.Typer(no_args_is_help=True)
pipeline_app = typer.Typer(no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(distill_app, name="distill")
app.add_typer(eval_app, name="eval")
app.add_typer(safety_app, name="safety")
app.add_typer(pipeline_app, name="pipeline")


@pipeline_app.command("run")
def pipeline_run(config: str = typer.Option(..., help="Path to pipeline YAML config")):
    cfg = load_pipeline_config(config)
    run_pipeline(cfg)
    rprint("[green]OK[/green] Pipeline finished.")


@data_app.command("generate")
def data_generate(config: str = typer.Option(..., help="Path to data generation YAML config")):
    cfg = load_yaml(config)
    setup_logging(cfg["run"]["output_dir"])
    teacher = build_teacher(type("C", (), cfg["teacher"])())  # minimal shim
    out_path = cfg["data_generation"]["output_path"]
    generate_synthetic(
        teacher=teacher,
        dataset=cfg["data_generation"]["dataset"],
        num_samples=cfg["data_generation"]["num_samples"],
        prompt_template=cfg["data_generation"]["prompt_template"],
        output_path=out_path,
        store_rationale=cfg["data_generation"]["store_rationale"],
    )
    rprint(f"[green]OK[/green] Wrote {out_path}")


@distill_app.command("sft")
def distill_sft(config: str = typer.Option(..., help="Path to distillation YAML config")):
    cfg = load_yaml(config)
    setup_logging(cfg["run"]["output_dir"])
    out = run_sft_lora(
        student_model=cfg["student"]["model"],
        data_path=cfg["train"]["data_path"],
        output_dir=cfg["train"]["output_dir"],
        max_seq_len=cfg["student"].get("max_seq_len", 2048),
        per_device_train_batch_size=cfg["train"].get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg["train"].get("gradient_accumulation_steps", 1),
        learning_rate=cfg["train"].get("learning_rate", 2e-4),
        num_train_epochs=cfg["train"].get("num_train_epochs", 1),
        warmup_ratio=cfg["train"].get("warmup_ratio", 0.03),
        logging_steps=cfg["train"].get("logging_steps", 10),
        save_steps=cfg["train"].get("save_steps", 200),
        lora_r=cfg["lora"].get("r", 16),
        lora_alpha=cfg["lora"].get("alpha", 32),
        lora_dropout=cfg["lora"].get("dropout", 0.05),
        lora_target_modules=cfg["lora"].get("target_modules", []),
    )
    rprint(f"[green]OK[/green] Saved checkpoint to {out}")


@eval_app.command("run")
def eval_run(
    config: str = typer.Option(..., help="Path to eval suite YAML config"),
    model_kind: str = typer.Option("hf_local", help="hf_local | openai_compatible | dummy"),
    model_name: str = typer.Option("sshleifer/tiny-gpt2", help="Model id or provider model id"),
    output_dir: str = typer.Option("runs/eval", help="Output directory"),
    max_tokens: int = typer.Option(256, help="Max tokens for local generation (hf_local)"),
):
    setup_logging(output_dir)
    model_cfg = type("C", (), {"kind": model_kind, "model": model_name, "temperature": 0.0, "max_tokens": max_tokens})()
    model = build_teacher(model_cfg)
    report = run_eval_suite(model, config, output_dir)
    rprint(json.dumps(report, indent=2))


@eval_app.command("lm-eval")
def eval_lm_eval(
    model: str = typer.Option(..., help="HF model id or local path to checkpoint"),
    tasks: str = typer.Option(..., help="Comma-separated lm-eval tasks, e.g. mmlu,humaneval"),
    output_dir: str = typer.Option("runs/lm_eval", help="Output directory"),
    batch_size: int = typer.Option(1, help="Batch size"),
    limit: Optional[int] = typer.Option(None, help="Limit examples per task (debugging)"),
    trust_remote_code: bool = typer.Option(False, help="Pass trust_remote_code to HF loader"),
):
    setup_logging(output_dir)
    out = run_lm_eval(
        model_name=model,
        tasks=[t.strip() for t in tasks.split(",") if t.strip()],
        output_dir=output_dir,
        batch_size=batch_size,
        limit=limit,
        trust_remote_code=trust_remote_code,
    )
    rprint(f"Wrote {out}")


@eval_app.command("compare")
def eval_compare(pre: str = typer.Option(...), post: str = typer.Option(...)):
    md = compare_reports(pre, post)
    rprint(md)


@safety_app.command("run")
def safety_run(
    suite: str = typer.Option(..., help="Path to safety suite YAML config"),
    model_kind: str = typer.Option("hf_local", help="hf_local | openai_compatible | dummy"),
    model_name: str = typer.Option("sshleifer/tiny-gpt2"),
    output_dir: str = typer.Option("runs/safety", help="Output directory"),
    max_tokens: int = typer.Option(256, help="Max tokens for local generation (hf_local)"),
):
    setup_logging(output_dir)
    model_cfg = type("C", (), {"kind": model_kind, "model": model_name, "temperature": 0.0, "max_tokens": max_tokens})()
    model = build_teacher(model_cfg)
    report = run_safety_suite(model, suite, output_dir)
    rprint(json.dumps(report, indent=2))
