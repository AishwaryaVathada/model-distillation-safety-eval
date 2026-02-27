from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str
    output_dir: str
    seed: int = 0


class TeacherConfig(BaseModel):
    kind: str = Field(description="dummy | openai_compatible | vllm_local")
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024


class StudentConfig(BaseModel):
    model: str
    dtype: str | None = None
    max_seq_len: int = 2048


class DataGenConfig(BaseModel):
    dataset: str = "toy"
    num_samples: int = 100
    store_rationale: bool = False
    prompt_template: str = "default_reasoning_distill"
    output_path: str | None = None


class DistillConfig(BaseModel):
    method: str = "sft"  # sft | noop
    # for sft, see TrainConfig below


class LoRAConfig(BaseModel):
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = []


class TrainConfig(BaseModel):
    data_path: str
    output_dir: str
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200


class ValidationGatesConfig(BaseModel):
    run_safety_suite: bool = True
    run_eval_suite: bool = True
    safety_suite: str = "configs/safety/suite_default.yaml"
    eval_suite: str = "configs/eval/suite_default.yaml"


class PipelineConfig(BaseModel):
    run: RunConfig
    teacher: TeacherConfig | None = None
    student: StudentConfig | None = None
    data_generation: DataGenConfig | None = None
    distill: DistillConfig | None = None
    train: TrainConfig | None = None
    lora: LoRAConfig | None = None
    safety: dict[str, Any] | None = None
    eval: dict[str, Any] | None = None
    validation_gates: ValidationGatesConfig | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    raw = load_yaml(path)
    return PipelineConfig.model_validate(raw)
