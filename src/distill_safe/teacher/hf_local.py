from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import Teacher, TeacherResponse

log = logging.getLogger(__name__)


@dataclass
class HFLocalGenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


class HFLocalTeacher(Teacher):
    """Wraps a Hugging Face causal LM (optionally a PEFT adapter checkpoint) as a Teacher-like interface.

    Supports:
      - HF base models: model="Qwen/Qwen2.5-0.5B-Instruct"
      - Local checkpoints: model="/path/to/checkpoint"
      - PEFT adapter dirs produced by LoRA SFT (detects adapter_config.json and loads base + adapter)

    For benchmark-grade evaluation, prefer lm-evaluation-harness.
    """

    def __init__(
        self,
        model: str,
        *,
        dtype: str | None = None,
        device_map: str | None = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError("Missing transformers/torch. Install requirements.txt.") from e

        self.model_id = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

        torch_dtype = None
        if dtype:
            d = dtype.lower()
            if d in ("bf16", "bfloat16"):
                torch_dtype = torch.bfloat16
            elif d in ("fp16", "float16"):
                torch_dtype = torch.float16
            elif d in ("fp32", "float32"):
                torch_dtype = torch.float32

        model_path = Path(model)
        is_peft_adapter = model_path.exists() and (model_path / "adapter_config.json").exists()

        if is_peft_adapter:
            # Load base + adapter
            try:
                from peft import PeftModel
            except Exception as e:
                raise RuntimeError("PEFT adapter detected but peft is not installed.") from e

            adapter_cfg = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
            base_id = adapter_cfg.get("base_model_name_or_path")
            if not base_id:
                raise ValueError("adapter_config.json missing base_model_name_or_path")

            self.tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token

            base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch_dtype, device_map=device_map)
            self.lm = PeftModel.from_pretrained(base, model)
            self.lm.eval()
            log.info("Loaded PEFT adapter: base=%s adapter=%s", base_id, model)
        else:
            self.tok = AutoTokenizer.from_pretrained(model, use_fast=True)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token

            self.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype, device_map=device_map)
            self.lm.eval()
            log.info("Loaded HF model: %s", model)

    def generate(self, prompt: str) -> TeacherResponse:
        import torch

        inputs = self.tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.lm.device) for k, v in inputs.items()}

        do_sample = self.temperature > 0.0
        with torch.no_grad():
            out = self.lm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tok.eos_token_id,
            )

        text = self.tok.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        return TeacherResponse(text=text, rationale=None, raw={"model": self.model_id})
