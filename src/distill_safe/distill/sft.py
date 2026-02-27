from __future__ import annotations

import inspect
import logging
from pathlib import Path

from ..utils.io import read_jsonl

log = logging.getLogger(__name__)


def run_sft_lora(
    *,
    student_model: str,
    data_path: str,
    output_dir: str,
    max_seq_len: int = 2048,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 1,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 200,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
) -> str:
    """Minimal LoRA SFT using TRL + PEFT.

    This is intentionally conservative and may need tuning per base model.
    """

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import TrainingArguments
        from trl import SFTTrainer
        try:
            from trl import SFTConfig
        except Exception:
            SFTConfig = None
    except Exception as e:
        raise RuntimeError(
            "Training dependencies missing. Install requirements.txt and ensure torch/trl/peft are available."
        ) from e

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(data_path)

    tok = AutoTokenizer.from_pretrained(student_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules or None,
    )

    def format_fn(example):
        # Train on rendered input -> teacher output
        return f"{example['input']}\n{example['output']}"

    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    use_modern_trl = "max_seq_length" not in trainer_sig and SFTConfig is not None

    if use_modern_trl:
        ds = Dataset.from_list([{"text": format_fn(r)} for r in rows])
        args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            report_to="none",
            dataset_text_field="text",
            max_length=max_seq_len,
            completion_only_loss=False,
            use_cpu=not torch.cuda.is_available(),
            bf16=torch.cuda.is_available(),
            fp16=False,
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            peft_config=peft_cfg,
            processing_class=tok,
            args=args,
        )
    else:
        ds = Dataset.from_list(rows)
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            report_to=[],
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            peft_config=peft_cfg,
            formatting_func=format_fn,
            max_seq_length=max_seq_len,
            tokenizer=tok,
            args=args,
        )

    log.info("Starting SFT: model=%s data=%s", student_model, data_path)
    trainer.train()
    trainer.save_model(output_dir)
    log.info("Saved checkpoint: %s", output_dir)
    return output_dir
