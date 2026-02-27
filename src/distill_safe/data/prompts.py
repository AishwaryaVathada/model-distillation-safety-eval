from __future__ import annotations

DEFAULT_TEMPLATE = """You are a teacher model. Produce a high-quality answer.

Question:
{prompt}

Answer:
"""

DEFAULT_REASONING_DISTILL_TEMPLATE = """You are a teacher model for distillation.
Return the final answer clearly and concisely.

Question:
{prompt}

Final answer:
"""


def render(template_name: str, prompt: str) -> str:
    if template_name == "default":
        return DEFAULT_TEMPLATE.format(prompt=prompt)
    if template_name == "default_reasoning_distill":
        return DEFAULT_REASONING_DISTILL_TEMPLATE.format(prompt=prompt)
    raise ValueError(f"Unknown prompt template: {template_name}")
