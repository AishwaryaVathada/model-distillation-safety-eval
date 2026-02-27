from __future__ import annotations

import os
from typing import Any

import requests

from .base import Teacher, TeacherResponse


class OpenAICompatibleTeacher(Teacher):
    """A minimal OpenAI-compatible chat completion client.

    This is intentionally implemented with `requests` to avoid SDK drift.
    Configure using environment variables:
      - OPENAI_BASE_URL (e.g., https://api.deepseek.com)
      - OPENAI_API_KEY
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: int = 120,
        extra_headers: dict[str, str] | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}

        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL is required for openai_compatible teacher.")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for openai_compatible teacher.")

    def generate(self, prompt: str) -> TeacherResponse:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        headers.update(self.extra_headers)

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()

        # OpenAI-compatible response structure
        choice0 = data["choices"][0]["message"]
        text = choice0.get("content", "") or ""
        return TeacherResponse(text=text, rationale=None, raw=data)
