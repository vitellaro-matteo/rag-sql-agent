"""LLM abstraction layer backed by LiteLLM.

Provides a single async ``complete()`` function that routes to any
LiteLLM-supported backend (OpenAI, Ollama, Anthropic, Azure, etc.)
based on the provider/model pair in settings.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from litellm import acompletion

from src.core.config import load_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"


def load_prompt(name: str) -> dict[str, str]:
    """Load a prompt template from config/prompts/<name>.yaml.

    Args:
        name: Filename stem (e.g. "router" → config/prompts/router.yaml).

    Returns:
        Dict with at least ``system`` and ``user`` keys containing template strings.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _build_model_string() -> str:
    """Map config provider/model to a LiteLLM model string."""
    cfg = load_settings().llm
    provider = cfg.provider.lower()
    model = cfg.model

    mapping: dict[str, str] = {
        "openai": model,
        "ollama": f"ollama/{model}",
        "anthropic": model,
        "azure": f"azure/{model}",
        "huggingface": f"huggingface/{model}",
    }
    return mapping.get(provider, model)


async def complete(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: str | None = None,
) -> str:
    """Send a chat completion request through LiteLLM.

    Args:
        system_prompt: The system message setting agent behavior.
        user_prompt: The user message with the actual request.
        temperature: Override config temperature.
        max_tokens: Override config max_tokens.
        response_format: Optional hint (e.g. "json_object") — only used
            when the provider supports structured output.

    Returns:
        The assistant's response text.

    Raises:
        RuntimeError: If the LLM call fails after internal retries.
    """
    cfg = load_settings().llm
    model = _build_model_string()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature if temperature is not None else cfg.temperature,
        "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
        "timeout": cfg.request_timeout,
        "num_retries": 2,
    }

    # Ollama doesn't support response_format
    if response_format and cfg.provider.lower() not in ("ollama",):
        kwargs["response_format"] = {"type": response_format}

    logger.debug(
        "llm_request",
        model=model,
        system_len=len(system_prompt),
        user_len=len(user_prompt),
    )

    try:
        response = await acompletion(**kwargs)
        text: str = response.choices[0].message.content or ""
        logger.debug("llm_response", chars=len(text))
        return text.strip()
    except Exception as exc:
        logger.error("llm_error", error=str(exc), model=model)
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response.

    Handles both raw JSON and Markdown-fenced JSON blocks.

    Args:
        text: Raw LLM output.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")
