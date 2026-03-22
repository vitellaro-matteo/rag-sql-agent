"""Abstract base class for all pipeline agents.

Provides a uniform interface for LLM-backed agents with built-in
prompt loading, structured response parsing, and trace recording.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.llm import complete, load_prompt, parse_json_response
from src.core.logging import AgentTrace, get_logger


class BaseAgent(ABC):
    """Base class every agent inherits from.

    Subclasses must implement ``run()`` and set ``name`` and ``prompt_name``.
    """

    name: str = "base"
    prompt_name: str = ""  # Filename stem in config/prompts/

    def __init__(self) -> None:
        self.logger = get_logger(f"agent.{self.name}")
        self._prompt: dict[str, str] | None = None

    @property
    def prompt(self) -> dict[str, str]:
        """Lazy-load the YAML prompt template for this agent."""
        if self._prompt is None and self.prompt_name:
            self._prompt = load_prompt(self.prompt_name)
        return self._prompt or {}

    async def call_llm(
        self,
        template_vars: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Render prompts from the YAML template, call the LLM, and parse JSON.

        Args:
            template_vars: Variables to interpolate into the prompt templates.
            temperature: Override the default temperature.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            ValueError: If the LLM response is not valid JSON.
        """
        system = self.prompt.get("system", "").format(**template_vars)
        user = self.prompt.get("user", "").format(**template_vars)

        raw = await complete(system, user, temperature=temperature)
        self.logger.debug("raw_llm_response", response=raw[:500])

        return parse_json_response(raw)

    @abstractmethod
    async def run(self, context: dict[str, Any], trace: AgentTrace) -> dict[str, Any]:
        """Execute this agent's logic.

        Args:
            context: Shared pipeline context dict, mutated by each agent.
            trace: Trace recorder for observability.

        Returns:
            Updated context dict.
        """
        ...
