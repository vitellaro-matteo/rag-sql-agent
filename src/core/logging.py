"""Structured logging via structlog with rich console rendering.

Every agent step is logged as a structured event so the full reasoning
chain is observable in both console and JSON outputs.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from rich.console import Console

from src.core.config import load_settings

_console = Console(stderr=True)
_configured = False


def setup_logging() -> None:
    """Configure structlog processors and stdlib bridge.

    Call once at application startup. Subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    cfg = load_settings().logging
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if cfg.format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy libraries
    for name in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named, bound structlog logger.

    Args:
        name: Logger name, typically the module path.

    Returns:
        A BoundLogger with the given name bound as context.
    """
    setup_logging()
    return structlog.get_logger(name)


class AgentTrace:
    """Accumulates structured trace events for a single pipeline run.

    The Streamlit UI reads these events to render the live reasoning chain.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        agent: str,
        action: str,
        detail: Any = None,
        *,
        status: str = "ok",
    ) -> None:
        """Append a trace event.

        Args:
            agent: Name of the agent producing this event.
            action: Short verb describing the step (e.g. "classify", "generate_sql").
            detail: Arbitrary payload (dict, string, etc.).
            status: One of "ok", "warning", "error".
        """
        import time

        event = {
            "ts": time.time(),
            "agent": agent,
            "action": action,
            "detail": detail,
            "status": status,
        }
        self.events.append(event)

        logger = get_logger(f"trace.{agent}")
        log_method = logger.info if status == "ok" else logger.warning
        log_method(action, **{k: v for k, v in event.items() if k != "ts"})
