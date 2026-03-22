"""CLI entry point for interactive query processing.

Run directly::

    python -m src.main
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.orchestrator import PipelineOrchestrator
from src.core.logging import AgentTrace, setup_logging

console = Console()


def _print_trace(trace: AgentTrace) -> None:
    """Render the agent trace as a rich table."""
    table = Table(title="Agent Trace", show_lines=True)
    table.add_column("Agent", style="cyan", width=18)
    table.add_column("Action", style="green", width=24)
    table.add_column("Status", width=8)
    table.add_column("Detail", max_width=60)

    for event in trace.events:
        status = event.get("status", "ok")
        style = "red" if status == "error" else "yellow" if status == "warning" else ""
        detail = str(event.get("detail", ""))[:120]
        table.add_row(
            event.get("agent", ""),
            event.get("action", ""),
            f"[{style}]{status}[/{style}]" if style else status,
            detail,
        )
    console.print(table)


async def _run() -> None:
    """Interactive REPL loop."""
    setup_logging()
    orch = PipelineOrchestrator()

    console.print(Panel(
        "[bold]RAG-SQL Multi-Agent System[/bold]\n"
        "Ask natural-language questions about the fintech database.\n"
        "Type [bold cyan]quit[/bold cyan] to exit. "
        "Prefix with [bold cyan]role:admin[/bold cyan] to change roles.",
        border_style="blue",
    ))

    try:
        await orch.initialize()
    except FileNotFoundError as exc:
        console.print(f"[red]Setup error:[/red] {exc}")
        console.print("Run: python -m scripts.seed_db && python -m scripts.build_index")
        sys.exit(1)

    role = "analyst"

    while True:
        try:
            raw = console.input("\n[bold blue]You>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        # Allow role switching via prefix
        if raw.lower().startswith("role:"):
            role = raw.split(":", 1)[1].strip()
            console.print(f"[yellow]Switched to role: {role}[/yellow]")
            continue

        trace = AgentTrace()
        result = await orch.process(raw, role=role, trace=trace)

        _print_trace(trace)

        explanation = result.get("explanation", "No response generated.")
        console.print(Panel(explanation, title="Answer", border_style="green"))

        # Show follow-ups if available
        suggestions = result.get("follow_up_suggestions", [])
        if suggestions:
            console.print("[dim]Suggested follow-ups:[/dim]")
            for s in suggestions:
                console.print(f"  [dim]→ {s}[/dim]")

    await orch.shutdown()
    console.print("[dim]Goodbye.[/dim]")


def main() -> None:
    """Entry point."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
