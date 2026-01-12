"""CLI Event Handler with Pattern Matching.

This module provides the CLIEventHandler class that processes typed Osprey events
using Python's pattern matching (match/case) for clean console output.

The handler provides minimal, focused output for CLI usage - showing capability
progress, results, and errors without the detailed block structure of the TUI.

Usage:
    from osprey.events import parse_event
    from osprey.interfaces.cli.event_handler import CLIEventHandler

    handler = CLIEventHandler(console)

    async for chunk in graph.astream(..., stream_mode="custom"):
        event = parse_event(chunk)
        if event:
            await handler.handle(event)
"""

from typing import TYPE_CHECKING

from rich.console import Console

from osprey.events import (
    CapabilitiesSelectedEvent,
    CapabilityCompleteEvent,
    CapabilityStartEvent,
    ErrorEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    OspreyEvent,
    PhaseCompleteEvent,
    PhaseStartEvent,
    PlanCreatedEvent,
    ResultEvent,
    StatusEvent,
    TaskExtractedEvent,
)

if TYPE_CHECKING:
    pass


# Phase name mapping for display
PHASE_DISPLAY_NAMES = {
    "task_extraction": "Task Extraction",
    "classification": "Classification",
    "planning": "Planning",
    "execution": "Execution",
    "response": "Response",
}


class CLIEventHandler:
    """Handles typed Osprey events for CLI output using pattern matching.

    This handler provides minimal, focused console output suitable for CLI usage.
    It shows capability progress, final results, and errors.

    Attributes:
        console: Rich Console for output
        verbose: Whether to show detailed status updates
        show_timing: Whether to show timing information
    """

    def __init__(
        self,
        console: Console | None = None,
        verbose: bool = False,
        show_timing: bool = True,
    ):
        """Initialize the CLI event handler.

        Args:
            console: Rich Console for output (creates new one if not provided)
            verbose: Whether to show detailed status updates
            show_timing: Whether to show timing information
        """
        self.console = console or Console()
        self.verbose = verbose
        self.show_timing = show_timing
        self._current_capability: str | None = None

    async def handle(self, event: OspreyEvent) -> None:
        """Process a typed event using pattern matching.

        Routes the event to appropriate output based on event type.

        Args:
            event: The typed OspreyEvent to process
        """
        match event:
            # Phase lifecycle events (verbose only)
            case PhaseStartEvent(phase=phase, description=desc) if self.verbose:
                display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
                self.console.print(f"[dim cyan]>> {display_name}[/dim cyan]")

            case PhaseCompleteEvent(
                phase=phase, success=success, duration_ms=duration
            ) if self.verbose:
                display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
                status = "[green]OK[/green]" if success else "[red]FAILED[/red]"
                timing = f" ({duration}ms)" if self.show_timing else ""
                self.console.print(f"[dim]   {display_name}: {status}{timing}[/dim]")

            # Task preparation events (verbose only)
            case TaskExtractedEvent(task=task) if self.verbose:
                preview = task[:80] + "..." if len(task) > 80 else task
                self.console.print(f"[dim cyan]   Task: {preview}[/dim cyan]")

            case CapabilitiesSelectedEvent(capability_names=names) if self.verbose:
                self.console.print(f"[dim cyan]   Capabilities: {', '.join(names)}[/dim cyan]")

            case PlanCreatedEvent(steps=steps) if self.verbose:
                self.console.print(f"[dim cyan]   Plan: {len(steps)} steps[/dim cyan]")

            # LLM events (verbose only)
            case LLMRequestEvent(prompt_preview=preview) if self.verbose:
                short_preview = preview[:50] + "..." if len(preview) > 50 else preview
                self.console.print(f"[dim]   LLM Request: {short_preview}[/dim]")

            case LLMResponseEvent(response_preview=preview) if self.verbose:
                short_preview = preview[:50] + "..." if len(preview) > 50 else preview
                self.console.print(f"[dim]   LLM Response: {short_preview}[/dim]")

            # Capability execution events (always shown)
            case CapabilityStartEvent(
                capability_name=name, step_number=step, total_steps=total, description=desc
            ):
                self._current_capability = name
                self.console.print(
                    f"[cyan]>> Step {step}/{total}: {name}[/cyan]"
                )
                if self.verbose and desc:
                    self.console.print(f"[dim]   {desc}[/dim]")

            case CapabilityCompleteEvent(
                capability_name=name, success=success, duration_ms=duration, error_message=err
            ):
                if success:
                    timing = f" ({duration}ms)" if self.show_timing else ""
                    self.console.print(f"[green]   OK{timing}[/green]")
                else:
                    self.console.print(f"[red]   FAILED: {err or 'Unknown error'}[/red]")
                self._current_capability = None

            # Status updates (verbose only, or errors always)
            case StatusEvent(message=msg, level="error"):
                self.console.print(f"[red]   Error: {msg}[/red]")

            case StatusEvent(message=msg, level="warning"):
                self.console.print(f"[yellow]   Warning: {msg}[/yellow]")

            case StatusEvent(message=msg, level=level) if self.verbose:
                # Show other status updates only in verbose mode
                if level == "success":
                    self.console.print(f"[green]   {msg}[/green]")
                else:
                    self.console.print(f"[dim]   {msg}[/dim]")

            # Result events (always shown)
            case ResultEvent(response=response, success=success):
                if success:
                    self.console.print(f"\n[bold green]{response}[/bold green]")
                else:
                    self.console.print(f"\n[bold red]Execution failed: {response}[/bold red]")

            # Error events (always shown)
            case ErrorEvent(error_type=err_type, error_message=msg, recoverable=recoverable):
                recovery = " (recoverable)" if recoverable else ""
                self.console.print(f"[bold red]Error{recovery}: {err_type}[/bold red]")
                self.console.print(f"[red]{msg}[/red]")

            case _:
                # Unknown event type - skip silently
                pass

    def handle_sync(self, event: OspreyEvent) -> None:
        """Synchronous version of handle for non-async contexts.

        Args:
            event: The typed OspreyEvent to process
        """
        import asyncio

        # Run the async handler synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is already running, just process directly
                # (pattern matching doesn't need async)
                self._handle_sync_impl(event)
            else:
                loop.run_until_complete(self.handle(event))
        except RuntimeError:
            # No event loop - process directly
            self._handle_sync_impl(event)

    def _handle_sync_impl(self, event: OspreyEvent) -> None:
        """Direct synchronous implementation of event handling.

        Args:
            event: The typed OspreyEvent to process
        """
        match event:
            case PhaseStartEvent(phase=phase, description=desc) if self.verbose:
                display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
                self.console.print(f"[dim cyan]>> {display_name}[/dim cyan]")

            case PhaseCompleteEvent(
                phase=phase, success=success, duration_ms=duration
            ) if self.verbose:
                display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
                status = "[green]OK[/green]" if success else "[red]FAILED[/red]"
                timing = f" ({duration}ms)" if self.show_timing else ""
                self.console.print(f"[dim]   {display_name}: {status}{timing}[/dim]")

            # Task preparation events (verbose only)
            case TaskExtractedEvent(task=task) if self.verbose:
                preview = task[:80] + "..." if len(task) > 80 else task
                self.console.print(f"[dim cyan]   Task: {preview}[/dim cyan]")

            case CapabilitiesSelectedEvent(capability_names=names) if self.verbose:
                self.console.print(f"[dim cyan]   Capabilities: {', '.join(names)}[/dim cyan]")

            case PlanCreatedEvent(steps=steps) if self.verbose:
                self.console.print(f"[dim cyan]   Plan: {len(steps)} steps[/dim cyan]")

            # LLM events (verbose only)
            case LLMRequestEvent(prompt_preview=preview) if self.verbose:
                short_preview = preview[:50] + "..." if len(preview) > 50 else preview
                self.console.print(f"[dim]   LLM Request: {short_preview}[/dim]")

            case LLMResponseEvent(response_preview=preview) if self.verbose:
                short_preview = preview[:50] + "..." if len(preview) > 50 else preview
                self.console.print(f"[dim]   LLM Response: {short_preview}[/dim]")

            case CapabilityStartEvent(
                capability_name=name, step_number=step, total_steps=total, description=desc
            ):
                self._current_capability = name
                self.console.print(f"[cyan]>> Step {step}/{total}: {name}[/cyan]")
                if self.verbose and desc:
                    self.console.print(f"[dim]   {desc}[/dim]")

            case CapabilityCompleteEvent(
                capability_name=name, success=success, duration_ms=duration, error_message=err
            ):
                if success:
                    timing = f" ({duration}ms)" if self.show_timing else ""
                    self.console.print(f"[green]   OK{timing}[/green]")
                else:
                    self.console.print(f"[red]   FAILED: {err or 'Unknown error'}[/red]")
                self._current_capability = None

            case StatusEvent(message=msg, level="error"):
                self.console.print(f"[red]   Error: {msg}[/red]")

            case StatusEvent(message=msg, level="warning"):
                self.console.print(f"[yellow]   Warning: {msg}[/yellow]")

            case StatusEvent(message=msg, level=level) if self.verbose:
                if level == "success":
                    self.console.print(f"[green]   {msg}[/green]")
                else:
                    self.console.print(f"[dim]   {msg}[/dim]")

            case ResultEvent(response=response, success=success):
                if success:
                    self.console.print(f"\n[bold green]{response}[/bold green]")
                else:
                    self.console.print(f"\n[bold red]Execution failed: {response}[/bold red]")

            case ErrorEvent(error_type=err_type, error_message=msg, recoverable=recoverable):
                recovery = " (recoverable)" if recoverable else ""
                self.console.print(f"[bold red]Error{recovery}: {err_type}[/bold red]")
                self.console.print(f"[red]{msg}[/red]")

            case _:
                pass
