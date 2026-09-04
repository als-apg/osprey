"""OSPREY Web Terminal Interface.

A browser-based split-pane interface with a real terminal (running Claude Code
via PTY) on the left and a live workspace file viewer on the right.

``run_web`` is exported lazily. Importing it eagerly meant that reaching *any*
module in this package — the notebook sidecar and its kernel-side helpers among
them — built the FastAPI application and pulled in uvicorn, in processes that
serve no HTTP at all. The module ``__getattr__`` keeps ``from
osprey.interfaces.web_terminal import run_web`` working unchanged while a
sibling import stays a sibling import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.interfaces.web_terminal.app import run_web

__all__ = ["run_web"]


def __getattr__(name: str) -> Any:
    """Resolve ``run_web`` on first access, leaving every other name unbound.

    Args:
        name: Attribute requested from the package.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: For any name this package does not export.
    """
    if name == "run_web":
        from osprey.interfaces.web_terminal.app import run_web

        return run_web
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
