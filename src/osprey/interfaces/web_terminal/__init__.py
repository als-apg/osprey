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

#: Public name -> the submodule of this package that defines it. Entries are
#: resolved on first attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {"run_web": ".app"}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
