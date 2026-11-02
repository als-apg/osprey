"""The framework's one door to the connectors package's session hold.

Every framework site that refuses a write of a held session imports from here,
at module scope, so an ``osprey-connectors`` install without the hold stops the
server at start rather than letting it run with a write path the hold does not
cover. The import is the check: nothing else in the framework names these
connectors functions.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

__all__ = ["execution_verdict", "refusal_detail", "session_hold_refusal"]


def _installed_connectors() -> str:
    """The installed ``osprey-connectors`` version, for the message an operator reads."""
    try:
        return metadata.version("osprey-connectors")
    except metadata.PackageNotFoundError:
        return "not installed"


try:
    from osprey_connectors.posture_store import execution_verdict, session_hold_refusal
except ImportError as exc:
    # A RuntimeError, not an ImportError: an optional-import guard must not read
    # a server that cannot hold a session's writes as a feature that is absent.
    raise RuntimeError(
        "OSPREY needs osprey-connectors with session_hold, which this install does not "
        f"have (osprey-connectors {_installed_connectors()}). Upgrade osprey-connectors "
        "and start the server again."
    ) from exc


def refusal_detail(detail: str | None, verdict: Any) -> str | None:
    """A surface's own *detail*, followed by what *verdict* adds (a hold's sources)."""
    return " ".join(part for part in (detail, verdict.detail) if part) or None
