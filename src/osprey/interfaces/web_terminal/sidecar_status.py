"""What the web terminal recorded about each panel sidecar's start.

One JSON document per sidecar panel, at
``<shared agent-data root>/panel_status/<panel_id>.json``. The web terminal
writes it whenever a sidecar starts, comes up, fails or dies; the terminal's own
routes read the in-memory copy, and ``osprey health`` — a separate process with
no address or credential for the sidecar — reads the document. A record says
what the terminal reported, never that the panel answers.

The document holds no credential. The shared root is readable by agent tooling,
so the launch token is scrubbed from the reason before it is written.

This module is the one place the operator-facing wording lives
(:func:`status_message`): the terminal's panel routes, the rail and the health
row all show the sentence it builds.

Standard library only, plus the panel labels and the sibling atomic JSON store,
so the health category can import it without building the web application.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast, get_args

from osprey.interfaces.web_terminal._json_store import read_json_object, write_json_atomic
from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS

logger = logging.getLogger(__name__)

__all__ = [
    "PANEL_STATUS_DIRNAME",
    "SidecarState",
    "SidecarStatus",
    "clear_status",
    "failure_reason",
    "read_status",
    "status_message",
    "status_path",
    "write_status",
]

#: The store's directory name under the shared agent-data root.
PANEL_STATUS_DIRNAME = "panel_status"

SidecarState = Literal["starting", "running", "failed"]

#: The longest reason a record carries; the terminal log keeps the full tail.
_REASON_LIMIT = 240


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SidecarStatus:
    """One sidecar's start outcome as the terminal recorded it.

    Attributes:
        state: ``starting``, ``running`` or ``failed``.
        reason: One line saying why the start failed; set only when ``failed``.
        recorded_at: UTC time the record was written, ISO 8601 with offset.
    """

    state: SidecarState
    reason: str | None
    recorded_at: str

    @classmethod
    def starting(cls) -> SidecarStatus:
        return cls("starting", None, _now())

    @classmethod
    def running(cls) -> SidecarStatus:
        return cls("running", None, _now())

    @classmethod
    def failed(cls, reason: str) -> SidecarStatus:
        return cls("failed", reason, _now())

    def to_json(self) -> dict[str, Any]:
        return {"state": self.state, "reason": self.reason, "recorded_at": self.recorded_at}

    @classmethod
    def from_json(cls, obj: Any) -> SidecarStatus | None:
        """Parse a record, or answer ``None`` for anything malformed."""
        if not isinstance(obj, dict):
            return None
        state = obj.get("state")
        reason = obj.get("reason")
        recorded_at = obj.get("recorded_at")
        if state not in get_args(SidecarState) or not isinstance(recorded_at, str):
            return None
        if reason is not None and not isinstance(reason, str):
            return None
        if state == "failed" and not reason:
            return None
        return cls(cast(SidecarState, state), reason, recorded_at)


def status_path(shared_root: Path, panel_id: str) -> Path:
    """The record's path for *panel_id* under *shared_root*."""
    return Path(shared_root) / PANEL_STATUS_DIRNAME / f"{panel_id}.json"


def write_status(shared_root: Path, panel_id: str, status: SidecarStatus) -> None:
    """Write *status* for *panel_id*, atomically; never raise.

    A record that cannot be written must never block or fail a panel start, so
    an ``OSError`` is logged and dropped.
    """
    path = status_path(shared_root, panel_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, status.to_json())
    except OSError:
        logger.warning("Could not record the %s sidecar status at %s", panel_id, path)


def clear_status(shared_root: Path, panel_id: str) -> None:
    """Remove the record for *panel_id*; a missing record is not an error."""
    try:
        status_path(shared_root, panel_id).unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not clear the %s sidecar status", panel_id, exc_info=True)


def read_status(shared_root: Path, panel_id: str) -> SidecarStatus | None:
    """Read the record for *panel_id*, or ``None`` when absent or damaged."""
    return SidecarStatus.from_json(read_json_object(status_path(shared_root, panel_id)))


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def failure_reason(message: str, stderr_tail: str, secret: str | None) -> str:
    """Condense a start failure into one line a record can carry.

    The first non-empty line of *message* (the terminal's own sentence), plus
    the last non-empty line of *stderr_tail* (the sidecar's last word) when the
    message does not already carry it. Every occurrence of *secret* is replaced
    with ``<token>``, and the result is cut to 240 characters.

    Args:
        message: The failure's message.
        stderr_tail: The sidecar's last stderr lines, possibly empty.
        secret: The launch token, when there is one.
    """
    first = next((line.strip() for line in message.splitlines() if line.strip()), "")
    last = _last_line(stderr_tail)
    reason = f"{first}: {last}" if last and last not in first else first
    if secret:
        reason = reason.replace(secret, "<token>")
    if len(reason) > _REASON_LIMIT:
        reason = reason[: _REASON_LIMIT - 1] + "…"
    return reason


def status_message(panel_id: str, status: SidecarStatus | None) -> str | None:
    """The operator-facing sentence for *status*, or ``None`` when there is none.

    ``failed`` reads ``<LABEL> failed to start: <reason>``; ``starting`` reads
    ``<LABEL> is starting``. A running sidecar, or one with no record, has no
    message.
    """
    if status is None:
        return None
    label = BUILTIN_PANEL_LABELS.get(panel_id, panel_id.upper())
    if status.state == "failed":
        return f"{label} failed to start: {status.reason}"
    if status.state == "starting":
        return f"{label} is starting"
    return None
