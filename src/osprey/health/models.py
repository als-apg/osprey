"""Data models for the health-check framework.

Defines the result types produced by every health probe and the aggregated
report emitted by the runner. ``CheckReport.to_dict()`` is the locked wire
contract rendered by the web dashboard (P2) and the agent MCP tool (P3): its
key set and shape must not change without a coordinated contract revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Status(StrEnum):
    """Outcome of a single health check."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIP = "skip"


STATUS_ICONS = {
    "ok": "\033[32m✓\033[0m",  # green checkmark
    "warning": "\033[33m⚠\033[0m",  # yellow warning
    "error": "\033[31m✗\033[0m",  # red X
    "skip": "\033[90m-\033[0m",  # dim dash
}


@dataclass
class CheckResult:
    """Result of a single health check.

    Attributes:
        name: Machine-readable identifier, e.g. ``"epics.beam_current"``.
        category: Category the check belongs to, e.g. ``"file_system"``.
        status: Outcome of the check.
        message: Human-readable one-line summary.
        value: Measured value, e.g. ``"401.2 mA"`` (optional).
        latency_ms: Elapsed time for the check in milliseconds (optional).
        details: Extended diagnostic or error information (optional).
    """

    name: str
    category: str
    status: Status
    message: str
    value: str = ""
    latency_ms: float = 0.0
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the wire shape, omitting empty optional fields.

        ``value`` and ``details`` are included only when truthy; ``latency_ms``
        only when greater than zero (rounded to one decimal place).
        """
        d: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "status": self.status.value,
            "message": self.message,
        }
        if self.value:
            d["value"] = self.value
        if self.latency_ms > 0:
            d["latency_ms"] = round(self.latency_ms, 1)
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class CheckReport:
    """Aggregated results from a health-check run.

    Attributes:
        results: The individual check results.
        elapsed_ms: Total wall-clock time for the run in milliseconds.
        deadline_hit: Whether a suite-level deadline was reached during the run.
    """

    results: list[CheckResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    deadline_hit: bool = False

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.status == Status.OK)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.status == Status.WARNING)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status == Status.ERROR)

    @property
    def skip_count(self) -> int:
        return sum(1 for r in self.results if r.status == Status.SKIP)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def exit_code(self) -> int:
        """Return the process exit code: 2 if any errors, 1 if any warnings, else 0."""
        if self.error_count > 0:
            return 2
        if self.warning_count > 0:
            return 1
        return 0

    def summary_line(self) -> str:
        """Return a one-line summary, e.g. ``"10/15 checks passed (5 skipped)"``."""
        parts = []
        if self.warning_count:
            parts.append(f"{self.warning_count} warning{'s' if self.warning_count != 1 else ''}")
        if self.error_count:
            parts.append(f"{self.error_count} error{'s' if self.error_count != 1 else ''}")
        if self.skip_count:
            parts.append(f"{self.skip_count} skipped")
        extra = f" ({', '.join(parts)})" if parts else ""
        return f"{self.ok_count}/{self.total} checks passed{extra}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the locked wire shape rendered by P2/P3."""
        return {
            "summary": self.summary_line(),
            "ok": self.ok_count,
            "warnings": self.warning_count,
            "errors": self.error_count,
            "skips": self.skip_count,
            "total": self.total,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "deadline_hit": self.deadline_hit,
            "results": [r.to_dict() for r in self.results],
        }
