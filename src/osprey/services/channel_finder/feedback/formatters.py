"""Compact formatting helpers for channel finder feedback records."""

from __future__ import annotations

from typing import Any


def format_selections(selections: dict[str, Any]) -> str:
    """Format a selections dict as a compact ``key=value`` chain."""
    return ", ".join(f"{k}={v}" for k, v in selections.items())


def format_success(record: dict[str, Any]) -> str:
    """Format a success record as a compact one-liner."""
    sel = format_selections(record.get("selections", {}))
    count = record.get("channel_count", 0)
    return f"- GOOD: {sel} \u2192 {count} channels"


def format_failure(record: dict[str, Any]) -> str:
    """Format a failure record as a compact one-liner."""
    sel = record.get("partial_selections", {})
    path_str = format_selections(sel) if sel else "(none)"
    reason = record.get("reason", "rejected by operator")
    return f"- BAD: {path_str} \u2014 {reason}"
