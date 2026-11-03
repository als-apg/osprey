"""The audit bodies a test's own process hands the agent record, in write order.

A safety decision is an ``audit`` entry in the agent record
(:mod:`osprey.audit.writer`). Reading the entries back through the record
orders them by their millisecond ``ts``, and two decisions one panel write
files back to back can share that millisecond. The tests that assert the
*sequence* of a write's decisions (attempt, then outcome) therefore read the
bodies as the writer hands them over, after the write was stored, in the
order it stored them. The write itself still goes to the test's spool zone.
"""

from __future__ import annotations

from typing import Any

import pytest


def capture_audit_bodies(monkeypatch: pytest.MonkeyPatch, surface: str) -> list[dict[str, Any]]:
    """Return a list that fills with every stored audit body on *surface*, in order."""
    from osprey.audit import writer

    bodies: list[dict[str, Any]] = []
    real_write = writer._write

    def write(body, **ids):
        result = real_write(body, **ids)
        if result is not None and body.get("surface") == surface:
            bodies.append(dict(body))
        return result

    monkeypatch.setattr(writer, "_write", write)
    return bodies
