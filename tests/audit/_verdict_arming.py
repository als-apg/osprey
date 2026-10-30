"""Put a test process in a state where the write verdict allows its writes.

An audit entry's ``posture`` is the write verdict for the recorded control target
(:func:`osprey.audit.posture.posture`), so a test that means "the writes
posture" has to arrange a verdict that allows: a clean verdict environment, a
tier that allows writes, and a record whose switch is on for the target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osprey.audit import posture
from osprey_connectors_tests._harness import (
    FIXTURE_IDENTITY,
    isolate_verdict_environment,
    pin_identity,
    reset_verdict_state,
    write_control_context,
)

#: A tier that allows writes, on a deployment whose live machine is real.
TIER_ALLOWED_SECTION = {"type": "epics", "writes": "allowed"}


def arm_writes(
    monkeypatch: Any, root: Path, target: str = "live", *, identity: str = FIXTURE_IDENTITY
) -> None:
    """Make :func:`osprey.audit.posture.verdict` allow writes to *target*.

    Clears the verdict environment and points the agent-data root at *root*
    (:func:`isolate_verdict_environment`), patches the audit package's section
    reader to a tier that allows writes, and writes a record pointed at
    *target* with its switch on, under *identity*'s directory — the login the
    test's records are filed as, pinned here so writer and reader agree.
    """
    isolate_verdict_environment(monkeypatch, root)
    pin_identity(monkeypatch, identity)
    monkeypatch.delenv(posture.CONTROL_TARGET_ENV_VAR, raising=False)
    monkeypatch.setattr(posture, "control_system_section", lambda: dict(TIER_ALLOWED_SECTION))
    write_control_context(root, target=target, switch={target: "on"})
    reset_verdict_state()
