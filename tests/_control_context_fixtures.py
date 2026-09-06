"""Writers for the control-context record and the per-server reports, for tests.

Every suite downstream of the control context needs the same two files on disk:
``control_target/control_context.json`` and one or more
``control_target/server_<pid>.json``. Spelling those payloads in each suite is
how a schema change becomes fifteen separate green-to-red investigations — so
they are spelled once, here, from the same dataclasses the production readers
parse. A field that moves fails every suite at once, which is the point.

The writers take an explicit agent-data *root* rather than reading the
environment: a test that has not stamped ``OSPREY_AGENT_DATA_ROOT`` yet still
has to be able to lay down the deployment it is about to point at.

Both writers drop the reader cache after they write. The cache is keyed on
``(mtime_ns, size, ino)``, and a test that rewrites a record twice inside one
clock tick is the exact case that keying cannot see; leaving a parsed record
behind would also reach the next test in the process.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from osprey_connectors import control_context

#: The target a record carries when a test does not care which one it is.
DEFAULT_TARGET = "live"

#: The generation that goes with it. Deliberately not 0: a test that asserts a
#: generation moved wants the before-value to be distinguishable from "unset".
DEFAULT_GENERATION = 1


class _Self:
    """Sentinel: own this record with a live web terminal at this PID."""


#: Default owner: the running test process, which is alive by construction.
#: Ownership is what makes a record actionable — a reader that finds a dead
#: owner is entitled to claim over it — so the value a test gets without asking
#: is the one that does not silently put every suite on the claim path.
SELF_OWNER: Any = _Self


def owner(
    kind: str = control_context.OWNER_WEB_TERMINAL,
    pid: int | None = None,
    port: int | None = None,
) -> control_context.Owner:
    """An owner record, defaulting to a live web terminal at this PID.

    Args:
        kind: One of :data:`control_context.OWNER_KINDS`.
        pid: The owning process. Defaults to this one, which is alive.
        port: The web terminal's port, or ``None``.
    """
    return control_context.Owner(kind=kind, pid=os.getpid() if pid is None else pid, port=port)


def write_control_context(
    root: Path,
    target: str = DEFAULT_TARGET,
    generation: int = DEFAULT_GENERATION,
    posture: Any = None,
    *,
    owned_by: Any = SELF_OWNER,
    last_switch: Mapping[str, Any] | None = None,
) -> Path:
    """Write a control-context record under *root* and return its path.

    Args:
        root: The agent-data root. The ``control_target/`` directory below it
            is created if it does not exist.
        target: The control target the deployment is pointed at.
        generation: How many times that has moved.
        posture: Per-target narrowings, ``{target: "sandbox"}``. Empty when
            omitted, which is "the deployment ceiling is in charge". Anything
            that is not a mapping — a bare string, a list, ``7`` — is written
            to the field VERBATIM, so a suite pinning what the two posture
            parsers drop can state the shape it means rather than hand-rolling
            a payload beside this writer.
        owned_by: :data:`SELF_OWNER` (a live web terminal at this PID),
            ``None`` for an ownerless record free to be claimed, or an
            :class:`control_context.Owner` built with :func:`owner`.
        last_switch: The terminus of the last switch request, verbatim.

    Returns:
        The path written.
    """
    verbatim = posture is not None and not isinstance(posture, Mapping)
    record = control_context.ControlContext(
        target=target,
        generation=generation,
        owner=owner() if owned_by is SELF_OWNER else owned_by,
        posture={} if verbatim else dict(posture or {}),
        last_switch=None if last_switch is None else dict(last_switch),
    )
    path = control_context.record_path_under(root)
    if verbatim:
        payload = record.to_payload()
        payload["posture"] = posture
        return write_payload(path, payload)
    control_context.write_record(record, path=path)
    control_context.invalidate_cache()
    return path


def write_server_report(
    root: Path,
    pid: int,
    *,
    session: str | None = None,
    applied_target: str | None = None,
    applied_generation: int | None = None,
    children: Sequence[int] = (),
    reachability: Mapping[str, Any] | None = None,
    last_switch: Mapping[str, Any] | None = None,
    last_posture_realign: Mapping[str, Any] | None = None,
    targets: Mapping[str, Any] | None = None,
    updated_at: str | None = None,
) -> Path:
    """Write one controls server's report under *root* and return its path.

    Every field but *pid* defaults to its "has not answered yet" value, so a
    report written with no keyword arguments is a server that has just started:
    null-bound, unprobed, mid-nothing. ``applied_target``/``applied_generation``
    left at ``None`` mean exactly that and never "baseline".

    Args:
        root: The agent-data root.
        pid: The reporting server's PID, which names the file.
        session: Its ``OSPREY_POSTURE_SESSION``, ``None`` for a bare ``claude``.
        applied_target: The target its connector host is actually on.
        applied_generation: The generation that target was reached at.
        children: The connector-host PIDs it spawned.
        reachability: Its last probe per target.
        last_switch: Its progress through the current switch.
        last_posture_realign: Its last posture realignment.
        targets: Per-target display metadata.
        updated_at: When it last wrote, ISO-8601.

    Returns:
        The path written.
    """
    report = control_context.ServerReport(
        server_pid=pid,
        session=session,
        applied_target=applied_target,
        applied_generation=applied_generation,
        children=tuple(children),
        reachability=dict(reachability or {}),
        last_switch=None if last_switch is None else dict(last_switch),
        last_posture_realign=(None if last_posture_realign is None else dict(last_posture_realign)),
        targets=dict(targets or {}),
        updated_at=updated_at,
    )
    path = control_context.report_path_under(root, pid)
    write_payload(path, report.to_payload())
    return path


def write_payload(path: Path, payload: Any) -> Path:
    """Write *payload* as JSON to *path*, atomically, and drop the read cache.

    The escape hatch for the degradation tests: a record or report that is not
    a valid one cannot be built from the dataclasses, so those suites hand the
    raw object — a string, a list, a mapping missing its schema — straight to
    this. Same atomic replace the production writer uses, so the inode moves
    and a same-size rewrite is still seen.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    control_context.invalidate_cache()
    return path
