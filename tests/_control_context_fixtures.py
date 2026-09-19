"""Writers for the control-context record and the per-server reports, for tests.

Every suite downstream of the control context needs the same two files on disk:
``control_target/<identity>/control_context.json`` and one or more
``control_target/<identity>/server_<pid>.json``. Spelling those payloads in each
suite is how a schema change becomes fifteen separate green-to-red
investigations — so they are spelled once, here, from the same dataclasses the
production readers parse. A field that moves fails every suite at once, which is
the point.

The writers take an explicit agent-data *root* rather than reading the
environment: a test that has not stamped ``OSPREY_AGENT_DATA_ROOT`` yet still
has to be able to lay down the deployment it is about to point at.

The ``<identity>`` hop is nobody's literal here either. Both writers reach it
through :func:`~osprey_connectors.control_context.record_path_under`, the one
place the hops are spelled, and a suite that needs the directory itself asks
:func:`state_dir_under` for it rather than joining the name again — a fixture
that created one directory while the reader under test resolved another is a
narrowing that silently never applies, and the file being absent is the only
symptom.

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
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV

#: The target a record carries when a test does not care which one it is.
DEFAULT_TARGET = "live"

#: The generation that goes with it. Deliberately not 0: a test that asserts a
#: generation moved wants the before-value to be distinguishable from "unset".
DEFAULT_GENERATION = 1

#: The identity a suite pins when it has to spell the state directory it means.
#: Deliberately not a plausible account name: a path that still carries the
#: developer's login is then a rung nobody pinned, not a coincidence that also
#: passes on the one machine it was written on.
FIXTURE_IDENTITY = "test-operator"


# -- paths ------------------------------------------------------------------


def state_dir_under(root: Path) -> Path:
    """The directory the record and the per-server reports land in, under *root*.

    For suites that need the directory rather than a file in it — one to create
    before a reader is aimed at it, one to list, one to hand a path seam. Taken
    from :func:`~osprey_connectors.control_context.record_path_under` rather than
    joined here, so the identity hop has exactly one spelling in the tests as
    well as in the code: a suite that built this path itself would keep passing
    after the layout moved, against a directory no reader resolves.

    Args:
        root: The agent-data root. Nothing is created.
    """
    return control_context.record_path_under(Path(root)).parent


def pin_identity(monkeypatch: Any, identity: str = FIXTURE_IDENTITY) -> str:
    """Pin the acting identity for one test, and return the name it now answers.

    For the assertions that spell the directory literally. Without a pin the
    segment is :func:`getpass.getuser`'s answer — the developer's login locally,
    an agent account in CI, :data:`~osprey_connectors.identity.UNKNOWN_IDENTITY`
    in a slim image with no passwd entry — so a literal path is three different
    strings on three machines.

    The lower rung is cleared as well as the upper one set: ``OSPREY_TERMINAL_USER``
    wins over ``OSPREY_AUDIT_IDENTITY`` in the ladder, and a suite run from
    inside a multi-user web terminal carries one.

    NOT for a test whose subject runs as a subprocess. ``tests/hooks/conftest.py``
    builds a curated environment from an allowlist that carries neither variable,
    so a pinned parent and its hook child would resolve two different
    directories, and the child would simply find no record. Those tests write
    through the shared writers and assert no literal.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture; the pin is reverted with it.
        identity: The name to pin. Defaults to :data:`FIXTURE_IDENTITY`.

    Returns:
        The identity now in force, for the caller to build its expected path from.
    """
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, identity)
    return identity


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
        root: The agent-data root. The acting identity's directory below it —
            :func:`state_dir_under` — is created if it does not exist.
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
