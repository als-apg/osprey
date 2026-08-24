"""Single-writer state file naming the control-system target a server is on.

The controls MCP server is the ONLY writer. Everything else — the Claude Code
hooks that render the ``Target:`` line, the roster, the switch tool — reads.
That asymmetry is the whole design: a reader never has to reconcile two
opinions, and a stale file is always the residue of a server that died rather
than a second writer's disagreement.

Path contract
-------------
The file lives at::

    <repo_root>/<agent_data_base_dir>/control_target/target_state_<server_pid>.json

resolved here through :func:`osprey_connectors.workspace.resolve_shared_data_root`
— that is ``project_root`` (the deployment repo holding ``profile.yml``) joined
with ``agent_data.base_dir``, defaulting to ``var/agent_data``. The *shared*
root, deliberately, not :func:`~osprey_connectors.workspace.resolve_agent_data_root`:
the session-scoped root appends ``sessions/<OSPREY_SESSION_ID>``, which a reader
outside the server's environment cannot reproduce.

The hook side re-states this rule in stdlib-only Python (hooks run outside the
osprey venv and cannot import anything from here), so every element of it is
fixed and greppable:

* anchor: the repo root — ``build/`` is disposable and ``data/`` is checksummed,
  so hooks resolve the repo root the way ``osprey_hook_log.get_repo_root`` does
  and never the project dir;
* base dir: the literal ``var/agent_data``. A project that overrides
  ``agent_data.base_dir`` moves this directory somewhere a stdlib-only hook does
  not look; the hook then finds no state and falls back to the deployment
  baseline, which is the documented fail-closed outcome rather than a wrong one;
* fixed subdirectory: :data:`STATE_DIR_NAME` (``control_target``);
* one file per server process, named :data:`STATE_FILE_PREFIX` + PID +
  :data:`STATE_FILE_SUFFIX`, discovered by the glob :data:`STATE_FILE_GLOB`.

Record shape
------------
::

    {
      "target": "live" | "va",
      "generation": 0,
      "server_pid": 4321,          # os.getpid() of the controls server
      "owner_ppid": 4200,          # os.getppid() at write-on-start: the Claude
                                   # Code process that spawned the server
      "targets": {
        "live": {"label": str, "endpoint": str, "real_machine": bool,
                 "probe_channel": str},   # optional; omitted when unconfigured
        "va":   {"label": str, "endpoint": str, "real_machine": bool,
                 "probe_channel": str}
      },
      "children": [5001, 5002]     # connector-host child PIDs, may be empty
    }

The per-target display metadata is rendered ONCE, here, by the single writer,
from a prepared mapping the caller passes in. Readers render the prompt line
straight from this file and never re-derive it from config: a hook that parsed
YAML to answer "which target am I on" would be a second opinion about identity.
``probe_channel`` is part of that metadata for the same reason — the approval
describer names the channel a switch would probe, and it names it from here. It
is optional and never fabricated: a target with no configured probe channel
carries no key, so a reader can tell "not configured" from "configured as".

``children`` records the connector-host children a server owns so that
:func:`sweep_stale` can hand a starting server the orphan PIDs left behind by a
dead predecessor.

Durability
----------
Every write is a temp file created in the same directory followed by
``os.replace`` (atomic only within one filesystem), unlinking the temp file if
the dump fails — the :class:`~osprey.bridges.core.store.JsonFileStore` pattern.
Every read tolerates a missing, unreadable, or corrupt file by returning
``None``: readers of this file are fail-closed by contract, and a half-written
record must degrade to "state unavailable", never to an exception in a hook.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from osprey_connectors.workspace import resolve_shared_data_root

logger = logging.getLogger("osprey.mcp_server.control_system.target_state")

#: The two target names. ``live`` is the real machine, ``va`` the virtual
#: accelerator; both are always present in the record's ``targets`` mapping so a
#: reader can describe the target it is *not* on without a second lookup.
TARGET_LIVE = "live"
TARGET_VA = "va"
TARGET_NAMES: tuple[str, str] = (TARGET_LIVE, TARGET_VA)

#: Fixed subdirectory of the agent-data root. Part of the path contract above —
#: the hook restates this literal.
STATE_DIR_NAME = "control_target"

#: One file per server process. The PID in the name is what makes a stale file
#: sweepable without opening it.
STATE_FILE_PREFIX = "target_state_"
STATE_FILE_SUFFIX = ".json"
STATE_FILE_GLOB = f"{STATE_FILE_PREFIX}*{STATE_FILE_SUFFIX}"

__all__ = [
    "STATE_DIR_NAME",
    "STATE_FILE_GLOB",
    "STATE_FILE_PREFIX",
    "STATE_FILE_SUFFIX",
    "TARGET_LIVE",
    "TARGET_NAMES",
    "TARGET_VA",
    "delete_on_shutdown",
    "is_process_alive",
    "publish_switch",
    "read",
    "read_file",
    "record_child_pids",
    "state_dir",
    "state_file_path",
    "sweep_stale",
    "write_on_start",
]


# -- paths -----------------------------------------------------------------


def state_dir() -> Path:
    """Directory holding every server's state file. Not created by reading."""
    return resolve_shared_data_root() / STATE_DIR_NAME


def state_file_path(server_pid: int | None = None) -> Path:
    """Path of the state file owned by *server_pid* (default: this process)."""
    pid = os.getpid() if server_pid is None else int(server_pid)
    return state_dir() / f"{STATE_FILE_PREFIX}{pid}{STATE_FILE_SUFFIX}"


def _pid_from_path(path: Path) -> int | None:
    """PID encoded in a state file's name, or ``None`` if it is not a number."""
    stem = path.name[len(STATE_FILE_PREFIX) : -len(STATE_FILE_SUFFIX)]
    try:
        return int(stem)
    except ValueError:
        return None


# -- liveness --------------------------------------------------------------


def is_process_alive(pid: int) -> bool:
    """Whether *pid* names a running process.

    ``os.kill(pid, 0)`` sends no signal and only asks the kernel whether the
    process exists — the ``osprey web`` PID-file check does the same. A
    ``PermissionError`` means the process exists but belongs to another user, so
    it counts as ALIVE: sweeping a file whose owner is merely unreachable would
    delete live state. Non-positive PIDs are rejected without calling
    ``os.kill`` at all, because 0 and negatives address process *groups*.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:  # pragma: no cover - platform oddity; assume alive
        return True
    return True


# -- record normalization --------------------------------------------------


def _normalize_target_meta(value: Any) -> dict[str, Any]:
    """Coerce one target's display metadata to the shape readers expect.

    ``label`` / ``endpoint`` / ``real_machine`` are always present — a reader
    branching on a missing key is a reader that can crash a hook. ``probe_channel``
    is OPTIONAL and passes through only when the caller supplied a real one: the
    approval describer renders it exclusively from this file, so an invented
    placeholder would show an operator a channel nobody probes.
    """
    meta = value if isinstance(value, dict) else {}
    normalized: dict[str, Any] = {
        "label": str(meta.get("label") or ""),
        "endpoint": str(meta.get("endpoint") or ""),
        "real_machine": bool(meta.get("real_machine", False)),
    }
    probe_channel = meta.get("probe_channel")
    if isinstance(probe_channel, str) and probe_channel:
        normalized["probe_channel"] = probe_channel
    return normalized


def _normalize_targets(targets_meta: Any) -> dict[str, dict[str, Any]]:
    """Coerce the caller's prepared metadata into both target slots.

    Both slots always exist, so a reader rendering the prompt line never has to
    branch on a missing key — an unsupplied target renders as empty strings,
    which reads as "unknown" rather than crashing the hook that displays it.
    """
    meta = targets_meta if isinstance(targets_meta, dict) else {}
    return {name: _normalize_target_meta(meta.get(name)) for name in TARGET_NAMES}


def _normalize_children(children: Any) -> list[int]:
    """Coerce a child-PID list to positive ints, dropping anything else."""
    if not isinstance(children, (list, tuple)):
        return []
    pids: list[int] = []
    for item in children:
        try:
            pid = int(item)
        except (TypeError, ValueError):
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return pids


# -- writing ---------------------------------------------------------------


def _atomic_write_json(path: Path, record: dict[str, Any]) -> None:
    """Write *record* to *path* atomically: temp file in the same dir, rename.

    The temp file must share the target's directory — ``os.replace`` is atomic
    only within one filesystem — and is unlinked if the dump fails so a failed
    write leaves neither a partial state file nor litter beside it.
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_on_start(
    baseline_target: str,
    targets_meta: dict[str, Any] | None = None,
    *,
    server_pid: int | None = None,
    owner_ppid: int | None = None,
    generation: int = 0,
    children: list[int] | None = None,
) -> list[int]:
    """Reset this server's state to the deployment baseline and sweep stale files.

    Called once, at server start, before anything can switch. It is a RESET and
    not a merge: no target selection survives the process that made it, so a
    fresh server always starts on the baseline the deployment config declares.

    ``owner_ppid`` defaults to ``os.getppid()`` *at this moment*, which is the
    Claude Code process that spawned the server. It is what lets a hook pick the
    right state file when two sessions share one checkout: the hook selects the
    file whose ``owner_ppid`` is on its own ancestor chain.

    Args:
        baseline_target: ``live`` or ``va`` — the deployment baseline.
        targets_meta: Prepared per-target display metadata, ``{"live": {...},
            "va": {...}}`` with ``label`` / ``endpoint`` / ``real_machine``.
            Rendered by the caller from config and written verbatim here.
        server_pid: Owning PID; defaults to this process.
        owner_ppid: Overrides the captured parent PID (tests, re-parenting).
        generation: Starting generation, ``0`` unless a caller says otherwise.
        children: Connector-host child PIDs already known at start.

    Returns:
        Orphan child PIDs recorded by dead predecessors, for the caller to kill.
    """
    pid = os.getpid() if server_pid is None else int(server_pid)
    orphans = sweep_stale(server_pid=pid)
    record = {
        "target": str(baseline_target),
        "generation": int(generation),
        "server_pid": pid,
        "owner_ppid": os.getppid() if owner_ppid is None else int(owner_ppid),
        "targets": _normalize_targets(targets_meta),
        "children": _normalize_children(children),
    }
    _atomic_write_json(state_file_path(pid), record)
    logger.debug("Target state initialized: %s (pid %s)", record["target"], pid)
    return orphans


def _update(server_pid: int | None, changes: dict[str, Any]) -> bool:
    """Merge *changes* into this server's record. ``False`` if there is none."""
    path = state_file_path(server_pid)
    record = read_file(path)
    if record is None:
        logger.error("No target-state record at %s to update; write_on_start first", path)
        return False
    record.update(changes)
    _atomic_write_json(path, record)
    return True


def publish_switch(
    target: str,
    generation: int,
    *,
    children: list[int] | None = None,
    server_pid: int | None = None,
) -> bool:
    """Publish a completed switch to *target* at *generation*.

    Writes what it is told. Whether a generation is bumped — a same-target
    respawn does not bump — is the switch lifecycle's rule, not this module's:
    this file records the outcome so readers agree on it, and does not arbitrate
    it. Display metadata written at start is preserved.

    Returns:
        ``True`` when the record was updated, ``False`` when no record exists
        (nothing was written, and the caller has a start-ordering bug).
    """
    changes: dict[str, Any] = {"target": str(target), "generation": int(generation)}
    if children is not None:
        changes["children"] = _normalize_children(children)
    return _update(server_pid, changes)


def record_child_pids(children: list[int] | None, *, server_pid: int | None = None) -> bool:
    """Record the connector-host child PIDs this server owns.

    Pass an empty list (or ``None``) to clear them — after the children have
    been reaped, so a later sweep does not report already-dead PIDs as orphans.
    """
    return _update(server_pid, {"children": _normalize_children(children)})


def delete_on_shutdown(*, server_pid: int | None = None) -> None:
    """Remove this server's state file. A missing file is success, not an error."""
    path = state_file_path(server_pid)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - unwritable state dir
        logger.warning("Could not remove target state file %s: %s", path, exc)


# -- reading ---------------------------------------------------------------


def read_file(path: Path) -> dict[str, Any] | None:
    """Load one state file, or ``None`` if it is absent, unreadable, or corrupt.

    Never raises. Readers of this file (hooks, roster) treat "no answer" as the
    deployment baseline, so every failure mode has to arrive as the same value.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        # ValueError covers JSONDecodeError and UnicodeDecodeError alike.
        return None
    return loaded if isinstance(loaded, dict) else None


def read(server_pid: int | None = None) -> dict[str, Any] | None:
    """Load this server's state record, or ``None``. Never raises."""
    return read_file(state_file_path(server_pid))


# -- sweeping --------------------------------------------------------------


def sweep_stale(*, server_pid: int | None = None) -> list[int]:
    """Delete every state file whose owning server is dead; return its orphans.

    A server exits without running :func:`delete_on_shutdown` whenever it is
    killed rather than asked to stop, and its connector-host children can
    outlive it. The next server to start therefore inherits two jobs: clear the
    dead file so readers stop seeing a target nobody is on, and collect the
    child PIDs it recorded so the caller can kill them.

    The file this process owns is left alone even though the process is
    obviously alive — checking one's own liveness is a way to get it wrong.

    Args:
        server_pid: PID whose file to preserve; defaults to this process.

    Returns:
        Orphan child PIDs, in discovery order, without duplicates.
    """
    own = state_file_path(server_pid)
    directory = state_dir()
    try:
        entries = sorted(directory.glob(STATE_FILE_GLOB))
    except OSError:
        return []

    orphans: list[int] = []
    for entry in entries:
        if entry.name == own.name:
            continue
        pid = _pid_from_path(entry)
        if pid is not None and is_process_alive(pid):
            continue
        record = read_file(entry)
        if record is not None:
            for child in _normalize_children(record.get("children")):
                if child not in orphans:
                    orphans.append(child)
        try:
            entry.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - unwritable state dir
            logger.warning("Could not remove stale target state file %s: %s", entry, exc)
            continue
        logger.info("Swept stale target state file %s (owner pid %s gone)", entry.name, pid)
    return orphans
