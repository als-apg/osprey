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

The root is overridden by :data:`~osprey.audit.posture.OSPREY_AGENT_DATA_ROOT`
when the environment carries it. The web terminal's spawn sites stamp the root
they resolved into every session child, paired with the session key, precisely
so that this writer and the readers below do not each derive the directory
their own way; a stamped session therefore has one anchor, and only a process
outside any session (a CLI run, a dispatch worker) falls back to the derivation
above.

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
      "target": "live" | "va" | "standin",
      "generation": 0,
      "server_pid": 4321,          # os.getpid() of the controls server
      "owner_ppid": 4200,          # os.getppid() at write-on-start: the Claude
                                   # Code process that spawned the server
      "targets": {
        "live":    {"label": str, "endpoint": str, "real_machine": bool,
                    "probe_channel": str},  # optional; omitted when unconfigured
        "va":      {"label": str, "endpoint": str, "real_machine": bool,
                    "probe_channel": str},
        "standin": {"label": str, "endpoint": str, "real_machine": bool,
                    "probe_channel": str}
      },
      "children": [5001, 5002],    # connector-host child PIDs, may be empty
      "last_switch": {             # last switch terminus, or null
        "request_id": str, "target": str, "status": str,
        "reason": str | None, "detail": str | None,
        "at": str                          # wall clock, ISO-8601
      },
      "reachability": {            # last prober sweep, or null
        "published_at": str,       # wall clock, ISO-8601
        "targets": {"live": {"<role>": {"state": str, "probed_at": str, ...}}}
      },
      "last_posture_realign": {"state": "pending" | "done", "at": str}
    }

There is one slot per name in :data:`TARGET_NAMES` and every slot is always
written, whether or not the deployment configures that target. An unconfigured
target is absent-as-empty: its key exists and its ``label`` and ``endpoint`` are
empty strings with ``real_machine`` false and no ``probe_channel`` — never a
missing key. That is how ``va`` has always behaved on a deployment with no
virtual accelerator, and ``standin`` behaves identically on a deployment with no
stand-in: readers keep rendering from a fixed set of keys, and empty strings read
as "unknown" rather than crashing the hook that displays them.

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

``last_switch``, ``reachability`` and ``last_posture_realign`` are the three
publication blocks the header chip reads. They are null on a fresh record —
:func:`write_on_start` resets them with everything else, because none of them
describes anything a NEW server has done yet — and each is merged in by its own
publisher (:func:`publish_last_switch`, :func:`publish_reachability`,
:func:`publish_posture_realign`). Every reader takes them with ``.get()`` and
tolerates both absence and null. Reachability carries wall-clock ``probed_at``
stamps rather than an age, because the reader is in another process: an age
computed here would be the age at write time, and the whole point of the block
is to let a reader tell a fresh sweep from a prober that stopped.

The other files in this directory
---------------------------------
Two more file families share :func:`state_dir`, both named for a PID so that
residue is sweepable without being opened:

* ``exec_inflight_<pid>_<run id>.json`` — one marker per python execution,
  written by the executor (a different server process) and read here by
  :func:`in_flight_executions`. The switch gate, the posture route and the
  reconciler all ask the same question through it.
* ``request_<server_pid>.json`` — a switch REQUEST, written by the web server
  and addressed to one controls server, consumed and removed by that server's
  reconciler. This is the one file in this directory the controls server does
  not write, and it is not a second opinion about identity: it says what an
  operator asked for, never what is true. It expires (:data:`REQUEST_TTL_S`).

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

import contextlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey_connectors import session_store
from osprey_connectors.workspace import resolve_shared_data_root

logger = logging.getLogger("osprey.mcp_server.control_system.target_state")

#: The control-target names. ``live`` is the real machine, ``va`` the virtual
#: accelerator, ``standin`` the live stand-in soft IOC; every one of them is
#: always present in the record's ``targets`` mapping so a reader can describe a
#: target it is *not* on without a second lookup. Restated here rather than
#: imported from :mod:`osprey_connectors.types` for the same reason the path
#: contract is restated in the hooks: this module is the readers' vocabulary.
TARGET_LIVE = "live"
TARGET_VA = "va"
TARGET_STANDIN = "standin"
TARGET_NAMES: tuple[str, ...] = (TARGET_LIVE, TARGET_VA, TARGET_STANDIN)

#: Fixed subdirectory of the agent-data root. Part of the path contract above —
#: the hook restates this literal.
STATE_DIR_NAME = "control_target"

#: One file per server process. The PID in the name is what makes a stale file
#: sweepable without opening it.
STATE_FILE_PREFIX = "target_state_"
STATE_FILE_SUFFIX = ".json"
STATE_FILE_GLOB = f"{STATE_FILE_PREFIX}*{STATE_FILE_SUFFIX}"

#: One in-flight marker per execution, written by the python executor before a
#: sandbox subprocess starts and removed in a ``finally``. It lives in
#: :func:`state_dir` because that is the directory the executor and the controls
#: server already share, and it is named for the process that will remove it so
#: a marker left by a killed executor can be ignored rather than wedging every
#: later switch. The executor restates these constants in stdlib terms (see
#: :mod:`osprey.mcp_server.python_executor.executor`) and a drift guard pins the
#: two spellings equal; they live HERE rather than in the switch tool because
#: the reconciler, the posture route and the tool are all readers of them.
INFLIGHT_FILE_PREFIX = "exec_inflight_"
INFLIGHT_FILE_SUFFIX = ".json"
INFLIGHT_FILE_GLOB = f"{INFLIGHT_FILE_PREFIX}*{INFLIGHT_FILE_SUFFIX}"

#: One switch request per server process, written by the WEB server and consumed
#: by the controls server's reconciler. It is the single exception to "the
#: controls server is the only writer in this directory", and it is safe because
#: it is a different file: a request is desired state addressed to one server,
#: never a second opinion about what the target IS. The server PID in the name
#: is both the address and what makes an unclaimed request sweepable.
REQUEST_FILE_PREFIX = "request_"
REQUEST_FILE_SUFFIX = ".json"
REQUEST_FILE_GLOB = f"{REQUEST_FILE_PREFIX}*{REQUEST_FILE_SUFFIX}"

#: How long a switch request stays actionable. A request the reconciler reaches
#: later than this is refused as ``request_expired`` rather than acted on: the
#: operator who clicked Switch is no longer watching, and a switch that lands
#: minutes after the gesture is a surprise rather than a service.
REQUEST_TTL_S = 30

__all__ = [
    "INFLIGHT_FILE_GLOB",
    "INFLIGHT_FILE_PREFIX",
    "INFLIGHT_FILE_SUFFIX",
    "REQUEST_FILE_GLOB",
    "REQUEST_FILE_PREFIX",
    "REQUEST_FILE_SUFFIX",
    "REQUEST_TTL_S",
    "STATE_DIR_NAME",
    "RequestSuperseded",
    "STATE_FILE_GLOB",
    "STATE_FILE_PREFIX",
    "STATE_FILE_SUFFIX",
    "TARGET_LIVE",
    "TARGET_NAMES",
    "TARGET_STANDIN",
    "TARGET_VA",
    "delete_on_shutdown",
    "in_flight_executions",
    "is_process_alive",
    "is_request_fresh",
    "publish_last_switch",
    "publish_posture_realign",
    "publish_reachability",
    "publish_switch",
    "read",
    "read_file",
    "read_request",
    "record_child_pids",
    "remove_request",
    "request_file_path",
    "state_dir",
    "state_file_path",
    "sweep_stale",
    "write_on_start",
    "write_request",
]


# -- paths -----------------------------------------------------------------


def state_dir() -> Path:
    """Directory holding every server's state file. Not created by reading.

    :data:`~osprey.audit.posture.OSPREY_AGENT_DATA_ROOT` wins when it is set.
    A session child is stamped with the root its spawning server resolved, and
    that stamp is the whole point: writer and readers derive this directory
    three different ways (config here, config again in the store reader, a
    repo-root guess plus the literal ``var/agent_data`` in the stdlib-only
    hooks), and a deployment that moves ``agent_data.base_dir`` makes them
    disagree. Preferring the stamp here is what makes "no state file under the
    stamped root" mean "no controls server for this session" rather than "the
    reader looked in the wrong place".

    The stamp is read through
    :func:`~osprey_connectors.session_store.stamped_agent_data_root` rather
    than off the environment here, because the posture store sits in this same
    directory and the two must not normalise one variable differently — a
    ``~`` or a padded value would otherwise put the state file and the store
    in different places.

    Unset — a CLI run, a dispatch worker, a server outside any web session —
    is the old derivation unchanged, so a caller that patches
    ``resolve_shared_data_root`` still sees exactly what it patched. The config
    half stays HERE and is deliberately not delegated: this function raises
    where the store reader answers ``None``, and the web terminal's switch
    route turns that raise into its ``store_unavailable`` 503.
    """
    stamped = session_store.stamped_agent_data_root()
    root = stamped if stamped is not None else resolve_shared_data_root()
    return root / STATE_DIR_NAME


def state_file_path(server_pid: int | None = None) -> Path:
    """Path of the state file owned by *server_pid* (default: this process)."""
    pid = os.getpid() if server_pid is None else int(server_pid)
    return state_dir() / f"{STATE_FILE_PREFIX}{pid}{STATE_FILE_SUFFIX}"


def _pid_from_name(name: str, prefix: str, suffix: str) -> int | None:
    """PID encoded in *name*, or ``None`` when it does not encode one.

    Shared by the state files and the request files: both are named for the
    process that owns them precisely so a sweeper can judge them without
    opening them, and a name that does not follow the rule is judged the same
    way in both families — no PID, therefore no live owner.
    """
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    stem = name[len(prefix) : -len(suffix)]
    try:
        return int(stem)
    except ValueError:
        return None


def _pid_from_path(path: Path) -> int | None:
    """PID encoded in a state file's name, or ``None`` if it is not a number."""
    return _pid_from_name(path.name, STATE_FILE_PREFIX, STATE_FILE_SUFFIX)


def request_file_path(server_pid: int | None = None) -> Path:
    """Path of the switch request addressed to *server_pid* (default: this one)."""
    pid = os.getpid() if server_pid is None else int(server_pid)
    return state_dir() / f"{REQUEST_FILE_PREFIX}{pid}{REQUEST_FILE_SUFFIX}"


def _now_iso() -> str:
    """Wall clock, ISO-8601, UTC. The stamp every reader ages a block from."""
    return datetime.now(UTC).isoformat()


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
    """Coerce the caller's prepared metadata into one slot per target name.

    Every name in :data:`TARGET_NAMES` gets a slot, so a reader rendering the
    prompt line never has to branch on a missing key — an unsupplied target
    renders as empty strings, which reads as "unknown" rather than crashing the
    hook that displays it. A deployment without a stand-in therefore still
    carries a ``standin`` slot, empty, exactly as one without a virtual
    accelerator carries an empty ``va``.
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
        baseline_target: One of :data:`TARGET_NAMES` — the deployment baseline.
        targets_meta: Prepared per-target display metadata, ``{"live": {...},
            "va": {...}, "standin": {...}}`` with ``label`` / ``endpoint`` /
            ``real_machine``. Rendered by the caller from config and written
            verbatim here; a target the caller omits is written as an empty slot.
        server_pid: Owning PID; defaults to this process.
        owner_ppid: Overrides the captured parent PID (tests, re-parenting).
        generation: Starting generation, ``0`` unless a caller says otherwise.
        children: Connector-host child PIDs already known at start.

    The reset covers the three publication blocks (``last_switch``,
    ``reachability``, ``last_posture_realign``) and any switch request already
    addressed to this PID, for the same reason it covers the target itself:
    nothing a predecessor published describes this process.

    Returns:
        Orphan child PIDs recorded by dead predecessors, for the caller to kill.
    """
    pid = os.getpid() if server_pid is None else int(server_pid)
    orphans = sweep_stale(server_pid=pid)
    # A request addressed to this PID can only be a dead predecessor's residue —
    # nothing has yet told the web server that THIS server exists — and acting on
    # it would move a fresh session on a gesture nobody made in it.
    remove_request(server_pid=pid)
    record = {
        "target": str(baseline_target),
        "generation": int(generation),
        "server_pid": pid,
        "owner_ppid": os.getppid() if owner_ppid is None else int(owner_ppid),
        "targets": _normalize_targets(targets_meta),
        "children": _normalize_children(children),
        # The three publication blocks start null: none of them describes
        # anything this server has done yet, and a stale switch outcome or an
        # inherited reachability sweep would be read as this server's own.
        "last_switch": None,
        "reachability": None,
        "last_posture_realign": None,
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


# -- publication blocks ----------------------------------------------------
#
# THE MERGE MUST BE ATOMIC WITH RESPECT TO THE EVENT LOOP. Each publisher below
# is deliberately SYNCHRONOUS and each is a single :func:`_update` call, which
# reads the record and writes it back with no suspension point in between.
# Never make one of these ``async``, and never insert an ``await`` between the
# read and the write: two coroutines interleaving there would each write back a
# record built from a copy taken before the other's change, and the loser's
# block would vanish. The publishers run beside one another — the reconciler
# publishes ``last_switch`` and ``last_posture_realign`` while the endpoint
# prober publishes ``reachability`` every sweep — so this is the ordinary case,
# not the exotic one. Callers on the loop pay one small blocking file write; the
# alternative is a lock nobody outside this process could take anyway.


def _normalize_last_switch(outcome: Any) -> dict[str, Any] | None:
    """Coerce a switch terminus to the block readers age and match on.

    Written through largely verbatim: the vocabulary is the switch lifecycle's
    (``request_id``, ``target``, ``status``, ``reason``, ``detail``), and
    restating it here would give this module an opinion about refusals it does
    not arbitrate. What is enforced is the part every reader depends on — a
    wall-clock ``at``, so a stale outcome can be aged out rather than shown
    forever, and a ``request_id`` a chip can match its own request against
    instead of parsing prose.
    """
    if not isinstance(outcome, dict):
        return None
    block = dict(outcome)
    if not block.get("at"):
        block["at"] = _now_iso()
    return block


def _normalize_reachability(rows: Any) -> dict[str, Any] | None:
    """Coerce a prober sweep to ``{published_at, targets: {target: {role: row}}}``.

    A row passes through with its own keys intact — ``probed_at``, ``gateway``,
    ``detail`` and whatever else the prober measured — and is kept only when it
    carries a non-empty ``state`` string. ``not_applicable`` is a state like any
    other and is preserved as one: "this role is not probed on this target" is
    an answer, and collapsing it into "unknown" would tell an operator the
    prober had failed when it had in fact decided.

    ``published_at`` stamps the sweep, not the individual probes; a reader
    computes ``age_s`` from a row's own ``probed_at`` and treats a row without
    one as unaged.
    """
    source = rows if isinstance(rows, dict) else {}
    targets: dict[str, dict[str, dict[str, Any]]] = {}
    for target, roles in source.items():
        if not isinstance(roles, dict):
            continue
        kept: dict[str, dict[str, Any]] = {}
        for role, row in roles.items():
            if not isinstance(row, dict):
                continue
            state = row.get("state")
            if not isinstance(state, str) or not state:
                continue
            kept[str(role)] = {**row, "state": state}
        if kept:
            targets[str(target)] = kept
    if not targets:
        return None
    return {"published_at": _now_iso(), "targets": targets}


def _normalize_posture_realign(state: Any) -> dict[str, Any] | None:
    """Coerce a realignment note to ``{state, at}``, stamping ``at`` if absent."""
    if not isinstance(state, dict):
        return None
    block = dict(state)
    value = block.get("state")
    if isinstance(value, str) and value:
        block["state"] = value
    if not block.get("at"):
        block["at"] = _now_iso()
    return block


def publish_last_switch(outcome: dict[str, Any] | None, *, server_pid: int | None = None) -> bool:
    """Publish how the last switch request ended.

    Every terminus goes through here — success, each refusal, an expiry — so a
    chip that POSTed a request has exactly one place to learn what became of it,
    matched by ``request_id`` rather than by guessing from the target.

    SYNCHRONOUS by contract; see the section comment above.

    Args:
        outcome: ``{request_id, target, status, reason, detail, at}``; ``at``
            is stamped here when the caller leaves it out. ``target`` is the
            target the request named, which is how the per-target popover finds
            the row an outcome belongs to. ``None`` clears the block.
        server_pid: Owning PID; defaults to this process.

    Returns:
        ``True`` when the record was updated, ``False`` when there is none.
    """
    return _update(server_pid, {"last_switch": _normalize_last_switch(outcome)})


def publish_reachability(rows: Any, *, server_pid: int | None = None) -> bool:
    """Publish one endpoint-prober sweep, per target and role.

    Called after EVERY sweep, unconditionally: age is the staleness signal, so a
    prober that keeps measuring the same thing still has to say when it last
    looked. A reader that finds nothing here renders ``unknown`` rather than
    assuming the endpoints are down.

    SYNCHRONOUS by contract; see the section comment above.

    Args:
        rows: ``{target: {role: {"state": str, "probed_at": str, ...}}}``.
            Rows without a usable ``state`` are dropped; an empty sweep clears
            the block.
        server_pid: Owning PID; defaults to this process.

    Returns:
        ``True`` when the record was updated, ``False`` when there is none.
    """
    return _update(server_pid, {"reachability": _normalize_reachability(rows)})


def publish_posture_realign(state: dict[str, Any] | None, *, server_pid: int | None = None) -> bool:
    """Publish whether the active target's posture change has been realigned yet.

    A posture narrowed on the target the session is ON only takes effect once
    the connector is rebuilt, and that rebuild waits for any execution in
    flight. ``pending`` is how the popover says so instead of showing a toggle
    that appears to have done nothing.

    SYNCHRONOUS by contract; see the section comment above.

    Args:
        state: ``{"state": "pending" | "done", "at": ...}``; ``at`` is stamped
            here when the caller leaves it out. ``None`` clears the block.
        server_pid: Owning PID; defaults to this process.

    Returns:
        ``True`` when the record was updated, ``False`` when there is none.
    """
    return _update(server_pid, {"last_posture_realign": _normalize_posture_realign(state)})


def record_child_pids(children: list[int] | None, *, server_pid: int | None = None) -> bool:
    """Record the connector-host child PIDs this server owns.

    Pass an empty list (or ``None``) to clear them — after the children have
    been reaped, so a later sweep does not report already-dead PIDs as orphans.
    """
    return _update(server_pid, {"children": _normalize_children(children)})


# -- switch requests (the web server writes, the reconciler consumes) -------


class RequestSuperseded(RuntimeError):
    """Another writer's request occupies the slot this one tried to claim.

    One controls server answers one request at a time, so the request file is a
    slot rather than a queue. Two operators clicking Switch at the same moment
    both pass the "is one already pending?" read — it happens before the write,
    with an ``await`` in between — and the second write would silently replace
    the first, leaving the first operator watching for a ``request_id`` no file
    carries any more.

    :func:`write_request` therefore reads its own write back and raises this
    when what landed is not what it wrote. The loser is told the same thing it
    would have been told a moment earlier: a request is pending. Exactly one
    request survives, and both operators get a true answer.
    """


def write_request(record: dict[str, Any]) -> Path:
    """Write a switch request addressed to the controls server it names.

    The one write in this directory the controls server does not make. It is
    addressed, not broadcast: the file is named for the ``server_pid`` in the
    record, and a reconciler drops any request whose body names a different
    process, so a request written for a server that has since been replaced can
    never be honoured by its successor.

    **The write is confirmed by reading it back.** ``os.replace`` cannot fail on
    a slot another writer already owns — it overwrites it — so exclusivity is
    established after the fact instead: the record that ends up in the file is
    the one that won, and every other writer raises
    :class:`RequestSuperseded`. Read-back rather than ``O_EXCL`` because a
    stale request is *legitimately* replaced (the route overwrites one the TTL
    has expired), and because ``O_EXCL`` would have to write into the final
    path directly, leaving a half-written request behind a crash — which no
    reader could tell from a claimed slot.

    Args:
        record: ``{request_id, target, server_pid, created_at, requested_by}``.
            ``created_at`` is stamped here when the caller leaves it out, so the
            TTL can always be evaluated.

    Returns:
        The path written.

    Raises:
        ValueError: The record names no usable ``server_pid``. A request nobody
            is addressed by is a programming error, not a runtime condition.
        RequestSuperseded: Another writer's request is in the slot. The caller
            answers this with ``request_pending``.
        OSError: The state directory is unwritable — the caller answers this
            with ``store_unavailable``, never by pretending the request landed.
    """
    if not isinstance(record, dict):
        raise ValueError("A switch request must be a mapping")
    try:
        pid = int(record["server_pid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("A switch request must name the server_pid it addresses") from exc
    if pid <= 0:
        raise ValueError(f"A switch request must name a real server_pid, not {pid!r}")

    payload = dict(record)
    payload["server_pid"] = pid
    if not payload.get("created_at"):
        payload["created_at"] = _now_iso()
    path = request_file_path(pid)
    _atomic_write_json(path, payload)

    landed = read_file(path)
    if not isinstance(landed, dict) or landed.get("request_id") != payload.get("request_id"):
        logger.info(
            "Switch request %s for pid %s was superseded by %r before it could be read back",
            payload.get("request_id"),
            pid,
            (landed or {}).get("request_id"),
        )
        raise RequestSuperseded(
            f"Another switch request occupies the slot for pid {pid}; this one did not land."
        )

    logger.debug("Wrote switch request %s for pid %s", payload.get("request_id"), pid)
    return path


def read_request(server_pid: int | None = None) -> dict[str, Any] | None:
    """The switch request addressed to *server_pid*, or ``None``. Never raises.

    Freshness is NOT applied here: a reconciler has to be able to tell "nobody
    asked" from "somebody asked too long ago", because only the second one owes
    the operator a ``request_expired`` answer. Ask :func:`is_request_fresh`.
    """
    return read_file(request_file_path(server_pid))


def remove_request(server_pid: int | None = None) -> None:
    """Remove the switch request addressed to *server_pid*.

    A missing file is success: the consumer removes a request once it has
    reached a terminus, and reaching the same terminus twice must not raise.
    """
    path = request_file_path(server_pid)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - unwritable state dir
        logger.warning("Could not remove switch request %s: %s", path, exc)


def is_request_fresh(
    record: Any,
    *,
    now: float | None = None,
    ttl_s: float = REQUEST_TTL_S,
) -> bool:
    """Whether *record* is still actionable, by its own ``created_at``.

    One spelling for both readers — the route that refuses a second request
    while one is pending, and the reconciler that expires one it reached too
    late — so the window an operator sees is the window that is enforced.

    Fail-closed: a record with no parseable ``created_at`` is NOT fresh. It
    cannot be aged, and acting on an unaged request is exactly the surprise the
    TTL exists to prevent. A stamp in the near future is tolerated up to the
    same TTL, because the two processes need not share a clock to the second.
    """
    if not isinstance(record, dict):
        return False
    stamp = record.get("created_at")
    if not isinstance(stamp, str) or not stamp:
        return False
    try:
        created = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    reference = datetime.now(UTC).timestamp() if now is None else float(now)
    return abs(reference - created.timestamp()) <= float(ttl_s)


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


# -- in-flight execution markers -------------------------------------------


def in_flight_executions() -> list[dict[str, Any]]:
    """Every live execution marker, oldest first. Never raises.

    A marker whose writing process is gone is swept: it is the residue of a
    killed executor, and treating it as live would make every later switch
    impossible with nothing an operator could stop. A marker that cannot be
    read is neither reported nor removed — it says nothing, and it is not this
    reader's file to delete.

    The answer is ADVISORY. It is a best-effort observation of another
    process's files, so a marker can be missing (the executor could not write
    it) or can appear a moment after this reader looked. Nothing about
    correctness rests on it: the guarantee that a run cannot be moved onto a
    machine nobody selected is the generation pin — an execution stamped at
    generation *n* has its writes refused once the session moves past it — and
    this check exists to turn that refusal into a question asked before the
    switch rather than an error discovered after it.

    It lives beside the state file rather than in the switch tool because it has
    three readers now: the tool's gate, the reconciler's gate, and the posture
    route that refuses to widen a posture out from under a running execution.
    """
    try:
        entries = sorted(state_dir().glob(INFLIGHT_FILE_GLOB))
    except OSError:  # pragma: no cover - unreadable state dir
        return []

    live: list[dict[str, Any]] = []
    for entry in entries:
        record = read_file(entry)
        if record is None:
            logger.debug("Unreadable execution marker %s; ignoring it", entry.name)
            continue
        pid = record.get("pid")
        if not isinstance(pid, int) or not is_process_alive(pid):
            with contextlib.suppress(OSError):
                entry.unlink(missing_ok=True)
            logger.debug("Swept execution marker %s (writer pid %r gone)", entry.name, pid)
            continue
        live.append(record)
    return live


# -- sweeping --------------------------------------------------------------


def sweep_stale(*, server_pid: int | None = None) -> list[int]:
    """Delete every state file whose owning server is dead; return its orphans.

    Switch requests addressed to a dead server are swept in the same pass (see
    :func:`_sweep_stale_requests`); they are not state files, so they are never
    reported as orphans and never contribute child PIDs.

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

    _sweep_stale_requests()
    return orphans


def _sweep_stale_requests() -> None:
    """Unlink every switch request addressed to a server that is not running.

    A request is desired state addressed to one process. Once that process is
    gone the request can never be answered, and leaving it would let a later
    server that happened to reuse the PID act on a gesture made at a different
    session — so it is swept on the same schedule as the state files, by the
    same liveness rule, without being opened.
    """
    try:
        entries = sorted(state_dir().glob(REQUEST_FILE_GLOB))
    except OSError:
        return

    for entry in entries:
        pid = _pid_from_name(entry.name, REQUEST_FILE_PREFIX, REQUEST_FILE_SUFFIX)
        if pid is not None and is_process_alive(pid):
            continue
        try:
            entry.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - unwritable state dir
            logger.warning("Could not remove stale switch request %s: %s", entry, exc)
            continue
        logger.info("Swept stale switch request %s (addressee pid %s gone)", entry.name, pid)
