"""Single-writer report file saying what one controls server is doing.

A controls MCP server is the ONLY writer of its own report, and it writes
nothing else here. What the deployment IS on — target, generation, posture —
belongs to the control-context record
(:mod:`osprey_connectors.control_context`), which one owner writes; this file
is one fleet member's observation of itself, and a reader that wants the fleet
reads all of them. That asymmetry is the whole design: a reader never has to
reconcile two opinions about identity, and a stale report is always the residue
of a server that died rather than a second writer's disagreement.

Path contract
-------------
The file lives at::

    <repo_root>/<agent_data_base_dir>/control_target/server_<server_pid>.json

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
* one file per server process, named :data:`REPORT_FILE_PREFIX` + PID +
  :data:`REPORT_FILE_SUFFIX`, discovered by the glob :data:`REPORT_FILE_GLOB`.

Report shape
------------
The ten fields :class:`osprey_connectors.control_context.ServerReport` parses,
and nothing else — a payload written here round-trips through
:func:`~osprey_connectors.control_context.parse_report` unchanged::

    {
      "server_pid": 4321,          # os.getpid() of the controls server
      "session": "abc123" | null,  # its OSPREY_POSTURE_SESSION, null for a
                                   # bare `claude` that was never stamped
      "applied_target": "live" | "va" | "standin" | null,
      "applied_generation": 7 | null,
      "targets": {                 # probe_channel and selected_role are both
                                   # optional; each is omitted when unresolved
        "live":    {"label": str, "endpoint": str, "real_machine": bool,
                    "probe_channel": str, "selected_role": str},
        "va":      {"label": str, "endpoint": str, "real_machine": bool,
                    "probe_channel": str, "selected_role": str},
        "standin": {"label": str, "endpoint": str, "real_machine": bool,
                    "probe_channel": str, "selected_role": str}
      },
      "children": [5001, 5002],    # connector-host child PIDs, may be empty
      "last_switch": {             # progress through the current switch, or null
        "generation": int, "status": "applying" | "applied" | "failed",
        "reason": str | None, "detail": str | None,
        "at": str,                 # wall clock, ISO-8601
        "expires_at": str          # applying only; see publish_last_switch
      },
      "reachability": {            # last prober sweep, {} when never probed
        "published_at": str,       # wall clock, ISO-8601
        "targets": {"live": {"<role>": {"state": str, "probed_at": str, ...}}}
      },
      "last_posture_realign": {"state": "pending" | "done", "at": str},
      "updated_at": str            # wall clock, ISO-8601, every write
    }

``applied_target`` and ``applied_generation`` are **null until the first child
has answered its init frame** — :func:`publish_switch` is the one publisher that
sets them, and it is called after a launch or a swap succeeded. Null is not
"the baseline": it is "this server has not got there yet", and a reader deciding
whether the fleet has converged has to be able to tell those apart.

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
``selected_role`` names the role whose gateway the ``endpoint`` beside it is,
and is optional on the same terms.

Rendered once does not mean rendered forever. Identity depends on the session's
posture, and a session can narrow itself after start, so :func:`publish_targets`
lets the same single writer re-render the block and republish it. That is still
one opinion — the writer's, restated — and readers keep rendering verbatim.

``children`` records the connector-host children a server owns so that
:func:`sweep_stale` can hand a starting server the orphan PIDs left behind by a
dead predecessor.

``last_switch``, ``reachability`` and ``last_posture_realign`` are the three
publication blocks the header chip reads. They are empty on a fresh report —
:func:`write_server_record` resets them with everything else, because none of
them describes anything a NEW server has done yet — and each is merged in by its
own publisher (:func:`publish_last_switch`, :func:`publish_reachability`,
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
* ``switch_request_<requested_by_pid>.json`` — a switch REQUEST, named for the
  process that ASKED and consumed by whichever process owns the control-context
  record. Any process may write one for itself; nobody writes one for anybody
  else. It is not a second opinion about identity: it says what an operator
  asked for, never what is true. It expires (:data:`REQUEST_TTL_S`).

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
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from osprey.audit.posture import posture_session
from osprey_connectors import control_context, session_store
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

#: Fixed subdirectory of the agent-data root. Part of the path contract above,
#: and taken from :mod:`osprey_connectors.control_context` for the same reason
#: the report names below are: the record, the posture store and these reports
#: share one directory, and two spellings of it would make it two. Only the
#: stdlib-only hooks, which can import neither, restate the literal.
STATE_DIR_NAME = control_context.STATE_DIR_NAME

#: One report file per server process. The PID in the name is what makes a
#: stale report sweepable without opening it. Taken from
#: :mod:`osprey_connectors.control_context` rather than re-spelled here: the
#: writer and the readers that glob for it must not be able to disagree about
#: which family of files this is.
REPORT_FILE_PREFIX = control_context.REPORT_FILE_PREFIX
REPORT_FILE_SUFFIX = control_context.REPORT_FILE_SUFFIX
REPORT_FILE_GLOB = control_context.REPORT_FILE_GLOB

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

#: One switch request per REQUESTING process, consumed by the process that owns
#: the control-context record. It is the single exception to "the controls
#: server is the only writer in this directory", and it is safe because it is a
#: different file: a request is desired state, never a second opinion about what
#: the target IS. The PID in the name is the requester's own, which is what
#: makes each requester's slot exclusively its own and its residue sweepable
#: without being opened — a request outlives nobody but the process that asked.
REQUEST_FILE_PREFIX = "switch_request_"
REQUEST_FILE_SUFFIX = ".json"
REQUEST_FILE_GLOB = f"{REQUEST_FILE_PREFIX}*{REQUEST_FILE_SUFFIX}"

#: How long a switch request stays actionable. A request the reconciler reaches
#: later than this is refused as ``request_expired`` rather than acted on: the
#: operator who clicked Switch is no longer watching, and a switch that lands
#: minutes after the gesture is a surprise rather than a service.
REQUEST_TTL_S = 30

#: How far a server has got through the generation it is currently binding to.
#: ``applying`` is the only one with a deadline: it says a swap is running right
#: now, which is what makes the fleet not converged, so it carries the
#: ``expires_at`` past which a reader stops letting it block everybody
#: (:func:`publish_last_switch`). ``applied`` and ``failed`` are termini and are
#: read together with ``applied_target`` / ``applied_generation``.
SWITCH_APPLYING = "applying"
SWITCH_APPLIED = "applied"
SWITCH_FAILED = "failed"

__all__ = [
    "INFLIGHT_FILE_GLOB",
    "INFLIGHT_FILE_PREFIX",
    "INFLIGHT_FILE_SUFFIX",
    "REPORT_FILE_GLOB",
    "REPORT_FILE_PREFIX",
    "REPORT_FILE_SUFFIX",
    "REQUEST_FILE_GLOB",
    "REQUEST_FILE_PREFIX",
    "REQUEST_FILE_SUFFIX",
    "REQUEST_TTL_S",
    "STATE_DIR_NAME",
    "SWITCH_APPLIED",
    "SWITCH_APPLYING",
    "SWITCH_FAILED",
    "RequestSuperseded",
    "TARGET_LIVE",
    "TARGET_NAMES",
    "TARGET_STANDIN",
    "TARGET_VA",
    "applying_bound_s",
    "delete_on_shutdown",
    "in_flight_executions",
    "is_process_alive",
    "is_request_fresh",
    "live_records",
    "publish_last_switch",
    "publish_posture_realign",
    "publish_reachability",
    "publish_switch",
    "publish_targets",
    "read",
    "read_file",
    "read_request",
    "record_child_pids",
    "record_pid",
    "remove_request",
    "report_file_path",
    "request_file_path",
    "session_record",
    "state_dir",
    "sweep_stale",
    "write_request",
    "write_server_record",
]


# -- paths -----------------------------------------------------------------


def state_dir() -> Path:
    """Directory holding every server's report file. Not created by reading.

    :data:`~osprey.audit.posture.OSPREY_AGENT_DATA_ROOT` wins when it is set.
    A session child is stamped with the root its spawning server resolved, and
    that stamp is the whole point: writer and readers derive this directory
    three different ways (config here, config again in the store reader, a
    repo-root guess plus the literal ``var/agent_data`` in the stdlib-only
    hooks), and a deployment that moves ``agent_data.base_dir`` makes them
    disagree. Preferring the stamp here is what makes "no report under the
    stamped root" mean "no controls server here" rather than "the reader looked
    in the wrong place".

    The stamp is read through
    :func:`~osprey_connectors.session_store.stamped_agent_data_root` rather
    than off the environment here, because the posture store sits in this same
    directory and the two must not normalise one variable differently — a
    ``~`` or a padded value would otherwise put the report and the store
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


def report_file_path(server_pid: int | None = None) -> Path:
    """Path of the report written by *server_pid* (default: this process).

    Named from the constants :mod:`osprey_connectors.control_context` owns, so
    the writer and every reader name the same file. The directory comes from
    :func:`state_dir` rather than from the library's own resolver because this
    module raises where the library answers ``None``, and the web terminal's
    switch route turns that raise into its ``store_unavailable`` 503.
    """
    pid = os.getpid() if server_pid is None else int(server_pid)
    return state_dir() / f"{REPORT_FILE_PREFIX}{pid}{REPORT_FILE_SUFFIX}"


def request_file_path(requested_by_pid: int | None = None) -> Path:
    """Path of the switch request written by *requested_by_pid* (default: this one).

    The pid is the asker's, not the answerer's: one slot per requester, so two
    processes asking at once never overwrite each other and a request is swept
    when the process that is waiting for its answer is gone.
    """
    pid = os.getpid() if requested_by_pid is None else int(requested_by_pid)
    return state_dir() / f"{REQUEST_FILE_PREFIX}{pid}{REQUEST_FILE_SUFFIX}"


def _now_iso() -> str:
    """Wall clock, ISO-8601, UTC. The stamp every reader ages a block from."""
    return datetime.now(UTC).isoformat()


# -- liveness --------------------------------------------------------------


def is_process_alive(pid: object) -> bool:
    """Whether *pid* names a running process.

    The rule itself lives in
    :func:`osprey_connectors.control_context.is_process_alive`, with everything
    else that judges a file in this directory by the process it is named for.
    It is re-exported here because this module is the vocabulary its readers —
    the banner, the reconciler, the roster — already import, and because a
    caller that patches this name is patching the answer for the whole pass.
    """
    return control_context.is_process_alive(pid)


# -- the records that describe a session -----------------------------------


def record_pid(record: Mapping[str, Any], field: str) -> int | None:
    """The PID a record's *field* carries, or ``None`` when it carries no PID.

    Strict: the one writer of these files (:func:`write_server_record`) emits an
    ``int``, so anything else — a string, a float, a ``bool`` (which IS an
    ``int`` to ``isinstance`` and ``1`` to ``os.kill``) — is a record nothing
    here wrote, and coercing it would let such a record match a real parent.
    """
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def live_records(entries: Iterable[Path]) -> list[dict[str, Any]]:
    """Every readable record among *entries* whose owning server still runs.

    A record whose ``server_pid`` is gone is residue: its target describes a
    server nobody is talking to any more, and the next server to start sweeps
    it (:func:`sweep_stale`). Unreadable entries are skipped, not raised — a
    half-written file is the documented "state unavailable" outcome.

    The read and the liveness filter both live in
    :func:`osprey_connectors.control_context.live_report_payloads`; the
    liveness predicate is handed over from here so that a caller which has
    patched or narrowed :func:`is_process_alive` gets one answer for the whole
    pass.

    Args:
        entries: Report paths, typically ``state_dir().glob(REPORT_FILE_GLOB)``.
            Taken rather than globbed here because one caller keys a cache on
            the same listing and must not glob twice.
    """
    return control_context.live_report_payloads(entries, is_alive=is_process_alive)


def session_record(
    entries: Sequence[Path],
    owner_ppid: int,
    *,
    require_generation: bool = False,
) -> dict[str, Any] | None:
    """The one live record this session owns, or ``None``.

    THE matcher for every process that is a child of the same Claude Code
    process as the controls server — the audit posture, the python executor,
    the banner — so that no two of them can disagree about which session they
    are in.

    Selection is *exact parent equality*: the controls MCP server that writes
    these files and the server this code runs in are both spawned by the same
    Claude Code process, so the record whose ``owner_ppid`` equals our parent
    is the one describing our session. Deliberately narrower than the
    ancestor-chain walk the stdlib-only hooks do (and keep, by design: a hook
    cannot import this module): a deployment that interposes a process breaks
    the equality and gets "no answer", never another session's target. Strict
    ``int`` on both sides, per :func:`record_pid`.

    Zero matches (no controls server, or a directory this deployment never
    created) and more than one (an ``owner_ppid`` collision after PID reuse)
    both answer ``None``, as does a record naming a target no reader knows. A
    record whose ``server_pid`` is gone is residue and is skipped
    (:func:`live_records`).

    Args:
        entries: State-file paths, as for :func:`live_records`.
        owner_ppid: This process's parent, ``os.getppid()``.
        require_generation: Also require an ``int`` ``generation`` — the
            executor stamps its sandbox with it, and a record without one
            cannot pin a run.

    Returns:
        The matching record, or ``None``.
    """
    matches: list[dict[str, Any]] = []
    for record in live_records(entries):
        if record_pid(record, "owner_ppid") != owner_ppid:
            continue
        if record.get("target") not in TARGET_NAMES:
            continue
        if require_generation:
            generation = record.get("generation")
            if isinstance(generation, bool) or not isinstance(generation, int):
                continue
        matches.append(record)

    if len(matches) != 1:
        if matches:
            logger.warning(
                "%d control-target records share owner_ppid %s; the session target is unknown",
                len(matches),
                owner_ppid,
            )
        elif entries:
            # Records exist but none is ours — another session's, or a parent
            # this process does not have. Worth seeing when a switch appears to
            # have had no effect.
            logger.debug(
                "%d control-target record(s) present, none owned by ppid %s",
                len(entries),
                owner_ppid,
            )
        return None
    return matches[0]


# -- record normalization --------------------------------------------------


#: Display keys written only when the caller supplied a non-empty string.
#: ``probe_channel`` is the channel a switch would probe; ``selected_role`` is
#: the role whose gateway the ``endpoint`` beside it belongs to. Neither is ever
#: fabricated, so a reader can tell "not resolved" from "resolved as".
_OPTIONAL_TARGET_KEYS = ("probe_channel", "selected_role")


def _normalize_target_meta(value: Any) -> dict[str, Any]:
    """Coerce one target's display metadata to the shape readers expect.

    ``label`` / ``endpoint`` / ``real_machine`` are always present — a reader
    branching on a missing key is a reader that can crash a hook. The keys in
    :data:`_OPTIONAL_TARGET_KEYS` pass through only when the caller supplied a
    real one: the approval describer renders them exclusively from this file,
    so an invented placeholder would show an operator a channel nobody probes
    or a role nobody selected.
    """
    meta = value if isinstance(value, dict) else {}
    normalized: dict[str, Any] = {
        "label": str(meta.get("label") or ""),
        "endpoint": str(meta.get("endpoint") or ""),
        "real_machine": bool(meta.get("real_machine", False)),
    }
    for key in _OPTIONAL_TARGET_KEYS:
        optional = meta.get(key)
        if isinstance(optional, str) and optional:
            normalized[key] = optional
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

    The rule lives in
    :func:`osprey_connectors.control_context.write_json_atomic`, with the
    record that shares this directory and is read the same way. It keeps a
    name here because the reports and requests are written through it and a
    test that makes a write fail patches this one.
    """
    control_context.write_json_atomic(path, record)


def write_server_record(
    targets_meta: dict[str, Any] | None = None,
    *,
    server_pid: int | None = None,
    session: str | None = None,
    children: list[int] | None = None,
) -> list[int]:
    """Write this server's report at start and sweep the dead ones.

    Called once, at server start, before anything can switch. It is a RESET and
    not a merge: nothing a predecessor at this PID published describes this
    process, and a report that inherited one would tell the fleet this server
    had already reached a target it has never launched a child for.

    It publishes **no target**. What the deployment is on belongs to the
    control-context record, which this server reads (and claims when it is
    ownerless) at the same start; ``applied_target`` and ``applied_generation``
    stay null here until :func:`publish_switch` says a child answered its init
    frame.

    Args:
        targets_meta: Prepared per-target display metadata, ``{"live": {...},
            "va": {...}, "standin": {...}}`` with ``label`` / ``endpoint`` /
            ``real_machine``. Rendered by the caller from config and written
            verbatim here; a target the caller omits is written as an empty slot.
        server_pid: Reporting PID; defaults to this process.
        session: This server's audit session id; defaults to the environment's
            ``OSPREY_POSTURE_SESSION``, and stays ``None`` for a bare ``claude``
            that was never stamped with one. A session-less server reports the
            same way every other one does — the field says whose launches a
            failure of this server's own would refuse, not whether it counts.
        children: Connector-host child PIDs already known at start.

    Returns:
        Orphan child PIDs recorded by dead predecessors, for the caller to kill.
    """
    pid = os.getpid() if server_pid is None else int(server_pid)
    orphans = sweep_stale(server_pid=pid)
    # A request in this PID's slot can only be a dead predecessor's residue —
    # this process has asked for nothing yet — and leaving it would let an owner
    # move the deployment on a gesture nobody made in this session.
    remove_request(requested_by_pid=pid)
    report = {
        "server_pid": pid,
        "session": posture_session() if session is None else str(session),
        # Null, not the baseline: this server has launched nothing yet, and a
        # reader that took a start-time guess for an observation would count an
        # unconverged fleet as converged.
        "applied_target": None,
        "applied_generation": None,
        "children": _normalize_children(children),
        # The three publication blocks start empty: none of them describes
        # anything this server has done yet, and a stale switch outcome or an
        # inherited reachability sweep would be read as this server's own.
        "reachability": {},
        "last_switch": None,
        "last_posture_realign": None,
        "targets": _normalize_targets(targets_meta),
        "updated_at": _now_iso(),
    }
    _atomic_write_json(report_file_path(pid), report)
    logger.debug("Server report initialized for pid %s (session %r)", pid, report["session"])
    return orphans


def _update(server_pid: int | None, changes: dict[str, Any]) -> bool:
    """Merge *changes* into this server's report. ``False`` if there is none.

    ``updated_at`` moves on every merge, because a reader deciding whether a
    server is still saying anything ages it from exactly this stamp.
    """
    path = report_file_path(server_pid)
    report = read_file(path)
    if report is None:
        logger.error("No server report at %s to update; write_server_record first", path)
        return False
    report.update(changes)
    report["updated_at"] = _now_iso()
    _atomic_write_json(path, report)
    return True


def publish_switch(
    target: str,
    generation: int,
    *,
    children: list[int] | None = None,
    server_pid: int | None = None,
) -> bool:
    """Report that a child is live on *target* at *generation*.

    THE moment ``applied_target`` and ``applied_generation`` stop being null.
    Called once a connector-host child has answered its init frame, never
    before: until then this server has an intention, and an intention published
    as an observation is how a fleet reads as converged while a swap is still
    running.

    Writes what it is told. Which generation a child is bound to is the owner's
    to mint and the switch lifecycle's to assign — this file records the outcome
    so readers agree on it, and does not arbitrate it. Display metadata written
    at start is preserved; :func:`publish_targets` is the one publisher that
    moves it.

    Returns:
        ``True`` when the report was updated, ``False`` when none exists
        (nothing was written, and the caller has a start-ordering bug).
    """
    changes: dict[str, Any] = {
        "applied_target": str(target),
        "applied_generation": int(generation),
    }
    if children is not None:
        changes["children"] = _normalize_children(children)
    return _update(server_pid, changes)


def publish_targets(
    targets_meta: dict[str, Any],
    *,
    server_pid: int | None = None,
) -> bool:
    """Replace the per-target display metadata with a freshly rendered mapping.

    :func:`write_server_record` renders the ``targets`` block once from the deployment
    config, which is the right answer only while the session's posture matches
    the config ceiling. A session that narrows itself to read-only afterwards is
    served by a gateway the start-time render never named, and every reader of
    this file renders verbatim by contract — so the writer, not the readers, is
    what has to say the new thing. This publisher is that move: the single writer
    re-renders identity from the session it is actually in and republishes it.

    Normalized exactly as at start (:func:`_normalize_targets`), so a slot the
    caller omits is written empty rather than dropped, and the whole block is
    REPLACED rather than merged — a half-updated ``targets`` would let one target
    name its old gateway beside another naming its new one.

    Args:
        targets_meta: Prepared per-target display metadata, the same shape
            :func:`write_server_record` takes.
        server_pid: Reporting PID; defaults to this process.

    Returns:
        ``True`` when the report was updated, ``False`` when none exists
        (nothing was written, and the caller has a start-ordering bug).
    """
    return _update(server_pid, {"targets": _normalize_targets(targets_meta)})


# -- publication blocks ----------------------------------------------------
#
# THE MERGE MUST BE ATOMIC WITH RESPECT TO THE EVENT LOOP. Each publisher below
# is deliberately SYNCHRONOUS and each is a single :func:`_update` call, which
# reads the report and writes it back with no suspension point in between.
# Never make one of these ``async``, and never insert an ``await`` between the
# read and the write: two coroutines interleaving there would each write back a
# record built from a copy taken before the other's change, and the loser's
# block would vanish. The publishers run beside one another — the reconciler
# publishes ``last_switch`` and ``last_posture_realign`` while the endpoint
# prober publishes ``reachability`` every sweep — so this is the ordinary case,
# not the exotic one. Callers on the loop pay one small blocking file write; the
# alternative is a lock nobody outside this process could take anyway.


def applying_bound_s(
    *,
    spawn_timeout_s: float,
    probe_timeout_s: float,
    drain_timeout_s: float,
    fallback_retry: bool = True,
) -> float:
    """How long the swap this server is running can legitimately take.

    The bound a reader needs and cannot compute: the timeouts are this
    process's, read from its own deployment config, and every other process —
    a kernel, a sibling server, the terminal — only ever sees the report. So
    the publisher computes the deadline and writes it
    (:func:`publish_last_switch`), and a reader compares its own clock to that.

    A swap drains the old child once and then runs the spawn-and-probe pair;
    when the write gateway will not answer the probe, the manager retries the
    whole pair through the read-only gateway, which is why the pair counts
    twice by default. Deliberately generous: a bound that expires early makes a
    healthy swap look stranded and lets other sessions launch into the middle
    of it, while a bound that expires late costs only a slower recovery from a
    server that died mid-swap.

    Args:
        spawn_timeout_s: Bound on "spawned and answered its init frame".
        probe_timeout_s: Bound on the readiness probe.
        drain_timeout_s: Bound on draining the child being replaced.
        fallback_retry: Whether a probe failure can be retried through the
            read-only gateway — the doubling above. A first launch that cannot
            retry passes ``False``.

    Returns:
        Seconds from the ``applying`` stamp to its ``expires_at``.
    """
    attempts = 2 if fallback_retry else 1
    return float(drain_timeout_s) + attempts * (float(spawn_timeout_s) + float(probe_timeout_s))


def _expires_at(stamp: str, expires_in_s: float) -> str:
    """*stamp* moved forward by *expires_in_s*, ISO-8601, UTC.

    Anchored on the block's own ``at`` rather than on ``now`` so the deadline
    and the stamp a reader ages from cannot disagree by the cost of the write.
    """
    try:
        base = datetime.fromisoformat(stamp)
    except ValueError:
        base = datetime.now(UTC)
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    return (base + timedelta(seconds=float(expires_in_s))).isoformat()


def _normalize_last_switch(outcome: Any, expires_in_s: float | None) -> dict[str, Any] | None:
    """Coerce a switch block to the shape readers age, bound and match on.

    Written through largely verbatim: the vocabulary is the switch lifecycle's
    (``generation``, ``status``, ``reason``, ``detail``), and restating it here
    would give this module an opinion about refusals it does not arbitrate.
    What is enforced is the part every reader depends on — a wall-clock ``at``,
    so a block can be aged rather than believed forever, and on an
    :data:`SWITCH_APPLYING` block the ``expires_at`` past which it stops
    blocking the whole deployment.

    An ``applying`` block published with no bound at all is written as it
    stands: a reader with no deadline keeps waiting, which is the fail-closed
    outcome, and inventing one here would be this module guessing at timeouts
    it does not hold. It is logged, because it is a caller's bug.
    """
    if not isinstance(outcome, dict):
        return None
    block = dict(outcome)
    if not block.get("at"):
        block["at"] = _now_iso()
    if block.get("status") == SWITCH_APPLYING and not block.get("expires_at"):
        if expires_in_s is None:
            logger.warning(
                "Publishing an %r switch block with no expires_at; readers will keep "
                "treating this server as mid-swap until it publishes a terminus",
                SWITCH_APPLYING,
            )
        else:
            block["expires_at"] = _expires_at(str(block["at"]), expires_in_s)
    return block


def _normalize_reachability(rows: Any) -> dict[str, Any]:
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

    A sweep that measured nothing publishes ``{}`` — "this server has not
    probed" — rather than a null, so the field always holds the mapping
    :class:`~osprey_connectors.control_context.ServerReport` describes.
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
        return {}
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


def publish_last_switch(
    outcome: dict[str, Any] | None,
    *,
    expires_in_s: float | None = None,
    server_pid: int | None = None,
) -> bool:
    """Publish how far this server has got through the generation it is binding.

    Every step goes through here — the ``applying`` this server publishes
    before it starts a swap, and the ``applied`` or ``failed`` it ends on — so a
    reader deciding whether the fleet has converged has exactly one place per
    server to look, matched by ``generation`` rather than by guessing from the
    target.

    SYNCHRONOUS by contract; see the section comment above.

    Args:
        outcome: ``{generation, status, reason, detail, at, expires_at}``;
            ``at`` is stamped here when the caller leaves it out. ``status`` is
            one of :data:`SWITCH_APPLYING`, :data:`SWITCH_APPLIED`,
            :data:`SWITCH_FAILED`. ``None`` clears the block.
        expires_in_s: Seconds an :data:`SWITCH_APPLYING` block stays
            authoritative, normally :func:`applying_bound_s` of this server's
            own spawn / probe / drain timeouts. Stamped into ``expires_at``
            relative to ``at``; ignored when the caller supplied one, and when
            the block is a terminus, which needs no deadline.
        server_pid: Reporting PID; defaults to this process.

    Returns:
        ``True`` when the report was updated, ``False`` when there is none.
    """
    return _update(server_pid, {"last_switch": _normalize_last_switch(outcome, expires_in_s)})


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
        server_pid: Reporting PID; defaults to this process.

    Returns:
        ``True`` when the report was updated, ``False`` when there is none.
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
        server_pid: Reporting PID; defaults to this process.

    Returns:
        ``True`` when the report was updated, ``False`` when there is none.
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
    """Another request occupies this requester's slot, carrying a different id.

    One process asks for one switch at a time, so its request file is a slot
    rather than a queue. Two callers in the same process clicking Switch at the
    same moment both pass the "is one already pending?" read — it happens
    before the write, with an ``await`` in between — and the second write would
    silently replace the first, leaving the first caller watching for a
    ``request_id`` no file carries any more.

    :func:`write_request` therefore reads its own write back and raises this
    when a *readable* record with a different ``request_id`` is what it finds.
    The loser is told the same thing it would have been told a moment earlier:
    a request is pending. Exactly one request survives, and both callers get a
    true answer.

    A slot that is EMPTY on read-back is not a supersession — under the
    inverted polarity the owner unlinks a request the moment it consumes it, so
    an absent file is the fastest possible success.
    """


def write_request(record: dict[str, Any]) -> Path:
    """Write a switch request on behalf of the process that asked for it.

    The one write in this directory a controls server does not have to make. It
    is not addressed to anybody: the file is named for the ``requested_by_pid``
    in the record, and whichever process owns the control-context record
    consumes it. A request therefore survives exactly as long as the process
    waiting for its answer, and can never be inherited by a successor at some
    other server's pid.

    **The write is confirmed by reading it back**, because ``os.replace``
    overwrites a slot rather than failing on it. What comes back has two
    meanings under the inverted polarity, and only one of them is a failure:

    * nothing readable in the slot — the owner already consumed the request, or
      the read itself degraded. The request landed; the caller learns its
      outcome from the record's ``last_switch``, which is where it was going to
      look anyway.
    * a readable record carrying a *different* ``request_id`` — a second call
      in this same process overwrote this one. That is a supersession, and the
      loser raises :class:`RequestSuperseded`.

    Read-back rather than ``O_EXCL`` because a stale request is *legitimately*
    replaced (a caller overwrites one the TTL has expired), and because
    ``O_EXCL`` would have to write into the final path directly, leaving a
    half-written request behind a crash — which no reader could tell from a
    claimed slot.

    Args:
        record: ``{request_id, target, requested_at, session, requested_by_pid}``.
            ``requested_at`` is stamped here when the caller leaves it out, so
            the TTL can always be evaluated. ``session`` is the requester's
            audit session id and stays ``None`` for a bare ``claude``.

    Returns:
        The path written, whether or not the file still exists — a consumed
        request is a landed one.

    Raises:
        ValueError: The record names no usable ``requested_by_pid``. A request
            from nobody could never be swept or answered, so it is a
            programming error rather than a runtime condition.
        RequestSuperseded: A different request_id is in this requester's slot.
            The caller answers this with ``request_pending``.
        OSError: The state directory is unwritable — the caller answers this
            with ``store_unavailable``, never by pretending the request landed.
    """
    if not isinstance(record, dict):
        raise ValueError("A switch request must be a mapping")
    try:
        pid = int(record["requested_by_pid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("A switch request must name the requested_by_pid that asked") from exc
    if pid <= 0:
        raise ValueError(f"A switch request must name a real requested_by_pid, not {pid!r}")

    payload = dict(record)
    payload["requested_by_pid"] = pid
    if not payload.get("requested_at"):
        payload["requested_at"] = _now_iso()
    path = request_file_path(pid)
    _atomic_write_json(path, payload)

    landed = read_file(path)
    if isinstance(landed, dict) and landed.get("request_id") != payload.get("request_id"):
        logger.info(
            "Switch request %s from pid %s was superseded by %r before it could be read back",
            payload.get("request_id"),
            pid,
            landed.get("request_id"),
        )
        raise RequestSuperseded(
            f"Another switch request occupies the slot for pid {pid}; this one did not land."
        )

    logger.debug("Wrote switch request %s from pid %s", payload.get("request_id"), pid)
    return path


def read_request(requested_by_pid: int | None = None) -> dict[str, Any] | None:
    """The switch request written by *requested_by_pid*, or ``None``. Never raises.

    Freshness is NOT applied here: a consumer has to be able to tell "nobody
    asked" from "somebody asked too long ago", because only the second one owes
    the operator a ``request_expired`` answer. Ask :func:`is_request_fresh`.
    """
    return read_file(request_file_path(requested_by_pid))


def remove_request(requested_by_pid: int | None = None) -> None:
    """Remove the switch request written by *requested_by_pid*.

    A missing file is success: the consumer removes a request once it has
    reached a terminus, and reaching the same terminus twice must not raise.
    """
    path = request_file_path(requested_by_pid)
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
    """Whether *record* is still actionable, by its own ``requested_at``.

    One spelling for both readers — the route that refuses a second request
    while one is pending, and the reconciler that expires one it reached too
    late — so the window an operator sees is the window that is enforced.

    Fail-closed: a record with no parseable ``requested_at`` is NOT fresh. It
    cannot be aged, and acting on an unaged request is exactly the surprise the
    TTL exists to prevent. A stamp in the near future is tolerated up to the
    same TTL, because the two processes need not share a clock to the second.
    """
    if not isinstance(record, dict):
        return False
    stamp = record.get("requested_at")
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
    """Remove this server's report. A missing file is success, not an error."""
    path = report_file_path(server_pid)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - unwritable state dir
        logger.warning("Could not remove the server report %s: %s", path, exc)


# -- reading ---------------------------------------------------------------


def read_file(path: Path) -> dict[str, Any] | None:
    """Load one file here, or ``None`` if it is absent, unreadable, or corrupt.

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
    """Load this server's report payload, or ``None``. Never raises."""
    return read_file(report_file_path(server_pid))


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

    It lives beside the reports rather than in the switch tool because it has
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
    """Delete every report whose server is dead; return its orphans.

    Switch requests left by a dead requester are swept in the same pass; they
    are not reports, so they are never counted as orphans and never
    contribute child PIDs.

    A server exits without running :func:`delete_on_shutdown` whenever it is
    killed rather than asked to stop, and its connector-host children can
    outlive it. The next server to start therefore inherits two jobs: clear the
    dead file so readers stop seeing a target nobody is on, and collect the
    child PIDs it recorded so the caller can kill them.

    The file this process owns is left alone even though the process is
    obviously alive — checking one's own liveness is a way to get it wrong.

    The reaping rule — a file is judged by the PID in its name, and a name that
    encodes none has no owner either — lives in
    :func:`osprey_connectors.control_context.sweep_dead`. What a swept file
    *meant* is this module's business, so the child PIDs a dead server left
    running are read here, through the ``salvage`` hook, before the file goes.

    Args:
        server_pid: PID whose file to preserve; defaults to this process.

    Returns:
        Orphan child PIDs, in discovery order, without duplicates.
    """
    own = report_file_path(server_pid)
    directory = state_dir()
    orphans: list[int] = []

    def collect_orphans(entry: Path) -> None:
        record = read_file(entry)
        if record is None:
            return
        for child in _normalize_children(record.get("children")):
            if child not in orphans:
                orphans.append(child)

    control_context.sweep_dead(
        directory,
        prefix=REPORT_FILE_PREFIX,
        suffix=REPORT_FILE_SUFFIX,
        keep=own.name,
        salvage=collect_orphans,
        is_alive=is_process_alive,
    )
    # A request is desired state, and the process that asked is the one waiting
    # for the answer. Once it is gone there is nobody left to tell, and leaving
    # the file would let an owner move the deployment for a session that has
    # ended — so it is swept on the same schedule and by the same liveness
    # rule, without being opened.
    control_context.sweep_dead(
        directory,
        prefix=REQUEST_FILE_PREFIX,
        suffix=REQUEST_FILE_SUFFIX,
        is_alive=is_process_alive,
    )
    return orphans
