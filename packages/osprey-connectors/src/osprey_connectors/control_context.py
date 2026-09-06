"""The control-context record — one file per deployment instance.

Which machine a deployment is pointed at, how many times that has moved, and
what the operator has narrowed are properties of the DEPLOYMENT, not of any
process tree inside it. They live in a single JSON file,
:data:`RECORD_FILENAME` inside :data:`STATE_DIR_NAME` under the agent-data
root, and this module is where it is written and read.

The file has exactly one writer at a time — its **owner**: the web terminal
when one is running, otherwise a controls MCP server. Everything else is a
reader: every chat session, the Simple view, every notebook kernel, every
executor sandbox, every sibling MCP server, and the stdlib-only hooks, which
cannot import this module and therefore restate the rules below. Treat
them as the contract; a change here is a change there.

**1. Where the file is.** :func:`record_path` — the record sits in the same
directory as the posture store, and the root resolves by that store's rule 1
(:func:`osprey_connectors.session_store.agent_data_root`) rather than by a
second derivation here: a record one process writes and another looks for
somewhere else is worse than no record at all. An unresolvable root makes
:func:`record_path` ``None``, which reads as "no record" and makes
:func:`write_record` raise — a route surfaces that as ``store_unavailable``
rather than writing where no reader will look.

**2. What a payload must carry.** ``schema`` (exactly
:data:`SCHEMA_VERSION`), ``target`` (a member of
:data:`~osprey_connectors.types.CONTROL_TARGETS`) and ``generation`` (a
non-negative integer) are the record's identity. If any of the three is
absent, of the wrong type, or unrecognised, there is no record: this is a
file that decides whether a real machine is written to, so a hand-edited or
future-version payload must not be honoured in part.

The other three fields are annotations on that identity and degrade on their
own, because each has a meaningful "not stated" value that a reader can act
on safely:

* ``owner`` → ``None``, which is *ownerless* — the state a claim is for. An
  owner that does not name one of :data:`OWNER_KINDS` and a live PID is not
  an owner anybody should defer to, so reading it as ownerless is what lets
  the next process take the file over instead of stalling behind a ghost.
* ``posture`` → ``{}``, no narrowings, the deployment ceiling in charge. It
  is one entry in the posture store's grammar (rule 2 there): a per-target
  map of which only ``sandbox`` leaves survive, or the bare string
  ``sandbox`` covering every target. Nothing here can widen anything.
* ``last_switch`` → ``None``. It is the terminus of the last switch request
  — applied or refused — and it is carried verbatim. This module owns the
  block's shape, because a requester polls it for its own ``request_id``
  (:func:`terminus` builds it, :func:`write_terminus` stores it and reads it
  back); the switch machinery owns what one says. **A refusal is a record
  write that moves nothing else**: the same target, the same generation, the
  same posture, a new ``last_switch``. That is the whole of how a refusal is
  reported.

**3. How it is written.** :func:`write_json_atomic` replaces a file rather
than rewriting it — temp file in the same directory, then ``os.replace`` — so
no reader ever sees half of one and two writers never interleave into one.
Every file in this directory is written through it, the record
(:func:`write_record`) and the per-server reports and requests alike.
Serialising the read-modify-write around a write is the owner's job, not this
module's.

**4. When a change is seen.** Every read re-stats the file and re-parses on
``(st_mtime_ns, st_size, st_ino)``, the same signature the posture store uses
and for the same reason: writes arrive as temp+rename, so two of them inside
one filesystem clock tick differ by inode when mtime and size do not.

**5. Who else lives in that directory.** The record says what the deployment
is pointed at; it does not say what any one controls server has managed to do
about it. That is the per-server **report**, ``server_<pid>.json``, one file
per controls server, written by that server through its publishers and read
here (:func:`read_report`, :func:`live_reports`). A report is an observation,
so it degrades field by field around its one identity, ``server_pid``, and a
report nobody can parse is skipped rather than fatal — the readers of these
files describe a fleet, and one bad file must not blank the others.

**6. Whether anything may act.** The record says where the deployment is
pointed; the reports say who has got there. :func:`converged` is the one place
those two are compared, and it is what the executor's stamp, a notebook cell
and a controls server's write tools ask before they touch a machine: a swap in
flight stops everyone, a server that could not follow stops only its own
session, and nothing else stops anybody.

Every file in that directory is named for the process that owns it, which is
what lets a reader judge one without opening it: :func:`is_process_alive` and
:func:`sweep_dead` are that rule and its consequence. Liveness lives here, at
the bottom, because the record's owner, the reports' authors and the request
files' addressees are all judged by it and there must be one answer.

This module imports nothing from ``osprey`` and holds no config: it runs
inside connector-host children, executor sandboxes and notebook kernels,
where the dependency budget is the lean connector chain. The import direction
is one-way — the posture store below it, never the reverse at module scope.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey_connectors import session_store
from osprey_connectors.types import CONTROL_TARGETS

logger = logging.getLogger("osprey_connectors.control_context")

__all__ = [
    "REPORT_APPLIED",
    "REPORT_APPLYING",
    "REPORT_FAILED",
    "OWNER_CONTROLS_SERVER",
    "OWNER_KINDS",
    "OWNER_WEB_TERMINAL",
    "RECORD_FILENAME",
    "REPORT_FILE_GLOB",
    "REPORT_FILE_PREFIX",
    "REPORT_FILE_SUFFIX",
    "SCHEMA_VERSION",
    "STATE_DIR_NAME",
    "SWITCH_APPLIED",
    "SWITCH_REFUSED",
    "ControlContext",
    "Owner",
    "ServerReport",
    "blocking_pids",
    "invalidate_cache",
    "is_process_alive",
    "live_owner",
    "live_report_payloads",
    "live_reports",
    "owned_here",
    "parse_owner",
    "parse_posture",
    "converged",
    "file_signature",
    "parse_record",
    "parse_report",
    "read_record",
    "read_report",
    "read_report_payload",
    "record_path",
    "record_path_under",
    "report_path",
    "report_path_under",
    "report_paths",
    "state_dir",
    "sweep_dead",
    "terminus",
    "write_json_atomic",
    "write_record",
    "write_terminus",
]

#: The payload version. There is exactly one, and a payload that does not
#: spell it is not this record: hooks and record change together, and a
#: deployment is regenerated rather than migrated.
SCHEMA_VERSION = 1

#: The record's filename inside :func:`state_dir`. The stdlib hooks restate
#: this literal.
RECORD_FILENAME = "control_context.json"

#: The directory the record shares with the posture store, the per-server
#: reports and the request files. Taken from the store rather than re-spelled
#: so one directory cannot be two directories.
STATE_DIR_NAME = session_store.STATE_DIR_NAME

#: The two kinds of process that can own the record. A web terminal outranks a
#: controls server — it is the one an operator is looking at.
OWNER_WEB_TERMINAL = "web_terminal"
OWNER_CONTROLS_SERVER = "controls_server"
OWNER_KINDS = frozenset({OWNER_WEB_TERMINAL, OWNER_CONTROLS_SERVER})

#: The two termini a switch request can reach in the record. ``applied`` moved
#: the target and minted a generation; ``refused`` moved nothing.
SWITCH_APPLIED = "applied"
SWITCH_REFUSED = "refused"

#: How far one controls server has got through the switch the record names.
#: ``applying`` is a swap in flight and stops every session while its bound
#: holds; ``applied`` is arrival, which the binding states as well; ``failed``
#: is a swap that will not land without an operator, and stops only the session
#: whose server it is. Distinct from the record's :data:`SWITCH_APPLIED` /
#: :data:`SWITCH_REFUSED`, which are a request's terminus rather than a
#: server's progress.
REPORT_APPLYING = "applying"
REPORT_APPLIED = "applied"
REPORT_FAILED = "failed"

#: One report file per controls server, in :func:`state_dir` beside the record.
#: The PID in the name is the address and is what makes a report left behind by
#: a killed server sweepable without opening it.
REPORT_FILE_PREFIX = "server_"
REPORT_FILE_SUFFIX = ".json"
REPORT_FILE_GLOB = f"{REPORT_FILE_PREFIX}*{REPORT_FILE_SUFFIX}"

#: The files this module has parsed, each against the signature that produced
#: it: the record, and one entry per server report. Keyed by path, because a
#: reader holds several of these files at once and a change to one of them is
#: not a reason to re-parse the others.
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[tuple[int, int, int], Any]] = {}

#: How many parsed files the cache holds before it is dropped whole. Reports
#: are named for PIDs, so a long-lived reader would otherwise keep an entry per
#: server that ever ran; dropping the lot costs one re-read each and cannot go
#: stale, because every read re-stats before it trusts what it holds.
_CACHE_MAX_ENTRIES = 128


# -- the record -------------------------------------------------------------


@dataclass(frozen=True)
class Owner:
    """The process currently allowed to write the record.

    Attributes:
        kind: One of :data:`OWNER_KINDS`.
        pid: The owning process's PID — how every other process decides
            whether the owner is still alive.
        port: The web terminal's port, so a follower can name where the owner
            is in a refusal an operator has to act on. ``None`` for a controls
            server, which serves nothing an operator can open.
    """

    kind: str
    pid: int
    port: int | None = None

    def to_payload(self) -> dict[str, Any]:
        """This owner as the record's ``owner`` object."""
        return {"kind": self.kind, "pid": self.pid, "port": self.port}


@dataclass(frozen=True)
class ControlContext:
    """One deployment's control context. Treat an instance as immutable.

    :func:`read_record` hands back the cached instance, so a caller that needs
    a changed record builds one with :func:`dataclasses.replace` rather than
    editing this one — an in-place edit would reach every other reader in the
    process.

    Attributes:
        target: The control target the deployment is pointed at.
        generation: How many times that has moved. Minted by the owner and
            by nobody else; a reader pins its work to it.
        owner: The process allowed to write this file, or ``None`` when the
            record is ownerless and free to be claimed.
        posture: Per-target narrowings, ``{target: "sandbox"}``. Only ever
            refuses; the deployment ceiling is elsewhere.
        last_switch: The terminus of the last switch request, verbatim, or
            ``None`` when no request has reached one.
    """

    target: str
    generation: int
    owner: Owner | None = None
    posture: dict[str, str] = field(default_factory=dict)
    last_switch: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """This record as the JSON object :func:`write_record` stores.

        The five contract fields and the schema, and nothing else: a payload
        round-trips through :func:`parse_record` unchanged, and a key some
        future version added does not survive a rewrite by this one.
        """
        return {
            "schema": SCHEMA_VERSION,
            "owner": None if self.owner is None else self.owner.to_payload(),
            "target": self.target,
            "generation": self.generation,
            "posture": dict(self.posture),
            "last_switch": None if self.last_switch is None else dict(self.last_switch),
        }


# -- path resolution --------------------------------------------------------


def state_dir() -> Path | None:
    """The directory the record lives in, or ``None`` when unresolvable.

    Rule 1 of the module contract, delegated whole to the posture store so
    that the two files in this directory cannot resolve their root two ways.
    """
    return session_store.state_dir()


def record_path_under(root: Path) -> Path:
    """The record's path under an explicit agent-data *root*.

    The one place the two path hops are spelled, for callers that hold a root
    rather than an environment — test fixtures, and any writer preparing a
    deployment before it is stamped.
    """
    return root / STATE_DIR_NAME / RECORD_FILENAME


def record_path() -> Path | None:
    """The record's path, or ``None`` when the agent-data root is unresolvable.

    ``None`` is not an error: for a reader it is indistinguishable from an
    empty deployment, and for a writer it is what a route reports as
    ``store_unavailable``.
    """
    directory = state_dir()
    return None if directory is None else directory / RECORD_FILENAME


# -- parsing ----------------------------------------------------------------


def _positive_int(value: Any) -> int | None:
    """*value* as a positive integer, or ``None``.

    ``bool`` is excluded on purpose: it is an ``int`` in Python, and a payload
    carrying ``true`` where a PID belongs states nothing about a process.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


def parse_owner(value: Any) -> Owner | None:
    """One ``owner`` object as an :class:`Owner`, or ``None`` when ownerless.

    Rule 2 of the module contract for this field: an owner has to name a kind
    this version knows and a plausible PID, because the answer decides whether
    another process defers to it or takes the record over. Anything less is
    ownerless — the claimable state — rather than an error.
    """
    if not isinstance(value, dict):
        return None
    kind = value.get("kind")
    pid = _positive_int(value.get("pid"))
    if kind not in OWNER_KINDS or pid is None:
        return None
    return Owner(kind=kind, pid=pid, port=_positive_int(value.get("port")))


def parse_posture(value: Any) -> dict[str, str]:
    """One ``posture`` value as ``{target: "sandbox"}`` — narrowings only.

    The posture store's entry grammar (its rule 2), applied by the store's own
    parser rather than restated here: this record now holds the entry that
    file used to hold per session, and two filters that disagree about which
    narrowings survive is a narrowing that silently does not apply.
    """
    return session_store.parse_store({"posture": value}).get("posture", {})


def parse_record(raw: Any) -> ControlContext | None:
    """Decode a record, or ``None`` when it states no usable context.

    Accepts the decoded JSON object or the raw text/bytes of one, so a reader
    that has already read the file and one that has not share this filter.
    Never raises: rule 2 of the module contract says every degraded outcome —
    missing, unparseable, wrong schema, unusable identity — arrives as the one
    value, because readers act on "no record" and must not have to tell the
    failures apart.
    """
    if isinstance(raw, str | bytes | bytearray):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001 — an unreadable record is "no record"
            logger.warning("Control-context record is not valid JSON; ignoring")
            return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema") is not SCHEMA_VERSION:
        # Identity comparison, so a JSON ``true`` cannot pass for schema 1.
        return None
    target = raw.get("target")
    if target not in CONTROL_TARGETS:
        return None
    generation = raw.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        return None
    last_switch = raw.get("last_switch")
    return ControlContext(
        target=target,
        generation=generation,
        owner=parse_owner(raw.get("owner")),
        posture=parse_posture(raw.get("posture")),
        last_switch=dict(last_switch) if isinstance(last_switch, dict) else None,
    )


# -- reading ----------------------------------------------------------------


def file_signature(path: Path | None) -> tuple[int, int, int] | None:
    """``(mtime_ns, size, inode)`` for *path*, or ``None`` when it is absent.

    Rule 4 of the module contract, and the same answer for everything that
    watches one of these files: the cache below re-parses on it, and the
    reconciler wakes on it. The inode is the third element because
    :func:`write_json_atomic` replaces the file rather than rewriting it, so
    two switches inside one filesystem clock tick differ there and nowhere
    else. An unresolvable path has no signature, which reads as "no file".
    """
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _read_cached(path: Path, parse: Callable[[Path], Any]) -> Any:
    """*path* through *parse*, re-read only when its signature moved. Rule 4.

    An absent file drops whatever the cache held for that path: a report whose
    server exited and whose file has been swept must not go on being answered
    from memory.
    """
    key = str(path)
    signature = file_signature(path)
    if signature is None:
        with _CACHE_LOCK:
            _CACHE.pop(key, None)
        return None
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    value = parse(path)
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX_ENTRIES:
            _CACHE.clear()
        _CACHE[key] = (signature, value)
    return value


def _read(path: Path) -> ControlContext | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        # ``UnicodeDecodeError`` is a ``ValueError`` rather than an ``OSError``
        # and so needs naming: a record that is not UTF-8 is unreadable for the
        # same reason a corrupt one is, and every reader here answers that with
        # "no record" instead of an exception raised inside a write path.
        logger.warning("Could not read the control-context record at %s", path, exc_info=True)
        return None
    return parse_record(raw)


def read_record(*, path: Path | None = None) -> ControlContext | None:
    """The deployment's control context, or ``None`` when there is none.

    ``None`` for every degraded outcome (rule 2): no agent-data root, no file,
    an unreadable or corrupt file, a payload from another schema, or one whose
    target or generation cannot be used. Callers fall back to the deployment
    baseline or refuse, and none of them has to tell those cases apart.

    Re-stats on every call and re-parses only when the signature moved (rule
    4). The returned record is the cached instance — treat it as immutable.

    Args:
        path: Read this file instead of :func:`record_path`. For a caller that
            has already resolved a root, and for tests.
    """
    target_path = record_path() if path is None else path
    if target_path is None:
        return None
    record: ControlContext | None = _read_cached(target_path, _read)
    return record


def invalidate_cache() -> None:
    """Forget every parsed file. For tests and for a process that moved roots."""
    with _CACHE_LOCK:
        _CACHE.clear()


# -- writing ----------------------------------------------------------------


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* to *path* as JSON, replacing any existing file. Rule 3.

    The temp file shares the target's directory — ``os.replace`` is atomic
    only within one filesystem — and is removed if the dump fails, so a failed
    write leaves neither half a file nor litter beside it.

    Every file in :func:`state_dir` is written this way, the record and the
    per-server reports and requests alike: they are read by processes that
    never coordinate with the writer, so a reader must never be able to see
    half of one.

    Raises:
        OSError: The write itself failed.
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
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


def write_record(record: ControlContext, *, path: Path | None = None) -> Path:
    """Store *record*, replacing any existing one atomically. Rule 3.

    This is a bare write. It does not read the file first, does not check who
    owns it, and does not serialise against a concurrent writer: the owner
    does all three around this call, because only the owner knows what it is
    changing and whether it is still entitled to.

    Args:
        record: The context to store.
        path: Write this file instead of :func:`record_path`.

    Returns:
        The path written.

    Raises:
        RuntimeError: The agent-data root does not resolve, so there is
            nowhere a reader would look for what this call would write.
        OSError: The write itself failed.
    """
    target_path = record_path() if path is None else path
    if target_path is None:
        raise RuntimeError(
            "No agent-data root: the control-context record has nowhere to be written"
        )
    write_json_atomic(target_path, record.to_payload())
    return target_path


# -- process liveness -------------------------------------------------------


def is_process_alive(pid: object) -> bool:
    """Whether *pid* names a running process.

    ``os.kill(pid, 0)`` sends no signal and only asks the kernel whether the
    process exists — the ``osprey web`` PID-file check does the same. A
    ``PermissionError`` means the process exists but belongs to another user, so
    it counts as ALIVE: sweeping a file whose owner is merely unreachable would
    delete live state. Non-positive PIDs are rejected without calling
    ``os.kill`` at all, because 0 and negatives address process *groups*.

    Anything that is not an ``int`` names no process and is ``False`` without
    a call either, so a field read off a file can be handed over unconverted.
    That includes ``bool``: ``True`` is ``1`` to ``os.kill``, and PID 1 is
    always alive, so a report carrying ``"server_pid": true`` would otherwise
    read as a running server forever.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
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


def _pid_from_filename(name: str, prefix: str, suffix: str) -> int | None:
    """The PID *name* encodes, or ``None`` when it encodes none.

    Shared by every family of file in :func:`state_dir`: each is named for the
    process that owns it precisely so a sweeper can judge it without opening
    it, and a name that does not follow the rule is judged the same way in all
    of them — no PID, therefore no live owner.
    """
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    stem = name[len(prefix) : -len(suffix)]
    try:
        return int(stem)
    except ValueError:
        return None


# -- ownership --------------------------------------------------------------


def owned_here(record: ControlContext | None) -> bool:
    """Whether THIS process owns *record*, and may therefore write it.

    PID equality and nothing else: the record names one owning process, and a
    second rule about the owner's *kind* would let two processes disagree
    about who writes one file. An ownerless record is owned by nobody, so it
    is not owned here either — it is claimed first, and claiming is a write.
    """
    return record is not None and record.owner is not None and record.owner.pid == os.getpid()


def live_owner(record: ControlContext | None) -> Owner | None:
    """*record*'s owner while its process still runs, else ``None``.

    The one question every would-be claimant asks first: an owner whose
    process is gone holds nothing, and its record is free. ``None`` covers
    every way of having no live owner — no record, no owner, a dead one — so
    a caller that only wants to know whether to claim need not tell them
    apart.
    """
    if record is None or record.owner is None:
        return None
    return record.owner if is_process_alive(record.owner.pid) else None


# -- the terminus of a switch ----------------------------------------------


def terminus(
    record: ControlContext,
    *,
    request_id: str,
    target: str,
    requested_at: Any,
    requested_by: str,
    status: str,
    reason: str | None,
    detail: str,
    generation: int | None,
) -> ControlContext:
    """*record* as it will be once a switch request has been answered.

    Every answer to a switch gesture ends here, whoever writes it: the owning
    server answering its own agent, and the reconciler answering a request
    another process filed. The block is what a requester polls for — it looks
    for its own ``request_id`` and reads the outcome beside it — so the shape
    is fixed here rather than at each terminus.

    *generation* is what the record carries AFTER the answer, and ``None`` is
    what makes the answer a refusal: a refusal moves neither target nor
    generation, and its block names no binding because there is none.

    Args:
        record: The record being answered against. Not modified.
        request_id: The id the requester is waiting on.
        target: The target that was asked for — recorded whether or not the
            deployment went there, because a refusal has to say what it
            refused.
        requested_at: The request's own stamp, copied verbatim.
        requested_by: Who to name in the block: an audit session where the
            requester has one, ``pid:<n>`` where it does not.
        status: :data:`SWITCH_APPLIED` or :data:`SWITCH_REFUSED`.
        reason: The refusal's machine-readable reason, ``None`` when applied.
        detail: The sentence an operator reads.
        generation: The record's generation after the answer, or ``None`` for
            a refusal.
    """
    moved = generation is not None
    return replace(
        record,
        target=target if moved else record.target,
        generation=generation if moved else record.generation,
        last_switch={
            "request_id": request_id,
            "target": target,
            "requested_at": requested_at,
            "requested_by": requested_by,
            "status": status,
            "reason": reason,
            "detail": detail,
            "generation": generation,
        },
    )


def write_terminus(updated: ControlContext, request_id: str) -> bool:
    """Write *updated*, read it back, and say whether the answer stuck.

    Read-verified because the record has one owner but two processes reach for
    it: a server answers its own agent in one place and consumes other
    processes' requests in another, and this call can also lose the record
    outright to a new owner between the write and the read. A lost update has
    to be discovered here rather than reported to an agent as a switch that
    happened.

    Three things have to hold for the answer to be believed: the record is
    still readable, this process still owns it, and the block in it is the one
    just written. The caller decides what a ``False`` means for it — the owner
    refuses its agent, the reconciler leaves the request file for the next
    pass — and neither may treat the switch as done.
    """
    try:
        write_record(updated)
    except Exception:
        logger.warning("Could not write the control-context record", exc_info=True)
        return False
    confirmed = read_record()
    return (
        confirmed is not None
        and owned_here(confirmed)
        and (confirmed.last_switch or {}).get("request_id") == request_id
    )


# -- the per-server reports -------------------------------------------------


@dataclass(frozen=True)
class ServerReport:
    """What one controls server says about itself. Treat an instance as immutable.

    Rule 5 of the module contract. Every field but ``server_pid`` is an
    observation that can be absent — a server that has just started has
    answered nothing yet — so each degrades on its own to a value a reader can
    act on rather than invalidating the report.

    Attributes:
        server_pid: The reporting server. Its identity, and how a reader knows
            whether the report still describes anything.
        session: The server's ``OSPREY_POSTURE_SESSION``, or ``None`` for a
            bare ``claude`` that was never stamped with one.
        applied_target: The target its connector host is actually on, or
            ``None`` until the first child has answered its init frame. Null is
            not "baseline": it is "this server has not got there yet".
        applied_generation: The generation that target was reached at, on the
            same terms.
        children: The connector-host PIDs it spawned, so a later server can
            reap them if this one is killed.
        reachability: Its last probe per target, ``{}`` when it has not probed.
        last_switch: Its progress through the current switch — ``applying``,
            ``applied`` or ``failed``, with the ``expires_at`` bound only this
            process could compute — or ``None``.
        last_posture_realign: Its last posture realignment, or ``None``.
        targets: Per-target display metadata, ``{}`` when it has none.
        updated_at: When it last wrote, ISO-8601, or ``None``.
    """

    server_pid: int
    session: str | None = None
    applied_target: str | None = None
    applied_generation: int | None = None
    children: tuple[int, ...] = ()
    reachability: dict[str, Any] = field(default_factory=dict)
    last_switch: dict[str, Any] | None = None
    last_posture_realign: dict[str, Any] | None = None
    targets: dict[str, Any] = field(default_factory=dict)
    updated_at: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """This report as the JSON object its file holds.

        The ten contract fields and nothing else, so a payload round-trips
        through :func:`parse_report` unchanged.
        """
        return {
            "server_pid": self.server_pid,
            "session": self.session,
            "applied_target": self.applied_target,
            "applied_generation": self.applied_generation,
            "children": list(self.children),
            "reachability": dict(self.reachability),
            "last_switch": None if self.last_switch is None else dict(self.last_switch),
            "last_posture_realign": (
                None if self.last_posture_realign is None else dict(self.last_posture_realign)
            ),
            "targets": dict(self.targets),
            "updated_at": self.updated_at,
        }


def _non_negative_int(value: Any) -> int | None:
    """*value* as a non-negative integer, or ``None``. ``bool`` excluded."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _text(value: Any) -> str | None:
    """*value* as a non-empty string, or ``None``.

    An empty string states nothing, and a reader that compared one to a session
    id would be comparing to a value no writer here ever means.
    """
    return value if isinstance(value, str) and value else None


def _child_pids(value: Any) -> tuple[int, ...]:
    """*value* as the positive child PIDs it names, in order, without repeats."""
    if not isinstance(value, list | tuple):
        return ()
    pids: list[int] = []
    for item in value:
        pid = _positive_int(item)
        if pid is not None and pid not in pids:
            pids.append(pid)
    return tuple(pids)


def _mapping(value: Any) -> dict[str, Any]:
    """*value* as a mapping, or an empty one. Anything else states nothing."""
    return dict(value) if isinstance(value, dict) else {}


def parse_report(raw: Any) -> ServerReport | None:
    """Decode one server report, or ``None`` when it describes no server.

    Rule 5. ``server_pid`` is the identity and the only field that can make the
    whole report worthless: without a plausible PID there is nothing to ask the
    kernel about, so the file says nothing a reader can use. Everything else
    degrades to its own "not stated" value.
    """
    if not isinstance(raw, dict):
        return None
    server_pid = _positive_int(raw.get("server_pid"))
    if server_pid is None:
        return None
    applied_target = raw.get("applied_target")
    last_switch = raw.get("last_switch")
    last_realign = raw.get("last_posture_realign")
    return ServerReport(
        server_pid=server_pid,
        session=_text(raw.get("session")),
        applied_target=applied_target if applied_target in CONTROL_TARGETS else None,
        applied_generation=_non_negative_int(raw.get("applied_generation")),
        children=_child_pids(raw.get("children")),
        reachability=_mapping(raw.get("reachability")),
        last_switch=dict(last_switch) if isinstance(last_switch, dict) else None,
        last_posture_realign=dict(last_realign) if isinstance(last_realign, dict) else None,
        targets=_mapping(raw.get("targets")),
        updated_at=_text(raw.get("updated_at")),
    )


def report_path_under(root: Path, server_pid: object) -> Path:
    """The report path of *server_pid* under an explicit agent-data *root*."""
    return root / STATE_DIR_NAME / f"{REPORT_FILE_PREFIX}{server_pid}{REPORT_FILE_SUFFIX}"


def report_path(server_pid: object) -> Path | None:
    """The report path of *server_pid*, or ``None`` without an agent-data root."""
    directory = state_dir()
    if directory is None:
        return None
    return directory / f"{REPORT_FILE_PREFIX}{server_pid}{REPORT_FILE_SUFFIX}"


def report_paths() -> list[Path]:
    """Every report file in :func:`state_dir`, sorted. Empty when there is none.

    Never raises: no agent-data root, no directory and an unreadable one all
    answer "no reports", the same as an empty deployment, because a caller
    describing a fleet has the same nothing to say in each case.
    """
    directory = state_dir()
    if directory is None:
        return []
    try:
        return sorted(directory.glob(REPORT_FILE_GLOB))
    except OSError:  # pragma: no cover - unreadable state dir
        return []


def _read_payload(path: Path) -> dict[str, Any] | None:
    """One report file as its JSON object, or ``None``. Never raises."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.debug("Could not read the server report at %s; skipping it", path)
        return None
    try:
        loaded = json.loads(raw)
    except Exception:  # noqa: BLE001 — a half-written report is skipped, not fatal
        logger.debug("Server report %s is not valid JSON; skipping it", path.name)
        return None
    return loaded if isinstance(loaded, dict) else None


def _read_entry(path: Path) -> tuple[dict[str, Any], ServerReport | None] | None:
    """One report file read and parsed once, for the cache to hold as a pair.

    Both layers land in the same cache slot so that a caller reading the
    payload and a caller reading the report neither open the file twice nor
    parse it twice, and so that the parsed report a reader holds is the same
    object until the file actually moves.
    """
    payload = _read_payload(path)
    if payload is None:
        return None
    return (payload, parse_report(payload))


def read_report_payload(path: Path) -> dict[str, Any] | None:
    """One report file as the object it holds, ``None`` when it holds none.

    The file layer under :func:`read_report`: the payload verbatim, including
    keys this version does not know, for a caller that is still reading a
    report by hand. Signature-cached like every read here (rule 4).
    """
    entry: tuple[dict[str, Any], ServerReport | None] | None = _read_cached(path, _read_entry)
    return None if entry is None else entry[0]


def read_report(path: Path) -> ServerReport | None:
    """One server's report, or ``None`` when the file states no usable one."""
    entry: tuple[dict[str, Any], ServerReport | None] | None = _read_cached(path, _read_entry)
    return None if entry is None else entry[1]


def live_report_payloads(
    entries: Iterable[Path],
    *,
    is_alive: Callable[[object], bool] = is_process_alive,
) -> list[dict[str, Any]]:
    """The payload of every report among *entries* whose server still runs.

    A report whose ``server_pid`` is gone is residue: it describes a server
    nobody is talking to any more, and the next server to start sweeps it
    (:func:`sweep_dead`). Unreadable entries are skipped rather than raised.

    Args:
        entries: Report paths, typically :func:`report_paths`. Taken rather
            than globbed here because a caller keying a cache on the listing
            must not glob twice.
        is_alive: The liveness predicate, so one pass over a fleet asks the
            same question the same way throughout.
    """
    live: list[dict[str, Any]] = []
    for entry in entries:
        payload = read_report_payload(entry)
        if payload is None:
            continue
        pid = _positive_int(payload.get("server_pid"))
        if pid is None or not is_alive(pid):
            continue
        live.append(payload)
    return live


def live_reports(
    entries: Iterable[Path] | None = None,
    *,
    is_alive: Callable[[object], bool] = is_process_alive,
) -> list[ServerReport]:
    """Every report of a running controls server, parsed.

    The fleet as a reader sees it: what each live server is on, at which
    generation, and how far through a switch it has got.

    Args:
        entries: Report paths; :func:`report_paths` when omitted.
        is_alive: As for :func:`live_report_payloads`.
    """
    paths = report_paths() if entries is None else entries
    reports: list[ServerReport] = []
    for entry in paths:
        report = read_report(entry)
        if report is None or not is_alive(report.server_pid):
            continue
        reports.append(report)
    return reports


# -- sweeping ---------------------------------------------------------------


def sweep_dead(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
    keep: str | None = None,
    salvage: Callable[[Path], None] | None = None,
    is_alive: Callable[[object], bool] = is_process_alive,
) -> list[Path]:
    """Unlink every ``prefix<pid>suffix`` file in *directory* whose PID is gone.

    A process that is killed rather than asked to stop leaves its file behind,
    and a later process that happens to reuse the PID would otherwise inherit
    it. The name is the whole judgement — a file whose name encodes no PID
    encodes no owner either, and goes the same way.

    Args:
        directory: Where to look. A missing or unreadable one sweeps nothing.
        prefix: Filename prefix of the family to sweep.
        suffix: Filename suffix of the family to sweep.
        keep: A filename to leave alone without probing it. This is how a
            process spares its own file: asking whether oneself is alive is a
            way to get it wrong.
        salvage: Called with each doomed file before it is unlinked, for a
            caller that needs what it said — the child PIDs a dead server left
            running are read here and nowhere else.
        is_alive: The liveness predicate.

    Returns:
        The paths removed, in discovery order.
    """
    try:
        entries = sorted(directory.glob(f"{prefix}*{suffix}"))
    except OSError:
        return []

    removed: list[Path] = []
    for entry in entries:
        if keep is not None and entry.name == keep:
            continue
        pid = _pid_from_filename(entry.name, prefix, suffix)
        if pid is not None and is_alive(pid):
            continue
        if salvage is not None:
            salvage(entry)
        try:
            entry.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - unwritable state dir
            logger.warning("Could not remove stale file %s: %s", entry, exc)
            continue
        logger.info("Swept stale file %s (owner pid %s gone)", entry.name, pid)
        removed.append(entry)
    return removed


# -- convergence ------------------------------------------------------------


def _switch_at(report: ServerReport, generation: int) -> dict[str, Any] | None:
    """*report*'s switch block when it is about *generation*, else ``None``.

    A block that names a different generation is about a swap this record has
    already moved past — the server re-adopts on its next tick — and a block
    that names none names no swap the fleet is coordinating on. Neither is
    this record's business; a server left behind by either is still judged by
    what it is bound to.
    """
    block = report.last_switch
    if not isinstance(block, dict):
        return None
    if _non_negative_int(block.get("generation")) != generation:
        return None
    return block


def _bound_epoch(block: dict[str, Any]) -> float | None:
    """The ``expires_at`` of *block* as epoch seconds, or ``None`` for no bound.

    The bound is a wall clock the publishing server computed from its own
    spawn, probe and drain timeouts — the only process that knows them — so
    this reader parses it and does not derive one. A naive stamp is read as
    UTC, the same rule the in-flight markers use, because the writer and the
    reader share a host even when they do not share a spelling.

    ``None`` is "no bound", which keeps blocking rather than expiring
    immediately: an unreadable deadline is not a reason to decide a swap
    finished.
    """
    stamp = block.get("expires_at")
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        moment = datetime.fromisoformat(stamp)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.timestamp()
    except (ValueError, OSError, OverflowError):
        return None


def _bound_off_record(report: ServerReport, record: ControlContext) -> bool:
    """Whether *report* says it arrived somewhere other than the record says.

    Both halves of the binding have to be stated and both have to match:
    ``applied_target`` and ``applied_generation`` are published together but
    parse apart, and null in either is "this server has not got there yet",
    never "baseline". A server that has not got there yet is behind, not
    wrong, and holds nothing an operator has to clear.
    """
    if report.applied_target is None or report.applied_generation is None:
        return False
    return (report.applied_target, report.applied_generation) != (record.target, record.generation)


def blocking_pids(
    record: ControlContext,
    reports: Iterable[ServerReport],
    session: str | None,
    *,
    now: float | None = None,
) -> tuple[int, ...]:
    """The controls servers that stop *session* acting on *record*, PID order.

    Empty when there is nothing in the way — which is what :func:`converged`
    reports. The PIDs are the refusal text: ``switch_in_progress`` names them
    so an operator can find the process that has to finish or be killed, and
    the two callers that refuse (the executor's stamp and the kernel's cell
    gate) both say which.

    Args:
        record: The deployment's control context. The generation in it is what
            the fleet coordinates on.
        reports: The reports of the servers that are still running, as
            :func:`live_reports` returns them. A report of a dead server
            describes nobody and is filtered out there, not here: this
            function opens no file and asks the kernel nothing.
        session: The reader's ``OSPREY_POSTURE_SESSION``. ``None`` (and the
            empty string) for a reader that owns no report — a bare ``claude``,
            a notebook kernel's ``kernel:<id>``, and the owner asking whether
            the fleet as a whole has settled.
        now: Epoch seconds to age the ``applying`` bounds against; the wall
            clock when omitted.
    """
    fleet = tuple(reports)
    reference = datetime.now(UTC).timestamp() if now is None else float(now)
    blocked: set[int] = set()
    for report in fleet:
        block = _switch_at(report, record.generation) or {}
        status = block.get("status")
        if status == REPORT_APPLYING:
            bound = _bound_epoch(block)
            if bound is None or reference <= bound:
                # A swap in flight: nobody launches anywhere until it lands,
                # because the server holding the connector is between two
                # targets. Past its bound it has stopped being in flight, and
                # only the session whose server it is is still held up by it.
                blocked.add(report.server_pid)
                continue
        if not session or report.session != session:
            continue
        if status in (REPORT_APPLYING, REPORT_FAILED) or _bound_off_record(report, record):
            blocked.add(report.server_pid)
    return tuple(sorted(blocked))


def converged(
    record: ControlContext,
    reports: Iterable[ServerReport],
    session: str | None,
    *,
    now: float | None = None,
) -> bool:
    """Whether *session* may act on *record* — rule 6 of the module contract.

    False in exactly two situations, and the arguments are the same in both:
    :func:`blocking_pids` names the servers behind them.

    **A swap is in flight.** Any live server reporting ``applying`` for the
    record's generation, within the ``expires_at`` bound it wrote, stops every
    session: that server is between two targets, and a launch admitted now
    would reach whichever one it happens to be holding. Past the bound the
    server is stuck rather than working, and it stops only its own session —
    a process killed mid-swap must not refuse a deployment for ever.

    **The reader's own server is not there.** A live report carrying this
    *session* that is bound somewhere other than the record's
    ``(target, generation)``, or that reports ``failed`` at that generation,
    stops that session and nothing else. That is the whole of what a failed
    swap costs: the session whose server could not follow is refused, its own
    MCP writes with it, while every other session, every notebook kernel and
    every bare ``claude`` carries on.

    A session that owns no report — ``None``, the empty string, a kernel's
    ``kernel:<id>`` — is therefore blocked only by the first situation. So is
    the owner's own check before it consumes a switch request, which passes
    ``None`` deliberately: it is asking whether the fleet has settled, not
    whether some session has.

    This is a judgement over parsed data and nothing else. It opens no file,
    signals no process and imports nothing from ``osprey``, because the
    executor stamps every sandbox with it and a notebook kernel gates every
    cell on it.
    """
    return not blocking_pids(record, reports, session, now=now)
