"""The control-context record — one file per identity.

Which machine a deployment is pointed at, how many times that has moved, and
what one operator has narrowed are properties of the deployment as that
operator holds it, not of any process tree inside it. They live in a single
JSON file, :data:`RECORD_FILENAME` inside the acting identity's directory
under :data:`STATE_DIR_NAME` (:func:`state_dir`), and this module is where it
is written and read.

The file has exactly one writer at a time — its **owner**: the web terminal
when one is running, otherwise a controls MCP server. Everything else is a
reader: every chat session, every notebook kernel, every
executor sandbox, every sibling MCP server, and the stdlib-only hooks, which
cannot import this module and therefore restate the rules below. Treat
them as the contract; a change here is a change there.

**1. Where the file is.** :func:`record_path` — the record sits in the same
directory as the posture store, and the root resolves by that store's rule 1
(:func:`osprey_connectors.posture_store.agent_data_root`) rather than by a
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

The other fields are annotations on that identity and degrade on their
own, because each has a meaningful "not stated" value that a reader can act
on safely:

* ``owner`` → ``None``, which is *ownerless* — the state a claim is for. An
  owner that does not name one of :data:`_OWNER_KINDS` and a live PID is not
  an owner anybody should defer to, so reading it as ownerless is what lets
  the next process take the file over instead of stalling behind a ghost.
* ``posture`` → ``{}``. A mirror of ``switch`` kept for old readers, never
  an input: :meth:`ControlContext.to_payload` writes ``sandbox`` for every
  machine in :data:`~osprey_connectors.types.CONTROL_TARGETS` whose position
  is off (:func:`posture_mirror`), whatever the record it was built
  from carried, and no write decision reads it back. A file written before
  the mirror may still hold an old narrowing; it is parsed by the posture
  store's grammar (rule 2 there) and ignored by the verdict.
* ``last_switch`` → ``None``. It is the terminus of the last switch request
  — applied or refused — and it is carried verbatim. This module owns the
  block's shape, because a requester polls it for its own ``request_id``
  (:func:`terminus` builds it, :func:`write_terminus` stores it and reads it
  back); the switch machinery owns what one says. **A refusal is a record
  write that moves nothing else**: the same target, the same generation, the
  same posture, a new ``last_switch``. That is the whole of how a refusal is
  reported.
* ``switch`` → ``{}``, every machine's writes off. A per-target map of which
  only ``"asked"`` and ``"allowed"`` leaves survive, and a target absent from
  it is off. The legacy ``"on"`` reads as ``"asked"``, and any write stores the
  whole map in the new spelling; any other value is dropped on parse, so a
  degraded switch can only refuse. :func:`switch_position` is the one reader of
  the map.
* ``switch_epoch`` → ``None``, a record written before epochs existed. An
  epoch is a ``time.time_ns()`` stamp that only moves forward; the person who
  widens a switch names the epoch they saw, so a stop or a lost record in
  between makes their gesture stale. ``None`` is never "older" than anything.

The schema stays :data:`SCHEMA_VERSION` with the two switch fields added:
every pre-existing reader, the stdlib hooks included, ignores keys it does
not know, so a record written by this version reads unchanged through them.

**3. How it is written.** :func:`write_json_atomic` replaces a file rather
than rewriting it, whole through ``safety_file.atomic_replace``, so no reader ever sees half of one and two writers never interleave into one.
Every file in this directory is written through it, the record
(:func:`write_record`) and the per-server reports and requests alike.

The record is written only under its lock: an exclusive ``fcntl.flock`` on
the sibling file ``<record path>.lock``, derived from the path being written
(so ``path=`` moves the lock with the file), waited on for a bounded time and
then refused with ``RuntimeError``. Inside the lock :func:`write_record`
re-reads the file on disk and writes the caller's record with the on-disk
``switch`` and ``switch_epoch`` substituted: whatever a caller read earlier,
no writer but the person's own switch gesture (:func:`set_switch`,
which reads, compares and writes in one hold) moves either field, so a
target switch or a refusal terminus racing a stop cannot turn a machine
back on. A write that finds the record missing or unreadable writes the
switch empty and stamps a fresh epoch, never 0. What a caller changes in
the other fields, and whether it is still entitled to, remains the owner's
business.

The exclusion holds among processes sharing one kernel — one host, or
containers on a Linux host over a bind mount. Across a macOS host and a VM
file share ``flock`` is not guaranteed to exclude anything; that residual is
closed by the sandboxed connector host, not here.

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
flight stops everyone, a server acquiring its first connector stops only its
own session, a server that could not follow stops only its own session, and
nothing else stops anybody.

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

import contextlib
import fcntl
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from osprey_connectors import posture_store
from osprey_connectors.identity import acting_identity
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.safety_file import atomic_replace
from osprey_connectors.types import CONTROL_TARGETS

logger = logging.getLogger("osprey_connectors.control_context")

__all__ = [
    "claimed",
    "REPORT_APPLIED",
    "REPORT_APPLYING",
    "REPORT_FAILED",
    "OWNER_CONTROLS_SERVER",
    "OWNER_WEB_TERMINAL",
    "RECORD_FILENAME",
    "REPORT_FILE_GLOB",
    "REPORT_FILE_PREFIX",
    "REPORT_FILE_SUFFIX",
    "SCHEMA_VERSION",
    "STATE_DIR_NAME",
    "SWITCH_APPLIED",
    "SWITCH_POSITIONS",
    "SWITCH_REFUSED",
    "ControlContext",
    "Owner",
    "ServerReport",
    "StoppedSince",
    "applied_detail",
    "blocking_pids",
    "invalidate_cache",
    "is_process_alive",
    "live_owner",
    "live_reports",
    "owned_here",
    "parse_posture",
    "posture_mirror",
    "converged",
    "file_signature",
    "parse_record",
    "parse_report",
    "read_record",
    "read_record_status",
    "read_report",
    "record_path",
    "report_paths",
    "set_switch",
    "state_dir",
    "sweep_dead",
    "switch_position",
    "terminus",
    "write_json_atomic",
    "unchanged_detail",
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

#: The tree of per-identity directories the record shares with the posture
#: store, the per-server reports and the request files — one directory per
#: identity, each holding every one of those families. Taken from the store
#: rather than re-spelled so one directory cannot be two directories.
STATE_DIR_NAME = posture_store.STATE_DIR_NAME

#: The two kinds of process that can own the record. A web terminal outranks a
#: controls server — it is the one an operator is looking at.
OWNER_WEB_TERMINAL = "web_terminal"
OWNER_CONTROLS_SERVER = "controls_server"
_OWNER_KINDS = frozenset({OWNER_WEB_TERMINAL, OWNER_CONTROLS_SERVER})

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

#: A machine's three positions. ``off`` is never stored: it is a target absent
#: from the map.
_SWITCH_OFF: Literal["off"] = "off"
_SWITCH_ASKED: Literal["asked"] = "asked"
_SWITCH_ALLOWED: Literal["allowed"] = "allowed"

#: The word a record written before the three positions arms a machine with.
#: Read, and taken by :func:`set_switch`, as :data:`_SWITCH_ASKED`; never written. Every other value a payload
#: carries under ``switch`` is dropped on parse (rule 2).
_SWITCH_LEGACY_ON = "on"

#: The record lock's file, beside the record: ``<record path>`` plus this.
#: Rule 3. Not one of the PID-named families, so :func:`sweep_dead` never
#: touches it.
_LOCK_SUFFIX = ".lock"

#: How long a record write waits for the lock before it refuses. A holder
#: does one read and one atomic replace; seconds of contention is a wedged
#: process, and a write path must answer rather than hang behind it.
_LOCK_TIMEOUT_S = 5.0

#: How often a waiting writer retries the lock.
_LOCK_POLL_S = 0.01


# -- the record -------------------------------------------------------------


@dataclass(frozen=True)
class Owner:
    """The process currently allowed to write the record.

    Attributes:
        kind: One of :data:`_OWNER_KINDS`.
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
        posture: The ``posture`` field as read off disk — the mirror of
            ``switch`` (:func:`posture_mirror`), or an old narrowing in a file
            written before it. A reading only: :meth:`to_payload` ignores it
            and derives the field from ``switch``, so nothing set here is
            ever stored, and no write decision reads it.
        last_switch: The terminus of the last switch request, verbatim, or
            ``None`` when no request has reached one.
        switch: The position the person has put each machine at,
            ``{target: "asked" | "allowed"}``; an absent target is off, and
            :func:`switch_position` is how it is read. :func:`write_record`
            never changes it — it keeps whatever the file on disk says.
        switch_epoch: The ``time.time_ns()`` stamp of the last stop or loss
            of the record, or ``None`` for a record written before epochs
            existed. Kept by :func:`write_record` like ``switch``.
    """

    target: str
    generation: int
    owner: Owner | None = None
    posture: dict[str, str] = field(default_factory=dict)
    last_switch: dict[str, Any] | None = None
    switch: Mapping[str, str] = field(default_factory=dict)
    switch_epoch: int | None = None

    def to_payload(self) -> dict[str, Any]:
        """This record as the JSON object :func:`write_record` stores.

        The seven contract fields and the schema, and nothing else: a payload
        round-trips through :func:`parse_record` unchanged, and a key some
        future version added does not survive a rewrite by this one. The
        ``posture`` field is the mirror of ``switch`` (:func:`posture_mirror`),
        never :attr:`posture`: a narrowing is not something a writer can store.
        """
        return {
            "schema": SCHEMA_VERSION,
            "owner": None if self.owner is None else self.owner.to_payload(),
            "target": self.target,
            "generation": self.generation,
            "posture": posture_mirror(self.switch),
            "last_switch": None if self.last_switch is None else dict(self.last_switch),
            "switch": dict(self.switch),
            "switch_epoch": self.switch_epoch,
        }


def posture_mirror(switch: Mapping[str, str]) -> dict[str, str]:
    """The ``posture`` field a record with *switch* stores — ``{target: "sandbox"}``.

    ``sandbox`` for every machine in
    :data:`~osprey_connectors.types.CONTROL_TARGETS` at off, so an old reader
    of the field refuses every machine the person has not turned on; ``asked``
    and ``allowed`` both store no entry, which is how the old grammar spells
    "writes". Derived on every write and never read by a write decision: the
    switch is the state, this is its shadow in the old grammar.

    Args:
        switch: The record's switch, ``{target: "asked" | "allowed"}``.
    """
    return {
        target: posture_store.POSTURE_SANDBOX
        for target in CONTROL_TARGETS
        if _switch_word(switch.get(target)) is None
    }


def switch_position(
    record: ControlContext | None, target: str
) -> Literal["off", "asked", "allowed"]:
    """Where the person has put *target* — the one reader of the ``switch`` map.

    Never raises. No record, a target absent from the map and an unknown value
    all read as off; a hand-built record still spelling ``"on"`` reads as asked,
    as a parsed one does.

    Args:
        record: The record, or ``None`` when there is none.
        target: The machine.
    """
    if record is None:
        return _SWITCH_OFF
    return _switch_word(record.switch.get(target)) or _SWITCH_OFF


def claimed(record: ControlContext, baseline: str) -> ControlContext:
    """*record* as a process claiming it holds it, given the deployment's *baseline*.

    A record nobody has switched follows the deployment's baseline; every
    applied switch mints ``generation + 1``, so generation 0 is a claim and
    never a switch. An unswitched record on another target moves to
    *baseline* and keeps its generation, owner and last switch terminus. A switched record, or one already
    on *baseline*, is returned as it is.
    """
    if record.generation != 0 or record.target == baseline:
        return record
    logger.warning(
        "control context: unswitched record on %r follows the deployment's baseline %r",
        record.target,
        baseline,
    )
    return replace(record, target=baseline)


# -- path resolution --------------------------------------------------------


def state_dir() -> Path | None:
    """The directory the record lives in, or ``None`` when unresolvable.

    Rule 1 of the module contract — the acting identity's directory, or the
    bind a container carries — delegated whole to the posture store so that
    the files in this directory cannot resolve their directory two ways.
    """
    return posture_store.state_dir()


def _record_path_under(root: Path) -> Path:
    """The record's path under an explicit agent-data *root*.

    The one place the three path hops are spelled, for a caller that holds a
    root rather than an environment and means the file of the identity it is
    itself acting as — a test fixture, and any same-process writer.

    The identity hop is :func:`~osprey_connectors.identity.acting_identity`,
    the same rung :func:`state_dir` ends on, so a caller that holds the root
    this process is stamped with names the file this process reads. A writer
    preparing ANOTHER identity's directory is not that caller and gets the
    wrong path here; it names the identity instead, through
    :func:`osprey.deployment.compose_generator.control_target_identity_dir`.
    The container bind deliberately does not apply: a caller passing a root has
    said which tree it means, and that is a statement about the host layout.
    """
    return root / STATE_DIR_NAME / acting_identity() / RECORD_FILENAME


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


def _parse_owner(value: Any) -> Owner | None:
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
    if kind not in _OWNER_KINDS or pid is None:
        return None
    return Owner(kind=kind, pid=pid, port=_positive_int(value.get("port")))


def parse_posture(value: Any) -> dict[str, str]:
    """One ``posture`` value as ``{target: "sandbox"}`` — narrowings only.

    The posture entry grammar (rule 2 of the store's contract), applied by that
    module's own parser rather than restated here: two filters that disagree
    about which narrowings survive is a narrowing that silently does not apply.
    """
    return posture_store.parse_posture_value(value)


def _switch_word(value: Any) -> Literal["asked", "allowed"] | None:
    """One stored ``switch`` value as the word it arms with, or ``None`` for off.

    Rule 2: an exact match, no folding and no trimming. The legacy ``"on"``
    is ``"asked"``.
    """
    if not isinstance(value, str):
        return None
    if value == _SWITCH_LEGACY_ON:
        return _SWITCH_ASKED
    if value == _SWITCH_ASKED:
        return _SWITCH_ASKED
    if value == _SWITCH_ALLOWED:
        return _SWITCH_ALLOWED
    return None


def _parse_switch(value: Any) -> dict[str, str]:
    """One ``switch`` value as ``{target: "asked" | "allowed"}`` (rule 2).

    Keys are kept whether or not they name a known target, so a turn-off of
    every machine can see and clear them; a reader only ever asks about the
    target it is writing to. Because every write stores what this returned, a
    map spelling ``"on"`` is rewritten in the new words by the next write.
    """
    if not isinstance(value, dict):
        return {}
    return {
        key: word
        for key, state in value.items()
        if isinstance(key, str) and (word := _switch_word(state)) is not None
    }


def _parse_epoch(value: Any) -> int | None:
    """One ``switch_epoch`` as a non-negative integer, or ``None`` (epoch-less)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


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
        except Exception:  # an unreadable record is "no record"
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
        owner=_parse_owner(raw.get("owner")),
        posture=parse_posture(raw.get("posture")),
        last_switch=dict(last_switch) if isinstance(last_switch, dict) else None,
        switch=_parse_switch(raw.get("switch")),
        switch_epoch=_parse_epoch(raw.get("switch_epoch")),
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


#: How a record read ended: a usable record, no file, or a file (or root)
#: that states nothing usable.
_RecordStatus = Literal["ok", "missing", "unreadable"]


def _read(path: Path) -> tuple[ControlContext | None, _RecordStatus]:
    """*path* parsed, with how the read ended. Uncached; never raises."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError):
        # ``UnicodeDecodeError`` is a ``ValueError`` rather than an ``OSError``
        # and so needs naming: a record that is not UTF-8 is unreadable for the
        # same reason a corrupt one is, and every reader here answers that with
        # "no record" instead of an exception raised inside a write path.
        logger.warning("Could not read the control-context record at %s", path, exc_info=True)
        return None, "unreadable"
    record = parse_record(raw)
    return (None, "unreadable") if record is None else (record, "ok")


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
    record, _status = read_record_status(path=path)
    return record


def read_record_status(
    *, path: Path | None = None
) -> tuple[ControlContext | None, Literal["ok", "missing", "unreadable"]]:
    """The record and how reading it ended — for a caller that must tell why.

    :func:`read_record` answers ``None`` for "nobody has written one yet" and
    for "something broke it" alike; a write verdict refuses both but names
    them differently. ``"missing"`` is no file at the path; ``"unreadable"``
    is every other way of having no usable record — an unresolvable
    agent-data root, an OS or decode failure, invalid JSON, a foreign schema,
    an unusable identity. ``"ok"`` comes with the record. Cached like
    :func:`read_record` (rule 4); never raises.

    Args:
        path: Read this file instead of :func:`record_path`.
    """
    target_path = record_path() if path is None else path
    if target_path is None:
        return None, "unreadable"
    cached: tuple[ControlContext | None, _RecordStatus] | None = _read_cached(target_path, _read)
    if cached is not None:
        return cached
    # No signature: absent, or a stat that failed for another reason.
    try:
        target_path.stat()
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "unreadable"
    # It appeared between the two stats; this read answers for that instant.
    return _read(target_path)


def invalidate_cache() -> None:
    """Forget every parsed file. For tests and for a process that moved roots."""
    with _CACHE_LOCK:
        _CACHE.clear()


# -- writing ----------------------------------------------------------------


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* to *path* as JSON, replacing any existing file. Rule 3.

    The file is replaced whole through ``safety_file.atomic_replace``, so a
    failed write leaves neither half a file nor litter beside it.

    Every file in :func:`state_dir` is written this way, the record and the
    per-server reports and requests alike: they are read by processes that
    never coordinate with the writer, so a reader must never be able to see
    half of one.

    The mode is load-bearing, not a preference: those readers run under other
    uids — a dispatch worker at uid 1000 reading what the host account wrote —
    and reach the file through the setgid group of the state directory, so the
    file is given 0640 whatever the umask: under umask 077 a record only the
    owner can read makes every owned lane write refuse
    ``control_context_unavailable``, while CI at umask 022 stays green and never
    sees it. Every directory this writer creates — the
    levels above the leaf as much as the leaf — is set to exactly 0o2770 rather
    than merely widened, because a directory left at the umask 0755 would let
    ``other`` list the identity names.

    Raises:
        OSError: The write itself failed.
    """
    _ensure_directory(path.parent)
    atomic_replace(
        path, json.dumps(payload, indent=2).encode("utf-8"), mode=0o640, sync="best_effort"
    )


def _ensure_directory(directory: Path) -> None:
    """Create *directory* and the levels above it at 0o2770. See :func:`write_json_atomic`."""
    # ``mkdir(parents=True)`` reports the leaf and says nothing about the levels
    # it creates on the way, so they are named before the call: the tree root is
    # one of them whenever this writer is the first to reach a deployment's tree,
    # and a root left at the umask defeats the mode set below it.
    created = []
    probe = directory
    while probe != probe.parent and not probe.exists():
        created.append(probe)
        probe = probe.parent
    try:
        directory.mkdir(parents=True)
    except FileExistsError:
        # Another writer got there first, so none of these levels are this
        # writer's to set a mode on.
        created = []
    for level in created:
        os.chmod(level, 0o2770)


def _lock_path(path: Path) -> Path:
    """The lock file for the record at *path*: ``<path>.lock``. Rule 3."""
    return path.with_name(path.name + _LOCK_SUFFIX)


def _open_lock_file(lock: Path) -> int:
    """Open *lock* for :func:`_record_lock`, creating it at 0o660 if absent.

    Neither open follows a symbolic link at the last component: the create
    uses ``O_EXCL | O_NOFOLLOW``, and the reopen of an existing lock uses
    ``O_NOFOLLOW``. A lock removed between the two is created once more.

    Raises:
        OSError: A symbolic link sits at *lock* (``errno.ELOOP``), or *lock*
            could not be created or opened.
    """
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
        except FileExistsError:
            pass
        else:
            try:
                # The create honoured the umask; this file is this writer's own.
                os.fchmod(fd, 0o660)
            except BaseException:
                os.close(fd)
                raise
            return fd
        try:
            return os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
        except FileNotFoundError:
            continue
    return os.open(lock, os.O_RDWR | os.O_NOFOLLOW)


@contextlib.contextmanager
def _record_lock(path: Path) -> Iterator[None]:
    """Hold the exclusive lock on the record at *path*. Rule 3.

    A bounded wait: a holder that does not let go inside
    :data:`_LOCK_TIMEOUT_S` is a wedged process, and the write refuses rather
    than hanging a route behind it. The lock file is given 0o660 like the
    record is given 0o640: writers under other uids reach it through the
    state directory's setgid group.

    Only a lock file this call created is given that mode. One that already
    exists is reused as it stands: it may be another uid's, and a mode set on
    it here would be a mode set on whatever the path resolves to. Neither open
    follows a symbolic link at the last component (``O_NOFOLLOW``), because the
    directory is group-writable by parties other than its owner: a link planted
    there would otherwise have this writer chmod and lock a file of the
    planter's choosing. A link refuses the write.

    Raises:
        RuntimeError: The lock was not acquired in time.
        OSError: A symbolic link sits where the lock file belongs
            (``errno.ELOOP``), or the lock file could not be created or opened.
    """
    _ensure_directory(path.parent)
    fd = _open_lock_file(_lock_path(path))
    try:
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"The control-context record at {path} stayed locked for "
                        f"{_LOCK_TIMEOUT_S:g} s; nothing was written"
                    ) from None
                time.sleep(_LOCK_POLL_S)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write_record_unlocked(record: ControlContext, path: Path) -> None:
    """Store *record* exactly as given. The caller holds :func:`_record_lock`.

    The one place the record's bytes are written. Private, and called only
    from inside this module with the lock held: a write that skipped the lock,
    or that wrote a ``switch`` it did not read, is the race rule 3 closes.
    """
    write_json_atomic(path, record.to_payload())


def write_record(record: ControlContext, *, path: Path | None = None) -> Path:
    """Store *record*, replacing any existing one atomically. Rule 3.

    Under the record lock, this re-reads the file on disk and stores *record*
    with the on-disk ``switch`` and ``switch_epoch`` in place of its own: no
    caller of this function changes either, whatever it read before. A file
    that is missing or unreadable is replaced with the switch empty and a
    fresh ``time.time_ns()`` epoch, never 0, so a widening begun before the
    record was lost names an older epoch and is refused.

    It does not check who owns the record: the owner decides what it is
    changing and whether it is still entitled to, around this call.

    Args:
        record: The context to store.
        path: Write this file instead of :func:`record_path`.

    Returns:
        The path written.

    Raises:
        RuntimeError: The agent-data root does not resolve, so there is
            nowhere a reader would look for what this call would write; or
            the record stayed locked past the bounded wait.
        OSError: The write itself failed.
    """
    target_path = record_path() if path is None else path
    if target_path is None:
        raise RuntimeError(
            "No agent-data root: the control-context record has nowhere to be written"
        )
    with _record_lock(target_path):
        on_disk, _status = _read(target_path)
        if on_disk is None:
            stored = replace(record, switch={}, switch_epoch=time.time_ns())
        else:
            stored = replace(record, switch=dict(on_disk.switch), switch_epoch=on_disk.switch_epoch)
        _write_record_unlocked(stored, target_path)
    return target_path


#: The keyword a person's gesture uses for "every machine": only ever off.
_ALL_TARGETS = "all"

#: A machine's three positions, narrowest first: a later one widens an earlier
#: one. The gesture's words and every reader's order. ``"off"`` is not stored
#: (rule 2).
SWITCH_POSITIONS: tuple[Literal["off"], Literal["asked"], Literal["allowed"]] = (
    _SWITCH_OFF,
    _SWITCH_ASKED,
    _SWITCH_ALLOWED,
)


class StoppedSince(Exception):
    """A widening refused because a stop, or a lost record, came after it began.

    Rule 3: a person widening a machine names the epoch they saw. A record
    whose epoch differs from it — newer or older —, a gesture that names none,
    a record that has no epoch, and no record at all each mean that what they
    saw is no longer what is on disk, so nothing is armed. The chip retries on
    its own, with :attr:`epoch`, only when :attr:`stamped`; otherwise it
    re-renders and needs a new click, so a widening that raced a stop stays off.

    Not a ``RuntimeError``: a caller that answers ``RuntimeError`` with "the
    store is unavailable" must not answer a stale gesture the same way.

    Attributes:
        epoch: The record's epoch now — the one a retried gesture names — or
            ``None`` when there is no usable record to retry against.
        stamped: ``True`` only when this refusal just stamped an epoch on an
            epoch-less record — nothing was stopped, so a retry is safe.
        reason: Always ``"stopped_since"``, a member of
            :data:`osprey_connectors.refusals.REASONS`.
    """

    def __init__(self, *, epoch: int | None, stamped: bool = False) -> None:
        self.epoch = epoch
        self.stamped = stamped
        self.reason = "stopped_since"
        super().__init__(refusal_phrase(self.reason))


def _gesture_targets(targets: Sequence[str] | Literal["all"], state: str) -> list[str] | None:
    """*targets* checked as named machines, or ``None`` for ``"all"``.

    Raises:
        ValueError: Anything that is not ``"all"`` or a non-empty sequence of
            names; ``"all"`` with anything but ``"off"``; ``"asked"`` or
            ``"allowed"`` for a name outside
            :data:`~osprey_connectors.types.CONTROL_TARGETS`.
    """
    if targets == _ALL_TARGETS:
        if state != _SWITCH_OFF:
            raise ValueError(f'set_switch("all", {state!r}): machines are turned on one at a time')
        return None
    if isinstance(targets, str) or not isinstance(targets, Sequence):
        raise ValueError(
            f'set_switch targets must be "all" or a sequence of names, not {targets!r}'
        )
    names = list(targets)
    if not names or not all(isinstance(name, str) for name in names):
        raise ValueError(
            f"set_switch targets must be a non-empty sequence of names, not {targets!r}"
        )
    if state != _SWITCH_OFF:
        unknown = [name for name in names if name not in CONTROL_TARGETS]
        if unknown:
            raise ValueError(f"set_switch cannot turn on unknown targets {unknown!r}")
    return names


def _widens(state: str, position: str) -> bool:
    """Whether moving a machine at *position* to *state* widens it."""
    return SWITCH_POSITIONS.index(state) > SWITCH_POSITIONS.index(position)


def set_switch(
    targets: Sequence[str] | Literal["all"],
    state: Literal["off", "on", "asked", "allowed"],
    *,
    epoch: int | None = None,
    path: Path | None = None,
) -> ControlContext | None:
    """The person's switch gesture — the one writer of ``switch``. Rule 3.

    Reads, compares and writes in one hold of the record lock, through the
    same private writer :func:`write_record` uses, so no other write lands
    between what this compared and what it stored.

    Each named target's move is classified inside the lock, against the map
    on disk, so a move believed to be a narrowing that raced a stop is judged
    a widening and refused.

    * A **widening** (one name moving up off → asked → allowed gates the whole
      call) requires *epoch*, the ``switch_epoch`` the person saw, and succeeds
      only when it equals the record's epoch — a compare-and-swap. It raises
      :class:`StoppedSince` carrying the on-disk epoch when the two differ
      (newer or older) or *epoch* is ``None``, when the record has no epoch
      (one is stamped and ``stamped`` is set, so the retried gesture succeeds),
      or when there is no usable record.
    * A **narrowing or a no-op** applies unconditionally and never refuses.
      ``"off"`` clears each named key, known target or not.
    * ``"all"`` (with ``"off"``) clears every key and moves the epoch to
      ``max(old + 1, time.time_ns())``, which makes every widening begun
      before it stale.

    Only a stop moves an existing epoch; a gesture on an epoch-less record
    stamps a fresh ``time.time_ns()``. Every other field is kept as on disk.

    Args:
        targets: The machines, or ``"all"``.
        state: ``"off"``, ``"asked"`` or ``"allowed"``; the legacy ``"on"``
            is taken as ``"asked"`` and never stored.
        epoch: The epoch the gesture was begun against. Ignored when nothing
            widens.
        path: Write this file instead of :func:`record_path`.

    Returns:
        The record as stored, or ``None`` when a non-widening gesture found no
        usable record — nothing is on, so nothing was written.

    Raises:
        ValueError: ``"all"`` with anything but ``"off"``, an unknown *state*,
            malformed *targets*, or an *epoch* that is neither ``None`` nor an
            integer. Nothing is written.
        StoppedSince: A widening that is stale, as above.
        RuntimeError: The agent-data root does not resolve, or the record
            stayed locked past the bounded wait.
        OSError: The write itself failed.
    """
    if state == _SWITCH_LEGACY_ON:
        # A framework released before the three positions sends "on".
        state = _SWITCH_ASKED
    if state not in SWITCH_POSITIONS:
        raise ValueError(f"set_switch state must be one of {SWITCH_POSITIONS}, not {state!r}")
    names = _gesture_targets(targets, state)
    if state != _SWITCH_OFF and (
        epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int))
    ):
        raise ValueError(f"set_switch epoch must be an integer or None, not {epoch!r}")
    target_path = record_path() if path is None else path
    if target_path is None:
        raise RuntimeError(
            "No agent-data root: the control-context record has nowhere to be written"
        )
    with _record_lock(target_path):
        current, _status = _read(target_path)
        if current is None:
            if state != _SWITCH_OFF:
                raise StoppedSince(epoch=None)
            return None
        on_disk_epoch = current.switch_epoch
        if names is None:
            stored = replace(
                current,
                switch={},
                switch_epoch=max((on_disk_epoch or 0) + 1, time.time_ns()),
            )
        else:
            if any(_widens(state, switch_position(current, name)) for name in names):
                if on_disk_epoch is None:
                    stamped = replace(current, switch_epoch=time.time_ns())
                    _write_record_unlocked(stamped, target_path)
                    raise StoppedSince(epoch=stamped.switch_epoch, stamped=True)
                if epoch != on_disk_epoch:
                    raise StoppedSince(epoch=on_disk_epoch)
            if state == _SWITCH_OFF:
                switch = {key: value for key, value in current.switch.items() if key not in names}
            else:
                switch = {**current.switch, **dict.fromkeys(names, state)}
            stored = replace(
                current,
                switch=switch,
                switch_epoch=time.time_ns() if on_disk_epoch is None else on_disk_epoch,
            )
        _write_record_unlocked(stored, target_path)
    return stored


# -- process liveness -------------------------------------------------------

_PROC_ROOT = "/proc"


def _is_thread_id(pid: int) -> bool:
    """Whether the kernel knows *pid* as a thread of some other process.

    Linux draws PIDs and thread ids from one number space, and ``/proc/<tid>``
    opens by number though no listing shows it. A process is its own thread
    group leader, so a ``Tgid`` that differs from *pid* means a thread. ``False``
    when ``/proc`` says nothing, which is every platform without one.
    """
    try:
        with open(f"{_PROC_ROOT}/{pid}/status", "rb") as status:
            for line in status:
                if line.startswith(b"Tgid:"):
                    return int(line.split()[1]) != pid
    except (OSError, ValueError, IndexError):
        return False  # no /proc, or one that does not read like Linux's: no claim either way
    return False


def is_process_alive(pid: object) -> bool:
    """Whether *pid* names a running process.

    ``os.kill(pid, 0)`` sends no signal and only asks the kernel whether the
    process exists — the ``osprey web`` PID-file check does the same. A
    ``PermissionError`` means the process exists but belongs to another user, so
    it counts as ALIVE: sweeping a file whose owner is merely unreachable would
    delete live state. Non-positive PIDs are rejected without calling
    ``os.kill`` at all, because 0 and negatives address process *groups*.

    ``os.kill`` also answers for a thread id, which is no process: a PID kept
    in a record that outlived its container can land on a thread of the
    process reading it, which would then follow itself for ever
    (:func:`_is_thread_id`). A real process on that PID is not told apart.

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
        pass  # another user's process still exists; the thread check below applies to it too
    except OSError:  # pragma: no cover - platform oddity; assume alive
        return True
    return not _is_thread_id(pid)


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


def unchanged_detail(target: str, generation: int) -> str:
    """The ``detail`` for a request that asked for the target of record.

    One sentence with one spelling, because four surfaces write it into the same
    field: the terminal owner's request half, the controls server's reconciler,
    the HTTP switch route and the agent's own ``control_target_set``. A record
    whose terminus read differently depending on which surface answered would be
    two vocabularies in one place, and the chip resolves a pending switch
    against what it finds there.

    It is an APPLIED terminus and mints no generation: the deployment is where
    the request asked for it to be, and a generation bumped for a switch that
    did not happen would refuse every write bound to the old one, for nothing.
    """
    return f"The control target is already {target!r} (generation {generation})."


def applied_detail(target: str, generation: int) -> str:
    """The ``detail`` for a switch that happened. *generation* is the new one.

    The companion to :func:`unchanged_detail`, written by the same four
    surfaces and for the same reason.
    """
    return f"The control target is now {target!r} (generation {generation})."


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
        generation=generation if generation is not None else record.generation,
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

    The write goes through :func:`write_record`, so it holds the record lock
    and keeps the on-disk switch: a terminus never turns a machine on or off.

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


def _report_path_under(root: Path, server_pid: object) -> Path:
    """The report path of *server_pid* under an explicit agent-data *root*.

    Through the same three hops as :func:`_record_path_under`, identity
    included: every family of file in this directory moves together, so a
    caller that finds the record under a root finds the reports beside it.
    """
    return (
        root
        / STATE_DIR_NAME
        / acting_identity()
        / f"{REPORT_FILE_PREFIX}{server_pid}{REPORT_FILE_SUFFIX}"
    )


def _report_path(server_pid: object) -> Path | None:
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
    except Exception:  # a half-written report is skipped, not fatal
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


def _read_report_payload(path: Path) -> dict[str, Any] | None:
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


def _live_report_payloads(
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
        payload = _read_report_payload(entry)
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
        is_alive: As for :func:`_live_report_payloads`.
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


def _holds_no_connector(report: ServerReport) -> bool:
    """Whether *report* states no connector on any of the three axes that could.

    A server between two targets is holding one of them, which is why a swap
    in flight stops the whole deployment. A server that has never got
    anywhere is holding nothing, so nothing it is doing can reach a target
    another session's launch would not.

    All three axes have to say so, and they say so together: the binding pair
    and the child PIDs are published in one write once a child has answered
    its init frame, so a report stating a connector on any one of them is not
    a server acquiring its first.
    """
    return (
        not report.children and report.applied_target is None and report.applied_generation is None
    )


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
        if status == REPORT_APPLYING and not _holds_no_connector(report):
            bound = _bound_epoch(block)
            if bound is None or reference <= bound:
                # A swap in flight: nobody launches anywhere until it lands,
                # because the server holding the connector is between two
                # targets. Past its bound it has stopped being in flight, and
                # only the session whose server it is is still held up by it.
                # A server holding no connector is not between two targets and
                # falls out above, on the same terms.
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
    A server that reports ``applying`` while it holds no connector at all —
    no children, nothing bound — is acquiring its first rather than swapping
    one, and stops only its own session.

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
