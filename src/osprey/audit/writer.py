"""The unified audit ledger writer — one append, keyed by (identity, surface).

:mod:`osprey.audit.envelope` says what a record *is*; this module is the only
place one becomes a file. Every safety surface — the MCP middleware, the HTTP
layers, the hooks, the sidecar, the framework writers — hands an envelope to
:func:`record_envelope` (or the fields to :func:`record`) and gets back the
path it landed in, or ``None``. A writer that must resolve its own directory
appends through :func:`append_envelope` instead.

**Routing is (identity, surface).** A record lands in
``var/audit/<identity>/<surface>.jsonl``, where the identity comes from
:func:`~osprey.utils.identity.acting_identity` — the shared ladder
``OSPREY_TERMINAL_USER`` → ``OSPREY_AUDIT_IDENTITY`` → local account →
``unknown``, and **never** the hostname. That helper also fills the envelope's
``actor``, so the two read the same answer from the same rung by default.

The directory component is always the **process** identity, whatever an
envelope's ``actor`` says: :func:`record_envelope` re-resolves it and never
routes on the field. That is deliberate — each container binds only
``var/audit/<its own identity>/`` read-write, so routing on a caller-supplied
actor would file records where the process cannot write them (and, where it
could, let one writer's records land in another's ledger). A record whose
``actor`` differs from its directory is the normal shape for a service that
acts for someone: the sidecar's login refusals file under ``sidecar/`` and name
*alice* as the subject.

Both path components are guarded. The *surface* is the untrusted half by
construction (a facility names its own MCP servers); the identity is guarded
too, because :func:`ledger_path` takes one as a parameter and neither the
envelope nor the ladder validates a caller-supplied value as a path component.
Both are cut to a file name's byte budget as well: a component over
``NAME_MAX`` costs every record the process writes, not one file's name.

**The maintenance marker moves the file, not the record.** The root maintenance
phase runs with ``OSPREY_AUDIT_WRITER=maintenance`` as a per-command env prefix
(never an export — that would survive ``exec gosu`` and misroute every app-side
record). While it is set, every record the process emits files under
``<identity>/maintenance.jsonl`` whatever surface decided, and each envelope
still names its real surface. That is what keeps the ledger's central
invariant: **one uid per file**. Root and the app user both refuse
``scaffold_restore`` writes in the same container under the same identity; if
they shared a file, the first root-owned line would make the rest of it
unwritable by the app user. Only the closed set :data:`WRITER_CONTEXTS` routes;
an unrecognised value is ignored rather than opening a ledger nobody reads.

**One bounded, unsynced append.** A record is one ``os.write`` of one encoded
line onto an ``O_APPEND`` descriptor. POSIX makes that append atomic with
respect to other appenders, so concurrent emitters never interleave a line, and
:data:`MAX_RECORD_BYTES` (2 KB) keeps every ordinary record inside the size
where that holds even on the less generous filesystems a deployment might mount
``var/`` from. Two deliberate exceptions, in the order they are taken:

* An oversize ``detail`` is replaced by :data:`DETAIL_DROPPED`. ``detail`` is
  documented as supplementary context, so it is the field that gives way.
* On :data:`~osprey.audit.envelope.SURFACE_EXECUTOR` the ``source`` is kept
  whole (up to the envelope's 8000-char bound, so a line can reach ~8 KB).
  There the refused code *is* the artifact under audit, and a record without it
  is an alert rather than an audit trail. It is still exactly one ``write()``
  call of one line; what the oversize gives up is only the guarantee that no
  other appender's line can land in the middle of it.
* An identifier-only record that is still over budget is written whole and
  logged. Trimming identifiers would leave a record that names the wrong
  thing, which is worse than a large one.

No fsync: the audit zone is durable by construction (``osprey build``
re-renders ``build/`` and never touches ``var/``; ``osprey reset`` keeps
``var/audit`` unless ``--purge-audit``), and a per-record sync would put disk
latency on the path of every tool call.

**Nothing here may cost the operation it records.** Every entry point that
resolves its own routing swallows its errors and returns ``Path | None``; the
event is logged *before* the durable write, so a read-only filesystem
downgrades the audit trail instead of erasing it. :func:`append_envelope`,
whose caller owns the path, hands its failures back instead — that caller
owns the degrade too. An invalid envelope is included in that: construction validation
lives in the envelope, and :func:`record` calls it from inside this boundary so
an emitter never needs a ``try`` of its own.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from osprey.audit.envelope import DECISION_REFUSED, SURFACE_EXECUTOR, AuditEnvelope
from osprey.utils.identity import acting_identity
from osprey.utils.logger import get_logger

logger = get_logger("audit_writer")

#: Names the writer context for the records a process emits. Set ONLY as a
#: per-command env prefix on the root maintenance heredoc, and stripped from
#: every MCP server spec (``registry.mcp.NON_PINNABLE_AUDIT_MARKERS``) so no
#: facility can repoint another process's records. Spelled here rather than
#: imported from the registry — this module is imported by the MCP middleware,
#: the HTTP layer and the services, and a registry import would hand all of
#: them the server catalogue. ``tests/audit/test_writer.py`` pins the two
#: spellings against each other, the way the registry's own markers are pinned
#: against their assigning modules.
AUDIT_WRITER_ENV: str = "OSPREY_AUDIT_WRITER"

#: The one recognised writer context: the root maintenance phase.
WRITER_MAINTENANCE: str = "maintenance"

#: The closed set of writer contexts. Closed on purpose: a stray inherited
#: value must not be able to invent a ledger, and the marker's whole job is to
#: separate two known writers rather than to name an arbitrary one.
WRITER_CONTEXTS: tuple[str, ...] = (WRITER_MAINTENANCE,)

#: Extension every ledger carries — one JSON object per line.
LEDGER_SUFFIX: str = ".jsonl"

#: Where a record goes when its surface is not usable as a file name. The
#: envelope keeps the real surface, so nothing is lost but the routing.
FALLBACK_SURFACE: str = "unrouted"

#: Longest ledger stem, so that ``<stem>.jsonl`` fits the 255-byte file name
#: every filesystem Osprey runs on allows. Counted in BYTES, not characters:
#: ``NAME_MAX`` is a byte limit, so a 250-character non-ASCII surface is a
#: 500-byte name that ext4 and overlayfs — the container's filesystems — refuse
#: with ``ENAMETOOLONG`` while APFS accepts it. The envelope bounds a surface at
#: 256 characters, which is one filename too long even in ASCII: a surface that
#: hits that bound would otherwise make its records unwritable. Truncating can
#: merge two pathologically long surfaces into one file; the records still name
#: their real surface, and a merged file beats a dropped record.
MAX_LEDGER_STEM_BYTES: int = 255 - len(".jsonl")

#: The size a single append stays inside so that concurrent appenders cannot
#: interleave. Chosen small rather than clever: every identifier-shaped record
#: fits with room to spare, and the two documented exceptions above are the
#: only records that do not.
MAX_RECORD_BYTES: int = 2048

#: What an oversize ``detail`` is replaced by. A fixed machine-ish marker, not
#: a truncation: a record that says its context was dropped is honest, while a
#: silently shortened one reads like the whole context.
DETAIL_DROPPED: str = "<dropped: record over the append bound>"

#: Mode requested for a freshly created ledger — owner writes, group reads.
#: Group read is what lets the later audit read path serve records the
#: maintenance phase wrote as root, without granting anyone a second writer.
#: A request, not a guarantee: the process umask still applies, and the
#: containerized zones are provisioned setgid host-side at render/up.
#: ``tests/audit/test_writer.py`` pins the value and the invariant behind it:
#: no second writer (no group- or other-write) and nothing for ``other`` at all.
LEDGER_FILE_MODE: int = 0o640

#: Mode a per-identity ledger directory is created with when the writer has to
#: create it itself. The deploy path owns this directory normally
#: (``compose_generator.ensure_audit_dir`` → ``_ensure_group_shared_dir``, whose
#: ``SHARED_CORPUS_DIR_MODE`` is this same value, pinned against it by a test —
#: the audit package must not import the deployment package); the fallback below
#: exists for when it did not, and the two creators of one directory must not
#: disagree. Group-write + setgid is what lets the root maintenance phase and
#: the dropped app user share the zone: without it, whichever ran first locks
#: the other out of that identity's ledger for the life of the deployment.
LEDGER_DIR_MODE: int = 0o2770

#: Append-only by construction: the descriptor cannot seek backwards, so a
#: record can never overwrite one already stored.
_OPEN_FLAGS: int = os.O_WRONLY | os.O_APPEND | os.O_CREAT

# Characters that would let a surface escape its identity's directory or split
# into several, plus the two names that are a path component syntactically but
# resolve elsewhere. Spelled here rather than imported from
# :mod:`osprey.utils.identity`, whose equivalent is private and whose job is
# the ladder rather than validation — and whose contract is that its own output
# already passed this test.
_UNSAFE_IN_COMPONENT: tuple[str, ...] = ("/", "\\", "\0")
_RESERVED_NAMES: tuple[str, ...] = (".", "..")


def audit_dir() -> Path:
    """The deployment's audit zone (``<repo>/var/audit``).

    Anchored on the project root resolver, so a record lands in the repo whose
    operation was audited rather than in whatever directory the emitting
    process happens to run from, and spelled with
    :data:`~osprey.utils.workspace.AUDIT_DIR_RELPATH` so the writer's path and
    the container's mounted path come from one constant.

    Imported lazily and kept as its own function for the same two reasons the
    P1 refusal ledger did it: tests get a single seam to redirect instead of
    having to stand up a project root, and this module stays importable from
    the MCP middleware and the HTTP layer without dragging the workspace
    resolver (and its config load) behind it.
    """
    from osprey.utils.workspace import (
        AUDIT_DIR_RELPATH,
        load_osprey_config,
        resolve_project_root,
    )

    return resolve_project_root(load_osprey_config()) / AUDIT_DIR_RELPATH


def writer_context() -> str | None:
    """The recognised writer context this process runs under, or ``None``.

    Read per call, never cached: the marker is a property of the *process
    phase* — the root maintenance step sets it for the duration of one command
    — and a value captured at import would outlive the phase that set it.
    """
    marker = os.environ.get(AUDIT_WRITER_ENV)
    if not isinstance(marker, str):
        return None
    candidate = marker.strip()
    if not candidate:
        return None
    if candidate not in WRITER_CONTEXTS:
        logger.warning(
            "Ignoring unrecognised %s=%r; records keep their per-surface routing",
            AUDIT_WRITER_ENV,
            candidate,
        )
        return None
    return candidate


def _safe_component(value: str) -> str | None:
    """Return *value* if it is usable as one path component, else ``None``."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or candidate in _RESERVED_NAMES:
        return None
    if any(bad in candidate for bad in _UNSAFE_IN_COMPONENT):
        return None
    return candidate


def _cut_to_name_bytes(value: str, kind: str) -> str:
    """*value*, cut to the longest UTF-8 prefix usable as one path component.

    Counted in BYTES and cut on the encoded form, dropping a partial character:
    ``NAME_MAX`` is a byte limit on ext4 and overlayfs — the filesystems a
    container mounts ``var/`` from — and a name that ends mid-sequence is not a
    name.

    BOTH halves of a ledger path come through here, the stem via
    :func:`ledger_name` and the identity directory via :func:`ledger_path`,
    because either one over the limit costs the same thing: ``ENAMETOOLONG`` on
    every record the process writes, swallowed by the never-raises boundary. A
    truncated name is the same trade the stem already makes — two pathological
    names can merge into one, the records still say who and what they were, and
    a merged file beats a dropped record.

    :data:`MAX_LEDGER_STEM_BYTES` bounds both. It is six bytes tighter than a
    directory strictly needs (a directory leaves no room for ``.jsonl``), which
    is one constant instead of two for a case nothing reaches by design.
    """
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_LEDGER_STEM_BYTES:
        return value
    logger.warning(
        "Audit %s %r is too long for a file name; filing under its first %d bytes",
        kind,
        value,
        MAX_LEDGER_STEM_BYTES,
    )
    return encoded[:MAX_LEDGER_STEM_BYTES].decode("utf-8", "ignore")


def ledger_name(surface: str) -> str:
    """The ledger file stem a record on *surface* is filed under.

    The writer context wins when one is active — that is the whole mechanism
    behind one-uid-per-file — and an unusable surface is quarantined into
    :data:`FALLBACK_SURFACE` rather than allowed to name a path.
    """
    context = writer_context()
    if context is not None:
        return context

    safe = _safe_component(surface)
    if safe is None:
        logger.warning(
            "Audit surface %r cannot name a file; filing under %r",
            surface,
            FALLBACK_SURFACE,
        )
        return FALLBACK_SURFACE
    return _cut_to_name_bytes(safe, "surface")


def ledger_path(surface: str, identity: str | None = None) -> Path:
    """Full path of the ledger a record on *surface* belongs in.

    *identity* defaults to :func:`~osprey.utils.identity.acting_identity`; it
    is a parameter only so a caller that already resolved the identity for the
    envelope's ``actor`` does not have to resolve it twice.

    A supplied *identity* gets the same path-component guard the surface gets,
    and an unusable one falls back to the ladder rather than naming a
    directory. The ladder's own output needs no guard, but this parameter is
    not the ladder: the value a caller has in hand is the envelope's ``actor``,
    which the envelope bounds and never validates as a path component — so an
    emitter that resolves a user from a request header could otherwise walk the
    ledger out of the audit zone with ``../``.

    It gets the surface's LENGTH treatment too. The envelope bounds an
    identifier at 256 characters, one byte past what a file name may be even in
    ASCII, so an unbounded identity is the same ``ENAMETOOLONG`` the stem is
    already cut for — and it would silently cost every record the process
    writes rather than one file's name.
    """
    if identity is None:
        who = acting_identity()
    else:
        who = _safe_component(identity)
        if who is None:
            who = acting_identity()
            logger.warning(
                "Audit identity %r cannot name a directory; filing under %r instead",
                identity,
                who,
            )
    directory = _cut_to_name_bytes(who, "identity")
    return audit_dir() / directory / f"{ledger_name(surface)}{LEDGER_SUFFIX}"


def _encode(record: dict[str, Any]) -> bytes:
    """Serialize one record to the bytes of one JSONL line.

    Compact separators because the byte budget is the point, and the default
    ``ensure_ascii`` because a ledger of pure ASCII lines is readable by every
    consumer that will ever tail it.
    """
    return (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8", "replace")


def _line_for(envelope: AuditEnvelope) -> bytes:
    """The bytes to append for *envelope*, degraded to fit where it can be.

    The order is the module docstring's: supplementary context gives way
    first, the executor's evidence never gives way, and identifiers never give
    way — an over-budget record that is true beats an in-budget one that names
    the wrong thing.
    """
    record = envelope.to_dict()
    line = _encode(record)
    if len(line) <= MAX_RECORD_BYTES:
        return line

    # Supplementary context gives way FIRST, executor records included: an
    # executor record with a small source and a full detail fits once the
    # detail goes, and keeping atomicity for it costs only the field the module
    # already classes as supplementary. The executor exception below is for the
    # source that is genuinely too big, not for every executor record.
    dropped = None
    if record.get("detail") is not None:
        record["detail"] = DETAIL_DROPPED
        dropped = _encode(record)
        if len(dropped) <= MAX_RECORD_BYTES:
            return dropped

    if record.get("source") is not None and record.get("surface") == SURFACE_EXECUTOR:
        # The documented exception: still one write() of one line, and the
        # refused code is why the record exists at all. The detail rides along
        # whole — dropping it did not buy the bound back, so it buys nothing.
        return line

    logger.warning(
        "Audit record for %s/%s exceeds %d bytes with identifiers alone; writing it whole",
        record.get("surface"),
        record.get("subject"),
        MAX_RECORD_BYTES,
    )
    return dropped if dropped is not None else line


def _create_identity_dir(parent: Path) -> None:
    """Create a per-identity ledger directory at the shared mode.

    Best-effort on the mode, never on the directory: the ``chmod`` is what
    actually sets :data:`LEDGER_DIR_MODE`, because ``mkdir``'s mode argument is
    masked by the process umask (0o2770 under the usual 022 arrives as 0o2750,
    with no group write) and the setgid bit does not survive it at all. A
    process that may create the directory but not chmod it still gets a
    directory — a slightly narrow one beats a dropped record.
    """
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(parent, LEDGER_DIR_MODE)
    except OSError:
        logger.debug(
            "Could not set %o on the audit directory %s", LEDGER_DIR_MODE, parent, exc_info=True
        )


def _append(path: Path, line: bytes) -> bool:
    """Append *line* to *path* in one write. Returns whether it landed intact.

    The directory is created only after an open fails for want of it: the
    containerized zones are provisioned host-side (setgid, group-owned) and the
    laptop case creates them once, so paying a ``mkdir`` on every record would
    buy a syscall per tool call for a condition that is true once. When it does
    fall to this path, the directory is created at the same mode the deploy
    path uses — see :func:`_create_identity_dir`.
    """
    try:
        fd = os.open(path, _OPEN_FLAGS, LEDGER_FILE_MODE)
    except FileNotFoundError:
        _create_identity_dir(path.parent)
        fd = os.open(path, _OPEN_FLAGS, LEDGER_FILE_MODE)

    try:
        written = os.write(fd, line)
        if written != len(line) and written > 0 and not line[:written].endswith(b"\n"):
            # Terminate the fragment while the descriptor is still open, so the
            # NEXT record starts on a line of its own instead of being appended
            # onto a half-record and lost with it. One torn write must cost one
            # record, not every record after it.
            #
            # Why here and not a tail check before the write: the fast path
            # stays exactly one os.write of one line (this second write only
            # happens on a path that already stored nothing and returns False),
            # while reading the tail would cost a read on every append and
            # could not be trusted anyway — another appender may land between
            # the check and the write, which O_APPEND is precisely what saves
            # us from.
            try:
                os.write(fd, b"\n")
            except OSError:
                logger.debug("Could not terminate a torn audit line in %s", path, exc_info=True)
    finally:
        os.close(fd)

    if written != len(line):
        # Only reachable on a filesystem that accepted a partial write without
        # raising. The line is torn, so the record is not stored — say so
        # rather than hand back a path that suggests it is.
        logger.warning(
            "Short append to %s (%d of %d bytes); the audit record is incomplete",
            path,
            written,
            len(line),
        )
        return False
    return True


def _log_event(envelope: AuditEnvelope) -> None:
    """Put the event on the server log before the durable write is attempted.

    Refusals warn — an unwritable audit zone then downgrades the trail instead
    of erasing it. Admitted calls only debug: the MCP middleware records every
    tools/call, and a warning apiece would bury the refusals the log exists to
    surface.
    """
    message = "Audit %s: surface=%s actor=%s subject=%s reason=%s (posture=%s/%s session=%s)"
    args = (
        envelope.decision,
        envelope.surface,
        envelope.actor,
        envelope.subject,
        envelope.reason,
        envelope.posture,
        envelope.posture_source,
        envelope.session,
    )
    if envelope.decision == DECISION_REFUSED:
        logger.warning(message, *args)
    else:
        logger.debug(message, *args)


def record_envelope(envelope: AuditEnvelope) -> Path | None:
    """Append *envelope* to its ledger and return the path, or ``None``.

    ``None`` means the record was not durably stored — an unwritable zone, a
    torn write, anything at all. The caller does not branch on it (a refusal is
    enforced by the caller, not by this record landing), but a test can tell
    "wrote" from "gave up quietly" without reading the directory.

    Never raises: see the module docstring.
    """
    try:
        _log_event(envelope)
        path = ledger_path(envelope.surface)
        return path if append_envelope(path, envelope) else None
    except Exception:
        logger.warning("Could not append to the audit ledger", exc_info=True)
        return None


def append_envelope(path: Path, envelope: AuditEnvelope) -> bool:
    """Append *envelope* to the ledger at *path* in one write; ``True`` if it landed intact.

    The seam for a writer that resolves its own directory — the auth sidecar,
    whose image has no project root for :func:`audit_dir` to anchor on and is
    told its zone by environment instead. It gets the line shaping (the byte
    budget, the ``detail`` degrade) and the ``O_APPEND`` write from the same
    two functions :func:`record_envelope` uses, so neither is re-implemented
    one service over; only the routing is the caller's.

    Unlike the two routing entry points this one **raises** on an I/O failure
    — a caller that owns the path also owns the degrade, and the sidecar's is
    to put the record on its log. The decision is not logged here either (the
    append's own warnings name only the path): the caller's log line is the
    one that names it. And *path* is taken as given: the component guards of
    :func:`ledger_path` are not applied, so a caller must never derive it from
    anything a request supplied.
    """
    return _append(path, _line_for(envelope))


def record(**fields: Any) -> Path | None:
    """Build an envelope from *fields* and append it; returns the path or ``None``.

    The entry point for emitters that have facts rather than an envelope. Two
    things it does that a bare constructor call cannot:

    * ``actor`` defaults to :func:`~osprey.utils.identity.acting_identity`, so
      no emitter re-implements the ladder or accidentally names the process
      account where the container has a service identity.
    * Construction happens *inside* the never-raises boundary, so an emitter
      does not need a ``try`` of its own around a schema that validates. An
      invalid envelope degrades to a warning and ``None`` — the audit trail
      loses a record, the operation it describes is untouched.

    See :class:`~osprey.audit.envelope.AuditEnvelope` for the fields.
    """
    try:
        fields.setdefault("actor", acting_identity())
        envelope = AuditEnvelope(**fields)
    except Exception:
        logger.warning(
            "Could not build an audit envelope for surface=%r subject=%r; record dropped",
            fields.get("surface"),
            fields.get("subject"),
            exc_info=True,
        )
        return None
    return record_envelope(envelope)
