"""Audit ledger read route — ``GET /api/audit/recent``.

:mod:`osprey.audit.writer` is the append half of the unified audit ledger; this
is the read half. It tails the JSONL files under **this container's own**
identity directory and serves the newest records back to the Config panel.

Three things about it are deliberate.

**It is behind the Config panel tier, using that tier's own gate.** A ledger
names every safety decision the deployment made — the subjects, the sessions,
the actors — so it is at least as sensitive as the ``config.yml`` and
``.claude/`` surfaces the panel edits. Rather than spell a second refusal, the
handler calls :func:`~osprey.interfaces.web_terminal.routes.config._require_config_panel`
directly: one switch (``web.config_panel.enabled``), one wording, nothing to
drift. The gate runs FIRST, ahead of any filesystem access and ahead of every
parameter check — which is why ``limit`` arrives as a **string** and is parsed
and clamped inside the handler instead of being annotated ``int`` or bounded by
``Query(ge=..., le=...)``. Either of those hands FastAPI a coercion or
validation step that answers 422 before the handler runs, telling a caller who
may not open the Config panel that the route exists and what it takes. Doing it
by hand also buys the kinder semantics: an operator who asks for more than the
ceiling gets the ceiling, not a refusal.

**The directory is the process's, never the request's.** Each container binds
only ``var/audit/<its own identity>/`` read-write, so its own subdirectory is
the only thing it can honestly serve; the deployment-wide view over every
identity is host-side, by design. :func:`identity_dir` therefore resolves
exactly where :func:`~osprey.audit.writer.ledger_path` does — the same
``audit_dir()`` seam and the same :func:`~osprey.utils.identity.acting_identity`
ladder — and no request field participates. The one client-supplied string, the
``surface`` filter, is validated as a single path component AND then matched
against the stems this directory actually contains, so the parameter never
becomes a path even if the guard were wrong.

**The read is bounded.** A ledger grows for the life of a deployment. Each file
is seeked to from the end and only a window derived from the requested limit is
read, so answering ``limit=10`` costs the same whether the ledger holds a
thousand records or a million. Seeking into the middle of a file makes a torn
first line the normal case, not an exception — it is dropped, and any other line
that does not parse into an object is skipped rather than failing the request.

The window is sized for the writer's ordinary 2 KB record bound plus slack for
its one documented oversize exception, so it holds at least ``limit`` records
for any realistic ledger. It is a bound, not a promise: a ledger consisting
entirely of maximum-size executor records can return fewer than ``limit``. That
is the honest trade — a reader that guaranteed the count would have to keep
seeking backwards until it had them, which is the unbounded read this avoids.
Nothing here may take the web terminal down to report an audit trail: an
unresolvable audit zone answers with an empty list and a log line, the same way
the writer swallows its own errors rather than costing the operation it records.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from osprey.audit import writer
from osprey.audit.writer import LEDGER_SUFFIX, MAX_RECORD_BYTES
from osprey.interfaces.web_terminal.routes.config import _require_config_panel
from osprey.utils.identity import acting_identity

logger = logging.getLogger(__name__)

router = APIRouter()

#: Records returned when the caller names no ``limit``. Sized for a panel that
#: shows a scrollable recent-activity list, not for bulk export — the ledger
#: files themselves are the export format, and the host-side view is the tool
#: for reading across identities.
DEFAULT_LIMIT: int = 50

#: Hard ceiling on ``limit``. Clamped to, never refused with: see the module
#: docstring on why nothing may answer before the gate.
MAX_LIMIT: int = 200

#: Floor on ``limit``. Zero and negative values clamp here rather than meaning
#: "everything" or "nothing" — a request that asks for records gets records.
MIN_LIMIT: int = 1

#: Most ledger files one response reads across. A directory holds one file per
#: surface, and a facility names its own MCP servers, so the count is not
#: bounded by anything Osprey controls. The newest-modified files win, which is
#: where recent records are.
MAX_LEDGERS: int = 64

#: Slack added to the tail window on top of ``limit * MAX_RECORD_BYTES``. The
#: writer's documented oversize exception lets one executor record reach ~8 KB,
#: so a window sized purely on the ordinary bound could land mid-record for
#: every line it wanted and return nothing at ``limit=1``. Two of those.
TAIL_SLACK_BYTES: int = 16 * 1024

#: A ledger stem the ``surface`` filter may name: one path component, no
#: separators, no dot-names, and short enough to be a file name. An ALLOWLIST
#: rather than a list of forbidden characters — the failure mode of a denylist
#: here is a path, and the set of legitimate surface names is small and known.
#:
#: Deliberately narrower than what the writer will accept: it names only
#: identifier-shaped stems (ASCII, <=128 bytes), while ``ledger_name`` allows
#: any component up to ``MAX_LEDGER_STEM_BYTES`` (249 bytes), non-ASCII
#: included. A stem outside this shape still appears in the unfiltered
#: ``ledgers`` list and is reachable through the unfiltered read — it just
#: cannot be named alone via ``surface``, which in practice only affects the
#: rare non-identifier-shaped MCP tool prefix.
_SAFE_SURFACE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

#: Names that pass a character check but are not a surface.
_RESERVED_STEMS = frozenset({".", ".."})


def identity_dir() -> Path:
    """This container's own ledger directory.

    Resolved through the writer's two seams so the reader can only ever read
    where the writer writes: ``tests/interfaces/web_terminal/test_audit_routes.py``
    pins it against ``ledger_path(...).parent`` rather than against a second
    literal, because two literals that agree today are two literals.

    ``audit_dir`` is reached through the module (``writer.audit_dir()``) rather
    than bound here by a from-import, so the writer's single documented test
    seam — ``monkeypatch.setattr(writer, "audit_dir", ...)`` — redirects the
    reader and the writer together. Bound locally, a suite that patched only
    the writer would silently read the developer's live ``var/audit``.
    """
    return writer.audit_dir() / acting_identity()


def _clamp(limit: int) -> int:
    """Bring *limit* inside ``[MIN_LIMIT, MAX_LIMIT]``."""
    return max(MIN_LIMIT, min(MAX_LIMIT, limit))


def _parsed_limit(limit: str) -> int:
    """*limit* as a clamped integer, or a 400-shaped refusal.

    Taken off the request as a **string** and parsed here rather than annotated
    ``int``: an ``int`` annotation hands FastAPI a coercion step that answers
    422 before the handler — and therefore before the tier gate — so a caller
    who may not open the Config panel could still learn the route exists and
    what it takes. Everything this route decides, it decides after the gate.

    Raises:
        HTTPException: 400 when the value is not an integer. Out of range is
            not an error (it clamps); not a number is.
    """
    try:
        return _clamp(int(limit))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"limit must be a whole number; values outside "
                f"[{MIN_LIMIT}, {MAX_LIMIT}] are clamped, not refused"
            ),
        ) from None


def _tail_bytes(path: Path, budget: int) -> tuple[bytes, bool]:
    """The last *budget* bytes of *path*, and whether that window is truncated.

    Seeks from the end so the cost is the window rather than the file. The
    second element is ``True`` exactly when the seek landed past byte 0 — the
    window is mid-file and its first line is a fragment — and ``False`` when
    the whole file fit inside the window and was read from its own start.
    That fact is computed from the seek position rather than from the length
    of the bytes read back: a file whose size is exactly *budget* reads back
    exactly *budget* bytes starting at byte 0 (whole file, no fragment), which
    is indistinguishable by length alone from a larger file read back from a
    non-zero offset (mid-file, real fragment) — both return exactly *budget*
    bytes. :func:`_records_from` is what drops the fragment when this is
    ``True``, and it drops it unconditionally rather than trying to tell a
    torn read from a torn write — both produce a line that is not a record.
    """
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        start = max(0, size - budget)
        handle.seek(start)
        return handle.read(), start > 0


def _records_from(path: Path, limit: int) -> list[dict[str, Any]]:
    """Up to *limit* records from the tail of *path*, newest first.

    Every failure mode below is expected traffic, not an error: a partial first
    line is what seeking into a file gives you, and a line that does not parse
    is what a torn append leaves behind. Both are skipped so that one bad line
    costs one record instead of the whole response.
    """
    budget = limit * MAX_RECORD_BYTES + TAIL_SLACK_BYTES
    try:
        raw, truncated = _tail_bytes(path, budget)
    except OSError:
        logger.warning("Could not read the audit ledger %s", path, exc_info=True)
        return []

    lines = raw.split(b"\n")
    if truncated and lines:
        # The window did not reach the start of the file, so the first element
        # is whatever the seek landed in the middle of. Dropped unconditionally
        # rather than only when it fails to parse: usually the tail of a JSON
        # object is not a JSON object and the parse below would have caught it
        # anyway, but a seek that lands exactly on a record boundary with junk
        # ahead of it on the same line yields a fragment that parses cleanly,
        # and serving it would manufacture a record whose start was never read.
        #
        # When the seek instead lands exactly on a line boundary this discards
        # a whole, valid record. That costs nothing when the file holds more
        # than ``limit`` records — the record given up is one the response was
        # never going to reach anyway. ``truncated`` is a property of the seek
        # position, not of ``len(raw)``, so this is exact even when the file
        # holds exactly one window's worth of bytes: that case reads from byte
        # 0 (``truncated`` is ``False``) and keeps its first line whole.
        lines = lines[1:]

    records: list[dict[str, Any]] = []
    for line in reversed(lines):
        if len(records) >= limit:
            break
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _ledgers(directory: Path) -> list[Path]:
    """The ledger files this response may read, newest-modified first.

    ``iterdir`` rather than a recursive walk, and an explicit ``is_file``: the
    zone holds one flat file per surface, and a directory that happens to end
    in ``.jsonl`` is not a ledger.

    ``is_file()`` follows symlinks, so it alone would enumerate a link named
    ``<anything>.jsonl`` and serve whatever it points at — another identity's
    ledger included. ``is_symlink()`` does not follow, so the pair keeps the
    read inside the directory the container is allowed to serve: a ledger is a
    real file that lives here, not a name that resolves elsewhere.
    """
    try:
        found = [
            entry
            for entry in directory.iterdir()
            if entry.name.endswith(LEDGER_SUFFIX) and not entry.is_symlink() and entry.is_file()
        ]
    except OSError:
        return []
    found.sort(key=lambda entry: (_mtime(entry), entry.name), reverse=True)
    return found[:MAX_LEDGERS]


def _mtime(path: Path) -> float:
    """Modification time of *path*, or ``0.0`` if it went away underneath us."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _select(directory: Path, surface: str | None) -> list[Path]:
    """The ledgers to read, after applying the ``surface`` filter.

    The filter is applied to the STEMS already enumerated from *directory* —
    the parameter is compared, never joined. That is the property that makes a
    traversal impossible rather than merely refused.
    """
    ledgers = _ledgers(directory)
    if surface is None:
        return ledgers
    return [entry for entry in ledgers if entry.name[: -len(LEDGER_SUFFIX)] == surface]


def _sort_key(record: dict[str, Any]) -> str:
    """Newest-first sort key for *record*: a non-string ``ts`` sorts oldest.

    The writer always emits ``ts`` as an ISO-8601 string, so this only matters
    for a hand-edited or otherwise foreign line — the only content this reader
    is tolerant of in the first place. Stringifying a non-string ``ts`` (e.g.
    an integer epoch) and comparing it lexicographically against real
    timestamps would let it sort newer than every genuine record; treating it
    as absent instead sorts it to the oldest end, which is the safer default.
    """
    ts = record.get("ts")
    return ts if isinstance(ts, str) else ""


def _validated_surface(surface: str | None) -> str | None:
    """*surface* if it can name one ledger, else a 400-shaped refusal.

    Raises:
        HTTPException: 400 when the value is not a single safe path component.
    """
    if surface is None:
        return None
    if surface in _RESERVED_STEMS or not _SAFE_SURFACE.match(surface):
        raise HTTPException(
            status_code=400,
            detail=(
                "surface must name a single audit ledger: letters, digits, dot, "
                "dash and underscore only, and no path separators"
            ),
        )
    return surface


@router.get("/api/audit/recent")
def recent_audit_records(
    request: Request,
    limit: str = str(DEFAULT_LIMIT),
    surface: str | None = None,
) -> dict[str, Any]:
    """Serve the newest audit records this container wrote, newest first.

    Args:
        request: Incoming request, for ``app.state`` and the tier gate.
        limit: How many records to return, taken as text and parsed after the
            gate; clamped into ``[MIN_LIMIT, MAX_LIMIT]`` rather than
            validated, so the gate stays the first thing that answers.
        surface: Optional ledger to read alone, matched against the stems this
            container has.

    Returns:
        ``{"identity", "ledgers", "limit", "records"}`` — the identity whose
        zone was read, the ledger stems it was read from, the effective limit,
        and the records themselves, verbatim as the writer stored them,
        newest-``ts``-first. A record without a usable ``ts`` (absent, ``null``,
        or not a string) sorts to the oldest end rather than being compared as
        text.

    Raises:
        HTTPException: 403 when the Config panel tier is disabled, 400 when
            *surface* is not a usable ledger name or *limit* is not a number.
    """
    _require_config_panel(request)

    wanted = _validated_surface(surface)
    effective = _parsed_limit(limit)

    try:
        directory = identity_dir()
    except Exception:
        # The audit zone is resolved through the workspace/config loader; a
        # deployment that cannot resolve one still gets a working panel.
        logger.warning("Could not resolve the audit zone; serving no records", exc_info=True)
        return {
            "identity": acting_identity(),
            "ledgers": [],
            "limit": effective,
            "records": [],
        }

    selected = _select(directory, wanted)
    records: list[dict[str, Any]] = []
    for ledger in selected:
        # Trimmed after every ledger, not just once at the end: each ledger's
        # own tail is already bounded to `effective` records, so leaving the
        # combined list untrimmed across MAX_LEDGERS ledgers would carry
        # MAX_LEDGERS * effective records at once. Trimming here keeps the
        # peak at one ledger's contribution plus `effective` records. The sort
        # is stable, so records sharing a second-resolution timestamp keep the
        # per-file newest-first order the tail already put them in, and ties
        # across ledgers keep the enumeration order ledgers were read in.
        records.extend(_records_from(ledger, effective))
        records.sort(key=_sort_key, reverse=True)
        del records[effective:]

    return {
        "identity": directory.name,
        "ledgers": [entry.name[: -len(LEDGER_SUFFIX)] for entry in selected],
        "limit": effective,
        "records": records,
    }
