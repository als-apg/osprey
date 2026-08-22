"""Feedback submission routes.

Two endpoints, one composer:

* ``POST /api/feedback`` writes one record on this deployment and returns the
  finished clipboard payload. Every channel writes a record — a GitHub or email
  submission leaves the same paper trail a Local one does, which is what makes
  ``osprey feedback list`` the complete submission history.
* ``POST /api/feedback/bundle`` renders the session context alone for the
  standalone Copy button and writes nothing.

Three things about these handlers are deliberate and easy to undo by accident:

* **They are sync ``def``, not ``async def``.** FastAPI runs a sync handler in
  the threadpool. Composing reads the session transcript, which fans out into
  every subagent file (``1 + N`` reads), and the event loop these routes share
  is the same one that pumps the PTY WebSocket — a blocking read on it stalls
  everyone's terminal.
* **The session id is validated before any transcript path is built.** It is
  client-supplied and lands in ``transcript_dir / f"{session_id}.jsonl"``
  unsanitised, so a non-UUID is refused with 422 up front.
* **Identity is taken from the server, never from the request body.** A report
  says who filed it because the deployment knows, not because the browser
  claimed it.
"""

from __future__ import annotations

import getpass
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.interfaces.web_terminal.feedback_composer import (
    assemble_payload,
    bundle_for_copy,
    collect_material,
    first_line,
    render_bundle,
    validate_session_id,
)
from osprey.interfaces.web_terminal.feedback_store import (
    new_record_id,
    prune_store,
    write_record,
)

router = APIRouter()
logger = logging.getLogger(__name__)

RECORD_CONTEXT_BUDGET_BYTES = 5 * 1024 * 1024
"""Byte budget for the context kept in the record — far larger than the
clipboard payload's, because the record is the copy that keeps everything the
truncated report had to leave out."""

EXCERPT_CHARS = 200
"""How much of the report the header carries, for the ``osprey feedback list``
table. The full text lives in the paired context document."""

DEFAULT_MAX_STORE_BYTES = 256 * 1024 * 1024
"""Mirrors ``app.DEFAULT_FEEDBACK_MAX_STORE_BYTES`` (256 MB). Spelled as a
literal here, as routes do for every app.state default, to keep routes from
importing the app module."""

MAX_REQUEST_BYTES = 4 * 1024 * 1024
"""Largest report + scrollback one submission may carry, in UTF-8 bytes.

Deliberately the same figure as ``client_max_body_size 4m`` in
``templates/modules/web_terminals/nginx.conf.j2``, and **the two must move
together** — ``tests/deployment/web_terminals/test_nginx_feedback_body_size.py``
pins the nginx side. They are not the same measurement, and the difference is
worth stating rather than smoothing over:

* nginx weighs the **raw HTTP body**, which is strictly larger than what is
  counted here (JSON framing, escapes, the other fields).
* This check weighs the **decoded fields that reach disk**, and only those:
  a send that leaves the context box unticked carries no scrollback to count.
* Starlette has already buffered and parsed the whole body by the time a
  handler runs, so this bounds the *record*, not the process's peak memory.

The skew runs in the safe direction: in a multi-user deployment nginx is the
stricter, binding constraint and refuses the request before OSPREY sees it.
This check is the single-user backstop — a bare uvicorn has no body limit at
all, and without it the same client could spool an unbounded record there and
not in production."""

OUTBOUND_CHANNELS = ("github", "email")
"""Channels whose payload leaves this host, and whose record therefore carries
the size and digest of the exact string the operator pasted."""


class FeedbackRequest(BaseModel):
    """Body of ``POST /api/feedback``.

    Pinned by ``static/js/feedback-client.js``. ``session_id`` and
    ``scrollback`` travel only when the context checkbox is ticked, and a
    session that does not exist yet omits ``session_id`` entirely — so both are
    optional here and both are ignored unless ``include_context`` is set.
    """

    text: str
    channel: Literal["local", "github", "email"]
    include_metadata: bool = True
    include_context: bool = False
    session_id: str | None = None
    scrollback: str | None = None


class BundleRequest(BaseModel):
    """Body of ``POST /api/feedback/bundle``.

    ``text_len`` is the UTF-8 **byte** count of the text box (the browser
    measures it with ``TextEncoder``); it only sizes the bundle, and is clamped
    server-side so it cannot lift the payload cap.

    ``include_metadata`` is the same checkbox the send route reads, so a
    copied bundle carries exactly the tiers a sent one would. It defaults to
    ``True`` because that is what an older client omitting the field used to
    get, and because the deployment tier is the bundle's only always-present
    section.
    """

    session_id: str | None = None
    text_len: int | None = None
    scrollback: str | None = None
    include_metadata: bool = True


def _validated_session(session_id: str | None) -> str | None:
    """Return a canonical session UUID, ``None`` for "no session", else 422."""
    if session_id is None or session_id == "":
        return None
    try:
        return validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid session_id: {exc}") from exc


def _encodable(text: str | None) -> str:
    """Return *text* with anything UTF-8 cannot encode replaced.

    A lone surrogate reaches here intact — JSON can carry ``\\ud800`` and
    pydantic will hand it over as a ``str`` — and every later step disagrees
    about it: the composer encodes with ``surrogatepass``, ``json.dump`` writes
    it happily, and Starlette's response renderer raises ``UnicodeEncodeError``
    *after* the handler returns, i.e. after the record is on disk and the store
    has been pruned.

    This covers the **request** half only: the report and the scrollback the
    browser sent. The other half arrives from disk — chat turns, tool and agent
    names, artifact titles — and is substituted inside
    ``feedback_composer._BundleWriter.result()``, which is the one place that
    can do it without invalidating the byte budget its sections were trimmed
    to. Between the two, every string this route stores, digests or returns is
    encodable. Note the substitute is ``"?"``, what the ``replace`` error
    handler emits on the *encode* side — not ``U+FFFD``.
    """
    return (text or "").encode("utf-8", "replace").decode("utf-8")


def _enforce_request_size(*parts: str) -> None:
    """Refuse a request whose payload exceeds :data:`MAX_REQUEST_BYTES`.

    Raises:
        HTTPException: 413 when the combined UTF-8 length is over the ceiling.
    """
    total = sum(len(part.encode("utf-8", "replace")) for part in parts)
    if total > MAX_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"feedback submission is too large ({total} bytes); "
                f"the limit is {MAX_REQUEST_BYTES} bytes"
            ),
        )


def _utc_now() -> datetime:
    """Current UTC instant.

    A seam, not a wrapper: the deployment tier of every bundle carries a
    "submitted" stamp, so a test comparing two renderings byte-for-byte has to
    be able to hold the clock still.
    """
    return datetime.now(UTC)


def _identity(request: Request) -> str:
    """Who filed this report, as the deployment sees it.

    ``OSPREY_TERMINAL_USER`` is what a multi-user deployment sets per container;
    a developer laptop has no such value, so the process account stands in. A
    container with neither (no ``pwd`` entry for the uid is normal in a slim
    image) reports ``"unknown"`` rather than failing the submission.
    """
    terminal_user = getattr(request.app.state, "terminal_user", "")
    if isinstance(terminal_user, str) and terminal_user.strip():
        return terminal_user.strip()
    try:
        return getpass.getuser() or "unknown"
    except Exception:  # noqa: BLE001 — an unresolvable account must not lose the report
        logger.debug("feedback: no process account available for identity", exc_info=True)
        return "unknown"


def _version() -> str:
    """The running OSPREY version, or ``"unknown"`` if it cannot be read."""
    try:
        from osprey import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 — a version lookup must not lose the report
        logger.debug("feedback: could not read the OSPREY version", exc_info=True)
        return "unknown"


def _feedback_dir(request: Request) -> Path:
    """Where records are written.

    Set by the lifespan to ``resolve_shared_data_root() / "feedback"``, which is
    deliberately *not* derived from ``workspace_dir`` (that follows
    ``web_terminal.watch_dir`` and would land records outside the agent-data
    volume the CLI reads). ``workspace_dir / "feedback"`` is the fallback.

    Both attributes are read the way every other route reads app.state — with
    a ``getattr`` default — so an app.state carrying neither gets a 5xx naming
    the missing configuration rather than an ``AttributeError`` from inside the
    handler.

    Raises:
        HTTPException: 500 when neither ``feedback_dir`` nor ``workspace_dir``
            is set, because then there is nowhere on this host to put a record.
    """
    configured = getattr(request.app.state, "feedback_dir", None)
    if configured:
        return Path(configured)
    workspace_dir = getattr(request.app.state, "workspace_dir", None)
    if workspace_dir:
        return Path(workspace_dir) / "feedback"
    raise HTTPException(
        status_code=500, detail="no feedback store is configured on this deployment"
    )


def _sources(request: Request) -> tuple[Any, Any]:
    """Build the transcript reader and artifact store for this request.

    Both are per-request and imported lazily, the convention every
    web-terminal route follows. The reader must be rooted at
    ``project_cwd``: one built on any other directory reports every session as
    missing, silently.
    """
    from osprey.mcp_server.workspace.transcript_reader import TranscriptReader
    from osprey.stores.artifact_store import ArtifactStore

    return (
        TranscriptReader(request.app.state.project_cwd),
        ArtifactStore(getattr(request.app.state, "workspace_dir", None)),
    )


def _metadata(
    request: Request, identity: str, now: datetime, *, include_metadata: bool
) -> dict[str, Any]:
    """The deployment tier of the bundle, in render order.

    Who and when are always present — the dialog's help popover says the report
    always carries the text, the timestamp and the username when it is known.
    The checkbox governs the rest: which OSPREY this is and what the deployment
    calls itself.

    ``now`` is passed in rather than read here so that one request stamps one
    instant: the record's ``created_at`` and the bundle's ``submitted`` line
    describe the same submission and must not disagree by a few microseconds.
    """
    metadata: dict[str, Any] = {
        "submitted": now.isoformat(),
        "user": identity,
    }
    if include_metadata:
        metadata["osprey version"] = _version()
        app_name = getattr(request.app.state, "app_name", "")
        if app_name:
            metadata["app"] = str(app_name)
    return metadata


def _excerpt(text: str) -> str:
    """First line of the report, capped — the ``feedback list`` table column.

    ``first_line`` also strips control characters, which matters more here than
    anywhere else in this module: the excerpt is printed into a maintainer's
    terminal by ``osprey feedback list``, and the text it is cut from was typed
    into a browser.
    """
    return first_line(text, EXCERPT_CHARS)


@router.post("/api/feedback")
def submit_feedback(request: Request, body: FeedbackRequest) -> dict[str, Any]:
    """Record one feedback submission and return the clipboard payload.

    The response is ``{id, payload, context_status}``: ``payload`` is the
    finished clipboard string — report text, separator, composed context —
    already capped at 60,000 UTF-8 bytes server-side, so the browser writes it
    verbatim and does no size arithmetic of its own.

    Every channel writes a record. Local *is* the delivery for an air-gapped
    deployment; GitHub and email additionally carry the byte count and SHA-256
    of the payload, so a maintainer can tell whether the issue they are reading
    is the whole of what was composed.

    Raises:
        HTTPException: 413 when the report and scrollback together exceed
            :data:`MAX_REQUEST_BYTES`, 422 for a malformed ``session_id``, 500
            when the record cannot be written — the record is the deliverable,
            so a failed write is never reported as success.
    """
    text = _encodable(body.text)
    scrollback = _encodable(body.scrollback) if body.include_context else ""
    _enforce_request_size(text, scrollback)

    # Resolved up front with the other refusals: a deployment with nowhere to
    # put records fails every submission, and it should not read the whole
    # transcript (1 + N files, once per subagent) before saying so.
    feedback_dir = _feedback_dir(request)

    session_id = _validated_session(body.session_id) if body.include_context else None
    identity = _identity(request)
    now = _utc_now()

    reader, store = _sources(request)
    material = collect_material(
        reader,
        store,
        session_id,
        scrollback,
        _metadata(request, identity, now, include_metadata=body.include_metadata),
    )

    record_context, context_truncated = render_bundle(material, RECORD_CONTEXT_BUDGET_BYTES)
    record_id = new_record_id()
    payload, _bundle, text_truncated = assemble_payload(text, material, record_id=record_id)

    header: dict[str, Any] = {
        "id": record_id,
        "identity": identity,
        "created_at": now.isoformat(),
        "channel": body.channel,
        "version": _version(),
        "app_name": str(getattr(request.app.state, "app_name", "") or ""),
        "session_id": session_id,
        "include_metadata": body.include_metadata,
        "include_context": body.include_context,
        "excerpt": _excerpt(text),
        "context_status": material.context_status,
        "text_truncated": text_truncated,
        "context_truncated": context_truncated,
    }
    if body.channel in OUTBOUND_CHANNELS:
        encoded = payload.encode("utf-8", "surrogatepass")
        header["bundle_bytes"] = len(encoded)
        header["bundle_sha256"] = hashlib.sha256(encoded).hexdigest()

    try:
        # The id is allocated here, not by the store: a truncated payload names
        # it, and `bundle_sha256` above digests those exact bytes.
        write_record(
            feedback_dir,
            header,
            {"text": text, "context": record_context, "scrollback": scrollback},
            record_id=record_id,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as a 5xx, never swallowed
        logger.warning("feedback: could not write a record to %s", feedback_dir, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"could not record the feedback submission: {exc}"
        ) from exc

    try:
        prune_store(feedback_dir, _max_store_bytes(request))
    except Exception:  # noqa: BLE001 — the record is written; a full store is not this user's problem
        logger.warning("feedback: could not prune %s", feedback_dir, exc_info=True)

    return {"id": record_id, "payload": payload, "context_status": material.context_status}


@router.post("/api/feedback/bundle")
def feedback_bundle(request: Request, body: BundleRequest) -> dict[str, str]:
    """Render the session context alone, writing no record.

    This backs the standalone "Copy session context" button. It uses the same
    composer and the same budget formula as the send route, so the copied
    bundle is byte-identical to the bundle section of the payload a send would
    produce for the same session, the same amount of typed text and the same
    metadata checkbox — one checkbox, one meaning, on both buttons.

    Raises:
        HTTPException: 413 when the scrollback exceeds
            :data:`MAX_REQUEST_BYTES`, 422 for a malformed ``session_id``.
    """
    scrollback = _encodable(body.scrollback)
    _enforce_request_size(scrollback)

    session_id = _validated_session(body.session_id)
    identity = _identity(request)

    reader, store = _sources(request)
    material = collect_material(
        reader,
        store,
        session_id,
        scrollback,
        _metadata(request, identity, _utc_now(), include_metadata=body.include_metadata),
    )
    return {"bundle": bundle_for_copy(material, body.text_len)}


def _max_store_bytes(request: Request) -> int:
    """The store ceiling from ``web.feedback.max_store_bytes``.

    ``bool`` is rejected along with every other unusable value, matching
    ``app.coerce_store_ceiling``: ``True`` is an ``int`` in Python, and a
    1-byte ceiling would quietly prune every context document in the store.
    """
    ceiling = getattr(request.app.state, "feedback_max_store_bytes", DEFAULT_MAX_STORE_BYTES)
    if isinstance(ceiling, int) and not isinstance(ceiling, bool) and ceiling > 0:
        return ceiling
    return DEFAULT_MAX_STORE_BYTES
