"""Feedback submission routes.

Two endpoints, one composer:

* ``POST /api/feedback`` writes one ``feedback`` entry to the agent record and
  returns the finished clipboard payload. Every channel writes an entry — a
  GitHub or email submission leaves the same paper trail a Local one does,
  which is what makes ``osprey feedback list`` the complete submission history.
  The entry carries the header, the text, the context and scrollback the user
  chose, and the keys of the session's entries in place of excerpts of them.
* ``POST /api/feedback/bundle`` renders the session context alone for the
  standalone Copy button and writes nothing.

Three things about these handlers are deliberate and easy to undo by accident:

* **They are sync ``def``, not ``async def``.** FastAPI runs a sync handler in
  the threadpool. Composing reads the session's timeline from the record,
  waiting on a drain of the session, and the event loop these routes share is
  the same one that pumps the PTY WebSocket — a blocking read on it stalls
  everyone's terminal.
* **The session id is validated before the record is asked.** It is
  client-supplied, so a non-UUID is refused with 422 up front.
* **Identity is taken from the server, never from the request body.** A report
  says who filed it because the deployment knows, not because the browser
  claimed it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from osprey.agent_record import writer as record_writer
from osprey.agent_record.client import RecordClient
from osprey.agent_record.contract import KIND_FEEDBACK
from osprey.interfaces.web_terminal import transcript_map
from osprey.interfaces.web_terminal.feedback_composer import (
    assemble_payload,
    bundle_for_copy,
    collect_material,
    first_line,
    render_bundle,
    validate_session_id,
)
from osprey.interfaces.web_terminal.feedback_destination import osprey_version
from osprey_connectors.identity import acting_identity

router = APIRouter()
logger = logging.getLogger(__name__)

RECORD_CONTEXT_BUDGET_BYTES = 256 * 1024
"""Byte budget for the context kept in the entry: the deployment tier and the
artifact inventory. The session's steps are not in it — the entry references
them by key — so this bounds a list of titles, not a transcript."""

EXCERPT_CHARS = 200
"""How much of the report the ``excerpt`` field carries, for the ``osprey
feedback list`` table. The full text is the entry's ``text``."""

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

OUTBOUND_CHANNELS = ("github", "gitlab", "email")
"""Channels whose payload leaves this host, and whose record therefore carries
the size and digest of the exact string the operator pasted."""

TRACKER_CHANNELS = ("github", "gitlab")
"""Channels that open a prefilled new-issue form on one of the deployment's
configured trackers (``web.feedback.trackers``); the record names which."""

MAX_TRACKER_CHARS = 512
"""Ceiling on the ``tracker`` field — an ``owner/name`` or a project URL."""


class FeedbackRequest(BaseModel):
    """Body of ``POST /api/feedback``.

    Pinned by ``static/js/feedback-client.js``. ``session_id`` and
    ``scrollback`` travel only when the context checkbox is ticked, and a
    session that does not exist yet omits ``session_id`` entirely — so both are
    optional here and both are ignored unless ``include_context`` is set.
    """

    text: str
    channel: Literal["local", "github", "gitlab", "email"]
    include_metadata: bool = True
    include_context: bool = False
    session_id: str | None = None
    scrollback: str | None = None
    tracker: str | None = Field(default=None, max_length=MAX_TRACKER_CHARS)
    """The tracker a ``github``/``gitlab`` send opened: the ``owner/name`` or
    the project URL, as configured. Ignored on the other channels."""


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


def _transcript_of(request: Request, session_id: str | None) -> str | None:
    """The transcript the session key points at: a ``/clear`` moves a key off its own."""
    if session_id is None:
        return None
    return transcript_map.get(request.app, session_id)


def _encodable(text: str | None) -> str:
    """Return *text* with anything UTF-8 cannot encode replaced.

    A lone surrogate reaches here intact — JSON can carry ``\\ud800`` and
    pydantic will hand it over as a ``str`` — and every later step disagrees
    about it: the composer encodes with ``surrogatepass``, ``json.dump`` writes
    it happily, and Starlette's response renderer raises ``UnicodeEncodeError``
    *after* the handler returns, i.e. after the entry is in the spool.

    This covers the **request** half only: the report and the scrollback the
    browser sent. The other half arrives from the record — chat turns, tool and agent
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

    ``app.state.terminal_user`` stays the first rung: the lifespan populates it
    from ``OSPREY_TERMINAL_USER``, but it is also settable directly, so it is
    the seam an embedder (and a test) can drive without touching the process
    environment. Everything below it is :func:`acting_identity`, the one ladder
    every audit surface reads, so a report and the records around it cannot name
    two different actors.

    **This changed the fallback deliberately.** A containerized web terminal
    with no terminal user used to file under the process account; it now files
    under ``OSPREY_AUDIT_IDENTITY``, the identity the deployment gave that
    container. The old answer — ``osprey`` or ``root`` — named nobody and
    disagreed with every other record the same container writes. A laptop, which
    sets neither marker, still gets its process account; a slim image with no
    ``pwd`` entry for its uid still reports ``"unknown"`` rather than failing the
    submission.
    """
    terminal_user = getattr(request.app.state, "terminal_user", "")
    if isinstance(terminal_user, str) and terminal_user.strip():
        return terminal_user.strip()
    return acting_identity()


def _version() -> str:
    """The running OSPREY version, or ``"unknown"`` if it cannot be read.

    Delegates so the record header and the deployment identity in the report
    body cannot disagree about which OSPREY this is.
    """
    return osprey_version()


def record_client() -> RecordClient:
    """The record client this process reads with, configured from its environment.

    Resolved per request and reached through the module, so a test replaces it
    with ``monkeypatch.setattr(feedback, "record_client", ...)``. Never raises:
    an unconfigured client answers every read unavailable, and the report then
    says the record did not answer.
    """
    return RecordClient.from_env(os.environ)


def _sources(request: Request) -> tuple[Any, Any]:
    """The record client and the artifact store for this request.

    Both are per-request; the store is imported lazily, the convention every
    web-terminal route follows.
    """
    from osprey.stores.artifact_store import ArtifactStore

    return record_client(), ArtifactStore(getattr(request.app.state, "workspace_dir", None))


def _new_entry_id() -> str:
    """A fresh entry id, allocated before composing: the payload names it."""
    return uuid.uuid4().hex


def _metadata(
    request: Request,
    identity: str,
    now: datetime,
    *,
    include_metadata: bool,
    session_id: str | None = None,
) -> dict[str, Any]:
    """The deployment tier of the bundle, in render order.

    Who and when are always present — the dialog's help popover says the report
    always carries the text, the timestamp and the username when it is known.
    The metadata checkbox governs which OSPREY this is, which build it is, what
    the deployment calls itself, and the submitting browser (from the
    ``User-Agent`` header — the popover names it, so it must not travel
    unticked). The session id rides with the context it belongs to: an outbound
    draft's body is just a pointer line, so this tier is where a maintainer
    finds the session a pasted report came from.

    The build lines and the escalation link are what let a facility maintainer
    forward a framework bug upstream without re-investigating the deployment
    first. They are resolved once at startup, not composed here, because
    nothing in them varies per report.

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
        # Lowercased to match this block's own case; the browser's builder
        # renders the same lines Title Case beside its own neighbours. One
        # definition, two local conventions — see DeploymentIdentity.
        deployment = getattr(request.app.state, "deployment_identity", None)
        if deployment is not None:
            for label, value in deployment.build_lines().items():
                metadata[label.lower()] = value
        app_name = getattr(request.app.state, "app_name", "")
        if app_name:
            metadata["app"] = str(app_name)
        browser = request.headers.get("user-agent", "").strip()
        if browser:
            metadata["browser"] = browser
        escalation_url = getattr(request.app.state, "feedback_escalation_url", "")
        if escalation_url:
            metadata["escalate to OSPREY"] = escalation_url
    if session_id:
        metadata["session"] = session_id
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
    verbatim and does no size arithmetic of its own. ``id`` is the entry id of
    the ``feedback`` entry.

    Every channel writes an entry. Local *is* the delivery for an air-gapped
    deployment; GitHub and email additionally carry the byte count and SHA-256
    of the payload, so a maintainer can tell whether the issue they are reading
    is the whole of what was composed.

    Raises:
        HTTPException: 413 when the report and scrollback together exceed
            :data:`MAX_REQUEST_BYTES`, 422 for a malformed ``session_id``, 500
            when the entry cannot be written — the entry is the deliverable,
            so a failed write is never reported as success.
    """
    text = _encodable(body.text)
    scrollback = _encodable(body.scrollback) if body.include_context else ""
    _enforce_request_size(text, scrollback)

    session_id = _validated_session(body.session_id) if body.include_context else None
    identity = _identity(request)
    now = _utc_now()
    transcript_id = _transcript_of(request, session_id)

    client, store = _sources(request)
    material = collect_material(
        client,
        store,
        session_id,
        scrollback,
        _metadata(
            request, identity, now, include_metadata=body.include_metadata, session_id=session_id
        ),
        transcript_id=transcript_id,
    )

    record_context, context_truncated = render_bundle(
        material, RECORD_CONTEXT_BUDGET_BYTES, excerpts=False
    )
    entry_id = _new_entry_id()
    payload, _bundle, text_truncated = assemble_payload(text, material, record_id=entry_id)

    entry: dict[str, Any] = {
        "id": entry_id,
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
    if body.channel in TRACKER_CHANNELS and body.tracker:
        entry["tracker"] = _encodable(body.tracker)
    if body.channel in OUTBOUND_CHANNELS:
        encoded = payload.encode("utf-8", "surrogatepass")
        entry["bundle_bytes"] = len(encoded)
        entry["bundle_sha256"] = hashlib.sha256(encoded).hexdigest()
    entry["text"] = text
    entry["context"] = record_context
    entry["scrollback"] = scrollback
    entry["refs"] = list(material.refs)

    # Filed under the session its context was read from, so the entry sits in
    # that session's timeline beside the steps it references.
    ids: dict[str, Any] = {"entry_id": entry_id, "ts": now}
    if session_id:
        ids["session_id"] = transcript_id or session_id
    # The id is allocated here, not by the writer: a truncated payload names
    # it, and `bundle_sha256` above digests those exact bytes.
    if record_writer.write(KIND_FEEDBACK, entry, **ids) is None:
        logger.warning("feedback: the feedback entry %s was not written to the spool", entry_id)
        raise HTTPException(
            status_code=500,
            detail="could not record the feedback submission: the agent record's spool "
            "did not take the entry (see the record's health)",
        )

    return {"id": entry_id, "payload": payload, "context_status": material.context_status}


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

    client, store = _sources(request)
    material = collect_material(
        client,
        store,
        session_id,
        scrollback,
        _metadata(
            request,
            identity,
            _utc_now(),
            include_metadata=body.include_metadata,
            session_id=session_id,
        ),
        transcript_id=_transcript_of(request, session_id),
    )
    return {"bundle": bundle_for_copy(material, body.text_len)}
