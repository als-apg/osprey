"""The per-user bar layout, over HTTP.

Three verbs on one document — ``{version, rev, header, status,
header_visible, status_visible}`` — the arrangement of one operator's header and status bar:

* ``GET /api/bar-items`` answers what this deployment would render right now.
* ``PUT /api/bar-items`` saves an arrangement, conditional on the revision the
  editor was holding.
* ``DELETE /api/bar-items`` removes it, returning the operator to whatever
  their deployment renders.

The persistence, the validation and the revision arithmetic all live in
:mod:`~osprey.interfaces.web_terminal.bar_items_store`, which is deliberately
free of HTTP. This module is the other half of that split — the status ladder.
Every refusal answers ``{"error": …, "message": …}``, and ``error`` is the
machine-readable token; there are five rungs:

* **409** ``rev_conflict`` — the revision the client sent is not the one on
  disk. Another tab saved in between. The body carries the document that IS
  stored, so the client can adopt it, re-apply the edit the operator just made
  and retry once, rather than overwriting a save nobody saw.
* **413** ``too_large`` — the body is past :data:`MAX_REQUEST_BYTES`, refused
  before it is parsed.
* **422** — the document is not one this build can store, with the store's own
  ``reason`` naming the class: ``malformed``, ``version``, ``unknown-type``,
  ``duplicate``, ``overflow`` or ``bad-option``. The last five are spelled
  exactly as ``bar-layout.js`` spells its client-side drop reasons, so a
  browser log and a server refusal read alike. One reason is
  **this module's own** and is not in the store's vocabulary: ``bad-rev``, for
  a body carrying
  no usable ``rev``. It is kept distinct from ``malformed`` deliberately — a
  missing protocol field is a client bug worth logging loudly, a malformed
  document is a normalizer disagreement worth showing the operator, and a
  client should not have to read prose to tell them apart.
* **503** ``store_unavailable`` when there is nowhere to write — no agent-data
  root, no catalog, no lock — and ``store_write_failed`` when the write itself
  failed on a read-only mount or a volume that did not arrive. Nothing was
  saved either way, and saying so is the whole point: a preferences write that
  silently does nothing is a layout the operator re-does on every visit.
* **200** — the document as persisted, at its assigned revision.

**Two spellings, and the rule behind them.** The 422 family is kebab-case,
because those six reasons are the words ``bar-layout.js`` already uses for its
own drops. Every other rung is snake_case, because those are the tokens this
app spells the same failures with everywhere else — ``routes/websocket.py``
answers ``store_unavailable`` and ``store_write_failed`` for the posture store,
and ``control-target-facts.js`` matches on the first by name. So a caller that
already handles one of them handles this route's too. The full union is derived
and pinned in ``TestTheRefusalVocabulary``; a token this route can emit and
nothing else knows about is exactly what that class exists to catch.

**Why these handlers are ``async def`` when the store is blocking.** The lock
that serializes two tabs is an :class:`asyncio.Lock` on ``app.state``, so the
critical section has to be awaited. The store's own reads and writes are then
pushed to the threadpool with
:func:`~starlette.concurrency.run_in_threadpool`, because this event loop is
the one that pumps every connected PTY WebSocket and a blocking read on it
stalls everyone's terminal.

**What the lock actually protects.** ``save_layout`` reads the stored revision,
compares it and writes — three steps that are not atomic together, even though
the write itself is an ``os.replace``. Two saves interleaving there both read
the same revision and both write, and the loser's arrangement disappears with
no 409 to tell it so. The lock also spans the refresh of
``app.state.bar_items_effective``, the cache the server-side first paint reads,
so the file and the cache cannot end up describing different arrangements.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from osprey.interfaces.web_terminal.bar_items_store import (
    BarLayoutConflict,
    BarLayoutInvalid,
    BarVocabulary,
    load_layout,
    reset_layout,
    save_layout,
)

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 64 * 1024
"""Largest layout document one save may carry, in bytes of raw body.

The sibling store's route (``routes/feedback.py``) caps its submission and
answers 413 above it, and this route wants the same shape for the same reason:
the per-host item cap bounds what gets *stored*, never what gets *parsed*, so
without a ceiling a multi-megabyte array is fully decoded before ``_validate``
counts its entries and refuses it.

The figure is generous rather than tight. Two bars of
:data:`~osprey.interfaces.web_terminal.app.MAX_BAR_ITEMS_PER_HOST` items, each
entry carrying a type and its completed options, is a few kilobytes; 64 KiB
leaves an order of magnitude for an item type that grows a richer option set
without anyone having to remember this constant exists. It is well under the
4 MB ``client_max_body_size`` the multi-user nginx applies, so this is the
stricter of the two and is the only one a bare uvicorn has at all.
"""


def _effective(request: Request) -> dict:
    """The document this request would render.

    Imported at call time, the way ``routes/panels.py`` reaches
    :data:`~osprey.interfaces.web_terminal.app.FAMILY_RAIL_DEFAULTS`: ``app.py``
    imports this package to mount the router, so a module-level import here
    would close the cycle.
    """
    from osprey.interfaces.web_terminal.app import effective_bar_layout

    return effective_bar_layout(request.app)


def _deployment_default(request: Request) -> dict:
    """This deployment's own arrangement, ignoring anything the operator saved.

    The fallback a *load* is given, and deliberately not :func:`_effective`:
    used as a load's default, the operator's cached document would be handed
    back as though it had been read from disk. What a client is owed when
    nothing is stored is what the deployment renders.
    """
    from osprey.interfaces.web_terminal.app import DEFAULT_BAR_LAYOUT

    configured = getattr(request.app.state, "bar_layout", None)
    return configured if isinstance(configured, dict) else DEFAULT_BAR_LAYOUT


def _refuse(status_code: int, error: str, message: str, **extra: Any) -> HTTPException:
    """The refusal body every rung of the ladder answers with.

    Returned rather than raised so a call site reads ``raise _refuse(...)`` and
    control flow is visible at that line — the shape ``routes/websocket.py``
    uses for the control gestures.
    """
    return HTTPException(
        status_code=status_code, detail={"error": error, "message": message, **extra}
    )


def _store_dir(request: Request) -> Path:
    """Where this operator's document lives.

    Raises:
        HTTPException: 503 when the lifespan resolved no store. A deployment
            with no agent-data root cannot keep a layout, and that is the same
            answer an unwritable one gets: nothing was saved.
    """
    configured = getattr(request.app.state, "bar_items_dir", None)
    if not configured:
        raise _refuse(
            503,
            "store_unavailable",
            "This deployment has nowhere to keep a bar layout, so nothing was saved.",
        )
    return Path(configured)


def _vocabulary(request: Request) -> BarVocabulary:
    """The item types, schema version and per-host cap a save is checked against.

    Built once by the lifespan from the same tables the server-side first paint
    renders from, so a document this route accepts is one that paint can draw.

    Raises:
        HTTPException: 503 when the lifespan built none — the same "nothing was
            saved" answer, because validating against a guessed catalog would
            be worse than refusing.
    """
    vocabulary = getattr(request.app.state, "bar_items_vocabulary", None)
    if not isinstance(vocabulary, BarVocabulary):
        raise _refuse(
            503,
            "store_unavailable",
            "This deployment did not resolve its bar-item catalog, so nothing was saved.",
        )
    return vocabulary


def _lock(request: Request) -> asyncio.Lock:
    """The lock serializing writes for this app.

    Created once by the lifespan, before any request can reach a route.

    A missing one is refused rather than replaced. Building a lock here would
    look like a safety net and would not be one: two concurrent requests would
    both take that branch, each build a private lock, and each overwrite the
    other's on state — nothing serialized, and a log line claiming otherwise.
    Refusing is the honest answer, and it is the same one :func:`_store_dir`
    and :func:`_vocabulary` give for the other pieces of lifespan state.

    Raises:
        HTTPException: 503 when the lifespan left no lock.
    """
    lock = getattr(request.app.state, "bar_items_lock", None)
    if not isinstance(lock, asyncio.Lock):
        raise _refuse(
            503,
            "store_unavailable",
            "This deployment cannot serialize bar-layout saves, so nothing was saved.",
        )
    return lock


def _requested_rev(body: Any) -> int:
    """The revision the editor believed it was holding.

    Required, never defaulted. A PUT with no revision is an unconditional
    overwrite, and the one thing this contract exists to prevent is a save
    landing on top of one its author never saw.

    The two refusals carry **different reasons on purpose**. A body that is not
    an object is ``malformed`` — the word the store itself uses for a document
    that is not a layout, and the right one to show an operator. A body that is
    an object but carries no usable ``rev`` is ``bad-rev``, this module's own
    reason: the document may be perfectly good and the *protocol field* is
    missing, which is a client bug, not a normalizer disagreement. Collapsing
    the two would leave the sync layer parsing prose to tell them apart.

    Raises:
        HTTPException: 422 ``malformed`` when the body is not an object; 422
            ``bad-rev`` when its ``rev`` is absent or is not a whole number.
            ``bool`` is an ``int`` in Python and ``True`` is not a revision.
    """
    if not isinstance(body, dict):
        raise _refuse(422, "malformed", "a bar layout must be an object")
    rev = body.get("rev")
    if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
        raise _refuse(
            422,
            "bad-rev",
            f"rev must be the revision this edit was made against, not {rev!r}",
        )
    return rev


async def _body(request: Request) -> Any:
    """The parsed request body.

    Read raw rather than through a pydantic model on purpose: the store owns
    this document's shape, and a model here would answer a malformed layout
    with FastAPI's validation body instead of the store's ``reason`` — two
    vocabularies for one refusal.

    The ceiling is checked twice: against ``Content-Length`` first, so an
    oversized save is refused without reading it, and against the bytes
    actually received, because a chunked request declares no length. Only then
    is anything parsed.

    Raises:
        HTTPException: 413 when the body is over :data:`MAX_REQUEST_BYTES`; 422
            when it is not JSON at all.
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
        raise _too_large(int(declared))

    raw = await request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        raise _too_large(len(raw))

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _refuse(422, "malformed", f"the request body is not JSON: {exc}") from exc


def _too_large(size: int) -> HTTPException:
    """The 413 for a body past :data:`MAX_REQUEST_BYTES`."""
    return _refuse(
        413,
        "too_large",
        f"this bar layout is {size} bytes; the most one save may carry is {MAX_REQUEST_BYTES}",
    )


@router.get("/api/bar-items")
async def read_bar_items(request: Request) -> dict:
    """Answer the arrangement this deployment would render right now.

    Served from ``app.state.bar_items_effective``, the cache the lifespan
    populates once and every accepted write below refreshes — the disk is not
    touched, because this is on the path of every reconnecting tab.

    A deployment with no store still answers: what renders when nothing is
    saved is exactly the deployment default, and a client that cannot read the
    layout cannot draw the bars at all.

    The document comes back by reference and is **serialized, never touched**.
    It is the process-wide cache every server-side first paint renders from, so
    a handler that annotated or normalized it in place would corrupt every
    later render for the life of the process. Writes go through the store,
    which hands back a copy of its own.

    Returns:
        ``{version, rev, header, status, header_visible, status_visible}``. ``rev`` is ``0``
        when the operator has saved nothing, which is what a first conditional
        save should send back.
    """
    return _effective(request)


@router.put("/api/bar-items")
async def write_bar_items(request: Request) -> dict:
    """Save this operator's arrangement, conditional on ``rev``.

    Everything from reading the stored revision to refreshing the render cache
    happens under ``app.state.bar_items_lock``. Two tabs saving in the same
    moment therefore serialize: the first lands at the next revision, the
    second is told its revision is stale and is handed the document that won.

    Returns:
        The document as persisted, at its assigned revision.

    Raises:
        HTTPException: 409 when ``rev`` is not the stored revision, with the
            current document under ``layout``; 413 when the body is over
            :data:`MAX_REQUEST_BYTES`; 422 ``bad-rev`` when no usable ``rev``
            was sent, or the store's own ``reason`` when the document is not
            one this build can store; 503 when the store cannot be written and
            nothing was saved.
    """
    body = await _body(request)
    expected_rev = _requested_rev(body)
    store_dir = _store_dir(request)
    vocabulary = _vocabulary(request)

    async with _lock(request):
        try:
            saved = await run_in_threadpool(
                save_layout,
                store_dir,
                body,
                vocabulary=vocabulary,
                expected_rev=expected_rev,
            )
        except BarLayoutInvalid as exc:
            raise _refuse(422, exc.reason, str(exc)) from exc
        except BarLayoutConflict as exc:
            # Read inside the lock, so the document handed back is the one that
            # actually beat this save rather than whatever a third writer left
            # behind while the refusal was being composed.
            current = await run_in_threadpool(
                load_layout,
                store_dir,
                vocabulary=vocabulary,
                default=_deployment_default(request),
            )
            raise _refuse(
                409,
                "rev_conflict",
                (
                    "This bar layout was saved elsewhere while you were editing it. "
                    "Reload the arrangement below and apply your change again."
                ),
                layout=current,
            ) from exc
        except OSError as exc:
            logger.warning(
                "Could not save the bar layout to %s; nothing was written", store_dir, exc_info=True
            )
            raise _refuse(
                503,
                "store_write_failed",
                (
                    "The bar layout could not be written, so nothing was saved. "
                    "Check the server's write access to the agent-data root."
                ),
            ) from exc

        request.app.state.bar_items_effective = saved

    return saved


@router.delete("/api/bar-items")
async def clear_bar_items(request: Request) -> dict:
    """Discard this operator's arrangement and return them to the default.

    A delete, never a write of the deployment's own layout: "saved nothing" and
    "saved a document that happens to match the deployment" behave differently
    from here on. Only the first follows the deployment when ``web.bar_items``
    is next edited, and only the first is what another reset would restore. The
    cache is set back to ``None`` for the same reason.

    Removing a document that is not there is not an error — a client resetting a
    layout it never saved gets the same answer as one resetting a layout it did.

    Returns:
        The deployment default, in the same envelope ``GET`` answers with.

    Raises:
        HTTPException: 503 when the document exists and cannot be removed. The
            arrangement is still stored, and the cache is left describing it.
    """
    store_dir = _store_dir(request)

    async with _lock(request):
        try:
            await run_in_threadpool(reset_layout, store_dir)
        except OSError as exc:
            logger.warning(
                "Could not remove the bar layout at %s; it is still stored",
                store_dir,
                exc_info=True,
            )
            raise _refuse(
                503,
                "store_write_failed",
                (
                    "The saved bar layout could not be removed, so it is still in place. "
                    "Check the server's write access to the agent-data root."
                ),
            ) from exc

        request.app.state.bar_items_effective = None
        # Read here, not after the lock: a save landing in that window would
        # make this reset answer the arrangement it just discarded. The whole
        # sequence stays inside the critical section the module docstring
        # claims for it.
        restored = _effective(request)

    return restored
