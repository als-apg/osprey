"""ARIEL Web API routes.

REST endpoints for search, entry management, status, and settings.
"""

from __future__ import annotations

import json as _json
import os
import re
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from osprey.imaging.formats import OCTET_STREAM
from osprey.interfaces.ariel.api.attachment_response import attachment_response
from osprey.interfaces.ariel.api.schemas import (
    AttachmentResponse,
    DiagnosticResponse,
    EmbeddingTableStatus,
    EntriesListResponse,
    EntryCreateRequest,
    EntryCreateResponse,
    EntryResponse,
    ExpandedTermResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from osprey.utils.config import get_facility_timezone, to_facility_iso
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search import ARIELSearchService
    from osprey.services.ariel_search.models import AttachmentInfo, EnhancedLogbookEntry

router = APIRouter(prefix="/api")
logger = get_logger("ariel")


def _parse_metadata_form(raw: str | None) -> dict[str, Any]:
    """Parse a JSON metadata string from a form field, returning {} on failure."""
    if not raw:
        return {}
    try:
        parsed = _json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def _localize_facility(dt: datetime | None) -> datetime | None:
    """Attach the facility timezone to a naive operator-provided datetime.

    Naive dates are facility-local wall-clock (never box-local / UTC) before
    they drive a ``TIMESTAMPTZ`` query.
    """
    from osprey.utils.config import localize_facility

    return localize_facility(dt)


def _require_service(request: Request) -> ARIELSearchService:
    """Get the ARIEL service or raise 503 if the database is unavailable.

    The database text is unchanged: this is reached only when the database is
    genuinely unreachable, never because the configuration is broken (a
    configuration problem leaves the service constructed and degrades only the
    search path). When both are true the stored configuration errors are
    appended, so an operator staring at a 503 sees every reason at once.
    """
    service = getattr(request.app.state, "ariel_service", None)
    if service is None:
        detail = (
            "Database unavailable — search, browse, and entry creation "
            "require a database connection. Drafts and settings still work."
        )
        errors = getattr(request.app.state, "config_errors", None) or []
        if errors:
            detail = f"{detail} Configuration errors: " + "; ".join(errors)
        raise HTTPException(status_code=503, detail=detail)
    return cast("ARIELSearchService", service)


_ATTACHMENT_ROUTE_PREFIX = "/api/attachments/"


def _safe_url(url: Any) -> str | None:
    """Return ``url`` when the page may link to it, else None.

    Only absolute http(s) urls and this API's own attachment routes pass; any
    other scheme (``javascript:``, ``file:``, ``data:``) or relative path is
    dropped so it never reaches an ``href`` or ``src``.
    """
    if not isinstance(url, str) or not url:
        return None
    if url.startswith(_ATTACHMENT_ROUTE_PREFIX):
        return url
    lowered = url.lower()
    if lowered.startswith(("http://", "https://")):
        return url
    return None


def _display_url(
    entry_id: str,
    summary: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    copy_state: bool,
) -> str | None:
    """Return where the web page shows or downloads one attachment from.

    ===========================================  ==========================================
    The attachment                               ``display_url``
    ===========================================  ==========================================
    viewable                                     ``/api/attachments/{id}/rendition``
    copied, not viewable                         ``/api/attachments/{id}``
    no copy state, native item                   ``/api/attachments/{id parsed from url}``
    anything else with an absolute http(s) url   that url
    anything else                                null
    ===========================================  ==========================================

    Args:
        entry_id: The entry the attachment belongs to.
        summary: The attachment's summary.
        item: The JSONB attachment item the summary was built from.
        copy_state: Whether the store holds attachment copy state.

    Returns:
        The display url, or None.
    """
    from osprey.services.ariel_search.attachments import attachment_id_for, is_native_item

    attachment_id = summary.get("attachment_id")
    if attachment_id and summary.get("viewable"):
        return f"{_ATTACHMENT_ROUTE_PREFIX}{attachment_id}/rendition"
    if attachment_id and summary.get("copy_status") == "copied":
        return f"{_ATTACHMENT_ROUTE_PREFIX}{attachment_id}"
    if not copy_state and is_native_item(item):
        return f"{_ATTACHMENT_ROUTE_PREFIX}{attachment_id_for(entry_id, item)}"
    return summary.get("url")


def _entry_to_response(
    entry: Mapping[str, Any],
    *,
    attachment_rows: list[dict[str, Any]] | None,
    model_id: str | None,
    file_source: bool,
    score: float | None = None,
    highlights: list[str] | None = None,
) -> EntryResponse:
    """Convert a database entry to the web response model.

    Every attachment is kept, with its caption and visible text whole, matched
    attachments first. ``_matched_via`` and ``_matched_attachment_ids`` on the
    entry come out as ``matched_via`` and ``matched_attachment_ids``.

    Args:
        entry: The entry dict.
        attachment_rows: The entry's ``attachment_files`` rows, or None when the
            store holds no copy state.
        model_id: The configured caption model id (``caption_model_id``).
        file_source: Whether the entry's source resolves relative attachment
            paths (``file_source_for``).
        score: The search score, if any.
        highlights: The search highlights, if any.

    Returns:
        The entry response.
    """
    from osprey.services.ariel_search.attachments.summaries import (
        build_attachment_summary_pairs,
    )

    entry_id = str(entry.get("entry_id") or "")
    pairs = build_attachment_summary_pairs(
        entry,
        attachment_rows,
        None,
        entry.get("_matched_attachment_ids", ()),
        file_source=file_source,
        full_captions=True,
        model_id=model_id,
    )
    attachments = []
    for summary, item in pairs:
        display_url = _display_url(entry_id, summary, item, copy_state=attachment_rows is not None)
        attachments.append(
            AttachmentResponse(
                **{
                    **summary,
                    "url": _safe_url(summary.get("url")),
                    "display_url": _safe_url(display_url),
                }
            )
        )

    # Render the three timestamp fields facility-local (ISO with offset) via the
    # shared egress helper, so the web wire format matches the MCP path
    # (serialize_entry) instead of leaking the DB's raw UTC. The schema fields are
    # ``str`` so Pydantic passes these pre-formatted strings through unchanged.
    return EntryResponse(
        entry_id=entry["entry_id"],
        source_system=entry["source_system"],
        timestamp=to_facility_iso(entry["timestamp"]),
        author=entry.get("author", ""),
        raw_text=entry["raw_text"],
        attachments=attachments,
        metadata=entry.get("metadata", {}),
        created_at=to_facility_iso(entry["created_at"]),
        updated_at=to_facility_iso(entry["updated_at"]),
        summary=entry.get("summary"),
        keywords=entry.get("keywords", []),
        score=score,
        highlights=highlights or [],
        matched_via=list(entry.get("_matched_via") or ()),
        matched_attachment_ids=list(entry.get("_matched_attachment_ids") or ()),
    )


async def _attachment_rows_by_entry(
    service: ARIELSearchService, entries: list[Mapping[str, Any]]
) -> Mapping[str, list[dict[str, Any]]] | None:
    """Read the attachment rows of a page of entries in one call.

    A ``DatabaseQueryError`` from the reader is treated as a store without copy
    state: the entries keep their fallback summaries and the process logs the
    schema-gap warning once, so a failing reader never costs a result.

    Args:
        service: The ARIEL service whose repository is read.
        entries: The entries of the response; no read is made when empty.

    Returns:
        The rows per entry id, or None when the store holds no copy state.

    Raises:
        TypeError: If the reader returned something other than None or a dict.
    """
    from osprey.services.ariel_search.database.repository import read_attachment_rows

    if not entries:
        return {}
    return await read_attachment_rows(service.repository, [e["entry_id"] for e in entries])


async def _entry_responses(
    service: ARIELSearchService,
    entries: list[Mapping[str, Any]],
    *,
    with_scores: bool = False,
) -> list[EntryResponse]:
    """Convert a page of entries to responses with their attachment summaries.

    Args:
        service: The ARIEL service; its config resolves the caption model id and
            the file-source flag, and its repository supplies the rows.
        entries: The entries, in output order.
        with_scores: Carry each entry's ``_score`` and ``_highlights`` across.

    Returns:
        One response per entry, in the given order.
    """
    from osprey.services.ariel_search.attachments.compose import caption_model_id
    from osprey.services.ariel_search.attachments.summaries import file_source_for

    mapping = await _attachment_rows_by_entry(service, entries)
    model_id = caption_model_id(service.config)
    file_source = file_source_for(service.config)
    return [
        _entry_to_response(
            e,
            attachment_rows=None if mapping is None else mapping.get(e["entry_id"], []),
            model_id=model_id,
            file_source=file_source,
            score=e.get("_score") if with_scores else None,
            highlights=e.get("_highlights") if with_scores else None,
        )
        for e in entries
    ]


def _capabilities_modes(service: ARIELSearchService) -> list[str]:
    """List the search module names the capabilities endpoint advertises.

    Reuses the capabilities builder rather than re-deriving the module list, so
    the modes the API routes are exactly the ones the UI offers as tabs.

    Args:
        service: The ARIEL service whose config decides which modules are enabled.

    Returns:
        Enabled search module names, in registry order.
    """
    from osprey.services.ariel_search.capabilities import get_capabilities as _get_caps

    capabilities = _get_caps(service.config)
    return [mode["name"] for mode in capabilities["categories"]["direct"]["modes"]]


def _resolve_search_mode(service: ARIELSearchService, requested: str | None) -> str:
    """Resolve a requested search mode against the enabled search modules.

    The API takes a module name rather than a fixed enum, so a mode that names
    no enabled module is rejected outright instead of silently falling back to
    keyword search.

    Args:
        service: The ARIEL service handling the request.
        requested: Mode name from the request body, or ``None`` for the default.

    Returns:
        The normalized name of an enabled search module.

    Raises:
        HTTPException: 400 if the mode is malformed, or names no enabled module.
    """
    from osprey.services.ariel_search.models import normalize_search_mode

    if requested is None:
        mode = service.config.resolve_default_search_mode()
    else:
        try:
            mode = normalize_search_mode(requested)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    available = _capabilities_modes(service)
    if mode not in available:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown search mode '{mode}'. "
                f"Available modes: {', '.join(available) if available else '(none enabled)'}"
            ),
        )
    return mode


def _validate_hybrid_overrides(advanced_params: dict[str, Any]) -> None:
    """Reject malformed hybrid per-query overrides before the search runs.

    The search panel sends ``rerank`` and ``include_images`` from toggles and
    ``candidate_limit`` from a number field, so real traffic is already
    well-formed; a hand-written HTTP caller is not. A string ``"false"`` is
    truthy and would silently run the slow reranked path, or the picture lane,
    the caller asked to skip, and a zero or negative width is a nonsense
    retrieval size. Only the type is checked here: whether ``include_images``
    takes effect is the search service's to resolve. The wording matches the config-side parser, so an operator who sets
    the same value badly in ``config.yml`` reads the same sentence either way.

    A missing key -- and an explicit ``null``, which is how JSON spells the same
    thing -- means "use the configured default" and is left alone.

    Args:
        advanced_params: The request's mode-specific parameters.

    Raises:
        HTTPException: 400 naming the offending key and the value it got.
    """
    rerank = advanced_params.get("rerank")
    if rerank is not None and not isinstance(rerank, bool):
        raise HTTPException(
            status_code=400,
            detail=f"rerank must be a boolean, got {rerank!r}",
        )

    include_images = advanced_params.get("include_images")
    if include_images is not None and not isinstance(include_images, bool):
        raise HTTPException(
            status_code=400,
            detail=f"include_images must be a boolean, got {include_images!r}",
        )

    candidate_limit = advanced_params.get("candidate_limit")
    if candidate_limit is not None and (
        not isinstance(candidate_limit, int)
        or isinstance(candidate_limit, bool)
        or candidate_limit < 1
    ):
        raise HTTPException(
            status_code=400,
            detail=f"candidate_limit must be a positive integer, got {candidate_limit!r}",
        )


def _invalid_capabilities(config_errors: list[str], remedy: str | None) -> dict:
    """Build the capabilities payload for a configuration that did not parse.

    Shape-compatible with the normal payload (the frontend's ``Capabilities``
    typedef requires ``vocabulary``), but built from nothing: no config, no
    service and no database are needed to render the banner that tells the
    operator which key to fix.

    Args:
        config_errors: Every stored configuration error, in collection order.
        remedy: The class-derived operator action.

    Returns:
        The degraded capabilities payload.
    """
    return {
        "status": "configuration_invalid",
        "config_errors": list(config_errors),
        "remedy": remedy,
        "categories": {"direct": {"modes": []}},
        "default_mode": None,
        "shared_parameters": [],
        "vocabulary": {"enabled": False, "concepts": 0, "expand_by_default": False},
    }


@router.get("/capabilities")
async def get_capabilities(request: Request) -> dict:
    """Return available search modes and their tunable parameters.

    The frontend calls this at startup to dynamically render mode tabs and
    advanced options — and, when the configuration is broken, the banner that
    explains why search is dead. It is therefore service-independent: a
    ``configuration_invalid`` state answers 200 with the degraded payload
    without touching the service. A ``configuration_warning`` state has a
    working service, so it returns the normal payload with the three
    configuration keys added; if that service is missing the database really
    is down and ``_require_service`` raises 503 as it does everywhere else.
    An app whose state carries no configuration fields at all answers with the
    normal payload, with ``config_panel_enabled`` False. The payload names the
    zone every entry timestamp in this API is rendered in, so the page can read
    those times in it.
    """
    from osprey.interfaces.ariel.app import CONFIG_STATUS_INVALID, CONFIG_STATUS_WARNING
    from osprey.services.ariel_search.capabilities import get_capabilities as _get_caps

    status = getattr(request.app.state, "config_status", None)
    errors = getattr(request.app.state, "config_errors", None) or []
    remedy = getattr(request.app.state, "config_remedy", None)
    service = getattr(request.app.state, "ariel_service", None)

    # The Config panel's server gate, reported so the frontend can take the
    # Settings entry out of the display menu. The server refusal is the real
    # gate; this is its other half, never the only half. An app whose state
    # carries no flag is refused by that gate, so absence reads as False here.
    panel_enabled = bool(getattr(request.app.state, "config_panel_enabled", False))
    # Never raises: an invalid configuration answers UTC, which is also the
    # zone to_facility_iso renders in then.
    facility_timezone = get_facility_timezone().key

    if status == CONFIG_STATUS_INVALID or (errors and status is None and service is None):
        return {
            **_invalid_capabilities(errors, remedy),
            "config_panel_enabled": panel_enabled,
            "facility_timezone": facility_timezone,
        }

    service = _require_service(request)
    payload = {
        **_get_caps(service.config),
        "config_panel_enabled": panel_enabled,
        "facility_timezone": facility_timezone,
    }
    if errors:
        payload = {
            **payload,
            "status": status or CONFIG_STATUS_WARNING,
            "config_errors": list(errors),
            "remedy": remedy,
        }
    return payload


@router.get("/publish-info")
async def get_publish_info(request: Request) -> dict:
    """Describe the configured logbook's write capability for the create form.

    Lets the UI adapt its credential prompt to the actual adapter instead of
    showing fixed text: a logbook that requires authentication asks for
    credentials, a no-auth logbook publishes without them, and a read-only
    adapter saves to ARIEL only. ``requires_auth`` is reported as
    ``supports_write and requires_write_auth`` — a read-only adapter cannot
    publish, so credentials are irrelevant there.
    """
    service = _require_service(request)

    from osprey.services.ariel_search.exceptions import AdapterNotFoundError
    from osprey.services.ariel_search.ingestion import get_adapter

    try:
        adapter = get_adapter(service.config)
    except AdapterNotFoundError:
        # No ingestion adapter configured — entries can only be saved locally.
        return {"supports_write": False, "requires_auth": False, "source_system": None}

    return {
        "supports_write": adapter.supports_write,
        "requires_auth": adapter.supports_write and adapter.requires_write_auth,
        "source_system": adapter.source_system_name,
    }


@router.get("/filter-options/{field_name}")
async def get_filter_options(request: Request, field_name: str) -> dict:
    """Return distinct values for a filterable field.

    Used by dynamic_select parameters to populate dropdown options.
    """
    service = _require_service(request)

    field_methods = {
        "authors": "get_distinct_authors",
        "source_systems": "get_distinct_source_systems",
    }

    method_name = field_methods.get(field_name)
    if not method_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown filter field: {field_name}. Available: {', '.join(field_methods)}",
        )

    try:
        method = getattr(service.repository, method_name)
        values = await method()
        return {
            "field": field_name,
            "options": [{"value": v, "label": v} for v in values],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _expanded_terms(result: Any) -> list[ExpandedTermResponse]:
    """Map a search result's applied vocabulary expansion onto the wire model.

    Reports only what the executed search actually contained. Defensive about
    the shape: a result object that carries no real groups (a stub, or a module
    that never learned about expansion) yields an empty list rather than an
    error.

    Args:
        result: The service's search result.

    Returns:
        One response model per expanded span, in the order the service reported.
    """
    groups = getattr(result, "expanded_terms", None) or ()
    if not isinstance(groups, (tuple, list)):
        return []
    terms: list[ExpandedTermResponse] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        alternatives = group.get("alternatives") or ()
        if not isinstance(alternatives, (tuple, list)):
            alternatives = ()
        terms.append(
            ExpandedTermResponse(
                original=str(group.get("original", "")),
                alternatives=[str(a) for a in alternatives],
            )
        )
    return terms


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, search_req: SearchRequest) -> SearchResponse:
    """Execute search query.

    Routes to the search module named by ``mode``; an unknown or disabled mode
    is rejected with 400 rather than falling back to another module.
    """
    from osprey.services.ariel_search.database.repository import schema_behind_diagnostics
    from osprey.services.ariel_search.exceptions import PatternError, VocabularyError

    service = _require_service(request)
    start_time = time.time()

    # Validated before the try block so the 400 is not swallowed into a 500.
    service_mode = _resolve_search_mode(service, search_req.mode)
    if service_mode == "hybrid":
        _validate_hybrid_overrides(search_req.advanced_params)

    try:
        # advanced_params takes precedence over top-level filter fields
        adv = search_req.advanced_params
        start_date = adv.pop("start_date", None) or search_req.start_date
        end_date = adv.pop("end_date", None) or search_req.end_date
        author = adv.pop("author", None) or search_req.author
        source_system = adv.pop("source_system", None) or search_req.source_system

        if isinstance(start_date, str) and start_date:
            start_date = _localize_facility(datetime.fromisoformat(start_date))
        if isinstance(end_date, str) and end_date:
            end_date = _localize_facility(datetime.fromisoformat(end_date))

        time_range = None
        if start_date or end_date:
            time_range = (start_date, end_date)

        # Re-inject non-date filters into advanced_params for downstream use
        if author:
            adv["author"] = author
        if source_system:
            adv["source_system"] = source_system

        result = await service.search(
            query=search_req.query,
            max_results=search_req.max_results,
            time_range=time_range,
            mode=service_mode,
            advanced_params=adv,
        )

        execution_time = int((time.time() - start_time) * 1000)

        # A fused hybrid result may hold image-only entries beyond the text
        # hits; the page shows at most max_results, and its sources name
        # exactly the entries shown.
        page: list[Mapping[str, Any]] = [*result.entries[: search_req.max_results]]
        sources = (
            [entry["entry_id"] for entry in page]
            if any("_matched_via" in entry for entry in page)
            else list(result.sources)
        )
        entries = await _entry_responses(service, page, with_scores=True)
        schema_behind = await schema_behind_diagnostics(service.repository)

        return SearchResponse(
            entries=entries,
            answer=result.answer,
            sources=sources,
            search_modes_used=list(result.search_modes_used),
            reasoning=result.reasoning,
            total_results=len(entries),
            execution_time_ms=execution_time,
            diagnostics=[
                DiagnosticResponse(
                    level=d.level.value,
                    source=d.source,
                    message=d.message,
                    category=d.category,
                )
                for d in (*result.diagnostics, *schema_behind)
            ],
            expanded_terms=_expanded_terms(result),
        )

    except VocabularyError as e:
        # The deployment asked for vocabulary expansion and cannot have it.
        # 503, not 500: the service is up, this one path is unavailable until
        # the operator acts, and the detail names the key and that action.
        first = e.errors[0] if e.errors else e.message
        raise HTTPException(
            status_code=503,
            detail=f"{e.config_key}: {first}. {e.remedy}",
        ) from e
    except PatternError as e:
        # Defensive: the keyword module turns a rejected pattern into an ERROR
        # diagnostic, so this should not surface — if it ever does, it is the
        # operator's regex, not a server fault.
        detail = f"Invalid search pattern: {e.message}"
        if e.pattern:
            detail = f"Invalid search pattern '{e.pattern}': {e.message}"
        raise HTTPException(status_code=400, detail=detail) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/entries", response_model=EntriesListResponse)
async def list_entries(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    author: str | None = None,
    source_system: str | None = None,
) -> EntriesListResponse:
    """List entries with pagination and filtering, newest first."""
    service = _require_service(request)

    try:
        # Operator-supplied query params are parsed naive by FastAPI; interpret
        # them as facility-local before they hit the TIMESTAMPTZ column.
        local_start = _localize_facility(start_date)
        local_end = _localize_facility(end_date)

        # Count with the same filters so total_pages reflects the filtered set,
        # not the whole table.
        total = await service.repository.count_entries(
            start=local_start,
            end=local_end,
            author=author,
            source_system=source_system,
        )

        offset = max(page - 1, 0) * page_size
        entries = await service.repository.search_by_time_range(
            start=local_start,
            end=local_end,
            limit=page_size,
            offset=offset,
            author=author,
            source_system=source_system,
        )

        entry_responses = await _entry_responses(service, list(entries))

        total_pages = (total + page_size - 1) // page_size

        return EntriesListResponse(
            entries=entry_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/entries/{entry_id}", response_model=EntryResponse)
async def get_entry(request: Request, entry_id: str) -> EntryResponse:
    """Get a single entry by ID."""
    service = _require_service(request)

    try:
        entry = await service.repository.get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")

        return (await _entry_responses(service, [entry]))[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _auth_required_response(exc: Exception) -> JSONResponse:
    """Build a 401 that asks the operator for logbook credentials.

    The ``code`` discriminator lets the frontend show a credential prompt
    instead of a generic error, and keep the form populated for resubmission.
    """
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc), "code": "auth_required"},
    )


async def _publish_or_local(
    service: ARIELSearchService,
    facility_request: Any,
    *,
    fallback_metadata: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Publish an entry via the facility adapter, or save local-only if read-only.

    Returns ``(entry_id, source_system, sync_status, message)`` for the text
    entry. Only ``NotImplementedError`` (the adapter genuinely cannot write)
    falls back to a direct local DB insert — this is the legitimate use of the
    fallback. ``AuthenticationRequiredError`` (credentials needed) and
    ``IngestionError`` (the publish attempt failed) propagate to the caller so
    nothing is silently saved.

    Args:
        service: The ARIEL search service.
        facility_request: A ``FacilityEntryCreateRequest`` for the text entry.
        fallback_metadata: Fields for the local-only insert (``author``,
            ``raw_text``, ``metadata``) used when the adapter is read-only.
    """
    try:
        result = await service.create_entry(facility_request)
        return (
            result.entry_id,
            result.source_system,
            result.sync_status.value,
            result.message,
        )
    except NotImplementedError:
        logger.warning("Facility adapter does not support writes, falling back to direct DB insert")

        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        entry: EnhancedLogbookEntry = {
            "entry_id": entry_id,
            "source_system": "ARIEL Web",
            "timestamp": now,
            "author": fallback_metadata.get("author") or "Anonymous",
            "raw_text": fallback_metadata["raw_text"],
            "attachments": [],
            "metadata": fallback_metadata["metadata"],
            "created_at": now,
            "updated_at": now,
        }

        await service.repository.upsert_entry(entry)

        return (
            entry_id,
            "ARIEL Web",
            "local_only",
            f"Entry {entry_id} created (saved locally, not published to external logbook)",
        )


@router.post("/entries", response_model=EntryCreateResponse)
async def create_entry(
    request: Request,
    entry_req: EntryCreateRequest,
) -> EntryCreateResponse | JSONResponse:
    """Create a new logbook entry.

    Delegates to the facility adapter when write support is available. A
    read-only adapter falls back to a local-only save; a logbook that requires
    credentials returns 401 (so the UI can prompt) and a genuine publish failure
    returns 502 — neither silently saves local-only.
    """
    service = _require_service(request)

    from osprey.services.ariel_search.exceptions import (
        AuthenticationRequiredError,
        IngestionError,
    )
    from osprey.services.ariel_search.models import FacilityEntryCreateRequest

    facility_request = FacilityEntryCreateRequest(
        subject=entry_req.subject,
        details=entry_req.details,
        author=entry_req.author,
        logbook=entry_req.logbook,
        shift=entry_req.shift,
        tags=entry_req.tags,
        auth_user=entry_req.auth_user,
        auth_password=entry_req.auth_password,
    )

    try:
        entry_id, source_system, sync_status, message = await _publish_or_local(
            service,
            facility_request,
            fallback_metadata={
                "author": entry_req.author,
                "raw_text": f"{entry_req.subject}\n\n{entry_req.details}",
                "metadata": {
                    "logbook": entry_req.logbook,
                    "shift": entry_req.shift,
                    "tags": entry_req.tags,
                    "created_via": "ariel-web",
                    **(entry_req.metadata or {}),
                },
            },
        )
    except AuthenticationRequiredError as e:
        return _auth_required_response(e)
    except IngestionError as e:
        raise HTTPException(status_code=502, detail=f"Publish failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return EntryCreateResponse(
        entry_id=entry_id,
        message=message,
        sync_status=sync_status,
        source_system=source_system,
    )


#: The one 404 detail both attachment routes answer; it never echoes the id.
_PICTURE_NOT_AVAILABLE = "picture not available"


def _require_attachment_id(attachment_id: str) -> None:
    """Answer 404 for an id that is not an attachment id, before any lookup."""
    from osprey.services.ariel_search.attachments import ATTACHMENT_ID_RE

    if re.fullmatch(ATTACHMENT_ID_RE, attachment_id) is None:
        raise HTTPException(status_code=404, detail=_PICTURE_NOT_AVAILABLE)


@router.get("/attachments/{attachment_id}")
async def get_attachment(request: Request, attachment_id: str) -> Response:
    """Serve the stored original bytes of an attachment.

    The bytes are served through :func:`attachment_response`, which sniffs them:
    a raster picture opens inline, anything else downloads as a file. The stored
    MIME type is never used for serving.
    """
    service = _require_service(request)
    _require_attachment_id(attachment_id)

    try:
        original = await service.repository.get_attachment_original(attachment_id)
        if not original:
            raise HTTPException(status_code=404, detail=_PICTURE_NOT_AVAILABLE)
        return attachment_response(bytes(original["data"]), original.get("filename") or "file")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/attachments/{attachment_id}/rendition")
async def get_attachment_rendition(request: Request, attachment_id: str) -> Response:
    """Serve the stored display rendition of a viewable picture.

    Only bytes already stored are served; this route never renders. An
    attachment that is not viewable (no finished copy, a skipped or non-picture
    row, no rendition yet) answers 404.
    """
    from osprey.services.ariel_search.attachments.formats import is_viewable

    service = _require_service(request)
    _require_attachment_id(attachment_id)

    try:
        row = await service.repository.get_rendition(attachment_id)
        if not row or not is_viewable(row):
            raise HTTPException(status_code=404, detail=_PICTURE_NOT_AVAILABLE)
        return attachment_response(bytes(row["rendition_bytes"]), row.get("filename") or "file")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _store_and_link_attachments(
    service: ARIELSearchService,
    entry_id: str,
    staged: list[tuple[str, str | None, bytes]],
) -> int:
    """Persist staged ``(filename, mime_type, data)`` files and link them on the entry.

    Files are stored in ARIEL's own attachment store and referenced on the entry
    record. The adapter write contract carries no attachments, so they are never
    pushed to an external logbook — they live in ARIEL only. Each picture is
    stored with its rendition, so it is viewable as soon as the request returns.

    Returns:
        The number of attachments stored.
    """
    from osprey.services.ariel_search.attachments import store_native_attachment

    attachment_infos: list[AttachmentInfo] = []
    for filename, mime_type, data in staged:
        attachment_infos.append(
            await store_native_attachment(
                service.repository,
                entry_id,
                filename=filename,
                declared_mime=mime_type,
                data=data,
            )
        )

    if attachment_infos:
        entry = await service.repository.get_entry(entry_id)
        if entry is not None:
            entry["attachments"] = attachment_infos
            await service.repository.upsert_entry(entry)

    return len(attachment_infos)


@router.post("/entries/upload", response_model=EntryCreateResponse)
async def create_entry_with_attachments(
    request: Request,
    subject: str = Form(...),
    details: str = Form(...),
    author: str | None = Form(None),
    logbook: str | None = Form(None),
    shift: str | None = Form(None),
    tags: str = Form(""),
    metadata: str | None = Form(None),
    auth_user: str | None = Form(None),
    auth_password: str | None = Form(None),
    files: list[UploadFile] = File(default=[]),
) -> EntryCreateResponse | JSONResponse:
    """Create a new logbook entry with file attachments via multipart form.

    The text body is published through the facility adapter with the same
    semantics as ``POST /entries`` — a logbook that requires credentials returns
    401, a genuine publish failure returns 502, and a read-only adapter falls
    back to a local-only save. Attachments are then stored in ARIEL. Because the
    adapter write contract carries no attachments, they are never published to an
    external logbook; when the text body does publish externally, the response
    says so explicitly rather than silently dropping the files.
    """
    service = _require_service(request)

    from osprey.services.ariel_search.attachments import (
        AttachmentValidationError,
        guess_mime_type,
        validate_file_size,
    )
    from osprey.services.ariel_search.exceptions import (
        AuthenticationRequiredError,
        IngestionError,
    )
    from osprey.services.ariel_search.models import FacilityEntryCreateRequest

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    parsed_metadata = _parse_metadata_form(metadata)

    # Read and validate files up front — before any publish — so a rejected file
    # can never leave a half-published entry behind.
    staged: list[tuple[str, str | None, bytes]] = []
    for upload_file in files:
        if not upload_file.filename:
            continue
        data = await upload_file.read()
        try:
            validate_file_size(len(data), upload_file.filename)
        except AttachmentValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # A browser re-uploading a blob it downloaded declares octet-stream; that
        # says nothing about the file, so the name decides instead.
        declared = upload_file.content_type
        if not declared or declared == OCTET_STREAM:
            declared = guess_mime_type(upload_file.filename)
        mime_type = declared
        staged.append((upload_file.filename, mime_type, data))

    facility_request = FacilityEntryCreateRequest(
        subject=subject,
        details=details,
        author=author,
        logbook=logbook,
        shift=shift,
        tags=tag_list,
        auth_user=auth_user,
        auth_password=auth_password,
        metadata=parsed_metadata,
    )

    # Publish the text body (or save local-only for a read-only adapter).
    try:
        entry_id, source_system, sync_status, message = await _publish_or_local(
            service,
            facility_request,
            fallback_metadata={
                "author": author,
                "raw_text": f"{subject}\n\n{details}",
                "metadata": {
                    "logbook": logbook,
                    "shift": shift,
                    "tags": tag_list,
                    "created_via": "ariel-web",
                    **parsed_metadata,
                },
            },
        )
    except AuthenticationRequiredError as e:
        return _auth_required_response(e)
    except IngestionError as e:
        raise HTTPException(status_code=502, detail=f"Publish failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Store attachments in ARIEL and link them on the entry.
    attachment_count = await _store_and_link_attachments(service, entry_id, staged)

    # If the text published externally, be explicit that the files did not —
    # the adapter write contract carries no attachments.
    if attachment_count and sync_status != "local_only":
        message = (
            f"{message}. {attachment_count} attachment(s) saved to ARIEL only; "
            "attachments are not published to the external logbook."
        )

    return EntryCreateResponse(
        entry_id=entry_id,
        message=message,
        sync_status=sync_status,
        source_system=source_system,
        attachment_count=attachment_count,
    )


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request) -> StatusResponse:
    """Get service status and health information."""
    service = _require_service(request)

    try:
        status = await service.get_status()

        return StatusResponse(
            healthy=status.healthy,
            database_connected=status.database_connected,
            database_uri=status.database_uri,
            entry_count=status.entry_count,
            embedding_tables=[
                EmbeddingTableStatus(
                    table_name=t.table_name,
                    entry_count=t.entry_count,
                    dimension=t.dimension,
                    is_active=t.is_active,
                )
                for t in status.embedding_tables
            ],
            active_embedding_model=status.active_embedding_model,
            enabled_search_modules=status.enabled_search_modules,
            enabled_enhancement_modules=status.enabled_enhancement_modules,
            last_ingestion=status.last_ingestion,
            errors=status.errors,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Settings endpoints — config.yml and Claude Code setup files
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Find the OSPREY project root (directory containing config.yml)."""
    candidates = [
        Path(os.environ.get("CONFIG_FILE", "")).parent if os.environ.get("CONFIG_FILE") else None,
        Path("/app"),
        Path.cwd(),
    ]
    for p in candidates:
        if p and (p / "config.yml").exists():
            return p
    return Path.cwd()


def _config_path(request: Request) -> Path:
    """Return the config.yml the running panel actually loaded.

    The lifespan stores the resolved path on ``app.state.config_path``, so the
    editor writes the same file the loader read (and the same file a relative
    ``ariel.vocabulary.path`` resolves against). The candidate search survives
    only as the fallback for an app that never set it.

    Args:
        request: The incoming request, for its app state.

    Returns:
        Path to config.yml.
    """
    configured = getattr(request.app.state, "config_path", None)
    if configured is not None:
        return Path(configured)
    return _find_project_root() / "config.yml"


@router.get("/config")
async def get_config(request: Request) -> dict:
    """Return the current config.yml as a dict and raw YAML.

    Gated on ``web.config_panel.enabled`` before anything else, reads
    included: the document this returns carries every provider ``base_url``
    and the paths the safety layers derive their allow and deny areas from, so
    a tier that may not edit the file may not read it out either.
    """
    from osprey.interfaces.web_terminal.routes.config import _require_config_panel

    _require_config_panel(request)

    path = _config_path(request)
    if not path.exists():
        raise HTTPException(status_code=404, detail="config.yml not found")
    raw = path.read_text()
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        parsed = {}
    return {"raw": raw, "parsed": parsed}


class ConfigUpdateRequest(BaseModel):
    content: str


@router.put("/config")
async def update_config(request: Request, req: ConfigUpdateRequest) -> dict:
    """Write new content to config.yml with backup + fsync.

    This replaces the whole document verbatim, which makes it the widest write
    surface ARIEL has onto the file that carries the write gate, the approval
    gate and the paths the safety layers derive their allow and deny areas from
    -- and the only one that could change any of them without naming them,
    simply by handing over different bytes. The protected set is consulted by
    every framework writer, so it is consulted here too, on exactly the terms
    the Web Terminal's ``PUT /api/config`` uses: the replacement must leave every
    protected key exactly as it found it, and one that does not is refused with
    the same 403, the same wording and the same ``http_config`` audit
    record. Both surfaces share one implementation rather than two copies of it,
    so they cannot drift into telling an operator different stories about the
    same rule.

    The check sits between the parse and the *first* thing that touches disk --
    ahead of the backup, which is itself a write derived from a file this
    request may turn out not to be allowed to replace.
    """
    # Both helpers are the Web Terminal config route's, imported rather than
    # restated: the refusal an operator sees, the audit record it leaves and the
    # document diff it is based on are then the same objects, not two spellings
    # of the same intent.
    from osprey.interfaces.web_terminal.routes.config import (
        _as_document,
        _changed_protected_keys,
        _current_document,
        _refuse_protected_keys,
        _require_config_panel,
    )

    # The tier gate runs FIRST, ahead of the path check and ahead of the
    # protected-set logic, exactly as it does on the terminal: a deployment
    # that has taken this panel away never gets as far as having a key to
    # judge, so it never reports a protected-key refusal instead.
    _require_config_panel(request)

    path = _config_path(request)
    if not path.exists():
        raise HTTPException(status_code=404, detail="config.yml not found")

    try:
        parsed = yaml.safe_load(req.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}") from e

    changed = _changed_protected_keys(_current_document(path), _as_document(parsed))
    if changed:
        raise _refuse_protected_keys(request, changed)

    # Into the agent-data state zone, not beside the config: this file lives in
    # the render, which the container split makes root-owned, and creating a new
    # file there would fail before a byte of the save was written. See
    # osprey.utils.config_writer.CONFIG_BACKUP_DIRNAME.
    from osprey.utils.config_writer import write_config_backup

    write_config_backup(path)

    # fsync for crash safety
    fd = os.open(str(path), os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
    try:
        os.write(fd, req.content.encode())
        os.fsync(fd)
    finally:
        os.close(fd)

    return {"status": "ok", "requires_restart": True}
