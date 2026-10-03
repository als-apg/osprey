"""Build the one attachment summary shape every read surface emits.

A summary describes one JSONB attachment item of an entry, joined with its
``attachment_files`` row when one exists::

    {attachment_id?, filename, mime_type, viewable, copy_status, skip_reason?,
     caption?, caption_source?, visible_text?, url?}

Listings add ``caption_truncated`` / ``visible_text_truncated`` when they cut a
field. Every upstream- or model-derived string passes through
:func:`~osprey.services.ariel_search.attachments.compose._inert`, and ``url`` is
re-quoted, so no emitted field ever contains ``[``.

The row-creation rule mirrored here is ``record_rows``: a row whose id is the
item's ``attachment_id_for`` always wins; with no row, an item whose url is
empty or not fetchable for the configured source never gets one
(``skipped / no_source_url``), and any other item is ``pending`` until backfill
records it.
"""

from __future__ import annotations

import json
import threading
import weakref
from collections.abc import Collection, Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from osprey.services.ariel_search.attachments import (
    attachment_id_for,
    fetchable_url,
    is_native_item,
)
from osprey.services.ariel_search.attachments.compose import (
    CAPTION_MAX_CHARS,
    FILENAME_MAX_CHARS,
    _inert,
    _model_caption,
)
from osprey.services.ariel_search.attachments.copy import (
    _attachment_list,
    _filename_for,
    validated_declared_type,
)
from osprey.services.ariel_search.attachments.fetch import is_file_source
from osprey.services.ariel_search.attachments.formats import OCTET_STREAM, is_viewable, kind
from osprey.utils.logger import get_logger

logger = get_logger("ariel")

#: Every key a summary may carry, in emission order.
SUMMARY_KEYS: tuple[str, ...] = (
    "attachment_id",
    "filename",
    "mime_type",
    "viewable",
    "copy_status",
    "skip_reason",
    "caption",
    "caption_source",
    "visible_text",
    "url",
)

#: Keys only listings add, marking a caption or visible text cut to the listing length.
LISTING_ONLY_KEYS: tuple[str, ...] = ("caption_truncated", "visible_text_truncated")

#: Listings cut ``caption`` and ``visible_text`` to this many characters.
LISTING_TEXT_MAX_CHARS = 200

#: Characters left unescaped when re-quoting an emitted url's path, query and fragment.
_URL_SAFE = "/%:@!$&'()*+,;=-._~"

_SKIP_NO_SOURCE = "no_source_url"

_cache_lock = threading.Lock()
_file_source_cache: tuple[weakref.ref[Any], bool] | None = None
_warned_adapter = False


def file_source_for(config: Any) -> bool:
    """Return whether ``config``'s ingestion adapter reads entries from local files.

    ``is_file_source(get_adapter(config))`` when ``config.ingestion`` is set,
    else False. Any exception resolving or constructing the adapter answers
    False and logs one WARNING per process naming ``ariel.ingestion.adapter``,
    so a read surface never fails on ingestion settings. The answer is reused
    for as long as the same config object is passed.
    """
    global _file_source_cache
    with _cache_lock:
        cached = _file_source_cache
        if cached is not None and cached[0]() is config:
            return cached[1]
    value = _resolve_file_source(config)
    try:
        ref = weakref.ref(config)
    except TypeError:
        return value
    with _cache_lock:
        _file_source_cache = (ref, value)
    return value


def _resolve_file_source(config: Any) -> bool:
    global _warned_adapter
    if not getattr(config, "ingestion", None):
        return False
    try:
        from osprey.services.ariel_search.ingestion.adapters import get_adapter

        return is_file_source(get_adapter(config))
    except Exception as exc:
        if not _warned_adapter:
            _warned_adapter = True
            logger.warning(
                "ariel.ingestion.adapter could not be resolved (%s: %s); "
                "relative attachment paths are treated as not fetchable",
                type(exc).__name__,
                exc,
            )
        return False


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, Mapping) else {}


def _emitted_url(url: Any) -> str | None:
    """Return an absolute http(s) url re-quoted so it carries no ``[``, ``]`` or whitespace."""
    if not isinstance(url, str) or not url:
        return None
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not hostname:
        return None
    netloc = parts.netloc
    if any(c in netloc for c in "[]") or any(c.isspace() or ord(c) < 0x20 for c in netloc):
        return None
    return urlunsplit(
        (
            parts.scheme,
            netloc,
            quote(parts.path, safe=_URL_SAFE),
            quote(parts.query, safe=_URL_SAFE + "?"),
            quote(parts.fragment, safe=_URL_SAFE + "?"),
        )
    )


def _mime(value: Any) -> str:
    return validated_declared_type(value) or OCTET_STREAM


def _filename(item: Mapping[str, Any], row: Mapping[str, Any] | None) -> str:
    name: Any = row.get("filename") if row is not None else None
    if not isinstance(name, str) or not name.strip():
        url = item.get("url")
        name = _filename_for(item, url if isinstance(url, str) else "")
    return _inert(name, FILENAME_MAX_CHARS)


def _set_text(summary: dict[str, Any], key: str, value: Any, *, full_captions: bool) -> None:
    text = _inert(value, CAPTION_MAX_CHARS)
    if not text.strip():
        return
    if not full_captions and len(text) > LISTING_TEXT_MAX_CHARS:
        summary[key] = text[:LISTING_TEXT_MAX_CHARS]
        summary[f"{key}_truncated"] = True
    else:
        summary[key] = text


def _summary(
    item: Mapping[str, Any],
    attachment_id: str | None,
    rows_by_id: Mapping[str, Mapping[str, Any]] | None,
    captions: Mapping[str, Any],
    *,
    file_source: bool,
    full_captions: bool,
    model_id: str | None,
) -> dict[str, Any]:
    row = rows_by_id.get(attachment_id) if rows_by_id is not None and attachment_id else None
    url = item.get("url")
    summary: dict[str, Any] = {}
    if row is not None:
        summary["attachment_id"] = attachment_id
        summary["filename"] = _filename(item, row)
        summary["mime_type"] = _mime(row.get("mime_type"))
        summary["viewable"] = is_viewable(row)
        summary["copy_status"] = _inert(row.get("copy_status"), FILENAME_MAX_CHARS)
        skip_reason = row.get("skip_reason")
        if skip_reason is not None:
            summary["skip_reason"] = _inert(skip_reason, FILENAME_MAX_CHARS)
    else:
        summary["filename"] = _filename(item, None)
        summary["mime_type"] = _mime(item.get("type"))
        summary["viewable"] = False
        no_source = rows_by_id is not None and (
            attachment_id is None
            or (
                not is_native_item(item)
                and not (isinstance(url, str) and fetchable_url(url, file_source=file_source))
            )
        )
        if no_source:
            summary["copy_status"] = "skipped"
            summary["skip_reason"] = _SKIP_NO_SOURCE
        else:
            summary["copy_status"] = "pending"

    stored = _model_caption(captions, attachment_id, model_id)
    if stored is not None and model_id:
        summary["caption_source"] = f"model:{_inert(model_id, FILENAME_MAX_CHARS)}"
        _set_text(summary, "caption", stored.get("caption"), full_captions=full_captions)
        _set_text(summary, "visible_text", stored.get("visible_text"), full_captions=full_captions)
    else:
        _set_text(summary, "caption", item.get("caption"), full_captions=full_captions)
        if "caption" in summary:
            summary["caption_source"] = "upstream"

    emitted = _emitted_url(row.get("source_url") if row is not None else None) or _emitted_url(url)
    if emitted is not None:
        summary["url"] = emitted

    order = SUMMARY_KEYS + LISTING_ONLY_KEYS
    return {key: summary[key] for key in order if key in summary}


def build_attachment_summary_pairs(
    entry: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | None,
    limit: int | None,
    matched_ids: Collection[str],
    *,
    file_source: bool,
    full_captions: bool = False,
    model_id: str | None = None,
) -> list[tuple[dict[str, Any], Mapping[str, Any]]]:
    """Return ``(summary, JSONB item)`` pairs for an entry's attachments, in final order.

    Args:
        entry: The entry dict; reads ``entry_id``, ``attachments`` and
            ``attachment_captions``.
        rows: The entry's ``attachment_files`` rows (no blobs), or None when
            there is no copy state, in which case every item is ``pending``
            with no id and not viewable.
        limit: Keep at most this many summaries; None keeps all.
        matched_ids: Attachment ids a search matched; they sort first.
        file_source: Whether the entry's source resolves relative paths
            (pass :func:`file_source_for`).
        full_captions: Emit caption and visible text in full; otherwise cut
            them to 200 characters and mark the cut field ``*_truncated``.
        model_id: The configured caption model id; its caption wins over the
            upstream one.

    Returns:
        Pairs ordered matched-first, then image-first, otherwise in list order,
        deduplicated by attachment id.
    """
    entry_id = str(entry.get("entry_id") or "")
    captions = _as_mapping(entry.get("attachment_captions"))
    rows_by_id: dict[str, Mapping[str, Any]] | None = None
    if rows is not None:
        rows_by_id = {}
        for row in rows:
            if isinstance(row, Mapping) and isinstance(row.get("attachment_id"), str):
                rows_by_id.setdefault(row["attachment_id"], row)
    matched = set(matched_ids or ())

    seen: set[str] = set()
    built: list[tuple[int, dict[str, Any], Mapping[str, Any], str | None]] = []
    for item in _attachment_list(entry.get("attachments")):
        if not isinstance(item, Mapping):
            continue
        attachment_id = attachment_id_for(entry_id, item)
        if attachment_id is not None:
            if attachment_id in seen:
                continue
            seen.add(attachment_id)
        summary = _summary(
            item,
            attachment_id,
            rows_by_id,
            captions,
            file_source=file_source,
            full_captions=full_captions,
            model_id=model_id,
        )
        built.append((len(built), summary, item, attachment_id))

    built.sort(
        key=lambda b: (
            0 if b[3] is not None and b[3] in matched else 1,
            0 if kind(b[1]["mime_type"]) == "image" else 1,
            b[0],
        )
    )
    pairs = [(summary, item) for _, summary, item, _ in built]
    if limit is not None:
        pairs = pairs[: max(limit, 0)]
    return pairs


def build_attachment_summaries(
    entry: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | None,
    limit: int | None,
    matched_ids: Collection[str],
    *,
    file_source: bool,
    full_captions: bool = False,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the attachment summaries of an entry; see :func:`build_attachment_summary_pairs`."""
    pairs = build_attachment_summary_pairs(
        entry,
        rows,
        limit,
        matched_ids,
        file_source=file_source,
        full_captions=full_captions,
        model_id=model_id,
    )
    return [summary for summary, _ in pairs]


__all__ = [
    "LISTING_ONLY_KEYS",
    "LISTING_TEXT_MAX_CHARS",
    "SUMMARY_KEYS",
    "build_attachment_summaries",
    "build_attachment_summary_pairs",
    "file_source_for",
]
