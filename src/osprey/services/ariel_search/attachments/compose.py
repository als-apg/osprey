"""Compose the searchable ``attachment_text`` of an entry from its picture captions.

Three pure helpers shared by ingest, attachment summaries and the caption module:

* :func:`_inert` makes an upstream- or model-derived string safe to interpolate:
  it can never open or close a ``[...]`` marker, never break a line, and never
  carry control characters.
* :func:`caption_model_id` resolves the one model id every caption reader and
  writer keys on.
* :func:`compose_attachment_text` is the only writer of the ``[picture ...]``
  markers, so no other code path can emit one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from osprey.services.ariel_search.attachments import attachment_id_for

FILENAME_MAX_CHARS = 120
CAPTION_MAX_CHARS = 1000

_BRACKETS = str.maketrans({"[": "(", "]": ")"})
_SPACE_CHARS_RE = re.compile("[\r\n\t  ]")
_CONTROL_CHARS_RE = re.compile("[\x00-\x1f\x7f-\x9f]")


def _inert(value: Any, max_chars: int) -> str:
    """Return ``value`` as a string that cannot forge a marker or break a line.

    Brackets become parentheses; CR, LF, TAB, U+2028 and U+2029 become a space;
    every other C0/C1 control character (and DEL) is removed; the result is cut
    to ``max_chars`` characters. A non-string value renders as ``''``.
    """
    if not isinstance(value, str):
        return ""
    text = value.translate(_BRACKETS)
    text = _SPACE_CHARS_RE.sub(" ", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    return text[: max(max_chars, 0)]


def caption_model_id(config: Any) -> str | None:
    """Return ``image_caption.model.model_id`` when configured, else None.

    The module's ``enabled`` flag is deliberately ignored: disabling the module
    stops new vision calls only, while existing captions stay keyed and visible.
    ``config`` is an ``ARIELConfig`` (read through
    ``get_enhancement_module_config``) or a raw ``ariel`` mapping.
    """
    module: Any = None
    getter = getattr(config, "get_enhancement_module_config", None)
    if callable(getter):
        module = getter("image_caption")
    elif isinstance(config, Mapping):
        modules = config.get("enhancement_modules")
        if isinstance(modules, Mapping):
            module = modules.get("image_caption")
    if not isinstance(module, Mapping):
        return None
    model = module.get("model")
    if not isinstance(model, Mapping):
        return None
    model_id = model.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    return model_id.strip()


def _model_caption(
    captions: Any, attachment_id: str | None, model_id: str | None
) -> Mapping | None:
    """Return the stored ``{caption, visible_text}`` for the item under ``model_id``, if any."""
    if attachment_id is None or model_id is None or not isinstance(captions, Mapping):
        return None
    per_item = captions.get(attachment_id)
    if not isinstance(per_item, Mapping):
        return None
    entry = per_item.get(model_id)
    if not isinstance(entry, Mapping) or "error" in entry:
        return None
    caption = entry.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        return None
    return entry


def compose_attachment_text(
    entry_id: str,
    attachments: Sequence[Any] | None,
    attachment_captions: Mapping[str, Any] | None,
    model_id: str | None,
) -> str | None:
    """Render an entry's picture captions as searchable text, one line per picture.

    For each attachment item, in list order: a caption stored under
    ``model_id`` for the item's id (found through ``attachment_id_for``) renders
    ``[picture <filename> - machine caption by <model_id>] <caption>`` (with
    `` Visible text: <visible_text>`` when non-empty); otherwise a non-empty
    upstream ``caption`` on the item renders
    ``[picture <filename> - upstream caption] <caption>``. ``{error}`` entries
    are ignored. Every interpolated field passes through :func:`_inert`.

    Returns None when no item contributes a line.
    """
    if not isinstance(attachments, Sequence) or isinstance(attachments, (str, bytes)):
        return None
    safe_model = _inert(model_id, FILENAME_MAX_CHARS) if model_id else None
    lines: list[str] = []
    for item in attachments:
        if not isinstance(item, Mapping):
            continue
        filename = _inert(item.get("filename"), FILENAME_MAX_CHARS)
        stored = _model_caption(attachment_captions, attachment_id_for(entry_id, item), model_id)
        if stored is not None and safe_model:
            line = (
                f"[picture {filename} - machine caption by {safe_model}] "
                f"{_inert(stored.get('caption'), CAPTION_MAX_CHARS)}"
            )
            visible = _inert(stored.get("visible_text"), CAPTION_MAX_CHARS)
            if visible.strip():
                line += f" Visible text: {visible}"
            lines.append(line)
            continue
        upstream = _inert(item.get("caption"), CAPTION_MAX_CHARS)
        if upstream.strip():
            lines.append(f"[picture {filename} - upstream caption] {upstream}")
    return "\n".join(lines) if lines else None


__all__ = [
    "CAPTION_MAX_CHARS",
    "FILENAME_MAX_CHARS",
    "_inert",
    "caption_model_id",
    "compose_attachment_text",
]
