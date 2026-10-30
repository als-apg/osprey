"""What a tool result shows in the Simple chat: a bounded, readable preview.

A tool's result is text, and the chat shows a preview of it in the tool's row.
The preview reaches the page twice: on the live stream (``operator_session``)
and again in a transcript's replay after a reload (``routes/session``). Both
build it here, so a reloaded page shows what the live page showed.

A result that is JSON keeps its shape. Long strings, long lists and long objects
are cut inside their values with a visible marker, and nesting deeper than
:data:`MAX_DEPTH` is replaced by one. The preview therefore stays valid JSON, and
the chat reads it as key and value rows. Any other text is cut at its head, and
so is text nested deeper than :data:`MAX_PARSE_DEPTH`, which is not read as JSON
at all. Either way the preview stays within a fixed budget, so one frame stays
small however large the result was.

The background-agent launch text opens with its mark (``Async agent launched``),
and every cut keeps the head of a string and of plain text, so the mark survives.

A refused write ends its text with the trailer ``[osprey-refusal: <reason>]``
(``osprey_connectors.refusals.refusal_text``). The trailer sits at the END, which a
head cut loses, so :func:`tool_result_refusal` reads it from the full text before
any preview is cut, and both surfaces call it.

The notes the CLI appends to a result for the model (``<system-reminder>…``)
are not part of the tool's output and are left out of the preview; the model
still receives them.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import Any

#: Longest preview of plain text, in characters. The old limit of 600 cut an
#: ordinary command's output after about ten lines. The result pane scrolls, so
#: the preview keeps about two dozen lines, and a long replay still ships a
#: bounded frame for each tool call.
PLAIN_TEXT_CHARS = 1500

#: The whole preview of a JSON result, in characters. The cuts inside its values
#: are tightened until it fits.
STRUCTURED_CHARS = 4000

#: Longest string a JSON result keeps whole; a longer one is cut to this length.
STRING_CHARS = 300

#: Most items a JSON result keeps from one list or object; the rest are counted.
MAX_ITEMS = 20

#: Deepest nesting a JSON result keeps; a container nested deeper is replaced by a marker.
MAX_DEPTH = 6

#: Deepest nesting a result is read as JSON at all; text nested deeper is plain
#: text. Decided here, before ``json.loads``, because how deep the parser can go
#: before it raises ``RecursionError`` differs between Python versions, and the
#: preview must not. Far below every version's limit, and far above any result.
MAX_PARSE_DEPTH = 100

#: A JSON string, matched whole so a bracket inside it is not a bracket, or one
#: bracket outside a string.
_STRING_OR_BRACKET = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[][{}]')

#: Most MCP envelopes peeled off one result. Each envelope wraps the one inside it.
MAX_ENVELOPES = 4

#: The cuts tried in turn while a JSON result is over budget, each tighter than the last.
_CUTS = ((STRING_CHARS, MAX_ITEMS), (150, 10), (60, 5), (20, 2))

#: The refusal trailer on a line of its own: ``\n[osprey-refusal: <reason>]``. It
#: ends the refusal text; the CLI may wrap that text, so it need not end the result.
_REFUSAL_TRAILER = re.compile(r"\r?\n\[osprey-refusal: ([a-z_]+)\][ \t]*(?=\r?\n|\Z)")

#: The lead the CLI puts before a PreToolUse hook's deny reason in a tool result.
_HOOK_ERROR_LEAD = re.compile(r"\APreToolUse:\S+ hook error: (?:\[[^\]\n]*\]: )?")

#: A note the CLI adds to a tool result for the model's eyes only.
_SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def tool_result_text(content: str | list[Any] | None) -> str:
    """The text a tool result's content carries, as the chat reads it.

    A string is the text. A list of blocks is its text blocks joined by newlines,
    with ``[image]`` standing in for an image block and ``[<type>]`` for any other
    block that is not text, so image bytes never reach the page. No content is an
    empty string. The live stream and a transcript's replay both read results
    through here.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(f"[{block.get('type') or 'block'}]")
    return "\n".join(parts)


def tool_result_refusal(text: str, is_error: bool) -> tuple[dict[str, str] | None, str]:
    """The refusal a tool result carries, and the text with its trailer stripped.

    Returns ``({reason, phrase}, stripped)`` for an error result whose text ends
    with the refusal trailer, and ``(None, text)`` for every other result. The
    trailer is read from the full text, before any preview cut. It is found on
    the text's own last line, or on the last line of a string inside a JSON
    result (an MCP tool error carries its message in a JSON envelope, where the
    line break is the two characters ``\\n``), the JSON possibly behind a
    short lead. The *phrase* is the line the
    trailer closes, less the CLI's ``PreToolUse:<tool> hook error:`` lead; it says
    what stopped the write and who can lift it. Only
    a reason in ``osprey_connectors.refusals.REASONS`` counts; a result that is
    not an error is never a refusal, whatever it quotes.
    """
    if not is_error or "[osprey-refusal: " not in text:
        return None, text
    found = _trailer(text)
    if found is not None:
        return found
    # A JSON result, possibly behind a short lead such as ``Error: ``.
    start = text.find("{")
    payload = _container(text[start:]) if start >= 0 else None
    if payload is None:
        return None, text
    hit: list[dict[str, str]] = []
    stripped_payload = _strip_trailers(payload, hit, depth=0)
    if not hit:
        return None, text
    return hit[0], text[:start] + json.dumps(stripped_payload, ensure_ascii=False)


def _trailer(text: str) -> tuple[dict[str, str], str] | None:
    """``({reason, phrase}, text without the trailer)`` when *text* carries one.

    The last trailer counts; a reason outside the closed set is no refusal.
    """
    from osprey_connectors.refusals import REASONS

    matches = list(_REFUSAL_TRAILER.finditer(text))
    if not matches or matches[-1].group(1) not in REASONS:
        return None
    match = matches[-1]
    line = text[: match.start()].rsplit("\n", 1)[-1]
    phrase = _HOOK_ERROR_LEAD.sub("", line).strip()
    return {"reason": match.group(1), "phrase": phrase}, text[: match.start()] + text[match.end() :]


def _strip_trailers(value: Any, hit: list[dict[str, str]], depth: int) -> Any:
    """*value* with the first trailing refusal found in its strings stripped."""
    if depth > MAX_DEPTH:
        return value
    if isinstance(value, str):
        if hit:
            return value
        found = _trailer(value)
        if found is not None:
            hit.append(found[0])
            return found[1]
        inner = _container(value)
        if inner is not None:
            stripped = _strip_trailers(inner, hit, depth + 1)
            if hit:
                return json.dumps(stripped, ensure_ascii=False)
        return value
    if isinstance(value, list):
        return [_strip_trailers(item, hit, depth + 1) for item in value]
    if isinstance(value, dict):
        return {key: _strip_trailers(item, hit, depth + 1) for key, item in value.items()}
    return value


def tool_result_preview(text: str) -> tuple[str, bool]:
    """Return the chat's preview of a tool result, and whether it was shortened.

    An MCP envelope is peeled off first. A JSON object or array left over is cut
    value by value and sent as compact JSON, with the cuts tightened until it fits
    :data:`STRUCTURED_CHARS`. If even the tightest cut is too large, the text is clipped to
    that budget. Anything else is cut at :data:`PLAIN_TEXT_CHARS`.
    """
    payload = _payload(_without_reminders(text))
    if isinstance(payload, str):
        return _head(payload, PLAIN_TEXT_CHARS)
    for string_chars, items in _CUTS:
        limits = _Limits(string_chars, items)
        preview = _dump(_shrink(payload, limits, depth=0))
        if len(preview) <= STRUCTURED_CHARS:
            return preview, limits.cut
    # Keys are never cut, so only a structure whose keys alone exceed the budget gets here.
    # Its depth is capped, unlike the payload's, so it is safe to serialize.
    tightest = _dump(_shrink(payload, _Limits(*_CUTS[-1]), depth=0))
    return _head(tightest, STRUCTURED_CHARS)


def _without_reminders(text: str) -> str:
    """*text* with the CLI's ``<system-reminder>`` notes taken out; unchanged when it has none."""
    cleaned = _SYSTEM_REMINDER.sub("", text)
    return text if cleaned == text else cleaned.strip()


def _payload(text: str) -> str | list[Any] | dict[str, Any]:
    """The value a result carries, with its MCP envelopes peeled off.

    Text that holds no JSON object or array is the value as it stands. An
    envelope is an object whose only key is ``result``, and its value is the
    tool's own result: text, or JSON held in a string.
    """
    value: Any = text
    for _ in range(MAX_ENVELOPES + 1):
        if isinstance(value, str):
            container = _container(value)
            if container is None:
                return value
            value = container
        if not (isinstance(value, dict) and list(value) == ["result"]):
            break
        value = value["result"]
    if isinstance(value, (dict, list, str)):
        return value
    return json.dumps(value)


def _container(text: str) -> dict[str, Any] | list[Any] | None:
    """The JSON object or array that ``text`` holds, or ``None`` when it holds none.

    Only text that opens with a bracket is read as JSON, so a bare number or a
    quoted word stays text, and only text nested no deeper than
    :data:`MAX_PARSE_DEPTH` is handed to the parser. ``NaN``, ``Infinity`` and
    numbers too large for a float become strings, because JSON has no way to
    send them.
    """
    stripped = text.strip()
    if not stripped.startswith(("{", "[")) or _nested_too_deep(stripped):
        return None
    try:
        value = json.loads(stripped, parse_constant=str, parse_float=_finite_float)
    except ValueError:
        return None
    return value if isinstance(value, (dict, list)) else None


def _nested_too_deep(text: str, bound: int = MAX_PARSE_DEPTH) -> bool:
    """Whether ``text`` opens more than ``bound`` containers inside one another.

    Brackets inside strings are not counted. Text with no more openers than
    ``bound`` in total cannot nest deeper and is not scanned.
    """
    if text.count("[") + text.count("{") <= bound:
        return False
    depth = 0
    for match in _STRING_OR_BRACKET.finditer(text):
        char = match.group()[0]
        if char in "[{":
            depth += 1
            if depth > bound:
                return True
        elif char in "]}":
            depth -= 1
    return False


def _finite_float(token: str) -> float | str:
    """A float literal, or its text when it overflows to infinity."""
    value = float(token)
    return value if math.isfinite(value) else token


@dataclass(slots=True)
class _Limits:
    """The cuts one attempt makes, and whether it has cut anything yet."""

    string_chars: int
    items: int
    cut: bool = False


def _shrink(value: Any, limits: _Limits, depth: int) -> Any:
    """``value`` with its long strings, long containers and deep nesting cut."""
    if isinstance(value, str):
        if len(value) <= limits.string_chars:
            return value
        limits.cut = True
        return f"{value[: limits.string_chars]}… +{len(value) - limits.string_chars} chars"
    if not isinstance(value, (dict, list)):
        return value
    if depth > MAX_DEPTH:
        limits.cut = True
        return "{…}" if isinstance(value, dict) else "[…]"
    if isinstance(value, list):
        kept: list[Any] = [_shrink(item, limits, depth + 1) for item in value[: limits.items]]
        if len(value) > limits.items:
            limits.cut = True
            kept.append(f"… +{len(value) - limits.items} more")
        return kept
    record = {
        key: _shrink(item, limits, depth + 1)
        for key, item in itertools.islice(value.items(), limits.items)
    }
    if len(value) > limits.items:
        limits.cut = True
        record.setdefault("…", f"+{len(value) - limits.items} more")
    return record


def _head(text: str, limit: int) -> tuple[str, bool]:
    """``text`` cut to ``limit`` characters, and whether it was cut."""
    return text[:limit], len(text) > limit


def _dump(value: Any) -> str:
    """Compact JSON for ``value``. Non-ASCII text stays as it is, so the budget counts what is read."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
