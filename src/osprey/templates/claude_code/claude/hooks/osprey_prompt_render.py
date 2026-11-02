"""Render an application's approval prompt from a restricted template.

An application may describe the action a human approves with a short template
over the tool call's ``input.*`` and the application's fetched ``state.*``. The
grammar is deliberately tiny: plain substitution, ``{{ input.a.b }}``, and one
filter, ``{{ state.x | default("..") }}``. Anything else — a block tag, a
comment, another filter, a call, an index, arithmetic, a root other than
``input`` or ``state``, or a dotted walk into something that is not a mapping —
refuses the whole template with :class:`PromptTemplateError`. No loops and no
conditionals: per-item detail arrives as display lines the application computed,
which :func:`render_display_lines` escapes and caps.

Every substituted value and every display line is escaped twice: a character
that is invisible or reshapes the text around it (a control, a format character
such as a bidi override or a zero-width space, a line or paragraph separator, a
private-use or unassigned code point) becomes a visible ``\\x`` token carrying
its codepoint (:func:`escape_control`), so a value can never forge a line or
reorder what the approver reads, and HTML specials become entities
(:func:`escape_html`). Each value and each display line is capped on its own,
never mid-escape, so the template's own text always shows; a prompt still over
:data:`MAX_OUTPUT_CHARS` is refused rather than cut. The number of display lines
is capped too.

:func:`input_digest` is the one fixed recipe that binds an approval to the
input it was shown: ``sha256(canonical_json({k: input[k] for k in over}))``.

This module uses the standard library only and imports no sibling: the Claude
Code hooks run under the system ``python3`` and import nothing from OSPREY, so
it ships beside them in ``.claude/hooks/``.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping

__all__ = [
    "MAX_DISPLAY_LINES",
    "MAX_DISPLAY_LINE_CHARS",
    "MAX_OUTPUT_CHARS",
    "MAX_VALUE_CHARS",
    "PromptTemplateError",
    "TRUNCATION_MARK",
    "canonical_json",
    "escape_control",
    "escape_html",
    "input_digest",
    "render_display_lines",
    "render_prompt",
]

#: Longest rendered prompt; a longer one is refused.
MAX_OUTPUT_CHARS = 4000
#: Longest single substituted value once escaped, marker included.
MAX_VALUE_CHARS = 500
#: Longest single display line, marker included.
MAX_DISPLAY_LINE_CHARS = 200
#: Most display lines shown; the rest are counted on one closing line.
MAX_DISPLAY_LINES = 20
#: Appended where a cap cut text short.
TRUNCATION_MARK = " …[truncated]"

#: Unicode categories escaped to a visible token: controls (C0 and C1), format
#: characters (bidi overrides, zero-width marks), line and paragraph separators,
#: private-use and unassigned code points.
_ESCAPED_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp", "Co", "Cn"})
_ROOTS = ("input", "state")

_PLACEHOLDER = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
_EXPRESSION = re.compile(
    r"""\s*
    (?P<path>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)
    \s*
    (?:\|\s*default\(\s*(?:"(?P<dq>[^"\\]*)"|'(?P<sq>[^'\\]*)')\s*\)\s*)?
    """,
    re.VERBOSE,
)


class PromptTemplateError(ValueError):
    """The template uses a construct the renderer refuses, or a value it cannot show."""


def _control_token(ch: str) -> str:
    if unicodedata.category(ch) in _ESCAPED_CATEGORIES:
        return f"\\x{ord(ch):02x}"
    return ch


def escape_control(text: object) -> str:
    """Escape every invisible or text-reshaping character to a visible ``\\x`` token.

    The token carries the full codepoint (``\\x0a``, ``\\x85``, ``\\x202e``). Keeps
    every value on one line, in the order it was written, and makes tampering
    legible to the approver rather than silently dropped.
    """
    return "".join(_control_token(ch) for ch in str(text))


def escape_html(text: object) -> str:
    """Escape HTML specials, quotes included."""
    return html.escape(str(text), quote=True)


def _escaped_capped(text: str, limit: int) -> str:
    """Escape *text* and cap it at *limit* characters, never splitting an escape."""
    tokens = [escape_html(_control_token(ch)) for ch in text]
    if sum(len(token) for token in tokens) <= limit:
        return "".join(tokens)
    budget = limit - len(TRUNCATION_MARK)
    kept: list[str] = []
    for token in tokens:
        budget -= len(token)
        if budget < 0:
            break
        kept.append(token)
    return "".join(kept) + TRUNCATION_MARK


def _scalar_text(value: object, where: str) -> str:
    if isinstance(value, Mapping) or (
        isinstance(value, Iterable) and not isinstance(value, (str, bytes))
    ):
        raise PromptTemplateError(f"{where} is a {type(value).__name__}, not a scalar")
    return str(value)


def _lookup(path: str, roots: Mapping[str, object]) -> tuple[bool, object]:
    """Walk *path* through *roots*; return ``(found, value)``."""
    head, *rest = path.split(".")
    if head not in _ROOTS:
        raise PromptTemplateError(f"{path!r}: only input.* and state.* may be substituted")
    current: object = roots[head]
    for depth, key in enumerate(rest):
        if key.startswith("__"):
            raise PromptTemplateError(f"{path!r}: dunder names are refused")
        if current is None:
            return False, None
        if not isinstance(current, Mapping):
            walked = ".".join([head, *rest[:depth]])
            raise PromptTemplateError(f"{path!r}: {walked} is not a mapping")
        if key not in current:
            return False, None
        current = current[key]
    return True, current


def _literal(text: str) -> str:
    """Return a literal template segment, refusing a stray ``{{`` or ``}}`` in it."""
    if "{{" in text or "}}" in text:
        raise PromptTemplateError("unbalanced placeholder braces")
    return text


def render_prompt(
    template: str,
    *,
    input: Mapping[str, object],
    state: Mapping[str, object],
) -> str:
    """Render *template* over *input* and *state*.

    A value that is missing or null, or sits under a null, takes the
    placeholder's ``default``; without one, it refuses. Each value is escaped and capped at :data:`MAX_VALUE_CHARS`.
    Raises :class:`PromptTemplateError` for any refused construct and for a
    prompt longer than :data:`MAX_OUTPUT_CHARS`.
    """
    if "{%" in template or "{#" in template:
        raise PromptTemplateError("block tags and comments are refused")
    roots = {"input": input or {}, "state": state or {}}
    parts: list[str] = []
    pos = 0
    for match in _PLACEHOLDER.finditer(template):
        parts.append(_literal(template[pos : match.start()]))
        pos = match.end()
        expr = _EXPRESSION.fullmatch(match.group(1))
        if expr is None:
            raise PromptTemplateError(f"refused placeholder {match.group(0)!r}")
        path = expr.group("path")
        default = expr.group("dq") if expr.group("dq") is not None else expr.group("sq")
        found, value = _lookup(path, roots)
        if not found or value is None:
            if default is None:
                raise PromptTemplateError(f"{path!r} has no value and no default")
            parts.append(_escaped_capped(default, MAX_VALUE_CHARS))
            continue
        parts.append(_escaped_capped(_scalar_text(value, path), MAX_VALUE_CHARS))
    parts.append(_literal(template[pos:]))
    rendered = "".join(parts)
    if len(rendered) > MAX_OUTPUT_CHARS:
        raise PromptTemplateError(
            f"the rendered prompt is {len(rendered)} characters, over {MAX_OUTPUT_CHARS}"
        )
    return rendered


def render_display_lines(lines: object) -> list[str]:
    """Escape and cap application-supplied display lines.

    *lines* must be a list. The first :data:`MAX_DISPLAY_LINES` must be scalars;
    each is escaped and capped to :data:`MAX_DISPLAY_LINE_CHARS`. The rest are
    never rendered: one closing line counts them.
    """
    if not isinstance(lines, list):
        raise PromptTemplateError(f"display lines must be a list, not {type(lines).__name__}")
    shown = [
        _escaped_capped(_scalar_text(line, f"display line {i}"), MAX_DISPLAY_LINE_CHARS)
        for i, line in enumerate(lines[:MAX_DISPLAY_LINES])
    ]
    hidden = len(lines) - MAX_DISPLAY_LINES
    if hidden > 0:
        shown.append(f"… {hidden} more line(s) not shown")
    return shown


def canonical_json(value: object) -> str:
    """Serialize *value* the one way both digest sides agree on.

    Sorted keys, no whitespace, ASCII-only, NaN and infinity refused.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def input_digest(tool_input: Mapping[str, object], over: Iterable[str]) -> str:
    """Return ``sha256(canonical_json({k: tool_input[k] for k in over}))`` as hex.

    A key in *over* that *tool_input* lacks raises :class:`KeyError`.
    """
    selected = {key: tool_input[key] for key in over}
    return hashlib.sha256(canonical_json(selected).encode("utf-8")).hexdigest()
