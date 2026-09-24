"""Rewrite standard Markdown into the formatting subset Microsoft Teams renders.

The OSPREY agent emits normal Markdown (which renders in the web terminal). A Teams
activity's ``text`` field renders a narrow Markdown subset, so this module provides
:func:`markdown_to_teams`, a *delivery-layer* transform applied to a local copy of the
answer text on the Teams posting path — the agent and its replayed history are never
touched (the "identical agent across channels" invariant). Keeping that copy local is
the posting caller's job; nothing here mutates its input.

Design: a **phased pipeline**, not a single regex pass, run in this fixed order:

    1. ``_mask_code``       — replace fenced blocks + inline code with inert
                              sentinels, so no later phase can rewrite code content.
                              Both kinds are stored verbatim.
    2. ``_convert_line_blocks`` — line-oriented transforms (tables, then headings).
    3. ``_convert_inline``  — span transforms (math).
    4. ``_restore_code``    — substitute the masked spans back, verbatim.

Masking first is the safety foundation: ``**``, ``#``, ``|``, ``$`` inside code are
restored unchanged. Malformed constructs (unterminated fence, piped table cell) fall
back to their original text rather than raising — the answer always lands.

Standard library only (``re``); zero imports from the rest of osprey.

Teams rendering targets
-----------------------
Teams already renders most of what the agent writes, so this transform is mostly a
pass-through; it exists for the two constructs Teams does *not* render — tables and
headings — plus LaTeX, which nothing in Teams understands.

    Construct              Markdown in              Teams target out
    ---------------------  -----------------------  -----------------------------
    bold                   ``**x**`` / ``__x__``    unchanged
    italic                 ``*x*`` / ``_x_``        unchanged
    bold+italic            ``***x***``              unchanged
    inline code            `` `x` ``                unchanged
    fenced code            ```` ```lang\n…``` ````  unchanged (language tag kept)
    bullet                 ``- `` / ``* `` / ``+ `` unchanged
    blockquote             ``> x``                  unchanged
    link                   ``[t](u)``               unchanged
    bare URL               ``https://…``            unchanged (Teams auto-links)
    heading                ``# x`` … ``###### x``   ``**x**`` (bold line, no hashes)
    table row              ``| H | v |``            ``**H**: v`` labeled lines
    math                   ``$x$`` / ``$$x$$``      ``x`` (delimiters stripped)
    mention                ``<@29:1>``              ``<at>Name</at>`` + a mention entity,
                                                    only for a named member of this
                                                    conversation (:func:`render_mentions`)

:func:`render_mentions` runs per posted message on the placeholder text, after
:func:`markdown_to_teams`, and never on the stored answer.

Unlike the Google Chat target, every construct here is a **fixpoint**: Teams bold is
``**x**``, which is not confusable with an italic ``*x*``, so re-feeding converted
text changes nothing. The transform is still only ever applied once in the real path.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def _mention_label(name: str | None) -> str:
    """A name made safe for an ``<at>`` tag: ``<``/``>`` become spaces and every run
    of whitespace one space, so the tag stays well-formed and the entity's ``text``
    is byte-equal to the message text. ``""`` for ``None``."""
    if not name:
        return ""
    return _WHITESPACE_RE.sub(" ", name.replace("<", " ").replace(">", " ")).strip()


def render_mentions(
    text: str,
    roster: Mapping[str, str | None],
    *,
    enabled: bool,
    placeholder: re.Pattern[str],
) -> tuple[str, list[dict[str, Any]]]:
    """Render the agent's mention placeholders in one message for Teams.

    A placeholder whose id is a key of ``roster`` (the members of this conversation)
    and whose member has a name becomes ``<at>Name</at>`` when ``enabled``, with one
    mention entity per occurrence, in text order. Anything else becomes plain
    ``@name`` (or ``@id`` when the roster has no name): a listed member with no name
    is never tagged, because the tag's label is shown to everyone. Only listed
    ``29:`` people are roster keys and Teams has no "everyone" mention for bots, so
    nothing but a listed person can render. ``placeholder`` is passed in (group 1 is
    the id) so this module stays free of osprey imports.

    Returns:
        The rendered text and its mention entities.
    """
    entities: list[dict[str, Any]] = []

    def one(m: re.Match[str]) -> str:
        ident = m.group(1)
        label = _mention_label(roster.get(ident))
        if enabled and ident in roster and label:
            tag = f"<at>{label}</at>"
            entities.append(
                {"type": "mention", "text": tag, "mentioned": {"id": ident, "name": label}}
            )
            return tag
        return "@" + (label or ident)

    return placeholder.sub(one, text), entities


def _make_sentinel(name: str) -> tuple[str, re.Pattern[str]]:
    """Build a ``(template, pattern)`` sentinel pair from one ``name``.

    ``template.format(i)`` emits the ``i``-th placeholder; ``pattern`` recovers the
    index back out of it. Deriving both from a single name keeps them from drifting
    apart. Placeholders contain only NUL bytes and digits, so no line-block or inline
    regex ever matches one.
    """
    return f"\x00{name}{{}}\x00", re.compile(rf"\x00{name}(\d+)\x00")


def _stash(store: list[str], template: str, content: str) -> str:
    """Append ``content`` to ``store`` and return its sentinel placeholder."""
    store.append(content)
    return template.format(len(store) - 1)


# Inert placeholder for a masked code span, so no later phase can rewrite code content
# and each masked span trivially survives the idempotence property.
_SENTINEL, _SENTINEL_RE = _make_sentinel("CODE")

# Fenced block: opening ``` + optional language on the same line, body, closing ```.
# Non-greedy body, DOTALL so the body may span lines. An unterminated fence simply
# fails to match and is left verbatim (malformed -> raw).
_FENCE_RE = re.compile(r"```[^\S\n]*[^\n`]*\n.*?```", re.DOTALL)
# Inline code span (single line), matched only after fenced blocks are masked.
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _mask_code(text: str) -> tuple[str, list[str]]:
    """Replace fenced blocks and inline code with sentinels.

    Returns ``(masked_text, spans)`` where ``spans[i]`` is the verbatim text the
    ``i``-th sentinel restores to. Teams renders a fenced block as preformatted text
    and ignores the language tag, so both kinds are stored exactly as written.
    """
    spans: list[str] = []

    def _take(m: re.Match[str]) -> str:
        return _stash(spans, _SENTINEL, m.group(0))

    text = _FENCE_RE.sub(_take, text)
    text = _INLINE_CODE_RE.sub(_take, text)
    return text, spans


def _restore_code(text: str, spans: list[str]) -> str:
    """Substitute masked code sentinels back with their verbatim spans."""
    if not spans:
        return text

    def _put(m: re.Match[str]) -> str:
        return spans[int(m.group(1))]

    return _SENTINEL_RE.sub(_put, text)


# Heading: 1-6 leading hashes, at most 3 spaces of indent, optional closing hashes.
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$")
# Table separator row: only ``| - : space`` chars, with at least one dash.
_TABLE_SEP_RE = re.compile(r"^\s*\|?[ \t:|-]*-[ \t:|-]*\|?\s*$")


def _split_row(line: str) -> list[str]:
    """Split a table row on unescaped ``|``, dropping optional outer pipes."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _render_table_row(headers: list[str], cells: list[str]) -> str:
    """One data row -> a single ``**Header**: value`` labeled line (pairs joined by ', ').

    A value paired with an empty header emits the value alone; a header paired with an
    empty value keeps its label so the row still says which column went missing.
    """
    pairs: list[str] = []
    for header, value in zip(headers, cells, strict=True):
        pairs.append(f"**{header}**: {value}".rstrip() if header else value)
    return ", ".join(p for p in pairs if p)


def _convert_tables(text: str) -> str:
    """Convert Markdown tables to labeled ``**Header**: value`` lines, one per row.

    A block is a header row (containing ``|``) immediately followed by a separator
    row, then contiguous data rows. Malformed blocks — an escaped ``\\|`` in a cell
    or a ragged column count — are emitted verbatim rather than raising.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        is_header = (
            "|" in line and i + 1 < n and "|" in lines[i + 1] and _TABLE_SEP_RE.match(lines[i + 1])
        )
        if not is_header:
            out.append(line)
            i += 1
            continue
        j = i + 2
        while j < n and lines[j].strip() and "|" in lines[j]:
            j += 1
        block = lines[i:j]
        headers = _split_row(line)
        data = block[2:]
        malformed = any("\\|" in ln for ln in block) or any(
            len(_split_row(row)) != len(headers) for row in data
        )
        if malformed or not data:
            out.extend(block)
        else:
            out.extend(_render_table_row(headers, _split_row(row)) for row in data)
        i = j
    return "\n".join(out)


def _convert_line_blocks(text: str) -> str:
    """Line-oriented block transforms: tables, then headings.

    A heading becomes a Teams-bold line. Bullets and blockquotes are deliberately
    absent: Teams renders ``-``, ``*``, ``+`` and ``>`` itself, so the pipeline leaves
    them alone. Operates line by line; masked-code sentinel lines never match the
    heading pattern, so code is left untouched.
    """
    text = _convert_tables(text)
    out: list[str] = []
    for line in text.split("\n"):
        h = _HEADING_RE.match(line)
        if h and h.group(1).strip():
            out.append(f"**{h.group(1).strip()}**")
            continue
        out.append(line)
    return "\n".join(out)


# Math delimiters, matched longest-first. Display ``$$…$$`` before inline ``$…$``.
# Inline ``$…$`` requires a non-space, non-digit right after the opening ``$`` and a
# non-space before the closing ``$`` so bare currency (``$5``, ``$5 and $10``) is left
# untouched — only genuine matched delimiter pairs are stripped.
_MATH_RES = (
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    re.compile(r"\$(?=\S)(?!\d)(.+?)(?<=\S)\$"),
)


def _convert_math(text: str) -> str:
    """Strip LaTeX/math delimiters, leaving the inner expression as plain text.

    Handles ``$$…$$``, ``\\[…\\]``, ``\\(…\\)`` and inline ``$…$``. A lone ``$``
    (currency) is never stripped. No Unicode/symbol conversion is done.
    """
    for pattern in _MATH_RES:
        text = pattern.sub(lambda m: m.group(1), text)
    return text


def _convert_inline(text: str) -> str:
    """Inline span transforms.

    Only math: Teams renders emphasis and links in their Markdown spelling already,
    so rewriting them would be a downgrade, not a translation.
    """
    return _convert_math(text)


def markdown_to_teams(text: str) -> str:
    """Rewrite ``text`` from standard Markdown into the subset Teams renders.

    Runs the phases (mask code -> line-block -> inline -> restore code) in order.
    Never raises on malformed input — the worst case emits the original block
    unchanged. Empty input returns empty. Every construct is a fixpoint, so a second
    application is a no-op, but the real delivery path applies it exactly once.
    """
    if not text:
        return text
    masked, code = _mask_code(text)
    masked = _convert_line_blocks(masked)
    masked = _convert_inline(masked)
    return _restore_code(masked, code)
