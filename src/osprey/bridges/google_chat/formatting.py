"""Rewrite standard Markdown into Google Chat's supported formatting subset.

The OSPREY agent emits normal Markdown (which renders in the web terminal). Google
Chat's plain-text message field supports only a narrow subset, so this module
provides :func:`markdown_to_chat`, a *delivery-layer* transform applied to a local
copy of the answer text on the Google Chat posting path — the agent and its replayed
history are never touched (the "identical agent across channels" invariant). Keeping
that copy local is the posting caller's job; nothing here mutates its input.

Design: a **phased pipeline**, not a single regex pass, run in this fixed order:

    1. ``_mask_code``       — replace fenced blocks + inline code with inert
                              sentinels, so no later phase can rewrite code content.
                              Fenced blocks are cleaned here (language tag dropped),
                              body kept verbatim.
    2. ``_convert_line_blocks`` — line-oriented transforms (headings, bullets, tables).
    3. ``_convert_inline``  — span transforms (links, math, emphasis).
    4. ``_restore_code``    — substitute the masked spans back, verbatim.

Masking first is the safety foundation: ``**``, ``#``, ``|``, ``_`` inside code are
restored unchanged. Malformed constructs (unterminated fence, piped table cell) fall
back to their original text rather than raising — the answer always lands.

Standard library only (``re``); zero imports from the rest of osprey.

Chat rendering targets
----------------------
Each target below is Google Chat's documented formatting subset for that construct.

    Construct              Markdown in            Chat target out
    ---------------------  ---------------------  ----------------------------
    bold                   ``**x**`` / ``__x__``  ``*x*``
    italic                 ``*x*``                ``_x_``
    bold+italic            ``***x***``            ``*_x_*``   (documented degrade)
    heading                ``# x`` … ``###### x``  ``*x*`` (bold line, hashes gone)
    bullet                 ``- `` / ``* `` / ``+ ``  ``- `` (normalized)
    fenced code            ```` ```lang\n…``` ````  ```` ```\n…``` ```` (lang dropped)
    inline code            `` `x` ``              `` `x` `` (verbatim)
    table row              ``| H | v |``          ``*H:* v`` labeled lines
    math                   ``$x$`` / ``$$x$$``     ``x`` (delimiters stripped)
    inline link            ``[t](u)``             ``t: u``   (bare ``u`` when t==u)
    reference link         ``[t][r]`` + ``[r]: u``  ``t: u``
    bare URL               ``https://…``          unchanged (Chat auto-links)

Chat's own single-underscore-italic means ``snake_case`` / URL underscores can render
oddly in the client; we deliberately leave them untouched (a known Chat limitation,
not something the transform should mangle).
"""

from __future__ import annotations

import re


def _make_sentinel(name: str) -> tuple[str, re.Pattern[str]]:
    """Build a ``(template, pattern)`` sentinel pair from one ``name``.

    ``template.format(i)`` emits the ``i``-th placeholder; ``pattern`` recovers the
    index back out of it. Deriving both from a single name keeps them from drifting
    apart. Placeholders contain only NUL bytes and digits, so no line-block / inline /
    emphasis / link / math regex ever matches one.
    """
    return f"\x00{name}{{}}\x00", re.compile(rf"\x00{name}(\d+)\x00")


def _stash(store: list[str], template: str, content: str) -> str:
    """Append ``content`` to ``store`` and return its sentinel placeholder."""
    store.append(content)
    return template.format(len(store) - 1)


# Inert placeholder for a masked code span, so no later phase can rewrite code content
# and each masked span trivially survives the idempotence property.
_SENTINEL, _SENTINEL_RE = _make_sentinel("CODE")

# Bold-protection sentinel. Chat bold is single-``*`` (``*x*``), which is
# indistinguishable from a Markdown italic span, so any Chat-bold we produce
# (headings, ``**bold**``) must be shielded from the italic pass. Producers stash
# the bold *content* and emit this placeholder; ``_restore_bold`` materializes it to
# ``*content*`` only after all italic conversion is done.
_BOLD, _BOLD_RE = _make_sentinel("BOLD")

# Fenced block: opening ``` + optional language on the same line, body, closing ```.
# Non-greedy body, DOTALL so the body may span lines. An unterminated fence simply
# fails to match and is left verbatim (malformed -> raw).
_FENCE_RE = re.compile(r"```[^\S\n]*([^\n`]*)\n(.*?)```", re.DOTALL)
# Inline code span (single line), matched only after fenced blocks are masked.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")


def _mask_code(text: str) -> tuple[str, list[str]]:
    """Replace fenced blocks and inline code with sentinels.

    Returns ``(masked_text, spans)`` where ``spans[i]`` is the verbatim text the
    ``i``-th sentinel restores to. Fenced blocks are stored in their cleaned Chat
    form (language tag dropped, body verbatim); inline code is stored unchanged.
    """
    spans: list[str] = []

    def _take_fence(m: re.Match[str]) -> str:
        body = m.group(2)
        return _stash(spans, _SENTINEL, f"```\n{body}```")

    def _take_inline(m: re.Match[str]) -> str:
        return _stash(spans, _SENTINEL, m.group(0))

    text = _FENCE_RE.sub(_take_fence, text)
    text = _INLINE_CODE_RE.sub(_take_inline, text)
    return text, spans


def _restore_code(text: str, spans: list[str]) -> str:
    """Substitute masked code sentinels back with their verbatim spans."""
    if not spans:
        return text

    def _put(m: re.Match[str]) -> str:
        return spans[int(m.group(1))]

    return _SENTINEL_RE.sub(_put, text)


def _restore_bold(text: str, bolds: list[str]) -> str:
    """Materialize bold sentinels to Chat bold ``*content*`` (runs after italic)."""
    if not bolds:
        return text

    def _put(m: re.Match[str]) -> str:
        return f"*{bolds[int(m.group(1))]}*"

    return _BOLD_RE.sub(_put, text)


# Heading: 1-6 leading hashes, at most 3 spaces of indent, optional closing hashes.
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$")
# Bullet: an unordered list marker (-, *, +) followed by whitespace. Requiring the
# whitespace is what keeps ``**bold**`` / ``*italic*`` at line start from matching.
_BULLET_RE = re.compile(r"^(\s*)[-*+][ \t]+(.*)$")
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


def _render_table_row(headers: list[str], cells: list[str], bolds: list[str]) -> str:
    """One data row -> a single ``*Header:* value`` labeled line (pairs joined by ', ').

    A header paired with an empty label emits the value alone. Header labels are
    routed through the bold store so the italic pass can't corrupt them.
    """
    pairs: list[str] = []
    for header, value in zip(headers, cells, strict=True):
        if header:
            label = _stash(bolds, _BOLD, f"{header}:")
            pairs.append(f"{label} {value}".rstrip())
        else:
            pairs.append(value)
    return ", ".join(p for p in pairs if p)


def _convert_tables(text: str, bolds: list[str]) -> str:
    """Convert Markdown tables to labeled ``*Header:* value`` lines, one per row.

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
            out.extend(_render_table_row(headers, _split_row(row), bolds) for row in data)
        i = j
    return "\n".join(out)


def _convert_line_blocks(text: str, bolds: list[str]) -> str:
    """Line-oriented block transforms: tables, then headings and bullets.

    Headings become a protected Chat-bold line; ``-``/``*``/``+`` list markers are
    normalized to ``- ``. Operates line by line; masked-code sentinel lines never
    match either pattern, so code is left untouched.
    """
    text = _convert_tables(text, bolds)
    out: list[str] = []
    for line in text.split("\n"):
        h = _HEADING_RE.match(line)
        if h and h.group(1).strip():
            out.append(_stash(bolds, _BOLD, h.group(1).strip()))
            continue
        b = _BULLET_RE.match(line)
        if b:
            out.append(f"{b.group(1)}- {b.group(2)}")
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


# Emphasis, applied strictly in this order so each pass never sees the previous
# pass's markers. Bold (incl. bold+italic) is consumed into the bold sentinel BEFORE
# italic runs, so italic never sees bold's stars.
_BOLD_ITALIC_STAR_RE = re.compile(r"\*\*\*(.+?)\*\*\*")
_BOLD_ITALIC_US_RE = re.compile(r"(?<!\w)___(.+?)___(?!\w)")
# ``**bold**`` with flanking guards: the open ``**`` is not preceded by a word char
# and not followed by whitespace (mirror for the close), so ``2**3 and 4**2`` (two
# arithmetic operators) is never paired into a spurious bold span.
_BOLD_STAR_RE = re.compile(r"(?<!\w)\*\*(?!\s)(.+?)(?<!\s)\*\*(?!\w)")
# ``__bold__`` only at token boundaries, so ``snake__case`` (intraword) is left alone.
_BOLD_US_RE = re.compile(r"(?<!\w)__(.+?)__(?!\w)")
# Single-star italic: the star may not be adjacent to another star or whitespace, so
# ``2**3`` and stray ``* spaced *`` stars are left untouched.
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?![\s*])(.+?)(?<![\s*])\*(?!\*)")


def _convert_emphasis(text: str, bolds: list[str]) -> str:
    """Convert ``*``-based emphasis to Chat's subset (underscore-italic left as-is).

    ``**bold**``/``__bold__`` -> ``*bold*`` (protected); ``*italic*`` -> ``_italic_``;
    ``***x***`` -> ``*_x_*`` (documented degrade). Markdown ``_italic_`` already IS
    Chat italic, and ``_`` in ``snake_case``/URLs is deliberately untouched (Chat's
    own between-underscore italic is a documented limitation, not ours to fix).
    """

    text = _BOLD_ITALIC_STAR_RE.sub(lambda m: _stash(bolds, _BOLD, f"_{m.group(1)}_"), text)
    text = _BOLD_ITALIC_US_RE.sub(lambda m: _stash(bolds, _BOLD, f"_{m.group(1)}_"), text)
    text = _BOLD_STAR_RE.sub(lambda m: _stash(bolds, _BOLD, m.group(1)), text)
    text = _BOLD_US_RE.sub(lambda m: _stash(bolds, _BOLD, m.group(1)), text)
    text = _ITALIC_STAR_RE.sub(lambda m: f"_{m.group(1)}_", text)
    return text


# Links. Reference definition line ``[ref]: url``; inline ``[text](url)`` (not an
# image ``![...]``); reference use ``[text][ref]`` (``[text][]`` -> ref == text).
_REF_DEF_RE = re.compile(r"^ {0,3}\[([^\]]+)\]:[ \t]+(\S+)[ \t]*$")
_INLINE_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")
_REF_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\[([^\]]*)\]")


def _link_target(label: str, url: str) -> str:
    """The Chat-legible target: bare ``url`` when label is empty or == url,
    else ``label: url``. A bare URL is never bracketed (Chat auto-links it)."""
    label = label.strip()
    if not label or label == url:
        return url
    return f"{label}: {url}"


def _convert_links(text: str) -> str:
    """Rewrite inline and reference-style Markdown links to the Chat target form.

    Reference definitions are collected and their lines dropped (they never render).
    A reference use with no matching definition is left unchanged. Bare URLs and
    Markdown images (``![alt](url)``) are left untouched.
    """
    defs: dict[str, str] = {}
    kept: list[str] = []
    for line in text.split("\n"):
        m = _REF_DEF_RE.match(line)
        if m:
            defs[m.group(1).strip().lower()] = m.group(2).strip()
        else:
            kept.append(line)
    text = "\n".join(kept)

    text = _INLINE_LINK_RE.sub(lambda m: _link_target(m.group(1), m.group(2)), text)

    def _ref(m: re.Match[str]) -> str:
        label = m.group(1).strip()
        ref = (m.group(2).strip() or label).lower()
        return _link_target(label, defs[ref]) if ref in defs else m.group(0)

    return _REF_LINK_RE.sub(_ref, text)


def _convert_inline(text: str, bolds: list[str]) -> str:
    """Inline span transforms. Order: links, math, then emphasis."""
    text = _convert_links(text)
    text = _convert_math(text)
    text = _convert_emphasis(text, bolds)
    return text


def markdown_to_chat(text: str) -> str:
    """Rewrite ``text`` from standard Markdown into Google Chat's formatting subset.

    Runs the phases (mask code -> line-block -> inline -> restore bold -> restore
    code) in order. Correct for a single application (the delivery-layer path).
    Never raises on malformed input — the worst case emits the original block
    unchanged. Empty input returns empty.

    Note on idempotence: Chat overloads single ``*`` for bold, so a produced Chat
    bold ``*x*`` is byte-identical to a Markdown italic ``*x*``. Re-feeding already
    converted bold/heading text would therefore re-read it as italic. Every other
    construct (code, italic->``_``, bullets, tables, math, links) is a fixpoint; the
    transform is only ever applied once in the real path.
    """
    if not text:
        return text
    masked, code = _mask_code(text)
    bolds: list[str] = []
    masked = _convert_line_blocks(masked, bolds)
    masked = _convert_inline(masked, bolds)
    masked = _restore_bold(masked, bolds)
    return _restore_code(masked, code)
