"""Rewrite standard Markdown into Slack's ``mrkdwn`` text format.

The OSPREY agent emits normal Markdown. Slack's message ``text`` field speaks
``mrkdwn``, a different dialect, so :func:`markdown_to_mrkdwn` is a *delivery-layer*
transform applied to the posted copy of an answer; the agent and its history are never
touched. It is pure, never raises, idempotent (``f(f(x)) == f(x)``) and stdlib only.

Design: a phased pipeline run in a fixed order.

    1. mask            -- fenced blocks, inline code, ``<@ID>`` mention placeholders
                          and Slack link forms already present become inert sentinels.
    2. links, math     -- ``[t](u)`` stashed as ``<u|t>``; math delimiters stripped.
    3. line blocks     -- tables, headings, bullets.
    4. inline          -- strikethrough, bold.
    5. escape          -- ``&``, ``<``, ``>`` everywhere still visible.
    6. restore         -- sentinels back to their verbatim spans.

Targets
-------

    Construct            Markdown in              mrkdwn out
    -------------------  -----------------------  ---------------------------
    bold                 ``**x**`` / ``__x__``    ``*x*``
    italic               ``_x_``                  ``_x_``
    single-star span     ``*x*``                  ``*x*`` (unchanged, see below)
    bold+italic          ``***x***`` / ``___x___``  ``*_x_*``
    strikethrough        ``~~x~~``                ``~x~``
    heading              ``# x`` .. ``###### x``  ``*x*`` (bold line)
    bullet               ``- `` / ``* `` / ``+ ``  ``- ``
    fenced code          ```` ```lang\\n...``` ````  ```` ```\\n...``` ```` (lang dropped)
    inline code          `` `x` ``                verbatim
    table row            ``| H | v |``            ``*H*: v`` (pairs joined by ``, ``)
    math                 ``$x$`` / ``$$x$$`` / ``\\(x\\)`` / ``\\[x\\]``  ``x``
    link                 ``[t](u)``               ``<u|t>`` (``<u>`` when t is empty or u)
    mention placeholder  ``<@U1>``                verbatim
    anything else ``<``  ``<!channel>``           ``&lt;!channel&gt;``

Only ``http(s)://`` and ``mailto:`` targets become Slack links; any other ``[t](u)``
is left as text, because Slack would show a non-URL link target as garbage.

Mentions
--------

:func:`render_mentions` runs per chunk *after* :func:`markdown_to_mrkdwn`. A ``<@ID>``
placeholder stays a live ``<@ID>`` only for a roster member when mentions are on, and
becomes plain ``@Name`` / ``@ID`` otherwise. Every ``<!...>`` control sequence
(``<!channel>``, ``<!here>``, ``<!everyone>``, ``<!subteam^...>``, any label, any other
form) is escaped to inert text, so the bridge can never broadcast-ping a channel.

Idempotence forces one deliberate choice: Markdown's single-star italic ``*x*`` is
byte-identical to mrkdwn bold, so it is left as ``*x*`` (Slack renders it bold).
Rewriting it to ``_x_`` would turn every bold this transform produces into italic on a
second pass. ``_x_`` remains the italic form.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = ["MENTION_PLACEHOLDER_RE", "markdown_to_mrkdwn", "render_mentions"]

MENTION_PLACEHOLDER_RE = re.compile(r"<@([^\s<>]+)>")
"""The agent's mention placeholder, ``<@ID>``.

A local copy of ``osprey.bridges.core.MENTION_PLACEHOLDER_RE`` (a test pins the two
equal) so this module stays standard-library only.
"""


def _make_sentinel(name: str) -> tuple[str, re.Pattern[str]]:
    """Build a ``(template, pattern)`` sentinel pair from one ``name``.

    Placeholders hold only NUL bytes, letters and digits, so no conversion or escape
    pattern ever matches one; NUL is stripped from the input so none can be forged.
    """
    return f"\x00{name}{{}}\x00", re.compile(rf"\x00{name}(\d+)\x00")


def _stash(store: list[str], template: str, content: str) -> str:
    """Append ``content`` to ``store`` and return its sentinel placeholder."""
    store.append(content)
    return template.format(len(store) - 1)


def _restore(text: str, store: list[str], pattern: re.Pattern[str], fmt: str = "{}") -> str:
    """Substitute sentinels back, repeating because a stashed span may hold another."""
    for _ in range(len(store) + 1):
        if not pattern.search(text):
            break
        text = pattern.sub(lambda m: fmt.format(store[int(m.group(1))]), text)
    return text


# Verbatim spans: code, mention placeholders, links (already Slack or produced here).
_KEEP, _KEEP_RE = _make_sentinel("KEEP")
# Produced bold content; materialized to ``*content*`` once bold conversion is done.
_BOLD, _BOLD_RE = _make_sentinel("BOLD")

_FENCE_RE = re.compile(r"```[^\S\n]*([^\n`]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_URL_PREFIX = r"(?:https?://|mailto:)"
# A Slack link already in the text, ``<url>`` or ``<url|label>``, with a URL target.
_SLACK_LINK_RE = re.compile(rf"<{_URL_PREFIX}[^\s<>|]+(?:\|[^<>\n]*)?>")
_BARE_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt);)")


def _escape(text: str) -> str:
    """Escape ``& < >`` for Slack, leaving the three entities already escaped alone."""
    text = _BARE_AMP_RE.sub("&amp;", text)
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _mask(text: str, keep: list[str]) -> str:
    """Stash fenced blocks, inline code, mention placeholders and Slack links."""
    text = _FENCE_RE.sub(lambda m: _stash(keep, _KEEP, f"```\n{m.group(2)}```"), text)
    text = _INLINE_CODE_RE.sub(lambda m: _stash(keep, _KEEP, m.group(0)), text)
    text = MENTION_PLACEHOLDER_RE.sub(lambda m: _stash(keep, _KEEP, m.group(0)), text)
    return _SLACK_LINK_RE.sub(
        lambda m: _stash(keep, _KEEP, _BARE_AMP_RE.sub("&amp;", m.group(0))), text
    )


_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+][ \t]+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[ \t:|-]*-[ \t:|-]*\|?\s*$")

# Every emphasis span refuses a marker run longer than its own and content that
# starts or ends with its marker character or whitespace, so a stray run of stars or
# tildes is never half-consumed (which would leave a new span for a second pass).
_BOLD_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*\*\*(?![\s*])(.+?)(?<![\s*])\*\*\*(?!\*)")
_BOLD_ITALIC_US_RE = re.compile(r"(?<![\w_])___(?![\s_])(.+?)(?<![\s_])___(?![\w_])")
# Flanking guards keep ``2**3 and 4**2`` (two operators) from pairing into bold.
_BOLD_STAR_RE = re.compile(r"(?<![\w*])\*\*(?![\s*])(.+?)(?<![\s*])\*\*(?![\w*])")
_BOLD_US_RE = re.compile(r"(?<![\w_])__(?![\s_])(.+?)(?<![\s_])__(?![\w_])")
_STRIKE_RE = re.compile(r"(?<![~\w])~~(?![\s~])(.+?)(?<![\s~])~~(?![~\w])")


def _plain_label(text: str) -> str:
    """Drop bold markers (and convert strikes) in text about to become a bold ``*...*``.

    Nested stars inside a produced bold would be re-read on a second pass, so a
    heading or table header keeps its words and loses its markers.
    """
    for pattern in (_BOLD_ITALIC_STAR_RE, _BOLD_STAR_RE, _BOLD_US_RE):
        text = pattern.sub(lambda m: m.group(1), text)
    text = _STRIKE_RE.sub(lambda m: f"~{m.group(1)}~", text)
    return text.strip("*").strip()


def _split_row(line: str) -> list[str]:
    """Split a table row on ``|``, dropping optional outer pipes."""
    s = line.strip().removeprefix("|").removesuffix("|")
    return [c.strip() for c in s.split("|")]


def _render_table_row(headers: list[str], cells: list[str], bolds: list[str]) -> str:
    """One data row -> ``*Header*: value`` pairs joined by ``, ``."""
    pairs: list[str] = []
    for header, value in zip(headers, cells, strict=True):
        label = _plain_label(header)
        pairs.append(f"{_stash(bolds, _BOLD, label)}: {value}".rstrip() if label else value)
    return ", ".join(p for p in pairs if p)


def _convert_tables(text: str, bolds: list[str]) -> str:
    """Convert Markdown tables to labelled lines; a malformed block stays verbatim."""
    lines = text.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not (
            "|" in line and i + 1 < n and "|" in lines[i + 1] and _TABLE_SEP_RE.match(lines[i + 1])
        ):
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
    """Tables, then headings (bold line) and bullets (normalized to ``- ``)."""
    text = _convert_tables(text, bolds)
    out: list[str] = []
    for line in text.split("\n"):
        h = _HEADING_RE.match(line)
        label = _plain_label(h.group(1)) if h else ""
        if label:
            out.append(_stash(bolds, _BOLD, label))
            continue
        b = _BULLET_RE.match(line)
        out.append(f"{b.group(1)}- {b.group(2)}" if b else line)
    return "\n".join(out)


_INLINE_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]*)\]\(([^)\s]+)\)")
_URL_RE = re.compile(rf"{_URL_PREFIX}[^\s<>|]+")


def _convert_links(text: str, keep: list[str]) -> str:
    """``[t](u)`` with a URL target -> a stashed ``<u|t>`` (label escaped)."""

    def one(m: re.Match[str]) -> str:
        label, url = m.group(1).strip(), m.group(2)
        if not _URL_RE.fullmatch(url):
            return m.group(0)
        url = _BARE_AMP_RE.sub("&amp;", url)
        if not label or label == m.group(2):
            return _stash(keep, _KEEP, f"<{url}>")
        label = _escape(_restore(label, keep, _KEEP_RE)).replace("\n", " ")
        return _stash(keep, _KEEP, f"<{url}|{label}>")

    return _INLINE_LINK_RE.sub(one, text)


# Display forms before inline; inline ``$`` needs a non-space, non-digit opener so
# currency (``$5 and $10``) is never stripped.
# The content may not hold its own opener, so stripping one pair never exposes another.
_MATH_RES = (
    re.compile(r"\$\$((?:(?!\$\$).)+?)\$\$", re.DOTALL),
    re.compile(r"\\\[((?:(?!\\\[).)+?)\\\]", re.DOTALL),
    re.compile(r"\\\(((?:(?!\\\().)+?)\\\)", re.DOTALL),
    re.compile(r"\$(?=\S)(?!\d)([^$]+?)(?<=\S)\$"),
)


def _strip_math(text: str) -> str:
    """Strip math delimiters until none are left.

    Stripping one form can expose another (``$$a\\(b$\\)``), so the passes repeat
    until the text stops changing; each pass shortens it, so the loop ends.
    """
    while True:
        before = text
        for pattern in _MATH_RES:
            text = pattern.sub(lambda m: m.group(1), text)
        if text == before:
            return text


def _convert_inline(text: str, bolds: list[str]) -> str:
    """Strikethrough, then bold (stashed so nothing re-reads it)."""
    text = _STRIKE_RE.sub(lambda m: f"~{m.group(1)}~", text)
    text = _BOLD_ITALIC_STAR_RE.sub(lambda m: _stash(bolds, _BOLD, f"_{m.group(1)}_"), text)
    text = _BOLD_ITALIC_US_RE.sub(lambda m: _stash(bolds, _BOLD, f"_{m.group(1)}_"), text)
    text = _BOLD_STAR_RE.sub(lambda m: _stash(bolds, _BOLD, m.group(1)), text)
    return _BOLD_US_RE.sub(lambda m: _stash(bolds, _BOLD, m.group(1)), text)


def markdown_to_mrkdwn(text: str) -> str:
    """Rewrite ``text`` from standard Markdown into Slack ``mrkdwn``.

    Pure and idempotent; never raises (a malformed construct is emitted as escaped raw
    text). ``<@ID>`` placeholders come out verbatim for the mention renderer; every
    other ``<...>`` is escaped. Empty input returns empty.
    """
    if not text:
        return text
    text = text.replace("\x00", "")
    keep: list[str] = []
    bolds: list[str] = []
    text = _mask(text, keep)
    text = _convert_links(text, keep)
    # Math goes before line blocks: a stripped delimiter can leave a bullet or table.
    text = _strip_math(text)
    text = _convert_line_blocks(text, bolds)
    text = _convert_inline(text, bolds)
    text = _restore(text, bolds, _BOLD_RE, "*{}*")
    text = _escape(text)
    return _restore(text, keep, _KEEP_RE)


# A placeholder, or a ``<!`` control opener up to its ``>`` (absent when unclosed).
_MENTION_OR_CONTROL_RE = re.compile(r"<@([^\s<>]+)>|<![^<>]*>?")
_WHITESPACE_RE = re.compile(r"\s+")


def _mention_label(name: str | None) -> str:
    """A roster name made inert for plain text: whitespace runs collapsed to one space,
    then ``& < >`` escaped so a name can never form a control sequence. ``""`` for
    ``None``."""
    if not name:
        return ""
    return _escape(_WHITESPACE_RE.sub(" ", name).strip())


def render_mentions(text: str, roster: Mapping[str, str | None], *, enabled: bool) -> str:
    """Render the agent's mention placeholders in one chunk for Slack.

    A ``<@ID>`` whose id is a key of ``roster`` (the channel's members) stays a live
    ``<@ID>`` when ``enabled``; Slack shows the name itself, so a listed member with no
    name still renders live. Anything else becomes plain ``@Name`` (listed, named) or
    ``@ID``. Every ``<!...>`` broadcast or group form, with or without ``|label`` and
    closed or not, is escaped to ``&lt;!...&gt;`` text that pings nobody. Pure; never
    raises.
    """
    if not text:
        return text

    def one(m: re.Match[str]) -> str:
        ident = m.group(1)
        if ident is None:
            return _escape(m.group(0))
        if enabled and ident in roster:
            return m.group(0)
        return "@" + (_mention_label(roster.get(ident)) or _escape(ident))

    return _MENTION_OR_CONTROL_RE.sub(one, text)
