"""Fence-aware message splitting, shared by every channel that caps a message.

Chat platforms reject a message body over a per-message character ceiling — Google
Chat at 4096, Microsoft Teams at 20 000 — and an agent answer routinely exceeds it,
so one answer has to travel as several messages. :func:`chunk_text` is that split,
and it is the same split everywhere: the rules that make a chunked answer readable
are properties of markdown and of prose, not of any one platform.

The ceiling itself is the one thing that is *not* shared, so ``limit`` is a
required argument with no default here. A default would have to be some
platform's number, and this package holds no channel knowledge by design (see
:mod:`osprey.bridges.core`) — an adapter names its own ceiling at the call site,
or wraps this function with it.

The split is fence-aware because half a ``` code block in each of two messages
renders as garbage in both. That is the only markup this module knows; any other
span that must travel whole (a rendered mention) is named by the caller through
``keep_whole``. Everything else is treated as plain text, so it runs safely over
text a caller has already transformed for its own channel.

Pure stdlib, pure functions, no state.
"""

from __future__ import annotations

import re


def _fence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans ``(start, end)`` of ``` fenced code blocks in ``text``.

    A block runs from a line whose first non-space characters are ```` ``` ```` to
    the next such line. An **unterminated** fence spans to the end of the text, so
    :func:`chunk_text` will not bisect that either — an answer cut off mid-block is
    exactly the case where the opening fence has no partner.

    Args:
        text: The text to scan.

    Returns:
        The spans, in order, non-overlapping. Empty when the text holds no fence.
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    in_fence = False
    start = 0
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            if in_fence:
                spans.append((start, offset + len(line)))
                in_fence = False
            else:
                in_fence = True
                start = offset
        offset += len(line) + 1  # +1 for the '\n' that split() consumed
    if in_fence:
        spans.append((start, len(text)))
    return spans


def chunk_text(text: str, limit: int, *, keep_whole: re.Pattern[str] | None = None) -> list[str]:
    """Split ``text`` into ``<=limit``-character chunks, preferring newline boundaries.

    Three properties, in priority order:

    1. every chunk fits the limit — this is the one the platform enforces, so it is
       never traded away;
    2. a ``` fenced block, or a span the caller names in ``keep_whole``, is never
       bisected — a split landing inside one is backed up to its start, so it
       travels whole in the next chunk (half a code block in each of two messages
       renders as garbage in both, and half a mention mentions nobody);
    3. otherwise the split is taken at the last newline in the window, so prose
       breaks between lines rather than mid-word.

    A single line — or a single fence or span — longer than ``limit`` cannot satisfy
    2 or 3 and is hard-split at the limit; a fence or span starting at offset 0 that
    overflows is the case where backing up would make no progress at all.

    Args:
        text: The message text, already transformed for its channel by the caller.
        limit: Maximum characters per chunk. Required, and deliberately so: the
            ceiling belongs to a channel, and this module knows none.
        keep_whole: A pattern whose matches must never be split across chunks,
            such as a channel's rendered mention. ``None`` names no span.

    Returns:
        The chunks in order, none of them empty — so a caller can never post an
        empty body by following this. The list itself is empty only for input that
        is empty or nothing but newlines, which is the same "there is no message
        here" case and is what a caller substitutes its own placeholder for.
        Trailing/leading newlines at the split points are stripped, so the chunks do
        not necessarily re-join to the original text.

    Raises:
        ValueError: If ``limit`` is not positive — a zero or negative ceiling would
            otherwise loop forever making no progress.
    """
    if limit <= 0:
        raise ValueError(f"chunk limit must be > 0; got {limit}")
    # Leading newlines are stripped up front, not only at each split: a window whose
    # last newline falls inside a leading run would otherwise cut a first chunk that
    # is all newlines, and rstrip would empty it — an empty body is a message these
    # platforms reject. The lstrip at the foot of the loop maintains the same
    # invariant for every later chunk, so no chunk can be empty.
    remaining = text.lstrip("\n")
    if not remaining:
        return []
    chunks: list[str] = []
    while len(remaining) > limit:
        window = remaining[:limit]
        # Prefer a line boundary within the window; fall back to a hard cut when
        # the window holds no newline (a single oversized line).
        split = window.rfind("\n")
        if split <= 0:
            split = limit
        for fence_start, fence_end in _fence_spans(remaining):
            if fence_start < split < fence_end:
                # Back the split up to the fence's start so the block stays whole.
                # A fence starting at 0 cannot be backed up any further, so the
                # hard split stands and that one block is bisected.
                if fence_start > 0:
                    split = fence_start
                break
        if keep_whole is not None:
            for m in keep_whole.finditer(remaining):
                if m.start() >= split:
                    break
                if m.start() < split < m.end():
                    # Same backing-up rule as a fence, and the same offset-0 exception.
                    if m.start() > 0:
                        split = m.start()
                    break
        chunks.append(remaining[:split].rstrip("\n"))
        remaining = remaining[split:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks
