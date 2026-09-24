"""The shared chunker: chunk sizes, split points, and fenced blocks kept whole.

Every property here is asserted at **both** shipped ceilings — Google Chat's 4096
and Microsoft Teams' 20 000 — because the chunker is now shared and a rule that
only holds at one limit is a rule that does not hold. The cases are written
relative to ``limit`` for that reason: nothing below hard-codes a length that
happens to straddle 4096.

Two properties get the most attention, because each breaks a live bridge in a way
a smaller test will not notice:

* no chunk is ever empty — an empty body is a message these platforms reject, so
  an answer whose first chunk collapses is an answer that fails to post at all;
* a ``` fenced block is never bisected, since half a code block in each of two
  messages renders as garbage in both.
"""

import re

import pytest

from osprey.bridges.core.text import _fence_spans, chunk_text

LIMITS = (4096, 20_000)
"""The per-message ceilings the shipped bridges pass: Chat's and Teams'."""


# --- chunk_text ------------------------------------------------------------


@pytest.mark.parametrize("limit", LIMITS)
def test_empty_text_yields_no_chunks(limit):
    assert chunk_text("", limit) == []


@pytest.mark.parametrize("limit", LIMITS)
def test_short_text_is_one_chunk(limit):
    assert chunk_text("hello", limit) == ["hello"]


@pytest.mark.parametrize("limit", LIMITS)
def test_long_text_splits_within_the_limit(limit):
    text = "x" * (2 * limit + 500)
    chunks = chunk_text(text, limit)

    assert len(chunks) == 3
    assert all(len(chunk) <= limit for chunk in chunks)
    assert "".join(chunks) == text


@pytest.mark.parametrize("limit", LIMITS)
def test_split_prefers_the_last_newline(limit):
    line = "a" * (limit - 1)
    chunks = chunk_text(line + "\n" + "b" * (limit - 1), limit)

    assert chunks == [line, "b" * (limit - 1)]


@pytest.mark.parametrize("limit", LIMITS)
def test_no_chunk_is_empty(limit):
    chunks = chunk_text(("line\n" * (limit // 2)).strip(), limit)

    assert chunks
    assert all(chunk for chunk in chunks)


@pytest.mark.parametrize("limit", LIMITS)
def test_leading_newlines_do_not_produce_an_empty_first_chunk(limit):
    # The window's last newline falls inside the leading run, so the first split
    # cuts nothing but newlines. rstrip would empty that chunk, and an empty message
    # body is rejected — the first post of a long answer would fail.
    body = "a" * (2 * limit + 500)
    chunks = chunk_text("\n\n\n" + body, limit)

    assert all(chunk for chunk in chunks)
    assert "".join(chunks) == body


@pytest.mark.parametrize("limit", LIMITS)
def test_text_that_is_only_newlines_yields_no_chunks(limit):
    # The same "there is no message here" case as empty input: the caller
    # substitutes its own placeholder rather than posting a blank line.
    assert chunk_text("\n" * (limit + 1), limit) == []


@pytest.mark.parametrize("limit", LIMITS)
def test_a_fence_is_never_bisected(limit):
    # A fence straddling the limit is backed up to its opening ``` and travels
    # whole in the next chunk.
    head = "A" * (limit - 46)
    fence = "```\n" + ("B" * 100) + "\n```"
    chunks = chunk_text(head + "\n" + fence + "\n" + ("C" * 100), limit)

    assert chunks[0] == head
    assert fence in chunks[1]
    assert chunks[1].count("```") == 2


@pytest.mark.parametrize("limit", LIMITS)
def test_an_oversized_fence_still_hard_splits(limit):
    # A fence that opens at offset 0 and overflows the limit cannot be kept
    # whole — backing the split up would make no progress — so the limit wins.
    body = "A" * (limit + 1000)
    chunks = chunk_text("```\n" + body + "\n```", limit)

    assert len(chunks) >= 2
    assert all(len(chunk) <= limit for chunk in chunks)
    assert "".join(chunks).count("A") == limit + 1000


def test_chunking_honors_the_limit_it_is_given():
    assert chunk_text("abcdefgh", 3) == ["abc", "def", "gh"]


def test_the_limit_is_required():
    # Deliberately not defaulted: the ceiling belongs to a channel, and a core
    # module that picked one would be picking some platform's.
    with pytest.raises(TypeError):
        chunk_text("anything")  # type: ignore[call-arg]


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_is_rejected(limit):
    # Would otherwise spin forever making no progress.
    with pytest.raises(ValueError, match="chunk limit"):
        chunk_text("anything", limit)


# --- _fence_spans ----------------------------------------------------------


def test_fence_spans_finds_nothing_in_plain_text():
    assert _fence_spans("just prose\nover two lines") == []


def test_fence_spans_covers_the_whole_block():
    text = "intro\n```\ncode\n```\ntail"
    (start, end) = _fence_spans(text)[0]

    assert text[start:end] == "```\ncode\n```"


def test_fence_spans_finds_each_block():
    assert len(_fence_spans("```\na\n```\nmid\n```\nb\n```")) == 2


def test_an_unterminated_fence_spans_to_the_end():
    # The truncated-answer case: no closing fence, so everything after the
    # opening one is still protected from a split.
    text = "intro\n```\ncode that never closes"
    (start, end) = _fence_spans(text)[0]

    assert (start, end) == (6, len(text))


def test_an_indented_fence_counts():
    assert _fence_spans("intro\n    ```\ncode\n    ```") != []


# --- keep_whole: spans the caller names are never bisected ----------------------

MENTION = re.compile(r"<users/[^\s<>]+>")


def test_a_keep_whole_span_is_never_bisected_by_a_hard_split():
    token = "<users/1234567890>"
    for offset in range(1, len(token)):
        text = "x" * (100 - offset) + token + "y" * 50
        chunks = chunk_text(text, 100, keep_whole=MENTION)
        assert all(len(chunk) <= 100 for chunk in chunks)
        assert "".join(chunks) == text
        for chunk in chunks:
            assert "<users/" not in chunk or token in chunk
            assert chunk.count("<") == chunk.count(">")


def test_keep_whole_does_not_change_a_split_that_misses_every_span():
    text = "a" * 30 + " <users/1> " + "b" * 200 + "\n" + "c" * 120
    assert chunk_text(text, 100, keep_whole=MENTION) == chunk_text(text, 100)


def test_a_keep_whole_span_at_offset_zero_that_overflows_is_hard_split():
    text = "<users/" + "9" * 200 + ">"
    chunks = chunk_text(text, 100, keep_whole=MENTION)
    assert [len(chunk) for chunk in chunks] == [100, 100, 8]
    assert "".join(chunks) == text
