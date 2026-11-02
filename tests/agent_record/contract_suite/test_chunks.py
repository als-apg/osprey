"""Chunked bodies on every backend: a body larger than the chunk limit
round-trips byte-identical (FR8), a re-sent chunked entry stores no duplicate
chunk, a re-send with a different chunk total keeps the first set, and a
missing chunk is returned flagged incomplete.
"""

from __future__ import annotations

from osprey.agent_record import contract as c

from . import _kit as k

#: The chunk limit the cases split at; every body here is many times larger.
LIMIT = 1024

KEY = "cc:alice:sess-big:u1:0"
SESSION = "sess-big"

#: Multi-byte text, nested values and key order that reassembly must keep.
BODY = {
    "text": "Strahlstrom ändert sich — ±0.5 mA, Δφ ≈ 3° 🚀 " * 400,
    "z_first": {"nested": [1, 2.5, None, True, "x"], "a_second": "kept in order"},
    "quote": 'a "quoted" \\ backslash and a newline\n',
}


def _answer(body: dict, t: float = 1) -> c.Entry:
    return k.cc(k.ALICE, SESSION, 1, c.KIND_ANSWER, t, body)


def test_a_body_over_the_limit_round_trips_byte_identical(backend, settle):
    chunks = c.split_entry(_answer(BODY), LIMIT)
    assert len(chunks) > 10
    assert len(c.serialize_body(BODY).encode()) > 10 * LIMIT
    assert backend.append(chunks).appended == len(chunks)
    settle()
    stored = [e for e in k.rows(backend) if e.key == KEY]
    assert sorted(e.chunk.index for e in stored) == list(range(len(chunks)))
    answer = k.reader(backend).full_entry(c.Own(k.ALICE), KEY)
    assert answer.available and answer.complete, answer.reason
    item = answer.data
    assert item.entry.chunk == c.UNSPLIT and not item.incomplete
    assert c.serialize_body(item.body) == c.serialize_body(BODY)
    assert list(item.body) == list(BODY)


def test_a_resent_chunked_entry_leaves_no_duplicate_chunk(backend, settle):
    chunks = c.split_entry(_answer(BODY), LIMIT)
    backend.append(chunks)
    again = backend.append(c.split_entry(_answer(BODY), LIMIT))
    assert again == c.AppendResult(appended=0, duplicates=len(chunks))
    settle()
    stored = [(e.key, e.chunk.index) for e in k.rows(backend)]
    assert len(stored) == len(set(stored)) == len(chunks)
    assert k.reader(backend).full_entry(c.All(), KEY).data.body == BODY


def test_a_resend_with_a_different_chunk_total_keeps_the_first_set(backend, settle):
    first = c.split_entry(_answer(BODY), LIMIT)
    longer = {**BODY, "text": BODY["text"] * 2}
    second = c.split_entry(_answer(longer, t=2), LIMIT)
    assert len(second) > len(first)
    backend.append(first)
    result = backend.append(second)
    assert result.duplicates == len(first)
    settle()
    answer = k.reader(backend).full_entry(c.All(), KEY)
    assert answer.complete, answer.reason
    assert c.serialize_body(answer.data.body) == c.serialize_body(BODY)
    assert answer.data.ts == k.at(1)


def test_a_shorter_resend_keeps_the_first_longer_set(backend, settle):
    longer = {**BODY, "text": BODY["text"] * 2}
    first = c.split_entry(_answer(longer), LIMIT)
    second = c.split_entry(_answer(BODY, t=2), LIMIT)
    assert len(second) < len(first)
    backend.append(first)
    assert backend.append(second) == c.AppendResult(appended=0, duplicates=len(second))
    settle()
    assert k.reader(backend).full_entry(c.All(), KEY).data.body == longer


def test_a_missing_chunk_is_returned_flagged_incomplete(backend, settle):
    chunks = c.split_entry(_answer(BODY), LIMIT)
    backend.append([ch for ch in chunks if ch.chunk.index != 3])
    backend.append([k.cc(k.ALICE, SESSION, 0, c.KIND_PROMPT, 0, {"text": "go"})])
    settle()
    reader = k.reader(backend)
    whole = reader.full_entry(c.All(), KEY)
    assert not whole.available and not whole.complete and whole.data is None
    timeline = reader.timeline(c.Own(k.ALICE), SESSION)
    assert timeline.available and not timeline.complete and "chunk" in timeline.reason
    broken = [n.item for n in timeline.data if n.item.key == KEY]
    assert len(broken) == 1 and broken[0].incomplete and broken[0].body is None
    assert [n.item.kind for n in timeline.data] == [c.KIND_PROMPT, c.KIND_ANSWER]
