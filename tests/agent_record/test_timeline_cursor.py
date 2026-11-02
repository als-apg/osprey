"""The snapshot ``after`` cursor of ``timeline`` and ``agent_timeline`` (P3-D2).

A timeline is paged by its top-level nodes in ``(ts, inserted_at, key,
chunk_index)`` order. The first page fixes ``as_of``; later pages read only
what arrived by it and start strictly after the cursor's node, so a page never
repeats or skips a node of the snapshot, and ``newer`` reports what arrived
after it. Backends stamp ``inserted_at`` in commit order (A2-2), so an entry
committed between two page reads, even with the clock stepped back, is outside
the snapshot.
"""

from __future__ import annotations

import base64
import json
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import reader as rd
from osprey.agent_record.backends import embedded
from osprey.agent_record.backends.embedded import EmbeddedBackend
from tests.agent_record.test_identities import _config
from tests.agent_record.test_reader import NOW, T0, _cc, _Clock, _flat

SESSION = "sess-paged"
A = "alice"
#: Every seeded entry's ts: one instant shared by the whole session.
TS = 10


def _seed() -> list[c.Entry]:
    """One session whose every entry shares one ``ts``: prompts and answers,
    calls joined to their decision and result, an Agent call with its subagent
    stream, and a subagent group whose call is not in the record."""
    out: list[c.Entry] = []
    n = 0

    def add(kind: str, body: dict, **fields) -> None:
        nonlocal n
        n += 1
        out.append(_cc(A, SESSION, n, kind, TS, body, **fields))

    for turn in range(6):
        add(c.KIND_PROMPT, {"text": f"ask {turn}"})
        add(c.KIND_THINKING, {"text": f"think {turn}"})
        tu = f"tu-{turn}"
        add(c.KIND_TOOL_CALL, {"id": tu, "name": "Read"}, tool_use_id=tu)
        add(c.KIND_DECISION, {"decision": "accept"}, tool_use_id=tu)
        add(c.KIND_TOOL_RESULT, {"content": f"file {turn}"}, tool_use_id=tu)
        add(c.KIND_ANSWER, {"text": f"answer {turn}"})
    add(c.KIND_TOOL_CALL, {"id": "tu-agent", "name": "Agent"}, tool_use_id="tu-agent")
    add(c.KIND_PROMPT, {"text": "sub task"}, agent_id="ag1")
    add(c.KIND_TOOL_CALL, {"id": "tu-sub", "name": "Read"}, tool_use_id="tu-sub", agent_id="ag1")
    add(c.KIND_TOOL_RESULT, {"content": "sub file"}, tool_use_id="tu-sub", agent_id="ag1")
    add(c.KIND_ANSWER, {"text": "sub done"}, agent_id="ag1")
    add(c.KIND_SUBAGENT, {"agent_id": "ag1"}, tool_use_id="tu-agent", agent_id="ag1")
    add(c.KIND_ANSWER, {"text": "orphan says"}, agent_id="ag9")
    add(c.KIND_ANSWER, {"text": "orphan again"}, agent_id="ag9")
    return out


@pytest.fixture
def config() -> dict:
    return _config(workers=2, bridges=("teams_bridge",), lanes=True)


@pytest.fixture
def backend(tmp_path: Path):
    with EmbeddedBackend(tmp_path / "record.duckdb") as b:
        b.append(_seed())
        yield b


@pytest.fixture
def reader(backend, config) -> rd.RecordReader:
    clock = _Clock()
    return rd.RecordReader(backend, config, clock=clock, sleep=clock.sleep, now=lambda: NOW)


def _node_ids(nodes) -> list[str]:
    """One id per top-level node: its item's key, or its group's agent id."""
    return [n.item.key if n.item is not None else f"group:{n.agent_id}" for n in nodes]


def _pages(read, limit: int, *, between=None, first=None):
    """Every page of one read, following ``next``; *between* runs after page 1."""
    pages = [first if first is not None else read(None, limit)]
    while pages[-1].next is not None:
        if between is not None and len(pages) == 1:
            between()
        pages.append(read(pages[-1].next, limit))
        assert len(pages) < 200, "paging did not end"
    return pages


def _timeline(reader, scope=None):
    scope = scope or c.All()
    return lambda after, limit: reader.timeline(scope, SESSION, after=after, limit=limit)


# --- every node once, in order ----------------------------------------------------


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 7, 50])
def test_paging_yields_every_top_level_node_once_and_in_order(reader, limit):
    whole = reader.timeline(c.All(), SESSION)
    assert whole.complete and whole.next is None and not whole.more and not whole.newer
    expected = _node_ids(whole.data)
    assert len(expected) == 6 * 4 + 1 + 1  # four nodes per turn, the Agent call, the group
    pages = _pages(_timeline(reader), limit)
    for page in pages:
        assert page.complete and not page.newer
        assert page.more == (page.next is not None)
        assert 0 < len(page.data) <= limit
    assert [i for p in pages for i in _node_ids(p.data)] == expected
    assert pages[-1].next is None


def test_timeline_order_is_ts_then_inserted_at_then_key(reader):
    positions = [n.position for n in reader.timeline(c.All(), SESSION).data]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_agent_timeline_pages_its_stream_the_same_way(reader, limit):
    whole = reader.agent_timeline(c.All(), SESSION, "ag1")
    assert len(whole.data) == 3

    def read(after, n):
        return reader.agent_timeline(c.All(), SESSION, "ag1", after=after, limit=n)

    pages = _pages(read, limit)
    assert [i for p in pages for i in _node_ids(p.data)] == _node_ids(whole.data)


def test_paging_under_own_sees_only_the_persons_nodes(reader):
    pages = _pages(_timeline(reader, c.Own("alice")), 4)
    assert [i for p in pages for i in _node_ids(p.data)] == _node_ids(
        reader.timeline(c.Own("alice"), SESSION).data
    )
    assert reader.timeline(c.Own("bob"), SESSION, limit=4).next is None


# --- a call and what joins to it stay on one page ---------------------------------


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_a_call_and_what_joins_to_it_stay_on_one_page(reader, limit):
    whole = reader.timeline(c.All(), SESSION)
    pages = _pages(_timeline(reader), limit)
    for page in pages:
        for node in page.data:
            if node.item is not None and node.item.kind == c.KIND_TOOL_CALL:
                assert node.result is not None, node.item.key
                if node.item.entry.tool_use_id != "tu-agent":
                    assert node.decision is not None, node.item.key
    paged = [k for p in pages for k in _flat(p.data)]
    assert paged == _flat(whole.data)
    assert len(paged) == len(set(paged)) == len(_seed())


# --- entries that arrive between page reads ---------------------------------------


def test_an_early_entry_arriving_between_pages_is_reported_through_newer(
    backend, reader, monkeypatch
):
    # The clock steps back: the new entry would be stamped before the snapshot
    # if the backend stamped from the clock alone.
    monkeypatch.setattr(embedded, "_now", lambda: T0 - timedelta(days=1))
    early = c.Entry(
        key=f"cc:{A}:{SESSION}:early:0",
        kind=c.KIND_PROMPT,
        ts=T0,
        identity=A,
        body={"text": "early"},
        session_id=SESSION,
    )
    snapshot = _node_ids(reader.timeline(c.All(), SESSION).data)
    pages = _pages(_timeline(reader), 3, between=lambda: backend.append([early]))
    assert not pages[0].newer
    assert all(p.newer for p in pages[1:])
    seen = [i for p in pages for i in _node_ids(p.data)]
    assert seen == snapshot
    assert len(seen) == len(set(seen))
    assert early.key not in seen
    reread = reader.timeline(c.All(), SESSION, limit=3)
    assert not reread.newer
    assert _node_ids(reread.data)[0] == early.key
    again = [i for p in _pages(_timeline(reader), 3, first=reread) for i in _node_ids(p.data)]
    assert again == [early.key, *snapshot]


def test_a_result_arriving_after_the_snapshot_is_not_joined_on_a_later_page(
    backend, reader, monkeypatch
):
    monkeypatch.setattr(embedded, "_now", lambda: T0 - timedelta(days=1))
    backend.append(
        [_cc(A, SESSION, 900, c.KIND_TOOL_CALL, 50, {"id": "tu-late"}, tool_use_id="tu-late")]
    )
    first = reader.timeline(c.All(), SESSION, limit=2)
    backend.append(
        [_cc(A, SESSION, 901, c.KIND_TOOL_RESULT, 51, {"content": "x"}, tool_use_id="tu-late")]
    )
    pages = _pages(_timeline(reader), 2, first=first)
    [late] = [n for p in pages for n in p.data if n.item and n.item.key.endswith(":u900:0")]
    assert late.result is None and pages[-1].newer
    [fresh] = [
        n
        for n in reader.timeline(c.All(), SESSION).data
        if n.item and n.item.key.endswith(":u900:0")
    ]
    assert fresh.result is not None


def test_concurrent_appenders_with_a_clock_stepping_back_never_skip_a_node(
    backend, config, monkeypatch
):
    """Two appenders write early-ts entries, the clock running backwards, while
    the timeline is paged; every pass yields exactly the entries arrived by its
    ``as_of``."""
    ticks = iter(range(10**9))
    lock = threading.Lock()

    def backwards():
        with lock:
            return T0 - timedelta(microseconds=next(ticks))

    monkeypatch.setattr(embedded, "_now", backwards)
    stop = threading.Event()

    def appender(name: str) -> None:
        for n in range(150):
            if stop.is_set():
                return
            backend.append(
                [
                    c.Entry(
                        key=f"cc:{A}:{SESSION}:{name}{n}:0",
                        kind=c.KIND_ANSWER,
                        ts=T0 - timedelta(seconds=n),
                        identity=A,
                        body={"text": f"{name} {n}"},
                        session_id=SESSION,
                    )
                ]
            )

    threads = [threading.Thread(target=appender, args=(name,)) for name in ("x", "y")]
    clock = _Clock()
    reader = rd.RecordReader(backend, config, clock=clock, sleep=clock.sleep, now=lambda: NOW)
    passes = []
    for t in threads:
        t.start()
    try:
        for _ in range(6):
            pages = _pages(_timeline(reader), 7)
            as_of = rd.TimelineCursor.decode(pages[0].next).as_of if pages[0].next else None
            passes.append((as_of, [k for p in pages for k in _flat(p.data)]))
    finally:
        stop.set()
        for t in threads:
            t.join()
    stored = backend.select(c.ScopeFilter(session_ids=frozenset({SESSION})), c.Query())
    stamps = [e.inserted_at for e in stored]
    assert len(set(stamps)) == len(stored) - len(_seed()) + 1  # appends never share a stamp
    for as_of, keys in passes:
        assert as_of is not None
        expected = {e.key for e in stored if e.inserted_at <= as_of}
        assert len(keys) == len(set(keys))
        assert set(keys) == expected


def test_appends_are_stamped_strictly_later_even_when_the_clock_steps_back(tmp_path, monkeypatch):
    times = iter([T0, T0 - timedelta(hours=1), T0 - timedelta(hours=2)])
    monkeypatch.setattr(embedded, "_now", lambda: next(times))
    with EmbeddedBackend(tmp_path / "r.duckdb") as backend:
        for n in range(3):
            backend.append(
                [c.Entry(key=f"sp:{A}:e{n}", kind=c.KIND_FEEDBACK, ts=T0, identity=A, body={})]
            )
        stamps = {e.key: e.inserted_at for e in backend.select(c.ScopeFilter(), c.Query())}
    assert stamps[f"sp:{A}:e0"] == T0
    assert stamps[f"sp:{A}:e1"] == T0 + timedelta(microseconds=1)
    assert stamps[f"sp:{A}:e2"] == T0 + timedelta(microseconds=2)


def test_a_reopened_record_stamps_after_what_it_holds(tmp_path, monkeypatch):
    path = tmp_path / "r.duckdb"
    with EmbeddedBackend(path) as backend:
        backend.append(
            [
                c.Entry(
                    key="sp:a:e0", kind=c.KIND_FEEDBACK, ts=T0, identity=A, body={}, inserted_at=T0
                )
            ]
        )
    monkeypatch.setattr(embedded, "_now", lambda: T0 - timedelta(days=1))
    with EmbeddedBackend(path) as backend:
        backend.append([c.Entry(key="sp:a:e1", kind=c.KIND_FEEDBACK, ts=T0, identity=A, body={})])
        stamps = {e.key: e.inserted_at for e in backend.select(c.ScopeFilter(), c.Query())}
    assert stamps["sp:a:e0"] == T0
    assert stamps["sp:a:e1"] == T0 + timedelta(microseconds=1)


# --- the cursor itself ------------------------------------------------------------


def test_a_cursor_round_trips():
    cursor = rd.TimelineCursor(T0, (T0, T0 + timedelta(microseconds=3), "cc:a:s:u1:0", 0))
    assert rd.TimelineCursor.decode(cursor.encode()) == cursor
    empty = rd.TimelineCursor(None)
    assert rd.TimelineCursor.decode(empty.encode()) == empty


def _b64(data) -> str:
    raw = json.dumps(data).encode() if not isinstance(data, bytes) else data
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.parametrize(
    "after",
    [
        "",
        "not a cursor",
        "%%%",
        _b64(b"\xff\xfe"),
        _b64(b"not json"),
        _b64([1, 2]),
        _b64({}),
        _b64({"as_of": None}),
        _b64({"as_of": "yesterday", "at": None}),
        _b64({"as_of": "2026-10-23T12:00:00", "at": None}),  # no zone
        _b64({"as_of": None, "at": [1, 2, 3, 4]}),
        _b64({"as_of": None, "at": ["2026-10-23T12:00:00+00:00"] * 2 + ["k", True]}),
        _b64({"as_of": None, "at": ["2026-10-23T12:00:00+00:00"] * 2 + ["k"]}),
        _b64({"as_of": None, "at": ["2026-10-23T12:00:00+00:00"] * 2 + [3, 0]}),
    ],
)
def test_a_malformed_cursor_is_a_value_error(reader, after):
    with pytest.raises(ValueError, match="not a timeline cursor"):
        reader.timeline(c.All(), SESSION, after=after)
    with pytest.raises(ValueError, match="not a timeline cursor"):
        reader.agent_timeline(c.All(), SESSION, "ag1", after=after)


def test_since_is_no_longer_a_timeline_argument(reader):
    with pytest.raises(TypeError):
        reader.timeline(c.All(), SESSION, since=T0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        reader.agent_timeline(c.All(), SESSION, "ag1", since=T0)  # type: ignore[call-arg]


def test_an_empty_session_has_no_next_and_reports_later_arrivals(backend, reader):
    first = reader.timeline(c.All(), "sess-empty", limit=0)
    assert first.data == [] and first.next is None and not first.newer
    stale = rd.TimelineCursor(None).encode()
    backend.append([_cc(A, "sess-empty", 1, c.KIND_PROMPT, 1, {"text": "hi"})])
    later = reader.timeline(c.All(), "sess-empty", after=stale)
    assert later.data == [] and later.newer and later.next is None


def test_a_page_json_carries_next_and_newer(reader):
    page = reader.timeline(c.All(), SESSION, limit=2).to_json()
    assert page["more"] is True and isinstance(page["next"], str) and page["newer"] is False
    last = reader.timeline(c.All(), SESSION).to_json()
    assert last["more"] is False and last["next"] is None
