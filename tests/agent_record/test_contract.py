"""The record contract: kinds and the allow-list, keys, entry validation,
chunking and reassembly, scrubbing, scopes, the backend protocol, ingest state
names, the spool line and segment names, and wait marks.

Also pins that the contract stays a stdlib-only leaf that does no I/O.
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey.agent_record import contract as c

TS = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def _entry(**overrides):
    fields = {"key": "sp:alice:e1", "kind": c.KIND_AUDIT, "ts": TS, "identity": "alice", "body": {}}
    fields.update(overrides)
    return c.Entry(**fields)


class FakeBackend:
    """An in-memory backend: first write wins on ``(key, chunk.index)``."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, int], c.Entry] = {}
        self.state: dict[str, dict] = {}

    def append(self, entries):
        appended = duplicates = 0
        for entry in entries:
            if entry.unique_key in self.rows:
                duplicates += 1
            else:
                self.rows[entry.unique_key] = entry
                appended += 1
        return c.AppendResult(appended, duplicates)

    def select(self, scope_filter, query):
        out = list(self.rows.values())
        if scope_filter.identities:
            out = [e for e in out if e.identity in scope_filter.identities]
        if query.keys:
            out = [e for e in out if e.key in query.keys]
        return out

    def delete_arrived_before(self, day):  # noqa: ARG002 - protocol signature
        return 0

    def export_parquet(self, arrival_day, dest):  # noqa: ARG002 - protocol signature
        return 0

    def get_state(self, name):
        return self.state.get(name)

    def put_state(self, name, value):
        self.state[name] = dict(value)

    def count_state(self, prefix, since=None):
        floor = None if since is None else c.state_time(since)
        return sum(
            1
            for name, value in self.state.items()
            if name.startswith(prefix)
            and (floor is None or value.get(c.STATE_TIME_FIELD, "") >= floor)
        )

    def health(self):
        return c.BackendHealth(ok=True, backend="fake")


# --- the module itself -----------------------------------------------------


def test_contract_imports_only_the_standard_library():
    tree = ast.parse(Path(c.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    non_stdlib = {r for r in roots if r not in sys.stdlib_module_names and r != "__future__"}
    assert non_stdlib == set()
    assert not {"os", "io", "shutil", "subprocess", "socket"} & roots


def test_package_init_re_exports_nothing():
    import osprey.agent_record as pkg

    tree = ast.parse(Path(pkg.__file__).read_text())
    assert all(isinstance(node, ast.Expr) for node in tree.body)


# --- kinds and the allow-list ----------------------------------------------


def test_kind_registry_is_the_fifteen_kinds_in_order():
    assert c.KINDS == (
        "prompt", "answer", "thinking", "tool_call", "tool_result", "subagent", "usage",
        "error", "decision", "audit", "run", "bridge_turn", "feedback", "debug", "api_call",
    )  # fmt: skip


def test_every_route_kind_is_registered_and_every_kind_has_a_route():
    routed = set().union(*c.ROUTE_KINDS.values())
    assert routed <= set(c.KINDS)
    assert routed == set(c.KINDS)


def test_run_only_from_dispatch_and_bridge_turn_only_from_bridges():
    spool = c.ROUTE_SPOOL
    assert c.kind_allowed("run", route=spool, source_class=c.SOURCE_DISPATCH)
    for cls in c.SOURCE_CLASSES - {c.SOURCE_DISPATCH}:
        assert not c.kind_allowed("run", route=spool, source_class=cls)
    assert c.kind_allowed("bridge_turn", route=spool, source_class=c.SOURCE_BRIDGE)
    for cls in c.SOURCE_CLASSES - {c.SOURCE_BRIDGE}:
        assert not c.kind_allowed("bridge_turn", route=spool, source_class=cls)
    assert c.kind_allowed("audit", route=spool, source_class=c.SOURCE_SIDECAR)


def test_a_kind_outside_its_route_is_refused():
    assert not c.kind_allowed("prompt", route=c.ROUTE_SPOOL, source_class=c.SOURCE_TERMINAL)
    assert not c.kind_allowed("audit", route=c.ROUTE_OTLP, source_class=c.SOURCE_TERMINAL)
    assert c.kind_allowed("decision", route=c.ROUTE_OTLP, source_class=c.SOURCE_TERMINAL)
    assert not c.kind_allowed("nonsense", route=c.ROUTE_SPOOL, source_class=c.SOURCE_TERMINAL)


def test_kind_allowed_refuses_unknown_route_or_class():
    with pytest.raises(ValueError, match="route"):
        c.kind_allowed("audit", route="ftp", source_class=c.SOURCE_TERMINAL)
    with pytest.raises(ValueError, match="source class"):
        c.kind_allowed("audit", route=c.ROUTE_SPOOL, source_class="martian")


def test_check_kind_names_the_registry():
    with pytest.raises(ValueError, match="prompt, answer"):
        c.check_kind("chatter")


# --- keys --------------------------------------------------------------------


def test_transcript_key_with_uuid():
    assert c.transcript_key("alice", "s1", "u-1", 2) == "cc:alice:s1:u-1:2"


def test_transcript_key_without_uuid_hashes_the_line():
    k1 = c.transcript_key("alice", "s1", None, 0, line=b'{"type":"summary"}')
    k2 = c.transcript_key("alice", "s1", None, 0, line=b'{"type":"summary"}')
    k3 = c.transcript_key("alice", "s1", None, 0, line=b'{"type":"other"}')
    assert k1 == k2 != k3
    assert k1.startswith("cc:alice:s1:")
    with pytest.raises(ValueError):
        c.transcript_key("alice", "s1", None, 0)


def test_otel_key_uses_tool_use_id_else_attributes_without_sequence():
    assert (
        c.otel_key("alice", "s1", "tool_decision", tool_use_id="toolu_1")
        == "otel:alice:s1:tool_decision:toolu_1"
    )
    a = c.otel_key("alice", "s1", "api_request", attributes={"x": 1, "event.sequence": 4, "y": 2})
    b = c.otel_key("alice", "s1", "api_request", attributes={"y": 2, "event.sequence": 9, "x": 1})
    other = c.otel_key("alice", "s1", "api_request", attributes={"x": 1, "y": 3})
    assert a == b != other


def test_spool_and_run_keys():
    assert c.spool_key("bob", "e-1") == "sp:bob:e-1"
    assert c.run_key("r1", "accept") == "run:r1:accept"
    assert c.run_key("r1", "terminal") == "run:r1:terminal"
    with pytest.raises(ValueError):
        c.run_key("r1", "midway")


@pytest.mark.parametrize("identity", ["", "a:b", "a/b", " a", "a b", "..", "a$b"])
def test_keys_refuse_an_unusable_identity(identity):
    with pytest.raises(ValueError):
        c.spool_key(identity, "e1")


def test_keys_always_start_with_the_identity():
    assert c.transcript_key("x", "s", "u", 0).split(":")[1] == "x"
    assert c.otel_key("x", "s", "e", tool_use_id="t").split(":")[1] == "x"
    assert c.spool_key("x", "e").split(":")[1] == "x"


# --- entries -----------------------------------------------------------------


def test_entry_identity_is_key_and_chunk_index():
    entry = _entry(chunk=c.Chunk(1, 3))
    assert entry.unique_key == ("sp:alice:e1", 1)
    assert _entry().chunk == c.Chunk(0, 1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("key", "", "no key"),
        ("kind", "", "no kind"),
        ("kind", "gossip", "unknown kind"),
        ("ts", None, "no ts"),
        ("ts", datetime(2026, 1, 1), "timezone"),
        ("identity", "", "no identity"),
        ("identity", "a:b", "source identity"),
        ("body", [1], "JSON object"),
        ("session_id", None, "string"),
    ],
)
def test_entry_validation_rejects(field, value, message):
    with pytest.raises(c.EntryInvalid, match=message):
        _entry(**{field: value})


def test_chunk_outside_its_set_is_invalid():
    with pytest.raises(c.EntryInvalid):
        c.Chunk(3, 3)
    with pytest.raises(c.EntryInvalid):
        c.Chunk(0, 0)


def test_parse_ts():
    assert c.parse_ts("2026-10-23T12:00:00Z") == TS
    assert c.parse_ts(TS) is TS
    with pytest.raises(c.EntryInvalid):
        c.parse_ts("2026-10-23T12:00:00")
    with pytest.raises(c.EntryInvalid):
        c.parse_ts(12)


# --- chunking ------------------------------------------------------------------


def test_small_body_is_returned_unsplit():
    entry = _entry(body={"text": "hi"})
    assert c.split_entry(entry, 1024) == [entry]


@pytest.mark.parametrize(
    "body",
    [
        {"text": "x" * 5000},
        {"text": "ünïcödé ✓ 漢字 " * 300, "n": 1, "z": [1, 2, {"q": None}]},
        {"text": 'quotes " and \\ backslashes\n\t' * 200},
        {"b": 2, "a": 1, "image": "A" * 3000},
    ],
)
def test_chunks_reassemble_byte_identical(body):
    entry = _entry(body=body)
    limit = 256
    chunks = c.split_entry(entry, limit)
    assert len(chunks) > 1
    assert [ch.chunk for ch in chunks] == [c.Chunk(i, len(chunks)) for i in range(len(chunks))]
    for ch in chunks:
        assert ch.key == entry.key
        assert len(c.serialize_body(ch.body).encode()) <= limit
    out = c.reassemble(chunks)
    assert not out.incomplete and out.ignored == 0
    assert c.serialize_body(out.body) == c.serialize_body(body)
    assert list(out.body) == list(body)


def test_split_refuses_a_limit_too_small_and_an_already_split_entry():
    with pytest.raises(ValueError):
        c.split_entry(_entry(body={"t": "x" * 100}), 5)
    with pytest.raises(ValueError):
        c.split_entry(_entry(chunk=c.Chunk(0, 2)), 100)


def test_reassemble_missing_chunk_is_incomplete():
    chunks = c.split_entry(_entry(body={"t": "x" * 2000}), 200)
    out = c.reassemble(chunks[:1] + chunks[2:])
    assert out.incomplete and out.body is None


def test_reassemble_without_chunk_zero_is_incomplete():
    chunks = c.split_entry(_entry(body={"t": "x" * 2000}), 200)
    assert c.reassemble(chunks[1:]).incomplete


def test_first_stored_chunk_zero_defines_the_set_and_extra_indexes_are_counted():
    first = c.split_entry(_entry(body={"t": "a" * 600}), 200)
    second = c.split_entry(_entry(body={"t": "b" * 3000}), 200)
    assert len(second) > len(first)
    stored = list(first) + [ch for ch in second if ch.chunk.index >= len(first)]
    out = c.reassemble(stored)
    assert not out.incomplete
    assert out.body == {"t": "a" * 600}
    assert out.ignored == len(second) - len(first)


def test_reassemble_unsplit_returns_the_body():
    entry = _entry(body={"k": 1})
    out = c.reassemble([entry])
    assert out.body == {"k": 1} and not out.incomplete


def test_reassemble_refuses_mixed_keys_or_nothing():
    with pytest.raises(ValueError):
        c.reassemble([])
    with pytest.raises(ValueError):
        c.reassemble([_entry(), _entry(key="sp:alice:e2")])


# --- scrub ---------------------------------------------------------------------


def test_scrub_masks_every_string_leaf_longest_first():
    secret = "abcdefgh12345678"  # gitleaks:allow - scrub fixture, not a secret
    inner = "abcdefgh"  # a prefix of the longer secret, also a secret itself
    body = {
        "text": f"token={secret}",
        "nested": {"list": ["x", f"pre-{inner}-post", 3, None], secret: "key kept"},
        "n": 12,
    }
    out = c.scrub(body, {inner, secret})
    assert out["text"] == "token=***"
    assert out["nested"]["list"] == ["x", "pre-***-post", 3, None]
    assert secret in out["nested"]  # mapping keys are not leaves
    assert body["text"] == f"token={secret}"  # the input is not changed


def test_scrub_ignores_short_secrets():
    assert c.scrub({"t": "1234567 here"}, {"1234567"}) == {"t": "1234567 here"}
    assert c.scrub({"t": "12345678 here"}, {"12345678"}) == {"t": "*** here"}


def test_scrub_with_no_secrets_returns_body():
    body = {"t": "x"}
    assert c.scrub(body, []) is body


# --- scope ---------------------------------------------------------------------


def test_scopes_validate():
    assert c.Reader("ops").kinds == frozenset(c.KINDS)
    assert c.Reader("ops", frozenset({"audit"}), 7).days_back == 7
    with pytest.raises(ValueError):
        c.Reader("ops", frozenset({"gossip"}))
    with pytest.raises(ValueError):
        c.Reader("ops", days_back=0)
    with pytest.raises(ValueError):
        c.Own("")
    with pytest.raises(ValueError):
        c.Source("a:b")
    with pytest.raises(ValueError):
        c.Narrow()
    with pytest.raises(ValueError):
        c.Narrow(session_id="s", run_ids=frozenset({"r"}))
    assert c.Source("teams-bridge", c.Narrow(run_ids={"r1"})).narrow.run_ids == {"r1"}
    assert c.All() == c.All()


# --- backend protocol -----------------------------------------------------------


def test_fake_backend_satisfies_the_protocol_and_first_write_wins():
    backend = FakeBackend()
    assert isinstance(backend, c.RecordBackend)
    assert isinstance(backend, c.StateStore)
    first = _entry(body={"v": 1})
    result = backend.append([first, _entry(body={"v": 2}), _entry(chunk=c.Chunk(1, 2))])
    assert result == c.AppendResult(appended=2, duplicates=1)
    assert backend.rows[("sp:alice:e1", 0)].body == {"v": 1}


def test_an_object_missing_a_method_is_not_a_backend():
    class Half:
        def append(self, entries): ...

    assert not isinstance(Half(), c.RecordBackend)


def test_backend_errors_are_one_family():
    assert issubclass(c.RecordBackendUnavailable, c.RecordBackendError)


# --- state names ------------------------------------------------------------------


def test_state_names():
    assert (
        c.transcript_state_name("alice", "-data-ws/s1.jsonl")
        == "transcript:alice:-data-ws/s1.jsonl"
    )
    assert (
        c.transcript_state_name("alice", "-ws/s1/subagents/agent-a.jsonl")
        == "transcript:alice:-ws/s1/subagents/agent-a.jsonl"
    )
    assert c.spool_state_name("bob", "h-1-2.open", 77) == "spool:bob:h-1-2.open:77"
    assert c.refused_state_name("bob", "e1") == "refused:bob:e1"


@pytest.mark.parametrize("relpath", ["/abs/s.jsonl", "../s.jsonl", "a/../b", "a//b", "", "a\\b"])
def test_transcript_state_name_refuses_paths_outside_projects(relpath):
    with pytest.raises(ValueError):
        c.transcript_state_name("alice", relpath)


def test_spool_state_name_refuses_a_non_segment():
    with pytest.raises(ValueError):
        c.spool_state_name("bob", "notes.txt", 1)


def test_state_values_round_trip():
    offset = c.TranscriptOffset(2, 1024)
    assert offset.to_state() == {"generation": 2, "offset": 1024}
    assert c.TranscriptOffset.from_state(offset.to_state()) == offset
    full = c.TranscriptOffset(3, 10, fully_read=True)
    assert c.TranscriptOffset.from_state(full.to_state()) == full
    assert c.SpoolCheckpoint.from_state({"offset": 5}) == c.SpoolCheckpoint(5)
    with pytest.raises(ValueError):
        c.TranscriptOffset.from_state({"generation": 1})
    with pytest.raises(ValueError):
        c.SpoolCheckpoint(-1)
    at = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    assert c.refused_state_value("run", "not allowed", at) == {
        "kind": "run",
        "reason": "not allowed",
        "at": "2026-10-23T12:00:00.000000Z",
    }
    with pytest.raises(ValueError):
        c.state_time(datetime(2026, 10, 23, 12, 0))


# --- spool segments ----------------------------------------------------------------


def test_segment_names_round_trip_with_dashes_in_host():
    name = c.segment_name("web-alice-7f", 42, 1700000000000000000)
    assert name == "web-alice-7f-42-1700000000000000000.open"
    parsed = c.parse_segment_name(name)
    assert parsed == c.SegmentName("web-alice-7f", 42, 1700000000000000000)
    sealed = c.parse_segment_name("web-alice-7f-42-1700000000000000000.sealed")
    assert sealed == c.SegmentName("web-alice-7f", 42, 1700000000000000000, sealed=True)


@pytest.mark.parametrize("name", ["x.jsonl", "h-1.open", "h-a-1.open", "a/h-1-2.open", ".errors"])
def test_non_segments_do_not_parse(name):
    assert c.parse_segment_name(name) is None


def test_segment_protocol_constants():
    assert (c.OPEN_SUFFIX, c.SEALED_SUFFIX) == (".open", ".sealed")
    assert c.SEGMENT_MAX_AGE_S == 60 and c.SEGMENT_IDLE_S == 60
    assert c.ABANDONED_OPEN_S == 3600
    assert c.OTLP_SPOOL_DIR == ".otlp"
    assert c.SPOOL_DIR_ENV == "OSPREY_SPOOL_DIR"


# --- spool lines -------------------------------------------------------------------


def _line(**overrides):
    data = {"entry_id": "e1", "kind": "audit", "ts": "2026-10-23T12:00:00Z", "body": {"a": 1}}
    data.update(overrides)
    return json.dumps(data)


def test_spool_line_round_trips_and_ignores_identity():
    line = c.parse_spool_line(_line(identity="mallory", session_id="s1", on_behalf_of="carol"))
    assert not hasattr(line, "identity")
    again = c.parse_spool_line(line.to_json())
    assert again == line
    assert "identity" not in json.loads(line.to_json())


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[1]",
        json.dumps({"kind": "audit", "ts": "2026-10-23T12:00:00Z", "body": {}}),
        _line(entry_id="has space"),
        _line(kind="gossip"),
        _line(ts="yesterday"),
        _line(body="text"),
        _line(session_id=5),
    ],
)
def test_bad_spool_lines_are_refused(raw):
    with pytest.raises(c.SpoolLineInvalid):
        c.parse_spool_line(raw)


def test_spool_entry_key_honours_line_key_only_when_receiver_owned():
    line = c.parse_spool_line(_line(key="otel:alice:s1:e:x"))
    assert c.spool_entry_key(line, "alice", receiver_owned=False) == "sp:alice:e1"
    assert c.spool_entry_key(line, "alice", receiver_owned=True) == "otel:alice:s1:e:x"
    with pytest.raises(c.SpoolLineInvalid):
        c.spool_entry_key(c.parse_spool_line(_line()), "alice", receiver_owned=True)


def test_run_lines_are_keyed_by_run_and_phase():
    accept = c.parse_spool_line(_line(kind="run", run_id="r1", body={"phase": "accept"}))
    assert c.spool_entry_key(accept, "dispatch-worker-1", receiver_owned=False) == "run:r1:accept"
    for bad in (_line(kind="run", body={"phase": "accept"}), _line(kind="run", run_id="r1")):
        with pytest.raises(c.SpoolLineInvalid):
            c.spool_entry_key(c.parse_spool_line(bad), "dispatch-worker-1", receiver_owned=False)


def test_entry_from_spool_line_drops_untrusted_on_behalf_of_and_bad_tokens():
    line = c.parse_spool_line(_line(on_behalf_of="carol", session_id="s1", run_id="bad id"))
    kept = c.entry_from_spool_line(
        line, identity="sidecar", receiver_owned=False, on_behalf_of_trusted=True
    )
    dropped = c.entry_from_spool_line(
        line, identity="alice", receiver_owned=False, on_behalf_of_trusted=False
    )
    assert kept.on_behalf_of == "carol" and dropped.on_behalf_of == ""
    assert kept.identity == "sidecar" and kept.key == "sp:sidecar:e1"
    assert kept.session_id == "s1" and kept.run_id == ""
    assert kept.ts == TS and kept.inserted_at is None


def test_write_result_is_entry_id_or_none():
    assert c.WriteResult == (str | None)


# --- wait marks ----------------------------------------------------------------------


def test_wait_mark_json_round_trip():
    drain = c.WaitMark.from_json({"drain": {"session_id": "s1"}})
    assert drain.drain == c.Drain("s1") and drain.transcript is None
    assert c.WaitMark.from_json(json.dumps(drain.to_json())) == drain
    anywhere = c.WaitMark.from_json({"drain": {"session_id": None}})
    assert anywhere.drain == c.Drain() and anywhere.to_json() == {"drain": {"session_id": None}}
    assert c.WaitMark.from_json({"drain": {}}) == anywhere
    data = {"transcript": {"session_id": "s1", "files": {"-ws/s1.jsonl": [0, 100]}}}
    transcript = c.WaitMark.from_json(data)
    assert transcript.transcript.files == {"-ws/s1.jsonl": (0, 100)}
    assert transcript.to_json() == data


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"drain": {}, "transcript": {}},
        {"spool": [["alice", "e1"]]},  # the removed form
        {"drain": None},
        {"drain": "s1"},
        {"drain": {"session_id": 7}},
        {"drain": {"session_id": "a b"}},
        {"drain": {"session_id": "s1", "identity": "bob"}},  # the caller never names one
        {"transcript": {"session_id": "s1", "files": {"/abs.jsonl": [0, 1]}}},
        {"transcript": {"session_id": "s1", "files": {"a.jsonl": [0]}}},
        {"transcript": {"files": {"a.jsonl": [0, 1]}}},
        {"other": 1},
        "not json",
    ],
)
def test_bad_wait_marks_are_refused(data):
    with pytest.raises(c.WaitMarkInvalid):
        c.WaitMark.from_json(data)


def test_a_wait_mark_names_exactly_one_form():
    with pytest.raises(c.WaitMarkInvalid):
        c.WaitMark()
    with pytest.raises(c.WaitMarkInvalid):
        c.WaitMark(drain=c.Drain(), transcript=c.TranscriptMark("s1", {}))


def test_a_drain_mark_is_never_checked_against_state():
    backend = FakeBackend()
    with pytest.raises(ValueError):
        c.WaitMark(drain=c.Drain("s1")).passed(backend, transcript_identity="alice")


def test_transcript_mark_passes_on_offset_or_a_fully_read_newer_generation():
    backend = FakeBackend()
    path = "-ws/s1.jsonl"
    mark = c.WaitMark(transcript=c.TranscriptMark("s1", {path: (1, 100)}))
    name = c.transcript_state_name("alice", path)
    assert not mark.passed(backend, transcript_identity="alice")
    backend.put_state(name, c.TranscriptOffset(1, 99).to_state())
    assert not mark.passed(backend, transcript_identity="alice")
    backend.put_state(name, c.TranscriptOffset(1, 100).to_state())
    assert mark.passed(backend, transcript_identity="alice")
    assert not mark.passed(backend, transcript_identity="bob")
    backend.put_state(name, c.TranscriptOffset(2, 10).to_state())
    assert not mark.passed(backend, transcript_identity="alice")
    backend.put_state(name, c.TranscriptOffset(2, 10, fully_read=True).to_state())
    assert mark.passed(backend, transcript_identity="alice")
    backend.put_state(name, c.TranscriptOffset(0, 5000, fully_read=True).to_state())
    assert not mark.passed(backend, transcript_identity="alice")


def test_backend_health_and_append_result_defaults():
    assert c.AppendResult() == c.AppendResult(0, 0)
    health = c.BackendHealth(ok=False, backend="embedded", detail="locked")
    assert health.entries is None and health.oldest_arrival_day is None
