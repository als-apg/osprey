"""Tests for the feedback composer: material, window, render and payload.

Four behaviours are pinned here:

* ``collect_material`` reads each source **exactly once** per request (the
  render tiers reuse the struct rather than re-reading), and degrades without
  raising when the session is dead or absent.
* ``select_artifacts`` builds the session's time window from *parsed*
  timestamps only and compares aware datetimes, never strings — transcripts
  stamp ``…Z`` while :class:`ArtifactEntry` stamps ``…+00:00``, and those two
  suffixes do not order together lexicographically.
* ``render_bundle`` renders the tiers in priority order into a UTF-8 **byte**
  budget, trimming each section oldest-first, and produces byte-identical
  output for identical input.
* ``assemble_payload`` is the one place the 60,000-byte clipboard cap is
  applied, it always leaves the context its 8,000-byte floor, and the bundle it
  embeds is byte-identical to what ``bundle_for_copy`` hands the standalone
  Copy button for the same text length.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock, call

import pytest

from osprey.interfaces.web_terminal.feedback_composer import (
    BUNDLE_TRIM_MARKER,
    CONTEXT_FLOOR_BYTES,
    CONTEXT_NO_SESSION,
    CONTEXT_OK,
    CONTEXT_TRANSCRIPT_NOT_FOUND,
    NO_WINDOW_ARTIFACTS_HEADER,
    PAYLOAD_CAP_BYTES,
    PAYLOAD_SEPARATOR,
    TAGGED_ARTIFACTS_HEADER,
    TEXT_TRUNCATION_MARKER,
    UNTAGGED_ARTIFACTS_HEADER,
    Material,
    assemble_payload,
    bundle_for_copy,
    clamp_text_len,
    collect_material,
    first_line,
    parse_timestamp,
    payload_bundle_budget,
    render_bundle,
    select_artifacts,
    validate_session_id,
)
from osprey.mcp_server.workspace.transcript_reader import TranscriptReader
from osprey.stores.artifact_store import ArtifactEntry, ArtifactStore

SESSION_ID = "8f14e45f-ceea-4d0a-b6f0-1c2d3e4f5a6b"
OTHER_SESSION_ID = "1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed"


def _entry(artifact_id: str, timestamp: str, session_id: str = "") -> ArtifactEntry:
    """Build a real :class:`ArtifactEntry` with only the fields under test set."""
    return ArtifactEntry(
        id=artifact_id,
        artifact_type="plot",
        title=artifact_id,
        description="",
        filename=f"{artifact_id}.png",
        mime_type="image/png",
        size_bytes=1,
        timestamp=timestamp,
        tool_source="test",
        session_id=session_id,
    )


def _reader(*, transcript_found: bool = True) -> Mock:
    """A ``TranscriptReader`` double with the three read methods stubbed."""
    reader = Mock(spec=TranscriptReader)
    reader.find_transcript_by_id.return_value = (
        Mock(name="transcript_path") if transcript_found else None
    )
    reader.read_session_by_id.return_value = [{"type": "tool_call", "timestamp": "t"}]
    reader.read_chat_history_by_id.return_value = [
        {"role": "user", "content": "hi", "timestamp": "t"}
    ]
    return reader


def _store(entries: list[ArtifactEntry] | None = None) -> Mock:
    store = Mock(spec=ArtifactStore)
    store.list_entries.return_value = entries if entries is not None else []
    return store


# --------------------------------------------------------------------------
# validate_session_id
# --------------------------------------------------------------------------


def test_validate_session_id_accepts_canonical_uuid():
    assert validate_session_id(SESSION_ID) == SESSION_ID


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "not-a-uuid",
        "../../etc/passwd",
        f"{SESSION_ID}/../../escape",
        f"{SESSION_ID}.jsonl",
        "8f14e45fceea4d0ab6f01c2d3e4f5a6b",  # hex, no hyphens
        f"urn:uuid:{SESSION_ID}",
        f"{{{SESSION_ID}}}",
        SESSION_ID.upper(),  # not the canonical lowercase filename
        None,
        12345,
    ],
)
def test_validate_session_id_rejects_non_uuid(candidate):
    with pytest.raises(ValueError):
        validate_session_id(candidate)


# --------------------------------------------------------------------------
# collect_material
# --------------------------------------------------------------------------


def test_collect_material_reads_each_source_exactly_once():
    reader = _reader()
    store = _store([_entry("a", "2026-08-20T10:00:00.000000+00:00")])

    material = collect_material(
        reader, store, SESSION_ID, scrollback="tail", metadata={"app_name": "OSPREY"}
    )

    assert reader.find_transcript_by_id.call_count == 1
    assert reader.read_session_by_id.call_count == 1
    assert reader.read_chat_history_by_id.call_count == 1
    assert store.list_entries.call_count == 1
    assert material.context_status == CONTEXT_OK
    assert material.session_id == SESSION_ID
    assert material.events == reader.read_session_by_id.return_value
    assert material.chat == reader.read_chat_history_by_id.return_value
    assert material.artifact_entries == store.list_entries.return_value
    assert material.scrollback == "tail"
    assert material.metadata == {"app_name": "OSPREY"}


def test_collect_material_does_not_use_store_session_filter():
    """Filtering is the composer's job — the store's OR-empty filter ignores the window."""
    reader = _reader()
    store = _store()

    collect_material(reader, store, SESSION_ID, scrollback="", metadata={})

    assert store.list_entries.call_args == call()


def test_collect_material_dead_session_marks_status_and_stops_reading():
    reader = _reader(transcript_found=False)
    store = _store()

    material = collect_material(
        reader, store, SESSION_ID, scrollback="tail", metadata={"app_name": "OSPREY"}
    )

    assert material.context_status == CONTEXT_TRANSCRIPT_NOT_FOUND
    assert reader.find_transcript_by_id.call_count == 1
    reader.read_session_by_id.assert_not_called()
    reader.read_chat_history_by_id.assert_not_called()
    store.list_entries.assert_not_called()
    assert material.events == []
    assert material.chat == []
    assert material.artifact_entries == []
    # The text and metadata the operator typed still survive.
    assert material.scrollback == "tail"
    assert material.metadata == {"app_name": "OSPREY"}
    assert material.session_id == SESSION_ID


def test_collect_material_without_session_is_metadata_only():
    reader = _reader()
    store = _store()

    material = collect_material(reader, store, None, scrollback="tail", metadata={"v": "1"})

    assert material.context_status == CONTEXT_NO_SESSION
    assert material.session_id is None
    reader.find_transcript_by_id.assert_not_called()
    reader.read_session_by_id.assert_not_called()
    reader.read_chat_history_by_id.assert_not_called()
    store.list_entries.assert_not_called()
    assert material.events == []
    assert material.chat == []
    assert material.artifact_entries == []
    assert material.metadata == {"v": "1"}
    assert material.scrollback == "tail"


def test_collect_material_copies_metadata():
    """The caller's dict must not alias the record that gets rendered."""
    reader = _reader()
    store = _store()
    metadata = {"app_name": "OSPREY"}

    material = collect_material(reader, store, SESSION_ID, scrollback="", metadata=metadata)
    metadata["app_name"] = "mutated"

    assert material.metadata == {"app_name": "OSPREY"}


def test_material_defaults_are_independent():
    assert Material().events == []
    Material().events.append({"a": 1})
    assert Material().events == []


# --------------------------------------------------------------------------
# parse_timestamp
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "2026-08-20T18:04:44.210Z",  # real Claude Code transcript stamp
        "2026-08-20T18:04:44.210000+00:00",  # ArtifactEntry stamp
    ],
)
def test_parse_timestamp_reads_both_suffixes_as_the_same_instant(raw):
    parsed = parse_timestamp(raw)
    assert parsed == datetime(2026, 8, 20, 18, 4, 44, 210000, tzinfo=UTC)


def test_parse_timestamp_assumes_utc_when_naive():
    assert parse_timestamp("2026-08-20T18:04:44") == datetime(2026, 8, 20, 18, 4, 44, tzinfo=UTC)


@pytest.mark.parametrize("raw", ["", None, "garbage", "2026-13-45T99:99:99Z", 17, []])
def test_parse_timestamp_returns_none_for_unusable(raw):
    assert parse_timestamp(raw) is None


# --------------------------------------------------------------------------
# select_artifacts
# --------------------------------------------------------------------------

WINDOW_EVENTS = ["2026-08-20T10:00:00.000Z", "2026-08-20T12:00:00.000Z"]


def test_select_artifacts_tags_this_session_and_windows_untagged():
    tagged_entry = _entry("tagged", "2026-08-19T00:00:00.000000+00:00", session_id=SESSION_ID)
    inside = _entry("inside", "2026-08-20T11:00:00.000000+00:00")
    outside_before = _entry("before", "2026-08-20T09:59:59.000000+00:00")
    outside_after = _entry("after", "2026-08-20T12:00:01.000000+00:00")
    other = _entry("other", "2026-08-20T11:00:00.000000+00:00", session_id=OTHER_SESSION_ID)

    tagged, windowed, window_ok = select_artifacts(
        [tagged_entry, inside, outside_before, outside_after, other],
        SESSION_ID,
        WINDOW_EVENTS,
    )

    assert window_ok is True
    assert [e.id for e in tagged] == ["tagged"]
    assert [e.id for e in windowed] == ["inside"]


def test_select_artifacts_excludes_other_session_tagged_entries_from_the_window():
    """An entry tagged for another session is never picked up by the window."""
    other = _entry("other", "2026-08-20T11:00:00.000000+00:00", session_id=OTHER_SESSION_ID)

    tagged, windowed, window_ok = select_artifacts([other], SESSION_ID, WINDOW_EVENTS)

    assert window_ok is True
    assert tagged == []
    assert windowed == []


def test_select_artifacts_compares_instants_not_strings():
    """``Z`` sorts after ``+`` lexicographically; the boundary entry proves parsing.

    ``'2026-08-20T10:00:00.000000+00:00' < '2026-08-20T10:00:00.000Z'`` as text,
    so a string comparison drops this entry even though it sits exactly on the
    window's lower edge.
    """
    boundary = _entry("boundary", "2026-08-20T10:00:00.000000+00:00")
    upper_edge = _entry("upper", "2026-08-20T12:00:00.000000+00:00")

    tagged, windowed, window_ok = select_artifacts(
        [boundary, upper_edge], SESSION_ID, WINDOW_EVENTS
    )

    assert window_ok is True
    assert tagged == []
    assert [e.id for e in windowed] == ["boundary", "upper"]


def test_select_artifacts_window_ignores_empty_and_malformed_event_stamps():
    """``read_session`` synthesizes ``timestamp: ""`` and sorts it first."""
    inside = _entry("inside", "2026-08-20T11:00:00.000000+00:00")
    outside = _entry("outside", "2026-08-20T13:00:00.000000+00:00")
    events = ["", None, "garbage", "2026-08-20T10:00:00.000Z", "2026-08-20T12:00:00.000Z", ""]

    tagged, windowed, window_ok = select_artifacts([inside, outside], SESSION_ID, events)

    assert window_ok is True
    assert tagged == []
    assert [e.id for e in windowed] == ["inside"]


def test_select_artifacts_reports_no_window_when_nothing_parses():
    tagged_entry = _entry("tagged", "2026-08-20T11:00:00.000000+00:00", session_id=SESSION_ID)
    untagged = _entry("untagged", "2026-08-20T11:00:00.000000+00:00")

    tagged, windowed, window_ok = select_artifacts(
        [tagged_entry, untagged], SESSION_ID, ["", None, "garbage"]
    )

    assert window_ok is False
    assert windowed == []
    assert [e.id for e in tagged] == ["tagged"]


def test_select_artifacts_reports_no_window_for_empty_event_list():
    _, windowed, window_ok = select_artifacts([], SESSION_ID, [])

    assert window_ok is False
    assert windowed == []


def test_select_artifacts_skips_untagged_entries_with_unusable_stamps():
    undated = _entry("undated", "")
    malformed = _entry("malformed", "not-a-timestamp")
    inside = _entry("inside", "2026-08-20T11:00:00.000000+00:00")

    _, windowed, window_ok = select_artifacts(
        [undated, malformed, inside], SESSION_ID, WINDOW_EVENTS
    )

    assert window_ok is True
    assert [e.id for e in windowed] == ["inside"]


def test_select_artifacts_without_session_id_tags_nothing():
    """An empty session id must not be read as "matches every untagged entry"."""
    untagged = _entry("untagged", "2026-08-20T11:00:00.000000+00:00")

    tagged, windowed, window_ok = select_artifacts([untagged], "", WINDOW_EVENTS)

    assert tagged == []
    assert window_ok is True
    assert [e.id for e in windowed] == ["untagged"]


def test_select_artifacts_single_event_makes_a_degenerate_window():
    exact = _entry("exact", "2026-08-20T10:00:00.000000+00:00")
    later = _entry("later", "2026-08-20T10:00:01.000000+00:00")

    _, windowed, window_ok = select_artifacts(
        [exact, later], SESSION_ID, ["2026-08-20T10:00:00.000Z"]
    )

    assert window_ok is True
    assert [e.id for e in windowed] == ["exact"]


# --------------------------------------------------------------------------
# render_bundle
# --------------------------------------------------------------------------

GENEROUS_BUDGET = 5_000_000


def _event(index: int, *, hour: int = 10, error: bool = False) -> dict:
    return {
        "type": "tool_call",
        "timestamp": f"2026-08-20T{hour:02d}:00:{index % 60:02d}.000Z",
        "tool": f"tool_{index}",
        "tool_name": f"tool_{index}",
        "full_tool_name": f"osprey__server__tool_{index}",
        "server": "server",
        "is_error": error,
        "result_summary": "connection refused\nsecond line" if error else "ok",
        "arguments": {},
    }


def _turn(index: int, role: str = "user", *, body: str = "") -> dict:
    return {
        "role": role,
        "content": body or f"turn {index} body",
        "timestamp": f"2026-08-20T11:00:{index % 60:02d}.000Z",
    }


def _material(**overrides) -> Material:
    """A fully populated, in-window :class:`Material` for render tests."""
    base = {
        "metadata": {"osprey_version": "1.2.3", "app_name": "OSPREY", "user": "operator"},
        "context_status": CONTEXT_OK,
        "events": [_event(0), _event(1), _event(2)],
        "chat": [_turn(0, "user"), _turn(1, "assistant"), _turn(2, "user")],
        "scrollback": "line one\nline two\nline three\n",
        "artifact_entries": [
            _entry("tag-1", "2026-08-20T10:00:30.000000+00:00", session_id=SESSION_ID),
            _entry("untagged-1", "2026-08-20T10:00:01.000000+00:00"),
        ],
        "session_id": SESSION_ID,
    }
    base.update(overrides)
    return Material(**base)


def _oversized_material() -> Material:
    """Material far larger than any budget under test (~6 MB)."""
    return Material(
        metadata={"osprey_version": "1.2.3", "app_name": "OSPREY"},
        context_status=CONTEXT_OK,
        events=[_event(i, error=i % 7 == 0) for i in range(4000)],
        chat=[_turn(i, "user" if i % 2 else "assistant", body="chat " * 300) for i in range(400)],
        scrollback="".join(f"scrollback line {i} {'x' * 80}\n" for i in range(60_000)),
        artifact_entries=[
            _entry(f"tag-{i}", f"2026-08-20T10:00:{i % 60:02d}.000000+00:00", session_id=SESSION_ID)
            for i in range(500)
        ]
        + [_entry(f"free-{i}", f"2026-08-20T10:00:{i % 60:02d}.000000+00:00") for i in range(500)],
        session_id=SESSION_ID,
    )


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


def test_render_bundle_emits_tiers_in_order():
    text, truncated = render_bundle(_material(), GENEROUS_BUDGET)

    positions = [
        text.index("## deployment"),
        text.index("## event log"),
        text.index("## chat history"),
        text.index("## terminal scrollback"),
        text.index(TAGGED_ARTIFACTS_HEADER),
        text.index(UNTAGGED_ARTIFACTS_HEADER),
    ]
    assert positions == sorted(positions)
    assert truncated is False


def test_render_bundle_renders_metadata_values():
    text, _ = render_bundle(_material(), GENEROUS_BUDGET)

    assert "- osprey_version: 1.2.3" in text
    assert "- app_name: OSPREY" in text
    assert "- user: operator" in text


def test_render_bundle_renders_events_with_error_detail():
    material = _material(events=[_event(0), _event(1, error=True)])

    text, _ = render_bundle(material, GENEROUS_BUDGET)

    assert "osprey__server__tool_0" in text
    assert "ERROR: connection refused" in text
    # Only the first line of a multi-line result summary is carried.
    assert "second line" not in text


def test_render_bundle_tolerates_undated_and_lifecycle_events():
    """``read_session`` synthesizes ``timestamp: ""`` for empty subagent files."""
    material = _material(
        events=[
            {"type": "agent_start", "timestamp": "", "agent_id": "abc", "agent_type": "explorer"},
            {"type": "agent_stop", "timestamp": "", "agent_id": "abc", "agent_type": "explorer"},
        ]
    )

    text, _ = render_bundle(material, GENEROUS_BUDGET)

    assert "agent_start" in text
    assert "explorer" in text


def test_render_bundle_renders_chat_newest_turn_first():
    material = _material(
        chat=[_turn(0, "user", body="oldest"), _turn(1, "assistant", body="newest")]
    )

    text, _ = render_bundle(material, GENEROUS_BUDGET)

    assert text.index("newest") < text.index("oldest")


def test_render_bundle_untagged_block_uses_the_exact_header():
    text, _ = render_bundle(_material(), GENEROUS_BUDGET)

    assert UNTAGGED_ARTIFACTS_HEADER == (
        "artifacts created on this deployment during this session's time window "
        "(not session-tagged; may include other terminal tabs or the chat panel)"
    )
    assert UNTAGGED_ARTIFACTS_HEADER in text
    assert NO_WINDOW_ARTIFACTS_HEADER not in text
    assert "untagged-1" in text


def test_render_bundle_labels_artifacts_when_no_window_is_derivable():
    material = _material(
        events=[{"type": "agent_start", "timestamp": "", "agent_id": "a", "agent_type": "x"}]
    )

    text, _ = render_bundle(material, GENEROUS_BUDGET)

    assert NO_WINDOW_ARTIFACTS_HEADER in text
    assert UNTAGGED_ARTIFACTS_HEADER not in text
    assert TAGGED_ARTIFACTS_HEADER not in text
    assert "tag-1" in text
    assert "untagged-1" not in text


def test_render_bundle_artifact_inventory_is_titles_and_ids_only():
    entry = ArtifactEntry(
        id="art-1",
        artifact_type="plot",
        title="Beam current",
        description="a description that must not travel",
        filename="secret-filename.png",
        mime_type="image/png",
        size_bytes=1,
        timestamp="2026-08-20T10:00:30.000000+00:00",
        tool_source="tool",
        session_id=SESSION_ID,
        metadata={"path": "/private/path"},
    )
    text, _ = render_bundle(_material(artifact_entries=[entry]), GENEROUS_BUDGET)

    assert "art-1" in text
    assert "Beam current" in text
    assert "a description that must not travel" not in text
    assert "secret-filename.png" not in text
    assert "/private/path" not in text


def test_render_bundle_omits_empty_sections():
    material = _material(events=[], chat=[], scrollback="", artifact_entries=[])

    text, truncated = render_bundle(material, GENEROUS_BUDGET)

    assert "## deployment" in text
    assert "## event log" not in text
    assert "## chat history" not in text
    assert "## terminal scrollback" not in text
    assert TAGGED_ARTIFACTS_HEADER not in text
    assert truncated is False


@pytest.mark.parametrize("status", [CONTEXT_TRANSCRIPT_NOT_FOUND, CONTEXT_NO_SESSION])
def test_render_bundle_replaces_context_tiers_with_a_status_note(status):
    material = _material(context_status=status)

    text, truncated = render_bundle(material, GENEROUS_BUDGET)

    assert "## deployment" in text
    assert "- app_name: OSPREY" in text
    assert "## session context" in text
    assert "## event log" not in text
    assert "## chat history" not in text
    assert "## terminal scrollback" not in text
    assert TAGGED_ARTIFACTS_HEADER not in text
    assert truncated is False
    note = text.split("## session context\n", 1)[1].strip()
    assert len(note.splitlines()) == 1


def test_render_bundle_status_note_names_the_dead_transcript():
    text, _ = render_bundle(_material(context_status=CONTEXT_TRANSCRIPT_NOT_FOUND), GENEROUS_BUDGET)

    assert "no transcript found" in text


@pytest.mark.parametrize("budget", [1_000, 60_000, 5_000_000])
def test_render_bundle_never_exceeds_the_byte_budget(budget):
    text, truncated = render_bundle(_oversized_material(), budget)

    assert _byte_len(text) <= budget
    assert truncated is True
    assert BUNDLE_TRIM_MARKER in text


def test_render_bundle_keeps_the_newest_events_and_marks_the_trim():
    material = _material(
        events=[_event(0), _event(1), _event(2), _event(3), _event(4)],
        chat=[],
        scrollback="",
        artifact_entries=[],
    )
    full, _ = render_bundle(material, GENEROUS_BUDGET)

    trimmed, truncated = render_bundle(material, _byte_len(full) - 60)

    assert truncated is True
    assert "tool_4" in trimmed
    assert "tool_0" not in trimmed
    # The marker sits where the older events were dropped.
    assert trimmed.index(BUNDLE_TRIM_MARKER) < trimmed.index("tool_4")


def test_render_bundle_keeps_the_newest_chat_turns_and_marks_the_trim():
    material = _material(
        events=[],
        chat=[_turn(i, body=f"turn-body-{i}") for i in range(6)],
        scrollback="",
        artifact_entries=[],
    )
    full, _ = render_bundle(material, GENEROUS_BUDGET)

    trimmed, truncated = render_bundle(material, _byte_len(full) - 60)

    assert truncated is True
    assert "turn-body-5" in trimmed
    assert "turn-body-0" not in trimmed
    # Chat renders newest-first, so the older end — and the marker — is the tail.
    assert trimmed.index("turn-body-5") < trimmed.index(BUNDLE_TRIM_MARKER)


def test_render_bundle_keeps_the_scrollback_tail():
    material = _material(
        events=[],
        chat=[],
        scrollback="".join(f"scrollback-{i}\n" for i in range(200)),
        artifact_entries=[],
    )

    trimmed, truncated = render_bundle(material, 900)

    assert truncated is True
    assert "scrollback-199" in trimmed
    assert "scrollback-0\n" not in trimmed
    assert BUNDLE_TRIM_MARKER in trimmed
    assert _byte_len(trimmed) <= 900


def test_render_bundle_later_tiers_yield_to_earlier_ones():
    """Tier order is a priority order: an oversized event log crowds out the rest."""
    material = _oversized_material()

    text, truncated = render_bundle(material, 4_000)

    assert truncated is True
    assert "## deployment" in text
    assert "## event log" in text
    assert TAGGED_ARTIFACTS_HEADER not in text


def test_render_bundle_is_byte_identical_for_identical_input():
    first, first_truncated = render_bundle(_material(), 4_096)
    second, second_truncated = render_bundle(_material(), 4_096)

    assert first.encode("utf-8") == second.encode("utf-8")
    assert first_truncated == second_truncated


def test_render_bundle_never_splits_a_multibyte_character():
    """The budget is UTF-8 bytes, so a naive character slice would overshoot it."""
    scrollback = "".join(f"мощность пучка {i} — стабильна ✅\n" for i in range(400))
    material = _material(events=[], chat=[], scrollback=scrollback, artifact_entries=[])
    assert _byte_len(scrollback) > len(scrollback)

    for budget in (200, 512, 1_337, 4_096):
        text, _ = render_bundle(material, budget)
        assert _byte_len(text) <= budget
        # Round-tripping proves no lone continuation byte survived the slice.
        assert text.encode("utf-8", "surrogatepass").decode("utf-8", "surrogatepass") == text


def test_render_bundle_handles_lone_surrogates_in_scrollback():
    """A lone surrogate reaches the composer through JSON and must not raise."""
    material = _material(events=[], chat=[], scrollback="before \ud800 after\n" * 50)

    text, _ = render_bundle(material, 300)

    assert _byte_len(text) <= 300


def test_render_bundle_returns_an_encodable_string_when_material_carries_surrogates():
    """The disk side of the surrogate problem: nothing hostile in the request.

    A browser that stringified an unpaired UTF-16 half writes ``\\ud800`` into
    the transcript, and it comes back here through ``read_chat_history_by_id``.
    Passing it on unrepaired moved the failure to the response renderer, which
    raises after the caller has already written its record.
    """
    material = _material(
        chat=[{"role": "user", "timestamp": "2026-08-20T10:00:10.000Z", "content": "oops \ud800"}],
        scrollback="tail \udfff\n",
    )

    text, _ = render_bundle(material, GENEROUS_BUDGET)

    text.encode("utf-8")  # must not raise
    assert "\ud800" not in text
    assert "\udfff" not in text
    assert "oops ?" in text


def test_render_bundle_substitution_never_exceeds_the_budget():
    # '?' is one byte where a surrogatepass-encoded half was three, so the
    # sanitized result can only shrink — the trimmed sections stay in budget.
    material = _material(events=[], chat=[], scrollback="before \ud800 after\n" * 50)

    for budget in (120, 300, 1_024):
        text, _ = render_bundle(material, budget)
        assert _byte_len(text) <= budget
        text.encode("utf-8")


def test_render_bundle_with_a_zero_budget_returns_nothing():
    text, truncated = render_bundle(_material(), 0)

    assert text == ""
    assert truncated is True


def test_render_bundle_with_nothing_to_say_is_not_truncated():
    text, truncated = render_bundle(Material(context_status=CONTEXT_OK), 0)

    assert text == ""
    assert truncated is False


# --------------------------------------------------------------------------
# assemble_payload / bundle_for_copy / clamp_text_len
#
# Every test here carries "payload" in its name: the gate selects them with
# ``-k payload``.
# --------------------------------------------------------------------------

SEPARATOR_BYTES = _byte_len(PAYLOAD_SEPARATOR)
MAX_TEXT_BYTES = PAYLOAD_CAP_BYTES - SEPARATOR_BYTES - CONTEXT_FLOOR_BYTES


def test_payload_cap_and_floor_are_the_proposal_numbers():
    assert PAYLOAD_CAP_BYTES == 60_000
    assert CONTEXT_FLOOR_BYTES == 8_000
    assert TEXT_TRUNCATION_MARKER == ("[text truncated — full text is in the Local record <id>]")


def test_payload_stays_under_the_cap_with_non_ascii_text():
    """The cap is bytes, not characters: this fixture's two counts differ."""
    text = "мощность пучка ✅ " * 6_000

    payload, bundle, truncated = assemble_payload(text, _oversized_material())

    assert len(text) != _byte_len(text)
    assert _byte_len(payload) <= PAYLOAD_CAP_BYTES
    assert truncated is True
    assert bundle in payload
    # A split multi-byte character would not survive the round trip.
    assert payload.encode("utf-8", "surrogatepass").decode("utf-8", "surrogatepass") == payload


@pytest.mark.parametrize(
    "text_bytes", [0, 1, 100, 8_000, 51_000, MAX_TEXT_BYTES - 1, MAX_TEXT_BYTES, 60_000, 200_000]
)
def test_payload_cap_holds_for_every_text_size(text_bytes):
    payload, _, _ = assemble_payload("x" * text_bytes, _oversized_material())

    assert _byte_len(payload) <= PAYLOAD_CAP_BYTES


def test_payload_truncates_a_100kb_text_and_keeps_the_context_floor():
    text = "x" * 100_000

    payload, bundle, truncated = assemble_payload(text, _oversized_material())

    assert truncated is True
    assert TEXT_TRUNCATION_MARKER in payload
    assert payload_bundle_budget(_byte_len(text)) == CONTEXT_FLOOR_BYTES
    assert _byte_len(bundle) <= CONTEXT_FLOOR_BYTES
    # An oversized session must actually fill the floor it was given.
    assert _byte_len(bundle) >= CONTEXT_FLOOR_BYTES - 200
    assert _byte_len(payload) <= PAYLOAD_CAP_BYTES


def test_payload_keeps_a_short_text_whole():
    payload, bundle, truncated = assemble_payload("the plot axes are swapped", _material())

    assert truncated is False
    assert payload.startswith("the plot axes are swapped")
    assert TEXT_TRUNCATION_MARKER not in payload
    assert payload.endswith(bundle)


def test_payload_puts_the_separator_between_the_text_and_the_context():
    payload, bundle, _ = assemble_payload("hello", _material())

    assert payload == "hello" + PAYLOAD_SEPARATOR + bundle


def test_payload_omits_the_separator_when_there_is_no_context():
    payload, bundle, truncated = assemble_payload("hello", Material())

    assert bundle == ""
    assert payload == "hello"
    assert truncated is False


def test_payload_keeps_the_id_placeholder_when_no_record_id_is_known():
    payload, _, _ = assemble_payload("x" * 100_000, _material())

    assert "[text truncated — full text is in the Local record <id>]" in payload


def test_payload_carries_the_record_id_when_one_is_given():
    payload, _, _ = assemble_payload(
        "x" * 100_000, _material(), record_id="fb-20260820T101500123-1a2b3c4d"
    )

    assert "full text is in the Local record fb-20260820T101500123-1a2b3c4d]" in payload
    assert "<id>" not in payload


def test_payload_cap_holds_even_with_an_absurdly_long_record_id():
    """The id must be composed IN, not substituted after: it costs bytes."""
    payload, _, truncated = assemble_payload(
        "x" * 100_000, _oversized_material(), record_id="fb-" + "9" * 500
    )

    assert truncated is True
    assert _byte_len(payload) <= PAYLOAD_CAP_BYTES


def test_payload_never_splits_a_multibyte_character_at_the_text_cut():
    payload, _, truncated = assemble_payload("✅" * 30_000, _material())

    assert truncated is True
    assert _byte_len(payload) <= PAYLOAD_CAP_BYTES
    assert payload.encode("utf-8").decode("utf-8") == payload
    assert "�" not in payload


def test_payload_tolerates_a_lone_surrogate_in_the_text():
    payload, _, _ = assemble_payload("bad \ud800 text " * 5_000, _material())

    assert _byte_len(payload) <= PAYLOAD_CAP_BYTES


def test_payload_is_byte_identical_for_identical_input():
    material = _oversized_material()

    first = assemble_payload("отчёт", material)
    second = assemble_payload("отчёт", material)

    assert first[0].encode("utf-8") == second[0].encode("utf-8")
    assert first[1:] == second[1:]


def test_payload_honours_a_smaller_cap_and_floor():
    payload, bundle, truncated = assemble_payload(
        "x" * 5_000, _oversized_material(), cap=2_000, floor=500
    )

    assert _byte_len(payload) <= 2_000
    assert truncated is True
    assert _byte_len(bundle) <= 500


def test_payload_with_a_dead_session_still_carries_text_and_metadata():
    material = _material(context_status=CONTEXT_TRANSCRIPT_NOT_FOUND)

    payload, bundle, truncated = assemble_payload("the plot is wrong", material)

    assert truncated is False
    assert "the plot is wrong" in payload
    assert "- app_name: OSPREY" in payload
    assert "no transcript found" in bundle


@pytest.mark.parametrize(
    "text",
    ["", "short", "x" * 40_000, "x" * 100_000, "мощность" * 5_000, "✅" * 20_000],
)
def test_payload_copy_bundle_is_byte_identical_to_the_payload_bundle(text):
    """The pinned criterion: the standalone Copy button and the send payload agree."""
    material = _oversized_material()

    _, bundle, _ = assemble_payload(text, material)
    copied = bundle_for_copy(material, _byte_len(text))

    assert copied.encode("utf-8", "surrogatepass") == bundle.encode("utf-8", "surrogatepass")


def test_payload_copy_bundle_ignores_a_text_len_above_the_text_allowance():
    material = _oversized_material()

    assert bundle_for_copy(material, PAYLOAD_CAP_BYTES) == bundle_for_copy(material, MAX_TEXT_BYTES)
    assert bundle_for_copy(material, 10**9) == bundle_for_copy(material, MAX_TEXT_BYTES)


@pytest.mark.parametrize("value", [None, -1, -10_000, "", "garbage", [], {}, "0", 0])
def test_payload_copy_bundle_clamps_an_unusable_text_len(value):
    material = _material()

    assert bundle_for_copy(material, value) == bundle_for_copy(material, 0)


def test_payload_copy_bundle_shrinks_as_the_typed_text_grows():
    material = _oversized_material()

    assert _byte_len(bundle_for_copy(material, 0)) > _byte_len(
        bundle_for_copy(material, MAX_TEXT_BYTES)
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        (0, 0),
        (-5, 0),
        (-(10**9), 0),
        (1234, 1234),
        ("1234", 1234),
        (3.9, 3),
        (10**9, 60_000),
        (60_001, 60_000),
        ("garbage", 0),
        ("12.5", 0),
        ("", 0),
        ([], 0),
        ({}, 0),
        (object(), 0),
    ],
)
def test_payload_text_len_clamp(value, expected):
    assert clamp_text_len(value) == expected


def test_payload_bundle_budget_reserves_the_floor_and_never_goes_negative():
    assert payload_bundle_budget(0) == PAYLOAD_CAP_BYTES - SEPARATOR_BYTES
    assert payload_bundle_budget(1_000) == PAYLOAD_CAP_BYTES - SEPARATOR_BYTES - 1_000
    assert payload_bundle_budget(MAX_TEXT_BYTES) == CONTEXT_FLOOR_BYTES
    assert payload_bundle_budget(10**9) == CONTEXT_FLOOR_BYTES
    assert payload_bundle_budget(-5) == PAYLOAD_CAP_BYTES - SEPARATOR_BYTES
    assert payload_bundle_budget(0, cap=10, floor=8_000, separator=PAYLOAD_SEPARATOR) >= 0


# ---- first_line: the shared excerpt/summary cutter ----


def test_first_line_keeps_the_first_line_and_caps_it():
    assert first_line("  hello there\nsecond line\n") == "hello there"
    assert first_line("x" * 500) == "x" * 200
    assert first_line("x" * 500, limit=10) == "x" * 10
    assert first_line("   \n  ") == ""


def test_first_line_strips_control_characters_before_capping():
    # The excerpt this produces is printed straight into a maintainer's terminal
    # by `osprey feedback list`, and the text it is cut from was typed in a
    # browser: an ESC sequence there would clear the triager's screen.
    cleaned = first_line("\x1b[2Jwiped\x07\x00 the\tscreen")
    assert "\x1b" not in cleaned
    assert "\x07" not in cleaned
    assert "\x00" not in cleaned
    assert cleaned == "[2Jwiped the\tscreen"


def test_first_line_spends_the_cap_on_content_not_escapes():
    assert first_line("\x1b[2J" + "y" * 200, limit=200) == "[2J" + "y" * 197
