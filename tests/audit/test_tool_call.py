"""The ``tool_call`` record: what a call ran against, and nothing it said."""

from __future__ import annotations

import json

import pytest

from osprey.audit import tool_call, writer


def test_the_surface_name() -> None:
    assert tool_call.SURFACE_TOOL_CALL == "tool_call"


def test_the_retired_surface_machinery_is_gone() -> None:
    """No setting, no payload bound, no artifact spill: nothing reads ``audit.tool_call.*``."""
    for name in (
        "settings",
        "capped",
        "serialize_result",
        "ENABLED_KEY",
        "MAX_INLINE_KEY",
        "DEFAULT_MAX_INLINE_BYTES",
        "ARTIFACT_ORIGIN",
        "IMAGE_SPILL_BYTES",
    ):
        assert not hasattr(tool_call, name), name


def test_append_record_is_one_entry(recorded_entries) -> None:
    body = _record(ts="2026-10-25T12:00:00+00:00")

    entry_id = writer.append_record(tool_call.SURFACE_TOOL_CALL, body)

    assert isinstance(entry_id, str) and entry_id
    (entry,) = recorded_entries("audit")
    assert entry.identity == "alice"
    assert entry.tool_use_id == "u"
    assert entry.body["surface"] == tool_call.SURFACE_TOOL_CALL
    assert dict(entry.body) == body


def test_append_record_never_raises(monkeypatch: pytest.MonkeyPatch, recorded_entries) -> None:
    def broken(*_args, **_kwargs):
        raise OSError("gone")

    monkeypatch.setattr(writer.record_writer, "_append", broken)
    assert writer.append_record("tool_call", {"a": 1}) is None
    assert recorded_entries("audit") == []


def _record(**overrides):
    fields = {
        "ts": "t",
        "actor": "a",
        "posture": "writes",
        "posture_source": "process",
        "session": None,
        "session_id": "c",
        "tool_use_id": "u",
        "server": "controls",
        "subject": "s",
        "approval": None,
        "target": "va",
        "generation": 3,
        "facts": {},
        "duration_ms": 12.5,
    }
    fields.update(overrides)
    return tool_call.build_record(**fields)


def test_the_record_keeps_its_documented_order() -> None:
    assert list(_record()) == [
        "ts",
        "surface",
        "actor",
        "posture",
        "posture_source",
        "session",
        "session_id",
        "tool_use_id",
        "duration_ms",
        "server",
        "subject",
        "approval",
        "target",
        "generation",
        "facts",
    ]


@pytest.mark.parametrize(
    "dropped", ["arguments", "arguments_ref", "result", "result_ref", "decision", "reason"]
)
def test_the_record_carries_no_payload_and_no_decision(dropped: str) -> None:
    assert dropped not in _record()
    with pytest.raises(TypeError):
        _record(**{dropped: "x"})


def test_the_duration_comes_before_the_facts() -> None:
    """A line cut short inside its facts still says how long the call took."""
    record = _record(facts={"old_values": {"A:1": {"value": 1}}})
    keys = list(record)
    assert keys.index("duration_ms") == keys.index("tool_use_id") + 1

    line = json.dumps(record, separators=(",", ":"))
    assert '"duration_ms":12.5' in line[: line.index('"facts"')]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"approval": None, "target": None, "generation": None, "facts": {}}, False),
        (
            {"approval": {"outcome": "approved"}, "target": None, "generation": None, "facts": {}},
            True,
        ),
        ({"approval": None, "target": "va", "generation": None, "facts": {}}, True),
        ({"approval": None, "target": None, "generation": 0, "facts": {}}, True),
        ({"approval": None, "target": None, "generation": None, "facts": {"ca_user": "x"}}, True),
    ],
)
def test_a_call_with_any_fact_is_recorded(fields: dict, expected: bool) -> None:
    assert tool_call.has_facts(**fields) is expected


def test_an_unset_session_takes_the_terminal_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSPREY_TERMINAL_SESSION", "dispatch-42")
    assert _record(session=None)["session"] == "dispatch-42"
    assert _record(session="pty-3")["session"] == "pty-3"


def test_no_session_anywhere_stays_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSPREY_TERMINAL_SESSION", raising=False)
    monkeypatch.setenv("OSPREY_TERMINAL_SESSION_LIFETIME", "43200")
    assert _record(session=None)["session"] is None
