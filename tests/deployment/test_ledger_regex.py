"""The ledger regexes against lines the real writers produce.

The gateway's operators take the fields the store keeps by position, never
by parsing the whole line: the anchor reads ``ts`` and ``surface``, then the
``tool_call`` regex or the envelope regex reads the rest. Every line here is
built by the writer that writes it in production — ``tool_call.build_record``,
``AuditEnvelope.to_dict`` with the writer's encoder, and the hook twin's
``emit_audit`` — so a writer that reorders its fields fails here.

Python's ``re`` reads these patterns as RE2 does (named groups, no
backtracking-only constructs); the pinned Collector runs them in
``test_telemetry_gateway_validate.py``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

import osprey
from osprey.audit.envelope import AuditEnvelope
from osprey.audit.tool_call import build_record
from osprey.audit.writer import _encode
from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_gateway as gw

_ANCHOR = re.compile(gw.ANCHOR_REGEX)
_TOOL_CALL = re.compile(gw.TOOL_CALL_REGEX)
_ENVELOPE = re.compile(gw.ENVELOPE_REGEX)

_TS = "2026-10-11T10:00:00.123Z"


def _parse(line: bytes | str) -> dict[str, str]:
    """What the operator chain writes to the attributes for *line*, as stanza would."""
    text = line.decode() if isinstance(line, bytes) else line
    text = text.rstrip("\n")
    anchor = _ANCHOR.match(text)
    if anchor is None:
        return {}
    fields = {k: v for k, v in anchor.groupdict().items() if v is not None}
    regex = _TOOL_CALL if fields["ledger_surface"] == "tool_call" else _ENVELOPE
    match = regex.match(text)
    if match is not None:
        fields.update({k: v for k, v in match.groupdict().items() if v is not None})
    return fields


def _tool_call(**overrides) -> dict:
    record = {
        "ts": _TS,
        "actor": "alice",
        "posture": "writes",
        "posture_source": "app",
        "session": "term-1",
        "session_id": "sess-1",
        "tool_use_id": "toolu_01",
        "server": "controls",
        "subject": "write_channel",
        "decision": "allowed",
        "reason": "posture",
        "approval": None,
        "target": None,
        "generation": None,
        "arguments": {"channel": "X"},
        "arguments_ref": None,
        "result": {"ok": True},
        "result_ref": None,
        "error": None,
        "is_error": False,
        "facts": {},
        "duration_ms": 12.5,
    }
    record.update(overrides)
    return build_record(**record)


def _envelope(**overrides) -> bytes:
    fields = {
        "ts": _TS,
        "surface": "approval",
        "actor": "alice",
        "posture": "writes",
        "posture_source": "app",
        "session": "term-1",
        "subject": "write_channel",
        "decision": "approved",
        "reason": "operator",
    }
    fields.update(overrides)
    return _encode(AuditEnvelope(**fields).to_dict())


def test_a_tool_call_line_gives_every_kept_field() -> None:
    fields = _parse(_encode(_tool_call()))
    assert fields == {
        "ledger_ts": _TS,
        "ledger_surface": "tool_call",
        "ledger_actor": "alice",
        "ledger_session": "term-1",
        "ledger_session_id": "sess-1",
        "ledger_tool_use_id": "toolu_01",
        "ledger_duration_ms": "12.5",
        "ledger_decision": "allowed",
    }


def test_a_null_tool_use_id_captures_none_from_the_arguments() -> None:
    line = _encode(
        _tool_call(tool_use_id=None, session_id=None, arguments={"tool_use_id": "nested"})
    )
    assert b'"tool_use_id":"nested"' in line
    fields = _parse(line)
    assert "ledger_tool_use_id" not in fields
    assert "ledger_session_id" not in fields
    assert fields["ledger_decision"] == "allowed"


def test_a_non_finite_duration_is_captured_as_written() -> None:
    fields = _parse(_encode(_tool_call(duration_ms=float("nan"))))
    assert fields["ledger_duration_ms"] == "NaN"
    assert _parse(_encode(_tool_call(duration_ms=3)))["ledger_duration_ms"] == "3"


def test_a_truncated_tool_call_line_keeps_duration_ms() -> None:
    cap = tc.gateway_max_log_size({})
    line = _encode(_tool_call(arguments={"blob": "x" * (2 * cap)})).rstrip(b"\n")
    cut = line[:cap]  # what the reader keeps under max_log_size_behavior: truncate
    assert len(cut) == cap and not cut.endswith(b"}")
    fields = _parse(cut)
    assert fields["ledger_duration_ms"] == "12.5"
    assert fields["ledger_tool_use_id"] == "toolu_01"
    assert fields["ledger_decision"] == "allowed"


def test_an_envelope_with_a_null_session_and_no_on_behalf_of() -> None:
    fields = _parse(_envelope(session=None))
    assert fields == {
        "ledger_ts": _TS,
        "ledger_surface": "approval",
        "ledger_actor": "alice",
        "ledger_decision": "approved",
    }


def test_an_envelope_keeps_on_behalf_of_after_the_session() -> None:
    fields = _parse(_envelope(surface="preflight", actor="queueserver", on_behalf_of="alice"))
    assert fields["ledger_on_behalf_of"] == "alice"
    assert fields["ledger_session"] == "term-1"
    assert fields["ledger_surface"] == "preflight"


def test_a_channel_finder_envelope_keeps_its_surface_and_time() -> None:
    fields = _parse(_envelope(surface="channel-finder", decision="allowed", tool_use_id="t-1"))
    assert fields["ledger_surface"] == "channel-finder"
    assert fields["ledger_ts"] == _TS
    assert fields["ledger_tool_use_id"] == "t-1"


def test_the_escaped_subject_does_not_end_the_match() -> None:
    line = _envelope(
        surface="auth_sidecar", subject='ev"il\\name', decision="refused", reason="login"
    )
    assert b'ev\\"il\\\\name' in line
    fields = _parse(line)
    assert fields["ledger_decision"] == "refused"
    assert fields["ledger_actor"] == "alice"


def test_an_executor_line_captures_the_real_tool_use_id_before_its_source() -> None:
    line = _envelope(
        surface="executor",
        decision="refused",
        detail="a detail",
        tool_use_id="toolu_real",
        source='print(\'"tool_use_id":"x"\')',
    )
    assert b'"source":' in line and b'\\"tool_use_id\\":\\"x\\"' in line
    fields = _parse(line)
    assert fields["ledger_tool_use_id"] == "toolu_real"
    assert fields["ledger_decision"] == "refused"


def _hook_module() -> ModuleType:
    path = (
        Path(osprey.__file__).parent
        / "templates"
        / "claude_code"
        / "claude"
        / "hooks"
        / "osprey_hook_log.py"
    )
    spec = importlib.util.spec_from_file_location("_ledger_regex_hook_log", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_hook_approval_keeps_its_decision_and_tool_use_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _hook_module()
    target = tmp_path / "approval.jsonl"
    monkeypatch.setattr(hook, "audit_ledger_path", lambda *a, **k: target)
    written = hook.emit_audit(
        "approval",
        {"tool_use_id": "toolu_hook"},
        hook.AUDIT_DECISION_ASK,
        "write_channel",
        "needs_approval",
        detail="limits",
    )
    assert written == str(target)
    line = target.read_bytes()
    assert json.loads(line)["tool_use_id"] == "toolu_hook"
    fields = _parse(line)
    assert fields["ledger_decision"] == hook.AUDIT_DECISION_ASK
    assert fields["ledger_tool_use_id"] == "toolu_hook"
    assert fields["ledger_surface"] == hook.audit_surface("approval")


def test_a_garbage_time_still_parses_the_surface() -> None:
    line = _envelope().replace(_TS.encode(), b"garbage")
    fields = _parse(line)
    assert fields["ledger_ts"] == "garbage"
    assert fields["ledger_surface"] == "approval"
    assert fields["ledger_decision"] == "approved"


def test_a_torn_line_keeps_what_precedes_the_tear() -> None:
    whole = _envelope(tool_use_id="toolu_9").rstrip(b"\n")
    torn = whole[: whole.index(b'"decision"') + 5]
    fields = _parse(torn)
    assert fields == {"ledger_ts": _TS, "ledger_surface": "approval"}
    assert not torn.endswith(b"}")


def test_a_line_that_is_not_a_ledger_line_matches_nothing() -> None:
    assert _parse(b"not json\n") == {}
    assert _parse(b'{"surface":"x","ts":"y"}\n') == {}
