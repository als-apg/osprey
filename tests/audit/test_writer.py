"""Tests for the audit writer.

The writer turns every safety decision into one ``audit`` entry in the agent
record, so these pin the properties every emitter downstream is allowed to
assume:

* **One decision, one entry.** The body is the envelope exactly, recorded once,
  under the identity of the process that wrote it — never the envelope's
  ``actor``.
* **The ids it can be joined on travel with it.** ``ts``, ``tool_use_id``, the
  terminal session and ``on_behalf_of`` are the line's ids as well as body
  fields.
* **A bounded body.** ≤2 KB, ``detail`` gives way first, the executor's source
  and every identifier never do.
* **Nothing here can cost the operation.** An unwritable spool, a failing
  record writer, the off switch, even an invalid envelope, degrade to ``None``
  and a log line.

Every test reads what it wrote through the record (``recorded_entries``), or
through the spool line the record consumes, never through a file of the
writer's own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.agent_record import writer as record_writer
from osprey.agent_record.contract import KIND_AUDIT, RECORD_OFF_ENV
from osprey.audit import writer
from osprey.audit.call import call_scope
from osprey.audit.envelope import (
    DECISION_ALLOWED,
    DECISION_REFUSED,
    MAX_SOURCE_CHARS,
    POSTURE_SOURCE_APP,
    POSTURE_SOURCE_PROCESS,
    SURFACE_EXECUTOR,
    TERMINAL_SESSION_ENV,
    AuditEnvelope,
)
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV

# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_markers(monkeypatch):
    """Clear the identity and session markers a developer shell may export."""
    for marker in (
        TERMINAL_USER_ENV,
        AUDIT_IDENTITY_ENV,
        TERMINAL_SESSION_ENV,
        RECORD_OFF_ENV,
        "CLAUDE_CODE_SESSION_ID",
        "OSPREY_TELEMETRY_SESSION_ID",
    ):
        monkeypatch.delenv(marker, raising=False)


#: A minimal legal record for the ``record(**fields)`` entry point.
_RECORD_FIELDS = {
    "surface": "mcp_python",
    "posture": "sandbox",
    "posture_source": POSTURE_SOURCE_PROCESS,
    "session": None,
    "subject": "mcp__python__execute",
    "decision": DECISION_ALLOWED,
    "reason": "allowed",
}


def make_envelope(**overrides) -> AuditEnvelope:
    """A minimal legal envelope, so each test's diff is the field it is about."""
    kwargs = {
        "surface": "http_config",
        "actor": "alice",
        "posture": "sandbox",
        "posture_source": POSTURE_SOURCE_APP,
        "session": "chat-7",
        "subject": "connectors.epics.writes_enabled",
        "decision": DECISION_REFUSED,
        "reason": "protected_key",
    }
    kwargs.update(overrides)
    return AuditEnvelope(**kwargs)


def _audit(recorded_entries) -> list:
    """Every ``audit`` entry the test's spool has produced."""
    return recorded_entries(KIND_AUDIT)


def _bodies(recorded_entries) -> list[dict]:
    return [dict(entry.body) for entry in _audit(recorded_entries)]


def _spool_lines(directory: Path) -> list[dict]:
    """The raw spool lines under *directory*, as the record consumer reads them."""
    record_writer.seal_all()
    lines: list[dict] = []
    for segment in sorted(directory.glob("*.sealed")):
        lines.extend(json.loads(line) for line in segment.read_text().splitlines() if line)
    return lines


def _encoded_size(body: dict) -> int:
    return len(writer._encode(body))


# --------------------------------------------------------------------------


class TestOneDecisionOneEntry:
    """The body is the envelope, recorded exactly once."""

    def test_record_envelope_records_the_envelope_once(self, recorded_entries):
        envelope = make_envelope()

        entry_id = writer.record_envelope(envelope)

        assert isinstance(entry_id, str) and entry_id
        (entry,) = _audit(recorded_entries)
        assert dict(entry.body) == envelope.to_dict()
        assert list(entry.body) == list(envelope.to_dict())

    def test_two_decisions_are_two_entries(self, recorded_entries):
        writer.record_envelope(make_envelope(subject="first"))
        writer.record_envelope(make_envelope(subject="second", surface="http_mutation"))

        bodies = sorted(_bodies(recorded_entries), key=lambda b: b["subject"])
        assert [(b["subject"], b["surface"]) for b in bodies] == [
            ("first", "http_config"),
            ("second", "http_mutation"),
        ]

    def test_session_is_present_even_when_null(self, recorded_entries):
        writer.record_envelope(make_envelope(session=None))

        (body,) = _bodies(recorded_entries)
        assert "session" in body
        assert body["session"] is None

    def test_the_identity_is_the_spool_directorys_not_the_actor(self, recorded_entries, spool_for):
        """A service acting for someone files under its own identity and names
        the person as the actor; the record never routes on the field."""
        spool_for("alice")
        writer.record_envelope(make_envelope(actor="mallory"))

        (entry,) = _audit(recorded_entries)
        assert entry.identity == "alice"
        assert entry.body["actor"] == "mallory"

    def test_two_identities_are_two_sources(self, recorded_entries, spool_for):
        spool_for("alice")
        writer.record_envelope(make_envelope(subject="from-alice"))
        spool_for("bob")
        writer.record_envelope(make_envelope(subject="from-bob"))

        seen = {(e.identity, e.body["subject"]) for e in _audit(recorded_entries)}
        assert seen == {("alice", "from-alice"), ("bob", "from-bob")}


class TestJoinIds:
    """The ids a reader joins on are the line's ids, not only body fields."""

    def test_ts_and_tool_use_id_are_the_entrys(self, recorded_entries):
        envelope = make_envelope(tool_use_id="toolu_01", ts="2026-10-25T10:15:02.123Z")

        writer.record_envelope(envelope)

        (entry,) = _audit(recorded_entries)
        assert entry.tool_use_id == "toolu_01"
        assert entry.ts.isoformat().startswith("2026-10-25T10:15:02.123")

    def test_the_terminal_session_is_the_processs(self, recorded_entries, monkeypatch):
        monkeypatch.setenv(TERMINAL_SESSION_ENV, "term-42")

        writer.record_envelope(make_envelope(session="posture-session"))

        (entry,) = _audit(recorded_entries)
        assert entry.terminal_session == "term-42"
        assert entry.body["session"] == "posture-session"

    def test_the_session_id_is_the_processs_conversation(self, recorded_entries, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "conv-proc")
        monkeypatch.delenv("OSPREY_TELEMETRY_SESSION_ID", raising=False)

        writer.record_envelope(make_envelope())

        (entry,) = _audit(recorded_entries)
        assert entry.session_id == "conv-proc"

    def test_on_behalf_of_travels_on_the_line(self, spool_zone):
        writer.record_envelope(make_envelope(on_behalf_of="bob"))

        (line,) = _spool_lines(spool_zone.root / spool_zone.terminal)
        assert line["on_behalf_of"] == "bob"
        assert line["body"]["on_behalf_of"] == "bob"

    def test_an_untrusted_writers_on_behalf_of_is_not_kept(self, recorded_entries):
        """Only the identities the trust list names may act for someone; a
        terminal's claim stays in the body and never becomes the entry's."""
        writer.record_envelope(make_envelope(on_behalf_of="bob"))

        (entry,) = _audit(recorded_entries)
        assert entry.on_behalf_of == ""

    def test_no_id_no_key(self, spool_zone):
        writer.record_envelope(make_envelope())

        (line,) = _spool_lines(spool_zone.root / spool_zone.terminal)
        assert "tool_use_id" not in line
        assert "on_behalf_of" not in line
        assert "terminal_session" not in line
        assert "session_id" not in line


class TestKwargsEntryPoint:
    """``record(**fields)`` for emitters that have facts, not an envelope."""

    def test_builds_and_writes(self, recorded_entries, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        assert writer.record(**_RECORD_FIELDS)

        (body,) = _bodies(recorded_entries)
        assert body["surface"] == "mcp_python"
        assert body["subject"] == "mcp__python__execute"

    def test_actor_defaults_to_the_acting_identity(self, recorded_entries, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc-mcp")

        writer.record(**_RECORD_FIELDS)

        (body,) = _bodies(recorded_entries)
        assert body["actor"] == "svc-mcp"

    def test_the_terminal_user_outranks_the_service_identity(self, recorded_entries, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc-mcp")
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")

        writer.record(**_RECORD_FIELDS)

        (body,) = _bodies(recorded_entries)
        assert body["actor"] == "alice"

    def test_an_explicit_actor_wins(self, recorded_entries, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc-mcp")

        writer.record(**_RECORD_FIELDS, actor="bob")

        (body,) = _bodies(recorded_entries)
        assert body["actor"] == "bob"

    def test_record_takes_the_tool_use_id_from_the_call_scope(self, recorded_entries):
        with call_scope("toolu_scope", "conv-1"):
            writer.record(**_RECORD_FIELDS)

        (entry,) = _audit(recorded_entries)
        assert entry.body["tool_use_id"] == "toolu_scope"
        assert entry.tool_use_id == "toolu_scope"
        assert entry.session_id == "conv-1"

    def test_an_explicit_tool_use_id_wins(self, recorded_entries):
        with call_scope("toolu_scope", "conv-1"):
            writer.record(**_RECORD_FIELDS, tool_use_id="toolu_explicit")

        (body,) = _bodies(recorded_entries)
        assert body["tool_use_id"] == "toolu_explicit"

    def test_no_scope_no_key(self, recorded_entries):
        writer.record(**_RECORD_FIELDS)

        (body,) = _bodies(recorded_entries)
        assert "tool_use_id" not in body

    def test_an_invalid_envelope_degrades_instead_of_raising(self, recorded_entries, caplog):
        with caplog.at_level("WARNING"):
            result = writer.record(**{**_RECORD_FIELDS, "posture_source": "made-up"})

        assert result is None
        assert _audit(recorded_entries) == []
        assert "record dropped" in caplog.text

    def test_a_missing_required_field_degrades(self, recorded_entries):
        fields = dict(_RECORD_FIELDS)
        del fields["subject"]

        assert writer.record(**fields) is None
        assert _audit(recorded_entries) == []

    def test_source_outside_the_executor_surface_degrades(self, recorded_entries):
        assert writer.record(**_RECORD_FIELDS, source="print(1)") is None
        assert _audit(recorded_entries) == []


class TestRecordBound:
    """≤2 KB per body, with the documented exceptions."""

    def test_an_ordinary_record_is_well_inside_the_bound(self, recorded_entries):
        writer.record_envelope(make_envelope(detail="channel=config"))

        (body,) = _bodies(recorded_entries)
        assert body["detail"] == "channel=config"
        assert _encoded_size(body) <= writer.MAX_RECORD_BYTES

    def test_an_oversize_detail_is_dropped_to_fit(self, recorded_entries):
        """``detail`` is supplementary by contract, so it is what gives way."""
        long_name = "n" * 256

        writer.record_envelope(
            make_envelope(
                actor=long_name,
                posture=long_name,
                session=long_name,
                role=long_name,
                detail="x" * 1024,
            )
        )

        (body,) = _bodies(recorded_entries)
        assert body["detail"] == writer.DETAIL_DROPPED
        assert _encoded_size(body) <= writer.MAX_RECORD_BYTES
        assert body["subject"] == "connectors.epics.writes_enabled"

    def test_identifiers_are_never_sacrificed_to_the_bound(self, recorded_entries):
        """A record whose identifiers were trimmed would name the wrong thing;
        over-budget-but-true beats in-budget-but-wrong."""
        long_name = "n" * 256

        writer.record_envelope(
            make_envelope(
                actor=long_name,
                posture=long_name,
                subject=long_name,
                reason=long_name,
                role=long_name,
                detail="x" * 1024,
            )
        )

        (body,) = _bodies(recorded_entries)
        assert body["subject"] == long_name
        assert body["reason"] == long_name

    def test_executor_source_is_written_whole(self, recorded_entries):
        """On the executor surface the refused code IS the artifact."""
        code = "caput('PV', 1)\n" * 400
        assert len(code) > writer.MAX_RECORD_BYTES

        writer.record_envelope(
            make_envelope(surface=SURFACE_EXECUTOR, source=code, detail="tool=execute")
        )

        (body,) = _bodies(recorded_entries)
        assert body["source"] == code
        assert body["detail"] == "tool=execute"

    def test_an_executor_record_drops_its_detail_before_its_bound(self, recorded_entries):
        """The executor exception buys the SOURCE, not the detail."""
        long_name = "n" * 256
        source = "caput('PV', 1)"

        writer.record_envelope(
            make_envelope(
                surface=SURFACE_EXECUTOR,
                source=source,
                detail="x" * 1024,
                actor=long_name,
                posture=long_name,
                subject=long_name,
                reason=long_name,
                role=long_name,
            )
        )

        (body,) = _bodies(recorded_entries)
        assert body["source"] == source
        assert body["detail"] == writer.DETAIL_DROPPED
        assert _encoded_size(body) <= writer.MAX_RECORD_BYTES

    def test_a_truncated_source_still_says_so(self, recorded_entries):
        writer.record_envelope(
            make_envelope(surface=SURFACE_EXECUTOR, source="c" * (MAX_SOURCE_CHARS + 10))
        )

        (body,) = _bodies(recorded_entries)
        assert body["source_truncated"] is True
        assert len(body["source"]) == MAX_SOURCE_CHARS

    def test_an_identifier_only_record_over_budget_is_written_and_flagged(
        self, recorded_entries, caplog
    ):
        long_name = "n" * 256
        envelope = make_envelope(
            surface=long_name,
            actor=long_name,
            posture=long_name,
            session=long_name,
            subject=long_name,
            decision=long_name,
            reason=long_name,
            role=long_name,
            detail="x" * 1024,
        )

        with caplog.at_level("WARNING"):
            writer.record_envelope(envelope)

        (body,) = _bodies(recorded_entries)
        assert body["subject"] == long_name
        assert body["detail"] == writer.DETAIL_DROPPED
        assert _encoded_size(body) > writer.MAX_RECORD_BYTES
        assert "exceeds" in caplog.text

    def test_the_caller_s_envelope_is_not_modified(self, recorded_entries):
        long_name = "n" * 256
        envelope = make_envelope(
            actor=long_name,
            posture=long_name,
            session=long_name,
            role=long_name,
            detail="x" * 1024,
        )

        writer.record_envelope(envelope)

        assert envelope.detail == "x" * 1024
        assert _bodies(recorded_entries)[0]["detail"] == writer.DETAIL_DROPPED

    def test_the_bound_is_two_kilobytes(self):
        assert writer.MAX_RECORD_BYTES == 2048


class TestAppendRecord:
    """The full ``tool_call`` record: not an envelope, not bounded."""

    def test_the_record_is_the_body_on_its_surface(self, recorded_entries):
        record = {
            "ts": "2026-10-25T10:15:04.000Z",
            "surface": "tool_call",
            "actor": "alice",
            "tool_use_id": "toolu_9",
            "result": "r" * (writer.MAX_RECORD_BYTES * 4),
        }

        assert writer.append_record("tool_call", record)

        (entry,) = _audit(recorded_entries)
        assert dict(entry.body) == record
        assert entry.tool_use_id == "toolu_9"

    def test_the_surface_argument_names_the_surface(self, recorded_entries):
        writer.append_record("tool_call", {"a": 1})

        (body,) = _bodies(recorded_entries)
        assert body == {"a": 1, "surface": "tool_call"}

    def test_nan_inside_facts_is_named_and_keeps_the_rest(self, spool_zone):
        record = {
            "surface": "tool_call",
            "facts": {"old_values": {"A:1": float("nan"), "A:2": 2.0}},
        }

        writer.append_record("tool_call", record)

        (line,) = _spool_lines(spool_zone.root / spool_zone.terminal)
        assert line["body"]["facts"]["old_values"] == {"A:1": "NaN", "A:2": 2.0}
        (value,) = [record["facts"]["old_values"]["A:1"]]
        assert value != value, "the caller's record was modified"

    def test_append_record_never_raises(self, monkeypatch):
        def broken(*args, **kwargs):
            raise OSError("gone")

        monkeypatch.setattr(writer.record_writer, "write", broken)
        assert writer.append_record("tool_call", {"a": 1}) is None


class TestNeverRaises:
    """Every failure degrades to ``None``; the operation is untouched."""

    def test_an_unwritable_spool_degrades_to_none(self, monkeypatch, tmp_path):
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("")
        monkeypatch.setenv("OSPREY_SPOOL_DIR", str(blocker / "alice"))

        assert writer.record_envelope(make_envelope()) is None
        assert writer.record(**_RECORD_FIELDS) is None

    def test_a_failing_record_writer_degrades_to_none(self, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(writer.record_writer, "write", broken)
        assert writer.record_envelope(make_envelope()) is None

    def test_the_off_switch_stores_nothing(self, recorded_entries, monkeypatch):
        monkeypatch.setenv(RECORD_OFF_ENV, "1")

        assert writer.record_envelope(make_envelope()) is None
        monkeypatch.delenv(RECORD_OFF_ENV)
        assert _audit(recorded_entries) == []

    def test_a_non_envelope_argument_degrades_to_none(self):
        assert writer.record_envelope("not an envelope") is None  # type: ignore[arg-type]


class TestServerLogCarriesTheEvent:
    """Logged before the write, so a broken spool downgrades rather than erases
    the trail."""

    def test_a_refusal_warns_even_when_the_write_fails(self, monkeypatch, caplog):
        def explode(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(writer.record_writer, "write", explode)
        with caplog.at_level("WARNING"):
            writer.record_envelope(make_envelope(subject="connectors.epics.writes_enabled"))

        assert "connectors.epics.writes_enabled" in caplog.text

    def test_an_allowed_record_does_not_warn(self, caplog):
        with caplog.at_level("WARNING"):
            writer.record_envelope(make_envelope(decision=DECISION_ALLOWED, reason="allowed"))

        assert caplog.text == ""


class TestNonFiniteNumbers:
    """A line is strict JSON, whatever number a record was handed."""

    @staticmethod
    def _strict(line: str) -> dict:
        def refuse(token: str) -> float:
            raise AssertionError(f"the line carries the bare token {token}")

        return json.loads(line, parse_constant=refuse)

    def test_every_non_finite_value_is_named(self):
        record = {
            "a": [float("inf"), (float("-inf"), 1.0)],
            "b": {"c": [{"d": float("nan")}]},
            float("inf"): "key",
        }
        stored = self._strict(writer._encode(record).decode())
        assert stored == {
            "a": ["Infinity", ["-Infinity", 1.0]],
            "b": {"c": [{"d": "NaN"}]},
            "Infinity": "key",
        }

    def test_a_finite_record_is_encoded_in_one_pass(self, monkeypatch):
        def unexpected(value):
            raise AssertionError(f"a finite record was walked: {value!r}")

        monkeypatch.setattr(writer, "_finite", unexpected)
        record = {"a": 1.5, "b": [1, 2.0, {"c": None}], "d": "x"}
        assert writer._encode(record) == b'{"a":1.5,"b":[1,2.0,{"c":null}],"d":"x"}\n'

    def test_an_unencodable_record_is_still_refused(self):
        with pytest.raises(TypeError):
            writer._encode({"a": object()})
