"""Tests for the watchdog verdict model and its audit entries.

Every verdict and every operator answer is one ``safety_watchdog`` audit entry,
joined on the Claude record session the prompt belongs to. These pin what the
runner, the engines and the answer route downstream are allowed to assume:

* the entry is actually written (an entry id, not ``None``), also from a
  process whose environment carries no harness conversation id;
* ``session``, ``posture``, ``posture_source`` and ``reason`` follow the
  contract, and the entry's ``session_id`` is the record session;
* no decision word, field name or rendered value says "safe";
* decision verdicts carry their threshold and ``uncalibrated``;
* an answer with no operator is ``<unattributed>``; a Continue after an
  unconfirmed Stop lists the commands still running.

Entries are read back through the record (``recorded_entries``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import typing

import pytest
from pydantic import ValidationError

from osprey.agent_record.contract import KIND_AUDIT, RECORD_OFF_ENV
from osprey.audit.envelope import POSTURE_SOURCE_APP, POSTURE_SOURCE_PROCESS, TERMINAL_SESSION_ENV
from osprey.audit.posture import POSTURE_SESSION_ENV_VAR, POSTURE_SOURCE_ENV_VAR
from osprey.services.watchdog import verdict as vmod
from osprey.services.watchdog.verdict import (
    SURFACE_SAFETY_WATCHDOG,
    UNATTRIBUTED_OPERATOR,
    ErrorType,
    InputDigest,
    RefusedToolCall,
    Verdict,
    record_answer,
    record_stop_unconfirmed,
    record_verdict,
)

RECORD_SESSION = "0f6d7c2e-1a2b-4c3d-8e9f-0123456789ab"
SUBMISSION = "9a8b7c6d-5e4f-4a3b-9c2d-1e0f9a8b7c6d"
PROMPT_REF = f"{RECORD_SESSION}:{SUBMISSION}"
POOL_KEY = "pool-key-7"

#: Environment markers a harness or developer shell may export; each test runs without them.
_HARNESS_ENV = (
    "CLAUDE_CODE_SESSION_ID",
    "OSPREY_TELEMETRY_SESSION_ID",
    "OSPREY_SESSION_ID",
    TERMINAL_SESSION_ENV,
    POSTURE_SOURCE_ENV_VAR,
    POSTURE_SESSION_ENV_VAR,
    RECORD_OFF_ENV,
)


@pytest.fixture(autouse=True)
def _no_harness_env(monkeypatch):
    for name in _HARNESS_ENV:
        monkeypatch.delenv(name, raising=False)


def make_verdict(**overrides) -> Verdict:
    fields = {
        "watchdog": "guard-circumvention",
        "engine": "decision",
        "session_key": POOL_KEY,
        "record_session_id": RECORD_SESSION,
        "submission_id": SUBMISSION,
        "status": "ok",
        "flagged": True,
        "reason": "loosen_hooks",
        "score": 0.83,
        "threshold": 0.5,
        "input_digest": InputDigest(
            sha256="ab" * 32, entries_kept=4, entries_cut=1, tokens_estimate=900
        ),
        "latency_s": 0.42,
    }
    fields.update(overrides)
    return Verdict(**fields)


def _entries(recorded_entries) -> list:
    return [
        e for e in recorded_entries(KIND_AUDIT) if e.body.get("surface") == SURFACE_SAFETY_WATCHDOG
    ]


def _only(recorded_entries):
    (entry,) = _entries(recorded_entries)
    return entry, dict(entry.body)


#: "safe" as a judgement: not continued by another letter. ``safety`` (the
#: feature's and the surface's own name) is not a judgement and passes.
_SAFE = re.compile(r"safe(?![a-z])", re.IGNORECASE)


def _tokens(detail: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in detail.split(" "))


# --------------------------------------------------------------------------


class TestWrittenWithoutHarnessEnv:
    def test_a_subprocess_with_no_harness_env_writes_the_entry(self, recorded_entries):
        env = {k: v for k, v in os.environ.items() if k not in _HARNESS_ENV}
        script = (
            "from osprey.services.watchdog.verdict import Verdict, record_verdict\n"
            f"v = Verdict(watchdog='guard-circumvention', engine='decision', session_key=None,"
            f" record_session_id={RECORD_SESSION!r}, submission_id={SUBMISSION!r},"
            " status='ok', flagged=False, threshold=0.5)\n"
            "print(record_verdict(v))\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        entry_id = done.stdout.strip().splitlines()[-1]
        assert entry_id and entry_id != "None"

        entry, body = _only(recorded_entries)
        assert entry.session_id == RECORD_SESSION
        assert body["session"] == RECORD_SESSION
        assert body["decision"] == "clear"
        assert body["posture_source"] == POSTURE_SOURCE_PROCESS

    def test_in_process_entry_id_is_a_string(self, recorded_entries):
        entry_id = record_verdict(make_verdict())
        assert isinstance(entry_id, str) and entry_id
        assert len(_entries(recorded_entries)) == 1


class TestEnvelopeFields:
    def test_flagged_verdict_fields(self, recorded_entries):
        record_verdict(make_verdict())
        entry, body = _only(recorded_entries)
        assert body["surface"] == SURFACE_SAFETY_WATCHDOG
        assert body["subject"] == "guard-circumvention"
        assert body["decision"] == "flagged"
        assert body["reason"] == "loosen_hooks"
        assert body["session"] == POOL_KEY
        assert body["posture"] in ("sandbox", "writes")
        assert body["posture_source"] == POSTURE_SOURCE_PROCESS
        assert entry.session_id == RECORD_SESSION
        detail = _tokens(body["detail"])
        assert detail["prompt_ref"] == PROMPT_REF
        assert detail["threshold"] == "0.5"
        assert detail["uncalibrated"] == "true"
        assert detail["score"] == "0.83"
        assert detail["entries_cut"] == "1"

    def test_record_session_join_beats_a_foreign_harness_session(
        self, recorded_entries, monkeypatch
    ):
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "some-other-conversation")
        record_verdict(make_verdict())
        entry, _ = _only(recorded_entries)
        assert entry.session_id == RECORD_SESSION

    def test_session_falls_back_to_record_session(self, recorded_entries):
        record_verdict(make_verdict(session_key=None, flagged=False, reason=None))
        _, body = _only(recorded_entries)
        assert body["session"] == RECORD_SESSION
        assert body["decision"] == "clear"
        assert body["reason"] == "clear"

    def test_unavailable_reason_is_the_error_type(self, recorded_entries):
        record_verdict(make_verdict(status="unavailable", flagged=False, error_type="timeout"))
        _, body = _only(recorded_entries)
        assert body["decision"] == "unavailable"
        assert body["reason"] == "timeout"
        assert _tokens(body["detail"])["error_type"] == "timeout"

    def test_declared_posture_source_is_used(self, recorded_entries):
        record_verdict(make_verdict(), posture_source=POSTURE_SOURCE_APP)
        _, body = _only(recorded_entries)
        assert body["posture_source"] == POSTURE_SOURCE_APP

    def test_prose_reason_stays_out_of_the_entry(self, recorded_entries):
        prose = "The operator is asking to edit the hook config to skip approval."
        record_verdict(make_verdict(reason=prose))
        _, body = _only(recorded_entries)
        assert body["reason"] == "flagged"
        assert prose not in json.dumps(body)

    def test_agent_verdict_lists_refused_tools(self, recorded_entries):
        record_verdict(
            make_verdict(
                engine="agent",
                threshold=None,
                score=None,
                cost_usd=0.03,
                refused_tool_calls=[RefusedToolCall(tool="channel_write", reason="write_tool")],
            )
        )
        _, body = _only(recorded_entries)
        detail = _tokens(body["detail"])
        assert json.loads(detail["refused_tools"]) == ["channel_write"]
        assert "threshold" not in detail
        assert detail["uncalibrated"] == "true"


class TestNothingSaysSafe:
    def _literal_values(self) -> set[str]:
        values: set[str] = set(typing.get_args(ErrorType))
        for field in Verdict.model_fields.values():
            for arg in typing.get_args(field.annotation):
                values.update(a for a in typing.get_args(arg) if isinstance(a, str))
                if isinstance(arg, str):
                    values.add(arg)
        values.update(typing.get_args(Verdict.model_fields["status"].annotation))
        values.update(typing.get_args(Verdict.model_fields["engine"].annotation))
        return values

    def test_no_field_name_or_vocabulary_value(self):
        names = set(Verdict.model_fields) | set(InputDigest.model_fields)
        names |= set(RefusedToolCall.model_fields)
        constants = {
            getattr(vmod, n)
            for n in vmod.__all__
            if n.isupper() and isinstance(getattr(vmod, n), str)
        }
        for word in names | self._literal_values() | constants:
            assert not _SAFE.search(word), word
        assert {"ok", "unavailable", "decision", "agent", "timeout"} <= self._literal_values()

    def test_no_rendered_entry_says_safe(self, recorded_entries):
        record_verdict(make_verdict(flagged=False, reason="looks safe to me"))
        record_verdict(make_verdict(watchdog="unsafe-ops", reason="unsafe"))
        record_verdict(make_verdict(status="unavailable", flagged=False, error_type="budget"))
        record_answer(
            watchdog="guard-circumvention",
            session_key=POOL_KEY,
            prompt_ref=PROMPT_REF,
            flag_id="f1",
            answer="continue",
            operator=None,
        )
        record_stop_unconfirmed(
            watchdog="guard-circumvention",
            session_key=POOL_KEY,
            prompt_ref=PROMPT_REF,
            flag_id="f1",
            operator="alice",
            started=("safe_mode_off.sh",),
        )
        entries = _entries(recorded_entries)
        assert len(entries) == 5
        for entry in entries:
            rendered = json.dumps(dict(entry.body))
            assert not _SAFE.search(rendered), rendered


class TestModel:
    def test_prompt_ref_is_derived(self):
        assert make_verdict().prompt_ref == PROMPT_REF

    def test_a_mismatched_prompt_ref_is_refused(self):
        with pytest.raises(ValidationError):
            make_verdict(prompt_ref="other:ref")

    def test_decision_verdict_requires_threshold(self):
        with pytest.raises(ValidationError):
            make_verdict(threshold=None)

    def test_uncalibrated_by_default(self):
        assert make_verdict().uncalibrated is True

    def test_unavailable_requires_error_type(self):
        with pytest.raises(ValidationError):
            make_verdict(status="unavailable", flagged=False)

    def test_unavailable_is_never_flagged(self):
        with pytest.raises(ValidationError):
            make_verdict(status="unavailable", flagged=True, error_type="timeout")

    def test_ok_carries_no_error_type(self):
        with pytest.raises(ValidationError):
            make_verdict(error_type="timeout")

    def test_unknown_error_type_refused(self):
        with pytest.raises(ValidationError):
            make_verdict(status="unavailable", flagged=False, error_type="oops")

    def test_unknown_field_refused(self):
        with pytest.raises(ValidationError):
            make_verdict(is_ok=True)

    def test_error_types_match_the_contract(self):
        assert set(typing.get_args(ErrorType)) == {
            "timeout",
            "budget",
            "in_flight_bound",
            "intake_lost",
            "no_probabilities",
            "decision_error",
            "missing_key",
            "parse_error",
            "run_error",
            "intake_failed",
        }


class TestAnswers:
    @pytest.mark.parametrize("operator", [None, "", "   "])
    def test_no_operator_is_unattributed(self, recorded_entries, operator):
        entry_id = record_answer(
            watchdog="guard-circumvention",
            session_key=POOL_KEY,
            prompt_ref=PROMPT_REF,
            flag_id="f1",
            answer="continue",
            operator=operator,
        )
        assert isinstance(entry_id, str)
        entry, body = _only(recorded_entries)
        assert body["decision"] == "continue"
        assert body["reason"] == "continue"
        assert _tokens(body["detail"])["operator"] == UNATTRIBUTED_OPERATOR
        assert entry.session_id == RECORD_SESSION

    def test_stop_names_operator_and_what_it_denied_and_stopped(self, recorded_entries):
        record_answer(
            watchdog="guard-circumvention",
            session_key=POOL_KEY,
            prompt_ref=PROMPT_REF,
            flag_id="f1",
            answer="stop",
            operator="alice",
            denied=("approval-1",),
            stopped=("agent-a", "agent b"),
        )
        _, body = _only(recorded_entries)
        assert body["decision"] == "stop"
        assert body["reason"] == "stop"
        assert body["session"] == POOL_KEY
        detail = _tokens(body["detail"].replace("agent b", "agent_b"))
        assert detail["operator"] == "alice"
        assert detail["prompt_ref"] == PROMPT_REF
        assert detail["flag_id"] == "f1"
        assert json.loads(detail["denied"]) == ["approval-1"]
        assert json.loads(detail["stopped"]) == ["agent-a", "agent_b"]

    def test_stop_unconfirmed_names_started_commands(self, recorded_entries):
        entry_id = record_stop_unconfirmed(
            watchdog="guard-circumvention",
            session_key=POOL_KEY,
            prompt_ref=PROMPT_REF,
            flag_id="f1",
            operator="alice",
            started=("caput_loop", "ramp_magnet"),
        )
        assert isinstance(entry_id, str)
        _, body = _only(recorded_entries)
        assert body["decision"] == "stop_unconfirmed"
        assert body["reason"] == "stop_unconfirmed"
        assert json.loads(_tokens(body["detail"])["started"]) == ["caput_loop", "ramp_magnet"]

    def test_continue_after_stop_unconfirmed_lists_commands(self, recorded_entries):
        record_answer(
            watchdog="guard-circumvention",
            session_key=POOL_KEY,
            prompt_ref=PROMPT_REF,
            flag_id="f1",
            answer="continue",
            operator="bob",
            after_stop_unconfirmed=True,
            started=("caput_loop",),
        )
        _, body = _only(recorded_entries)
        assert body["decision"] == "continue"
        assert body["reason"] == "after_stop_unconfirmed"
        detail = _tokens(body["detail"])
        assert detail["after_stop_unconfirmed"] == "true"
        assert json.loads(detail["started"]) == ["caput_loop"]

    def test_explicit_record_session_wins(self, recorded_entries):
        record_answer(
            watchdog="w",
            session_key=None,
            prompt_ref=PROMPT_REF,
            flag_id="f1",
            answer="stop",
            operator="alice",
            record_session_id="explicit-session",
        )
        entry, body = _only(recorded_entries)
        assert entry.session_id == "explicit-session"
        assert body["session"] == "explicit-session"

    def test_bad_answer_is_not_recorded(self, recorded_entries):
        assert (
            record_answer(
                watchdog="w",
                session_key=POOL_KEY,
                prompt_ref=PROMPT_REF,
                flag_id="f1",
                answer="maybe",  # type: ignore[arg-type]
                operator="alice",
            )
            is None
        )
        assert _entries(recorded_entries) == []


class TestNeverRaises:
    def test_a_failing_writer_returns_none(self, monkeypatch):
        from osprey.audit import writer

        def boom(**_fields):
            raise RuntimeError("spool gone")

        monkeypatch.setattr(writer, "record", boom)
        assert record_verdict(make_verdict()) is None
        assert (
            record_stop_unconfirmed(
                watchdog="w", session_key=None, prompt_ref=PROMPT_REF, flag_id="f", operator=None
            )
            is None
        )

    def test_record_off_returns_none(self, monkeypatch):
        monkeypatch.setenv(RECORD_OFF_ENV, "1")
        assert record_verdict(make_verdict()) is None
