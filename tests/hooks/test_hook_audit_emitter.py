"""The hook half of the unified audit ledger.

A hook is a fresh ``python3`` process with no ``osprey`` on its path, so it
cannot call :mod:`osprey.audit.writer`. It reimplements the minimal subset of
the writer instead — which is exactly the drift risk this file exists to pin:
the schema tests below assert the emitter's field set, bounds, closed sets and
timestamp format against :class:`osprey.audit.envelope.AuditEnvelope` itself,
importing osprey *in the test only*. If the envelope grows a required field,
these fail rather than letting the hook ledger quietly become a different
format that happens to share a directory.

The behavioural half runs the hooks the way Claude Code does — as subprocesses
over stdin/stdout — with a runner local to this file rather than the shared
``hook_runner`` fixture, because the audit markers
(``OSPREY_TERMINAL_USER``, ``OSPREY_POSTURE_SOURCE``, ...) are deliberately
absent from ``conftest``'s curated environment and this file owns none of it.
"""

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from osprey.audit import envelope as osprey_envelope

from .conftest import HOOKS_DIR, import_hook

# --------------------------------------------------------------------------
# Local subprocess runner
# --------------------------------------------------------------------------

#: The environment a hook subprocess starts from here. Deliberately built up
#: from nothing rather than copied from ``os.environ``: an audit record is a
#: function of the marker variables, and a leaked one from the developer's own
#: session would make a record read correct for the wrong reason.
_BASE_VARS = ("PATH", "PYTHONPATH", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "LANG")


def run_hook(hook_name, payload, env=None, cwd=None):
    """Run a hook script over stdin/stdout; return ``(rc, stdout, stderr)``."""
    child_env = {name: os.environ[name] for name in _BASE_VARS if name in os.environ}
    child_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(HOOKS_DIR / hook_name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=child_env,
        cwd=str(cwd) if cwd else None,
    )


def decision_of(result):
    """The ``permissionDecision`` a hook run put on stdout, or ``None``."""
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)["hookSpecificOutput"]["permissionDecision"]
            except Exception:
                continue
    return None


def ledger(root, actor, surface):
    """Path of one hook ledger under *root*'s audit zone."""
    return Path(root) / "var" / "audit" / actor / f"{surface}.jsonl"


def records(path):
    """Every record in a ledger, parsed."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@pytest.fixture
def repo(tmp_path):
    """A deployment repo root a hook resolves through ``get_repo_root``."""
    (tmp_path / "profile.yml").write_text("name: audit-emitter-test\n")
    return tmp_path


@pytest.fixture
def project_env(repo):
    """Environment putting a hook subprocess inside :func:`repo`, as ``alice``."""
    return {
        "CLAUDE_PROJECT_DIR": str(repo),
        "OSPREY_TERMINAL_USER": "alice",
    }


@pytest.fixture
def emitter():
    """The hook logging module, imported in-process for its helpers."""
    return import_hook("osprey_hook_log")


# --------------------------------------------------------------------------
# Schema parity with the envelope the rest of Osprey emits
# --------------------------------------------------------------------------


class TestSchemaParity:
    """The hook record and the writer record are one format, pinned both ways."""

    def test_required_fields_match_the_envelope(self, emitter):
        assert emitter.AUDIT_REQUIRED_FIELDS == osprey_envelope.AuditEnvelope.REQUIRED_FIELDS

    def test_posture_sources_are_the_envelopes_closed_set(self, emitter):
        assert emitter.AUDIT_POSTURE_SOURCES == osprey_envelope.POSTURE_SOURCES

    def test_process_is_spelled_as_the_envelope_spells_it(self, emitter):
        assert emitter.POSTURE_SOURCE_PROCESS == osprey_envelope.POSTURE_SOURCE_PROCESS

    def test_refused_is_spelled_as_the_envelope_spells_it(self, emitter):
        assert emitter.AUDIT_DECISION_REFUSED == osprey_envelope.DECISION_REFUSED

    def test_field_bounds_match_the_envelope(self, emitter):
        assert emitter.AUDIT_MAX_FIELD_CHARS == osprey_envelope.MAX_FIELD_CHARS
        assert emitter.AUDIT_MAX_DETAIL_CHARS == osprey_envelope.MAX_DETAIL_CHARS

    def test_the_append_bound_matches_the_writers(self, emitter):
        """The per-field bounds do NOT imply this one.

        Six 256-character identifiers plus a 1024-character ``detail`` encode
        to roughly 2.4 KB, so a hook that applied only the field bounds would
        emit lines past the size the writer treats as load-bearing — and hooks
        share ``var/audit/<identity>/`` with the middleware that respects it.
        """
        from osprey.audit import writer as osprey_writer

        assert emitter.AUDIT_MAX_RECORD_BYTES == osprey_writer.MAX_RECORD_BYTES

    def test_the_dropped_detail_marker_is_the_writers(self, emitter):
        """One marker, so a reader greps one string across both halves."""
        from osprey.audit import writer as osprey_writer

        assert emitter.AUDIT_DETAIL_DROPPED == osprey_writer.DETAIL_DROPPED

    def test_the_posture_env_spellings_match_the_frameworks(self, emitter):
        """The hook is the one recorder that cannot import these names.

        Every other in-process recorder is pinned symbol-to-symbol. Without
        this, renaming ``OSPREY_POSTURE_SESSION`` framework-side (where the
        spawn sites and ``posture.py`` are mutually pinned and would move
        together, staying green) leaves every hook record carrying
        ``session: null`` — the join key gone for the whole hook surface.
        """
        from osprey.audit import posture as osprey_posture

        assert emitter.POSTURE_SOURCE_ENV == osprey_posture.POSTURE_SOURCE_ENV_VAR
        assert emitter.POSTURE_SESSION_ENV == osprey_posture.POSTURE_SESSION_ENV_VAR
        assert emitter.EXECUTION_MODE_ENV == osprey_posture.POSTURE_ENV_VAR

    def test_audit_relpath_matches_the_frameworks(self, emitter):
        from osprey_connectors.workspace import AUDIT_DIR_RELPATH

        assert emitter.AUDIT_DIR_RELPATH == AUDIT_DIR_RELPATH

    def test_a_record_carries_exactly_the_envelopes_keys_in_order(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_POSTURE_SOURCE", "spawn")
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", "sess-1")
        path = emitter.emit_audit(
            "writes-check",
            {"tool_name": "mcp__controls__channel_write"},
            decision="refused",
            subject="mcp__controls__channel_write",
            reason="posture",
            detail="server=bluesky",
        )
        emitted = records(path)[0]

        reference = osprey_envelope.AuditEnvelope(
            surface="hook_writes_check",
            actor="alice",
            posture="writes",
            posture_source="spawn",
            session="sess-1",
            subject="mcp__controls__channel_write",
            decision="refused",
            reason="posture",
            detail="server=bluesky",
            ts=emitted["ts"],
        ).to_dict()

        assert list(emitted) == list(reference)
        assert emitted == reference

    def test_a_record_without_detail_omits_the_key_like_the_envelope(
        self, emitter, repo, monkeypatch
    ):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        path = emitter.emit_audit(
            "limits",
            {"tool_name": "mcp__controls__channel_write"},
            decision="refused",
            subject="mcp__controls__channel_write",
            reason="limits_violation",
        )
        emitted = records(path)[0]
        assert "detail" not in emitted
        assert list(emitted) == ["ts", *osprey_envelope.AuditEnvelope.REQUIRED_FIELDS]

    def test_the_timestamp_is_the_envelopes_format(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        stamp = records(path)[0]["ts"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stamp)
        reference = osprey_envelope.utc_timestamp()
        assert len(stamp) == len(reference)


# --------------------------------------------------------------------------
# Routing: identity, surface, path
# --------------------------------------------------------------------------


class TestRouting:
    """Where a record lands, and under whose name."""

    def test_the_terminal_user_wins_the_ladder(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "dispatch")
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert Path(path) == ledger(repo, "alice", "hook_limits")
        assert records(path)[0]["actor"] == "alice"

    def test_the_audit_identity_is_the_second_rung(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "dispatch")
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert Path(path) == ledger(repo, "dispatch", "hook_limits")

    def test_a_blank_rung_falls_through(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "   ")
        monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "dispatch")
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert Path(path) == ledger(repo, "dispatch", "hook_limits")

    def test_a_rung_that_is_not_one_path_component_falls_through(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "../elsewhere")
        monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "dispatch")
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert Path(path) == ledger(repo, "dispatch", "hook_limits")

    def test_the_actor_field_and_the_directory_are_the_same_answer(
        self, emitter, repo, monkeypatch
    ):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "bob")
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert Path(path).parent.name == records(path)[0]["actor"]

    def test_the_surface_is_the_hook_name_with_one_spelling(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        path = emitter.emit_audit(
            "writes-check", {}, decision="refused", subject="tool", reason="posture"
        )
        assert Path(path).name == "hook_writes_check.jsonl"
        assert records(path)[0]["surface"] == "hook_writes_check"

    def test_records_append_rather_than_replace(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        for subject in ("one", "two"):
            path = emitter.emit_audit(
                "limits", {}, decision="refused", subject=subject, reason="limits_violation"
            )
        assert [record["subject"] for record in records(path)] == ["one", "two"]

    def test_one_record_is_one_line(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        path = emitter.emit_audit(
            "limits",
            {},
            decision="refused",
            subject="tool",
            reason="limits_violation",
            detail="a\nb",
        )
        assert len(Path(path).read_text().splitlines()) == 1


# --------------------------------------------------------------------------
# Posture provenance — the child-side rule
# --------------------------------------------------------------------------


class TestPostureProvenance:
    """``posture_source`` is read from the marker, never inferred."""

    def test_absent_marker_is_process(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.delenv("OSPREY_POSTURE_SOURCE", raising=False)
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert records(path)[0]["posture_source"] == "process"

    def test_a_blank_marker_is_process(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_POSTURE_SOURCE", "  ")
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert records(path)[0]["posture_source"] == "process"

    def test_a_marker_outside_the_closed_set_is_process(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_POSTURE_SOURCE", "guessed")
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert records(path)[0]["posture_source"] == "process"

    @pytest.mark.parametrize("source", ["spawn", "live", "app", "process"])
    def test_a_recognised_marker_is_carried_verbatim(self, emitter, repo, monkeypatch, source):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_POSTURE_SOURCE", source)
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert records(path)[0]["posture_source"] == source

    def test_the_sandbox_posture_is_read_from_the_execution_mode(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert records(path)[0]["posture"] == "sandbox"

    @pytest.mark.parametrize("mode", ["readwrite", "", "READONLY"])
    def test_only_the_exact_readonly_string_sandboxes(self, emitter, repo, monkeypatch, mode):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", mode)
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert records(path)[0]["posture"] == "writes"

    def test_the_session_is_the_exported_posture_key(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", "chat-42")
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert records(path)[0]["session"] == "chat-42"

    def test_no_posture_key_records_a_null_session(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        record = records(path)[0]
        assert record["session"] is None
        assert "session" in record


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------


class TestBounds:
    """No caller-supplied string can inflate the ledger."""

    def test_detail_is_bounded(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        path = emitter.emit_audit(
            "limits",
            {},
            decision="refused",
            subject="tool",
            reason="limits_violation",
            detail="x" * 5000,
        )
        assert len(records(path)[0]["detail"]) == emitter.AUDIT_MAX_DETAIL_CHARS

    def test_identifier_fields_are_bounded(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="s" * 5000, reason="r" * 5000
        )
        record = records(path)[0]
        assert len(record["subject"]) == emitter.AUDIT_MAX_FIELD_CHARS
        assert len(record["reason"]) == emitter.AUDIT_MAX_FIELD_CHARS

    def test_a_long_session_marker_is_bounded(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", "s" * 5000)
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="limits_violation"
        )
        assert len(records(path)[0]["session"]) == emitter.AUDIT_MAX_FIELD_CHARS

    def test_a_record_at_every_field_bound_still_exceeds_the_append_bound(self, emitter):
        """Why the append bound is a SEPARATE check, not a consequence.

        Every field below is exactly at its own documented limit, and the line
        they encode to is over 2 KB. If this ever stops being true the degrade
        below becomes unreachable and the tests for it become decorative.
        """
        record = {
            "ts": "2026-01-01T00:00:00Z",
            "surface": "hook_limits",
            "actor": "a" * emitter.AUDIT_MAX_FIELD_CHARS,
            "posture": "sandbox",
            "posture_source": "spawn",
            "session": "s" * emitter.AUDIT_MAX_FIELD_CHARS,
            "subject": "b" * emitter.AUDIT_MAX_FIELD_CHARS,
            "decision": "refused",
            "reason": "r" * emitter.AUDIT_MAX_FIELD_CHARS,
            "detail": "d" * emitter.AUDIT_MAX_DETAIL_CHARS,
        }
        assert len(emitter._audit_encode(record)) > emitter.AUDIT_MAX_RECORD_BYTES

    def test_an_oversize_record_drops_detail_and_fits(self, emitter, repo, monkeypatch):
        """The writer's degrade order, restated: supplementary context first.

        The bound is what keeps one ``O_APPEND`` write un-interleavable by the
        other emitters sharing this directory, so a record that cannot fit
        gives up its ``detail`` rather than its atomicity.
        """
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", "s" * emitter.AUDIT_MAX_FIELD_CHARS)
        # A long-but-legal identity: the actor is a bounded envelope field too,
        # and it is what pushes an otherwise ordinary deny record over budget.
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "a" * 200)
        path = emitter.emit_audit(
            "limits",
            {},
            decision="refused",
            subject="b" * emitter.AUDIT_MAX_FIELD_CHARS,
            reason="r" * emitter.AUDIT_MAX_FIELD_CHARS,
            detail="d" * emitter.AUDIT_MAX_DETAIL_CHARS,
        )
        raw = Path(path).read_bytes()
        assert len(raw) <= emitter.AUDIT_MAX_RECORD_BYTES
        assert records(path)[0]["detail"] == emitter.AUDIT_DETAIL_DROPPED

    def test_an_in_budget_record_keeps_its_detail(self, emitter, repo, monkeypatch):
        """The negative control: the degrade fires on size, not on presence."""
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        path = emitter.emit_audit(
            "limits", {}, decision="refused", subject="tool", reason="posture", detail="key=x"
        )
        assert records(path)[0]["detail"] == "key=x"


# --------------------------------------------------------------------------
# Append mechanics
# --------------------------------------------------------------------------


class _ShortWriteOs:
    """``os`` with the first :func:`os.write` truncated, everything else real.

    Substituted for the emitter module's own ``os`` global so only this
    emitter's writes are affected — patching :func:`os.write` itself would
    truncate pytest's own output too.
    """

    def __init__(self, real, cut):
        self._real = real
        self._cut = cut
        self.writes = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def write(self, fd, data):
        self.writes.append(data)
        if len(self.writes) == 1:
            return self._real.write(fd, data[: self._cut])
        return self._real.write(fd, data)


class TestAppendMechanics:
    """One torn write costs one record, exactly as it does in the writer."""

    def test_a_short_write_terminates_its_own_fragment(self, emitter, repo, monkeypatch):
        """Otherwise the NEXT record is appended onto the fragment and both die.

        A filesystem that accepts a partial write without raising leaves half a
        JSON object with no newline. The writer answers this by writing the
        terminator while the descriptor is still open; the hook shares the
        directory and must not be the emitter that swallows its successor.
        """
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")

        shim = _ShortWriteOs(os, cut=20)
        with pytest.MonkeyPatch.context() as short_write:
            short_write.setattr(emitter, "os", shim)
            torn = emitter.emit_audit(
                "limits", {}, decision="refused", subject="torn", reason="posture"
            )

        assert torn is None, "a torn record must not report itself as stored"
        assert len(shim.writes) == 2, "the fragment was left unterminated"
        assert shim.writes[1] == b"\n"

        assert (
            emitter.emit_audit("limits", {}, decision="refused", subject="next", reason="posture")
            is not None
        )

        lines = ledger(repo, "alice", "hook_limits").read_bytes().split(b"\n")
        with pytest.raises(ValueError):
            json.loads(lines[0])
        assert json.loads(lines[1])["subject"] == "next"


# --------------------------------------------------------------------------
# Never-raise
# --------------------------------------------------------------------------


class TestNeverRaises:
    """A hook's decision must survive anything the audit trail cannot do."""

    @pytest.mark.skipif(
        os.name != "posix" or os.geteuid() == 0,
        reason="root writes through a mode-500 directory regardless",
    )
    def test_an_unwritable_audit_zone_returns_none(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        blocked = repo / "var"
        blocked.mkdir()
        blocked.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            assert (
                emitter.emit_audit(
                    "limits", {}, decision="refused", subject="tool", reason="limits_violation"
                )
                is None
            )
        finally:
            blocked.chmod(stat.S_IRWXU)

    def test_an_audit_zone_blocked_by_a_file_returns_none(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        (repo / "var").write_text("not a directory")
        assert (
            emitter.emit_audit(
                "limits", {}, decision="refused", subject="tool", reason="limits_violation"
            )
            is None
        )

    def test_a_repo_root_that_cannot_be_resolved_returns_none(self, emitter, monkeypatch):
        monkeypatch.setattr(emitter, "get_repo_root", lambda *_: (_ for _ in ()).throw(OSError))
        assert (
            emitter.emit_audit(
                "limits", {}, decision="refused", subject="tool", reason="limits_violation"
            )
            is None
        )

    def test_an_unserialisable_detail_returns_none_rather_than_raising(
        self, emitter, repo, monkeypatch
    ):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        assert (
            emitter.emit_audit(
                "limits",
                {},
                decision="refused",
                subject="tool",
                reason="limits_violation",
                detail=object(),
            )
            is None
        )

    def test_a_non_mapping_hook_input_does_not_raise(self, emitter, repo, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
        path = emitter.emit_audit(
            "limits", None, decision="refused", subject="tool", reason="limits_violation"
        )
        assert records(path)[0]["subject"] == "tool"


# --------------------------------------------------------------------------
# The deny and ask sites, run as Claude Code runs them
# --------------------------------------------------------------------------


class TestWritesCheckDenies:
    """Both posture refusals and the kill switch record."""

    def test_the_posture_deny_records(self, repo, project_env):
        env = {**project_env, "OSPREY_EXECUTION_MODE": "readonly"}
        result = run_hook(
            "osprey_writes_check.py",
            {"tool_name": "mcp__controls__channel_write", "tool_input": {}},
            env=env,
        )
        assert result.returncode == 0
        assert decision_of(result) == "deny"
        record = records(ledger(repo, "alice", "hook_writes_check"))[0]
        assert record["decision"] == "refused"
        assert record["reason"] == "posture"
        assert record["subject"] == "mcp__controls__channel_write"
        assert record["posture"] == "sandbox"

    def test_a_wildcard_matched_tool_records_the_posture_deny(self, repo, project_env, tmp_path):
        """A facility-custom server's tools reach the gate through the registry's
        ``mcp__<server>__.*`` matcher, and refuse under the same reason word."""
        hook_config = tmp_path / "hook_config.json"
        hook_config.write_text(json.dumps({"write_tools": ["mcp__bluesky__.*"]}))
        env = {
            **project_env,
            "OSPREY_EXECUTION_MODE": "readonly",
            "OSPREY_HOOK_CONFIG": str(hook_config),
        }
        result = run_hook(
            "osprey_writes_check.py",
            {"tool_name": "mcp__bluesky__queue_add", "tool_input": {}},
            env=env,
        )
        assert decision_of(result) == "deny"
        record = records(ledger(repo, "alice", "hook_writes_check"))[0]
        assert record["reason"] == "posture"
        assert record["subject"] == "mcp__bluesky__queue_add"

    def test_the_kill_switch_deny_records(self, repo, project_env):
        (repo / "config.yml").write_text("control_system:\n  writes_enabled: false\n")
        env = {**project_env, "OSPREY_CONFIG": str(repo / "config.yml")}
        result = run_hook(
            "osprey_writes_check.py",
            {"tool_name": "mcp__controls__channel_write", "tool_input": {}},
            env=env,
        )
        assert decision_of(result) == "deny"
        record = records(ledger(repo, "alice", "hook_writes_check"))[0]
        assert record["decision"] == "refused"
        assert record["reason"] == "writes_disabled"
        assert record["posture"] == "writes"

    def test_an_allowed_call_records_nothing(self, repo, project_env):
        (repo / "config.yml").write_text("control_system:\n  writes_enabled: true\n")
        env = {**project_env, "OSPREY_CONFIG": str(repo / "config.yml")}
        result = run_hook(
            "osprey_writes_check.py",
            {"tool_name": "mcp__controls__channel_write", "tool_input": {}},
            env=env,
        )
        assert result.returncode == 0
        assert not (repo / "var" / "audit").exists()

    def test_the_posture_deny_survives_an_unwritable_audit_zone(self, repo, project_env):
        blocked = repo / "var"
        blocked.mkdir()
        blocked.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            result = run_hook(
                "osprey_writes_check.py",
                {"tool_name": "mcp__controls__channel_write", "tool_input": {}},
                env={**project_env, "OSPREY_EXECUTION_MODE": "readonly"},
            )
            assert result.returncode == 0
            assert decision_of(result) == "deny"
        finally:
            blocked.chmod(stat.S_IRWXU)

    def test_the_record_does_not_reach_stdout(self, repo, project_env):
        result = run_hook(
            "osprey_writes_check.py",
            {"tool_name": "mcp__controls__channel_write", "tool_input": {}},
            env={**project_env, "OSPREY_EXECUTION_MODE": "readonly"},
        )
        assert json.loads(result.stdout.strip())["hookSpecificOutput"]["permissionDecision"] == (
            "deny"
        )


class TestOtherHookDenies:
    """Every other hook that refuses records the same way."""

    def test_the_memory_guard_deny_records(self, repo, project_env):
        result = run_hook(
            "osprey_memory_guard.py",
            {"tool_name": "Write", "tool_input": {"file_path": str(repo / "secrets.txt")}},
            env=project_env,
        )
        assert decision_of(result) == "deny"
        record = records(ledger(repo, "alice", "hook_memory_guard"))[0]
        assert record["decision"] == "refused"
        assert record["subject"] == str(repo / "secrets.txt")

    def test_the_memory_guard_records_a_missing_path(self, repo, project_env):
        result = run_hook(
            "osprey_memory_guard.py",
            {"tool_name": "Write", "tool_input": {}},
            env=project_env,
        )
        assert decision_of(result) == "deny"
        record = records(ledger(repo, "alice", "hook_memory_guard"))[0]
        assert record["reason"] == "no_path"
        assert record["subject"] == "Write"

    def test_an_allowed_memory_write_records_nothing(
        self, repo, project_env, hook_home, monkeypatch
    ):
        """The negative control the emitter needs: an ALLOW writes no record.

        The target is derived through the guard's own :func:`resolve_memory_dir`
        rather than spelled out here. A hand-written path is how this test used
        to be vacuous — ``~/.claude/notes.md`` is not the allowed directory
        (that is ``<config>/projects/<slug>/memory/``), so the hook denied, the
        assertion sat inside an ``if`` that never ran, and an audit record on
        the allow branch would have gone unnoticed.

        ``CLAUDE_CONFIG_DIR`` is set to a fully resolved path — the container's
        own shape — because the guard compares the RESOLVED target against an
        unresolved memory directory, and a ``$HOME`` reached through a symlink
        would make the two disagree for reasons that have nothing to do with
        the rule under test.
        """
        config_dir = hook_home.resolve() / ".claude"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
        memory = import_hook("osprey_memory_guard").resolve_memory_dir(str(repo))
        memory.mkdir(parents=True, exist_ok=True)

        result = run_hook(
            "osprey_memory_guard.py",
            {"tool_name": "Write", "tool_input": {"file_path": str(memory / "notes.md")}},
            env={
                **project_env,
                "HOME": str(hook_home),
                "USERPROFILE": str(hook_home),
                "CLAUDE_CONFIG_DIR": str(config_dir),
            },
        )
        assert result.returncode == 0
        assert decision_of(result) == "allow"
        assert not (repo / "var" / "audit").exists()

    def test_the_limits_deny_records(self, repo, project_env):
        database = repo / "channel_limits.json"
        database.write_text(
            json.dumps({"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}})
        )
        config = repo / "config.yml"
        config.write_text(
            yaml.dump(
                {
                    "control_system": {
                        "type": "mock",
                        "writes_enabled": True,
                        "limits_checking": {
                            "enabled": True,
                            "database_path": str(database),
                            "allow_unlisted_channels": False,
                        },
                    }
                }
            )
        )
        env = {**project_env, "OSPREY_CONFIG": str(config), "CONFIG_FILE": str(config)}
        result = run_hook(
            "osprey_limits.py",
            {
                "tool_name": "mcp__controls__channel_write",
                "tool_input": {"operations": [{"channel": "TEST:PV", "value": 999.0}]},
            },
            env=env,
        )
        assert decision_of(result) == "deny", result.stderr
        record = records(ledger(repo, "alice", "hook_limits"))[0]
        assert record["decision"] == "refused"
        assert record["reason"] == "limits_violation"
        assert record["subject"] == "mcp__controls__channel_write"
        assert record["detail"] == "violations=1"

    def test_the_limits_record_carries_no_channel_value(self, repo, project_env):
        """The violation lines name the refused VALUE; the ledger never does."""
        database = repo / "channel_limits.json"
        database.write_text(
            json.dumps({"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}})
        )
        config = repo / "config.yml"
        config.write_text(
            yaml.dump(
                {
                    "control_system": {
                        "type": "mock",
                        "writes_enabled": True,
                        "limits_checking": {
                            "enabled": True,
                            "database_path": str(database),
                            "allow_unlisted_channels": False,
                        },
                    }
                }
            )
        )
        env = {**project_env, "OSPREY_CONFIG": str(config), "CONFIG_FILE": str(config)}
        result = run_hook(
            "osprey_limits.py",
            {
                "tool_name": "mcp__controls__channel_write",
                "tool_input": {"operations": [{"channel": "TEST:PV", "value": 987654.0}]},
            },
            env=env,
        )
        assert decision_of(result) == "deny", result.stderr
        assert "987654" not in ledger(repo, "alice", "hook_limits").read_text()

    def test_the_approval_ask_records(self, repo, project_env, tmp_path):
        config = repo / "config.yml"
        config.write_text(
            yaml.dump(
                {
                    "approval": {
                        "enabled": True,
                        "default_policy": "selective",
                        "requires_approval": ["channel_write", "execute"],
                    },
                    "control_system": {"writes_enabled": True},
                }
            )
        )
        hook_config = tmp_path / "approval_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "server_prefixes": ["mcp__controls__", "mcp__python__"],
                    "approval_prefixes": ["mcp__controls__", "mcp__python__"],
                }
            )
        )
        env = {
            **project_env,
            "OSPREY_CONFIG": str(config),
            "CONFIG_FILE": str(config),
            "OSPREY_HOOK_CONFIG": str(hook_config),
        }
        result = run_hook(
            "osprey_approval.py",
            {
                "tool_name": "mcp__controls__channel_write",
                "tool_input": {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
            },
            env=env,
        )
        assert decision_of(result) == "ask", result.stderr
        record = records(ledger(repo, "alice", "hook_approval"))[0]
        assert record["decision"] == "ask"
        assert record["subject"] == "mcp__controls__channel_write"
        assert record["reason"] == "approval_required"

    def test_an_allowed_call_records_no_ask(self, repo, project_env, tmp_path):
        config = repo / "config.yml"
        config.write_text(
            yaml.dump(
                {
                    "approval": {"enabled": False},
                    "control_system": {"writes_enabled": True},
                }
            )
        )
        hook_config = tmp_path / "approval_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "server_prefixes": ["mcp__controls__", "mcp__python__"],
                    "approval_prefixes": ["mcp__controls__", "mcp__python__"],
                }
            )
        )
        env = {
            **project_env,
            "OSPREY_CONFIG": str(config),
            "CONFIG_FILE": str(config),
            "OSPREY_HOOK_CONFIG": str(hook_config),
        }
        result = run_hook(
            "osprey_approval.py",
            {
                "tool_name": "mcp__controls__channel_write",
                "tool_input": {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
            },
            env=env,
        )
        assert result.returncode == 0
        assert decision_of(result) != "ask"
        assert not ledger(repo, "alice", "hook_approval").exists()

    def test_the_ask_decision_word_is_outside_the_envelopes_two(self, emitter):
        """``ask`` is the third word this surface needs; the envelope allows it."""
        assert emitter.AUDIT_DECISION_ASK not in osprey_envelope.DECISIONS
        assert (
            osprey_envelope.AuditEnvelope(
                surface="hook_approval",
                actor="alice",
                posture="writes",
                posture_source="process",
                session=None,
                subject="tool",
                decision=emitter.AUDIT_DECISION_ASK,
                reason="approval_required",
            ).decision
            == "ask"
        )
