"""Tests for the unified audit envelope.

Covers the four things downstream emitters and the writer rely on: that the
closed set of ``posture_source`` values is enforced rather than guessed, that
every field is bounded by construction, that ``source`` cannot leak onto a
non-executor surface, and that ``to_dict`` emits the exact JSON shape the
per-surface JSONL files are made of — required keys always, optionals only
when set.

The module also stays a stdlib-only leaf: the MCP middleware, the HTTP layer
and the hooks all import it, so it must not drag the ``osprey`` package tree
(or an import cycle) along behind it.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import osprey.audit.envelope as envelope_module
from osprey.audit.envelope import (
    DECISION_ALLOWED,
    DECISION_REFUSED,
    DECISIONS,
    MAX_DETAIL_CHARS,
    MAX_FIELD_CHARS,
    MAX_SOURCE_CHARS,
    POSTURE_SOURCE_APP,
    POSTURE_SOURCE_LIVE,
    POSTURE_SOURCE_PROCESS,
    POSTURE_SOURCE_SPAWN,
    POSTURE_SOURCES,
    SURFACE_EXECUTOR,
    AuditEnvelope,
    utc_timestamp,
)

pytestmark = pytest.mark.unit


def make_envelope(**overrides: object) -> AuditEnvelope:
    """Build a minimal legal envelope, with *overrides* applied.

    Every test that is not about a specific field wants the same boring
    record; spelling it once keeps each test's diff to the field it is about.
    """
    kwargs: dict = {
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
    return AuditEnvelope(**kwargs)  # type: ignore[arg-type]


class TestClosedSets:
    """``posture_source`` is stated by the caller and checked here."""

    def test_posture_sources_are_the_four_declared_values(self) -> None:
        """The set is the contract the spawn sites and emitters share."""
        assert POSTURE_SOURCES == (
            POSTURE_SOURCE_SPAWN,
            POSTURE_SOURCE_LIVE,
            POSTURE_SOURCE_APP,
            POSTURE_SOURCE_PROCESS,
        )
        assert set(POSTURE_SOURCES) == {"spawn", "live", "app", "process"}

    @pytest.mark.parametrize("value", POSTURE_SOURCES)
    def test_every_declared_source_is_accepted(self, value: str) -> None:
        """Parametrized off the module's own tuple, so a new value is covered."""
        assert make_envelope(posture_source=value).posture_source == value

    @pytest.mark.parametrize("value", ["", "SPAWN", "session", "unknown", "sandbox", "writes"])
    def test_unknown_source_is_refused(self, value: str) -> None:
        """Including the posture values themselves — provenance is not the posture."""
        with pytest.raises(ValueError, match="posture_source"):
            make_envelope(posture_source=value)

    def test_decision_vocabulary_is_advisory(self) -> None:
        """Canonical spellings exist, but a surface may need its own word."""
        assert DECISIONS == (DECISION_ALLOWED, DECISION_REFUSED)
        assert make_envelope(decision="clamped").decision == "clamped"


class TestRequiredFields:
    """A record that names nobody is worse than no record."""

    @pytest.mark.parametrize(
        "name", ["surface", "actor", "posture", "subject", "decision", "reason"]
    )
    def test_empty_required_field_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match=name):
            make_envelope(**{name: ""})

    def test_session_may_be_none(self) -> None:
        """Null only where no posture-store key exists; posture_source says why."""
        envelope = make_envelope(session=None, posture_source=POSTURE_SOURCE_PROCESS)
        assert envelope.session is None
        assert envelope.to_dict()["session"] is None

    def test_required_fields_match_the_dataclass(self) -> None:
        """``REQUIRED_FIELDS`` drives ``to_dict``; it must not drift from the fields."""
        declared = {f.name for f in fields(AuditEnvelope)}
        assert set(AuditEnvelope.REQUIRED_FIELDS) <= declared

    def test_fields_are_keyword_only(self) -> None:
        """``actor`` and ``subject`` are both bare strings; a swap must not compile."""
        with pytest.raises(TypeError):
            AuditEnvelope("http_config", "alice")  # type: ignore[misc]

    def test_envelope_is_frozen(self) -> None:
        """A record cannot be edited between the decision and the write."""
        with pytest.raises(FrozenInstanceError):
            make_envelope().surface = "other"  # type: ignore[misc]


class TestBounds:
    """Bounded by construction, so the writer never re-applies a limit."""

    def test_declared_bounds(self) -> None:
        assert MAX_FIELD_CHARS == 256
        assert MAX_DETAIL_CHARS == 1024
        assert MAX_SOURCE_CHARS == 8000

    @pytest.mark.parametrize(
        "name", ["surface", "actor", "posture", "subject", "decision", "reason", "role", "session"]
    )
    def test_identifier_fields_are_truncated_silently(self, name: str) -> None:
        """An identifier this long is malformed; flagging it would cost every record a key."""
        envelope = make_envelope(**{name: "x" * (MAX_FIELD_CHARS + 500)})
        assert getattr(envelope, name) == "x" * MAX_FIELD_CHARS
        assert "source_truncated" not in envelope.to_dict()

    def test_detail_is_truncated_without_a_flag(self) -> None:
        """``detail`` is supplementary context, so a cut one is still usable."""
        envelope = make_envelope(detail="d" * (MAX_DETAIL_CHARS + 1))
        record = envelope.to_dict()
        assert envelope.detail == "d" * MAX_DETAIL_CHARS
        assert record["detail"] == "d" * MAX_DETAIL_CHARS
        assert "detail_truncated" not in record

    def test_detail_at_the_limit_is_kept_whole(self) -> None:
        envelope = make_envelope(detail="d" * MAX_DETAIL_CHARS)
        assert envelope.detail == "d" * MAX_DETAIL_CHARS

    def test_source_is_truncated_and_flagged(self) -> None:
        """A cut script that did not say so would read like the whole script."""
        envelope = make_envelope(surface=SURFACE_EXECUTOR, source="s" * (MAX_SOURCE_CHARS + 1))
        assert envelope.source == "s" * MAX_SOURCE_CHARS
        assert envelope.source_truncated is True
        assert envelope.to_dict()["source_truncated"] is True

    def test_short_source_is_not_flagged(self) -> None:
        envelope = make_envelope(surface=SURFACE_EXECUTOR, source="import os")
        assert envelope.source_truncated is False
        assert "source_truncated" not in envelope.to_dict()

    def test_source_truncated_is_derived_not_supplied(self) -> None:
        """An emitter that could set the flag independently could lie with it."""
        with pytest.raises(TypeError):
            make_envelope(surface=SURFACE_EXECUTOR, source="x", source_truncated=True)


class TestSourceIsExecutorOnly:
    """The one field that carries a payload is fenced to the one surface that needs it."""

    def test_executor_surface_may_carry_source(self) -> None:
        envelope = make_envelope(surface=SURFACE_EXECUTOR, source="caput('X', 1)")
        assert envelope.source == "caput('X', 1)"

    @pytest.mark.parametrize("surface", ["http_config", "setup_patch", "scaffold_restore", "hook"])
    def test_other_surfaces_may_not(self, surface: str) -> None:
        with pytest.raises(ValueError, match=SURFACE_EXECUTOR):
            make_envelope(surface=surface, source="caput('X', 1)")

    def test_other_surfaces_are_fine_without_source(self) -> None:
        assert make_envelope(surface="setup_patch").source is None


class TestToDict:
    """The exact JSON object a line of ``var/audit/<identity>/<surface>.jsonl`` holds."""

    def test_minimal_record_shape(self) -> None:
        record = make_envelope().to_dict()
        assert set(record) == {
            "ts",
            "surface",
            "actor",
            "posture",
            "posture_source",
            "session",
            "subject",
            "decision",
            "reason",
        }

    def test_ts_is_first_key(self) -> None:
        """Insertion order is what a human scanning the JSONL reads first."""
        assert next(iter(make_envelope().to_dict())) == "ts"

    def test_optional_keys_are_omitted_when_unset(self) -> None:
        record = make_envelope().to_dict()
        for key in ("detail", "role", "source", "source_truncated"):
            assert key not in record

    def test_optional_keys_are_present_when_set(self) -> None:
        record = make_envelope(
            surface=SURFACE_EXECUTOR, detail="tool=execute", role="operator", source="x = 1"
        ).to_dict()
        assert record["detail"] == "tool=execute"
        assert record["role"] == "operator"
        assert record["source"] == "x = 1"

    def test_record_is_json_serializable(self) -> None:
        """The writer appends ``json.dumps(record)``; nothing here may defeat it."""
        record = make_envelope(session=None, role="operator").to_dict()
        assert json.loads(json.dumps(record)) == record

    def test_posture_source_and_session_are_top_level(self) -> None:
        """Never smuggled into the budgeted ``detail`` — they are the join keys."""
        record = make_envelope(posture_source=POSTURE_SOURCE_SPAWN, session="pty-3").to_dict()
        assert record["posture_source"] == "spawn"
        assert record["session"] == "pty-3"


class TestTimestamp:
    """Second resolution with a literal Z, matching the ledgers this replaces."""

    def test_format(self) -> None:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_timestamp())

    def test_envelope_stamps_itself(self) -> None:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", make_envelope().ts)

    def test_supplied_ts_is_kept(self) -> None:
        """The writer may replay a record; a stamped one is not re-stamped."""
        assert make_envelope(ts="2026-01-01T00:00:00Z").ts == "2026-01-01T00:00:00Z"


class TestLeafModule:
    """Importable without the ``osprey`` package tree, like ``utils.sensitive_env``."""

    def test_imports_only_stdlib(self) -> None:
        path = Path(envelope_module.__file__ or "")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not [name for name in imported if name.split(".")[0] == "osprey"]

    def test_importable_standalone(self) -> None:
        """No writer, no config, no import cycle — just the schema."""
        result = subprocess.run(
            [sys.executable, "-c", "import osprey.audit.envelope"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
