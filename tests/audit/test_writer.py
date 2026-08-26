"""Tests for the unified audit ledger writer.

The writer is the one place a record becomes a file, so these pin the four
properties every emitter downstream is allowed to assume:

* **Routing is (identity, surface).** One file per pair, so a record's home is
  a function of who acted and which surface decided — never of the emitter.
* **The maintenance marker moves the FILE, not the record.** A root-run phase
  files under ``<identity>/maintenance.jsonl`` while its envelopes keep saying
  ``scaffold_restore``; that is what keeps one uid per file.
* **The append is one bounded ``write()``.** One syscall, one line, no fsync,
  ≤2 KB except for the documented executor-source case.
* **Nothing here can cost the operation.** A broken audit zone, an unwritable
  file, even an invalid envelope, degrade to ``None`` and a log line.

The cross-module spelling pin at the bottom is the same settlement
``tests/registry/test_marker_nonpinnability.py`` makes for the other three
markers: the registry reserved ``OSPREY_AUDIT_WRITER`` before its assigner
existed, and this is the module that finally reads it.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from osprey.audit import writer
from osprey.audit.envelope import (
    DECISION_ALLOWED,
    DECISION_REFUSED,
    MAX_SOURCE_CHARS,
    POSTURE_SOURCE_APP,
    POSTURE_SOURCE_PROCESS,
    SURFACE_EXECUTOR,
    AuditEnvelope,
)
from osprey.utils.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV, UNKNOWN_IDENTITY

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture
def audit_root(tmp_path, monkeypatch):
    """Redirect the audit zone at its single seam, and clear the markers.

    The env is cleared rather than trusted: a developer machine that happens
    to export ``OSPREY_TERMINAL_USER`` would otherwise silently re-route every
    record in this file.
    """
    for marker in (TERMINAL_USER_ENV, AUDIT_IDENTITY_ENV, writer.AUDIT_WRITER_ENV):
        monkeypatch.delenv(marker, raising=False)
    target = tmp_path / "var" / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: target)
    return target


def _records(path: Path) -> list[dict]:
    """The JSON objects in a JSONL ledger, in file order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


# --------------------------------------------------------------------------


class TestLedgerRouting:
    """Key = (identity, surface): ``var/audit/<identity>/<surface>.jsonl``."""

    def test_path_is_identity_then_surface(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "dispatch-worker-1")

        assert writer.ledger_path("http_config") == (
            audit_root / "dispatch-worker-1" / "http_config.jsonl"
        )

    def test_record_lands_at_that_path(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "sidecar")

        written = writer.record_envelope(make_envelope())

        assert written == audit_root / "sidecar" / "http_config.jsonl"
        assert written.is_file()

    def test_terminal_user_outranks_the_service_identity(self, audit_root, monkeypatch):
        """The writer's file ladder is the identity module's ladder, not a copy."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "shared-service")
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")

        assert writer.ledger_path("http_config").parent.name == "alice"

    def test_unresolvable_identity_still_files_the_record(self, audit_root, monkeypatch):
        """An unnamed account degrades the attribution, never the record."""
        monkeypatch.setattr(writer, "acting_identity", lambda: UNKNOWN_IDENTITY)

        written = writer.record_envelope(make_envelope())

        assert written == audit_root / UNKNOWN_IDENTITY / "http_config.jsonl"

    def test_two_surfaces_one_identity_are_two_files(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        writer.record_envelope(make_envelope(surface="http_config"))
        writer.record_envelope(make_envelope(surface="scaffold_restore"))

        assert sorted(p.name for p in (audit_root / "alice").iterdir()) == [
            "http_config.jsonl",
            "scaffold_restore.jsonl",
        ]

    def test_two_identities_one_surface_are_two_files(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        writer.record_envelope(make_envelope())
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "bob")
        writer.record_envelope(make_envelope())

        assert sorted(p.name for p in audit_root.iterdir()) == ["alice", "bob"]

    def test_the_identity_is_read_per_call(self, audit_root, monkeypatch):
        """Not cached at import: the markers are set per process, and a value
        frozen at import time would be whatever the first importer saw."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "before")
        first = writer.record_envelope(make_envelope())
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "after")
        second = writer.record_envelope(make_envelope())

        assert first.parent.name == "before"
        assert second.parent.name == "after"

    def test_the_directory_is_the_process_identity_not_the_actor(self, audit_root, monkeypatch):
        """A service that acts for someone files under ITSELF and names them.

        Each container binds only its own ``var/audit/<identity>/`` read-write,
        so routing on a caller-supplied ``actor`` would file records where the
        process cannot write them. The sidecar's refusal of alice's login is
        the shape: ``sidecar/`` on disk, ``alice`` in the record.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "sidecar")

        written = writer.record_envelope(make_envelope(actor="alice"))

        assert written == audit_root / "sidecar" / "http_config.jsonl"
        (record,) = _records(written)
        assert record["actor"] == "alice"


class TestSuppliedIdentity:
    """``ledger_path(identity=...)`` is a shortcut, not a hole in the zone."""

    def test_one_path_component_is_honoured(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        assert writer.ledger_path("http_config", identity="bob") == (
            audit_root / "bob" / "http_config.jsonl"
        )

    @pytest.mark.parametrize(
        "identity",
        ["../../../tmp/pwned", "..", ".", "nested/name", "with\\sep", "with\x00nul", "", "   "],
        ids=["parent", "dotdot", "dot", "separator", "backslash", "nul", "empty", "blank"],
    )
    def test_anything_that_is_not_one_component_falls_back_to_the_ladder(
        self, audit_root, monkeypatch, identity
    ):
        """The value a caller has in hand is the envelope's ``actor``, which
        the envelope bounds but never validates as a path component. An
        emitter that reads a user out of a request header must not be able to
        walk the ledger out of the audit zone.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        path = writer.ledger_path("http_config", identity=identity)

        assert path == audit_root / "alice" / "http_config.jsonl"
        assert audit_root in path.parents

    def test_an_identity_too_long_for_a_directory_name_is_shortened(self, audit_root, monkeypatch):
        """The other half of the surface's byte cut, on the other component.

        The envelope bounds an identifier at 256 characters — already one byte
        past what a file name may be — so an unbounded identity produces a
        directory ``mkdir`` refuses with ``ENAMETOOLONG``. The never-raises
        boundary would swallow that, which costs EVERY record the process
        writes rather than one file's name.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        path = writer.ledger_path("http_config", identity="x" * 300)

        assert path.parent.name == "x" * writer.MAX_LEDGER_STEM_BYTES
        assert len(path.parent.name.encode("utf-8")) <= 255
        assert audit_root in path.parents

    def test_a_record_under_a_too_long_identity_still_lands(self, audit_root, monkeypatch):
        """Behavioural half: the record is stored, under the shortened name."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "x" * 300)

        written = writer.record_envelope(make_envelope())

        assert written is not None, "a long identity dropped the record"
        assert written.is_file()
        assert len(written.parent.name.encode("utf-8")) <= 255

    def test_a_non_ascii_identity_is_shortened_by_bytes(self, audit_root, monkeypatch):
        """``NAME_MAX`` counts bytes, so the cut is on the encoded form and
        never leaves a partial character behind."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "é" * 200)

        written = writer.record_envelope(make_envelope())

        assert written is not None
        assert written.is_file()
        assert len(written.parent.name.encode("utf-8")) <= 255
        assert set(written.parent.name) == {"é"}, "the cut split a character"

    def test_the_record_lands_in_the_zone_too(self, audit_root, monkeypatch):
        """Not just the path helper: nothing escapes on the writing path."""
        monkeypatch.setattr(writer, "acting_identity", lambda: "alice")
        escape = audit_root.parent / "http_config.jsonl"

        written = writer.record_envelope(make_envelope(actor="../../../tmp/pwned"))

        assert written == audit_root / "alice" / "http_config.jsonl"
        assert not escape.exists()


class TestRecordShape:
    """The line is exactly the envelope's own JSON, one per line."""

    def test_line_is_the_envelope_to_dict(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        envelope = make_envelope(detail="channel=config", role="operator")

        written = writer.record_envelope(envelope)

        (record,) = _records(written)
        assert record == envelope.to_dict()

    def test_session_is_present_even_when_null(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        written = writer.record_envelope(make_envelope(session=None))

        (record,) = _records(written)
        assert record["session"] is None

    def test_appends_rather_than_replaces(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        for index in range(3):
            writer.record_envelope(make_envelope(subject=f"key.{index}"))

        records = _records(audit_root / "alice" / "http_config.jsonl")
        assert [r["subject"] for r in records] == ["key.0", "key.1", "key.2"]

    def test_existing_content_is_preserved(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        path = audit_root / "alice" / "http_config.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"ts": "old"}\n')

        writer.record_envelope(make_envelope())

        assert [r.get("ts") for r in _records(path)][0] == "old"
        assert len(_records(path)) == 2

    def test_every_line_ends_with_a_newline(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        written = writer.record_envelope(make_envelope())

        assert written.read_bytes().endswith(b"\n")


class TestKwargsEntryPoint:
    """``record(**fields)`` builds the envelope inside the never-raises boundary."""

    def test_builds_and_writes(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        written = writer.record(
            surface="mcp_python",
            posture="sandbox",
            posture_source=POSTURE_SOURCE_PROCESS,
            session=None,
            subject="mcp__python__execute",
            decision=DECISION_ALLOWED,
            reason="allowed",
        )

        (record,) = _records(written)
        assert record["subject"] == "mcp__python__execute"

    def test_actor_defaults_to_the_acting_identity(self, audit_root, monkeypatch):
        """No emitter re-implements the ladder: an omitted actor is resolved
        by the same helper that names the file."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "dispatch-worker-1")

        written = writer.record(
            surface="mcp_python",
            posture="sandbox",
            posture_source=POSTURE_SOURCE_PROCESS,
            session=None,
            subject="mcp__python__execute",
            decision=DECISION_REFUSED,
            reason="posture",
        )

        (record,) = _records(written)
        assert record["actor"] == "dispatch-worker-1"

    def test_an_explicit_actor_wins(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "sidecar")

        written = writer.record(
            surface="login",
            actor="sidecar",
            posture="sandbox",
            posture_source=POSTURE_SOURCE_APP,
            session=None,
            subject="alice",
            decision=DECISION_REFUSED,
            reason="claim_mismatch",
        )

        (record,) = _records(written)
        assert record["actor"] == "sidecar"

    def test_an_invalid_envelope_degrades_instead_of_raising(self, audit_root, monkeypatch):
        """Construction validates; the writer's boundary is what keeps a
        construction bug from failing the operation being described."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        assert (
            writer.record(
                surface="http_config",
                posture="sandbox",
                posture_source="made-up",
                session=None,
                subject="key",
                decision=DECISION_REFUSED,
                reason="protected_key",
            )
            is None
        )
        assert not audit_root.exists()

    def test_a_missing_required_field_degrades(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        assert writer.record(surface="http_config") is None

    def test_source_outside_the_executor_surface_degrades(self, audit_root, monkeypatch):
        """The schema's payload guard survives the writer's boundary: the
        record is dropped rather than the leak being written."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        assert (
            writer.record(
                surface="http_config",
                posture="sandbox",
                posture_source=POSTURE_SOURCE_APP,
                session=None,
                subject="key",
                decision=DECISION_REFUSED,
                reason="protected_key",
                source="caput('PV', 1)",
            )
            is None
        )
        assert not audit_root.exists()


class TestMaintenanceRouting:
    """The process marker moves the FILE; the envelope keeps its surface."""

    def test_marker_routes_to_the_maintenance_ledger(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        monkeypatch.setenv(writer.AUDIT_WRITER_ENV, writer.WRITER_MAINTENANCE)

        written = writer.record_envelope(make_envelope(surface="scaffold_restore"))

        assert written == audit_root / "alice" / "maintenance.jsonl"

    def test_the_record_still_names_the_real_surface(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        monkeypatch.setenv(writer.AUDIT_WRITER_ENV, writer.WRITER_MAINTENANCE)

        written = writer.record_envelope(make_envelope(surface="scaffold_restore"))

        (record,) = _records(written)
        assert record["surface"] == "scaffold_restore"

    def test_the_marker_does_not_move_the_identity(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        monkeypatch.setenv(writer.AUDIT_WRITER_ENV, writer.WRITER_MAINTENANCE)

        assert writer.ledger_path("scaffold_restore").parent.name == "alice"

    def test_every_surface_in_the_phase_lands_in_one_file(self, audit_root, monkeypatch):
        """The marker is a process context, not a per-call argument: a whole
        maintenance phase collapses into one file whatever it touches."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        monkeypatch.setenv(writer.AUDIT_WRITER_ENV, writer.WRITER_MAINTENANCE)

        writer.record_envelope(make_envelope(surface="scaffold_restore"))
        writer.record_envelope(make_envelope(surface="setup_patch"))

        assert [p.name for p in (audit_root / "alice").iterdir()] == ["maintenance.jsonl"]
        assert len(_records(audit_root / "alice" / "maintenance.jsonl")) == 2

    @pytest.mark.parametrize("value", ["", "   "], ids=["empty", "blank"])
    def test_an_unset_shaped_marker_routes_normally(self, audit_root, monkeypatch, value):
        """A rendered-but-blank ``environment:`` entry is the unset case."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        monkeypatch.setenv(writer.AUDIT_WRITER_ENV, value)

        assert writer.ledger_path("http_config").name == "http_config.jsonl"

    def test_an_unrecognised_writer_name_routes_normally(self, audit_root, monkeypatch):
        """The marker's values are a closed set: an inherited stray value
        must not silently open a new ledger nobody reads."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        monkeypatch.setenv(writer.AUDIT_WRITER_ENV, "somebody-else")

        written = writer.record_envelope(make_envelope())

        assert written == audit_root / "alice" / "http_config.jsonl"

    def test_maintenance_is_the_only_recognised_writer(self):
        assert writer.WRITER_CONTEXTS == (writer.WRITER_MAINTENANCE,)


class TestOneUidPerFile:
    """The regression the invariant exists for: root and app never share a file."""

    def _restore_refusal(self):
        return make_envelope(
            surface="scaffold_restore",
            posture="sandbox",
            posture_source=POSTURE_SOURCE_PROCESS,
            session=None,
            subject="build/config.yml",
            reason="reserved_path",
        )

    def test_root_phase_and_app_run_land_in_different_files(self, audit_root, monkeypatch):
        """Same container, same identity, same surface — split by the marker,
        which is exactly what makes each file single-uid."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "operator")

        monkeypatch.setenv(writer.AUDIT_WRITER_ENV, writer.WRITER_MAINTENANCE)
        root_run = writer.record_envelope(self._restore_refusal())

        monkeypatch.delenv(writer.AUDIT_WRITER_ENV)
        app_run = writer.record_envelope(self._restore_refusal())

        assert root_run != app_run
        assert root_run.name == "maintenance.jsonl"
        assert app_run.name == "scaffold_restore.jsonl"

    def test_different_process_identities_land_in_different_files(self, audit_root, monkeypatch):
        """The other topology: no marker, but the ladder resolves the root
        process and the app process to different names."""
        monkeypatch.setattr(writer, "acting_identity", lambda: "root")
        root_run = writer.record_envelope(self._restore_refusal())

        monkeypatch.setattr(writer, "acting_identity", lambda: "osprey")
        app_run = writer.record_envelope(self._restore_refusal())

        assert root_run != app_run
        assert root_run.parent.name == "root"
        assert app_run.parent.name == "osprey"


class TestAppendMechanics:
    """One ``write()`` on an O_APPEND descriptor, and no fsync."""

    def test_append_envelope_hands_an_io_failure_back(self, audit_root, monkeypatch):
        """The seam for a caller that owns the path raises rather than swallows:
        that caller owns the degrade, and a ``try`` added here would take it away."""

        def exploding_open(path, flags, mode=0o777):
            raise OSError("the audit zone is gone")

        monkeypatch.setattr(writer.os, "open", exploding_open)
        with pytest.raises(OSError):
            writer.append_envelope(audit_root / "alice" / "x.jsonl", make_envelope())

    def test_append_envelope_reports_a_short_append_as_not_stored(self, audit_root, monkeypatch):
        monkeypatch.setattr(writer.os, "write", lambda fd, data: 1)
        path = audit_root / "alice" / "x.jsonl"
        assert writer.append_envelope(path, make_envelope()) is False

    def test_one_write_call_per_record(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        calls = []
        real_write = os.write

        def counting_write(fd, data):
            calls.append(data)
            return real_write(fd, data)

        monkeypatch.setattr(writer.os, "write", counting_write)
        writer.record_envelope(make_envelope())

        assert len(calls) == 1
        assert calls[0].endswith(b"\n")

    def test_the_descriptor_is_opened_append_only(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        seen = {}
        real_open = os.open

        def recording_open(path, flags, mode=0o777):
            seen["flags"] = flags
            return real_open(path, flags, mode)

        monkeypatch.setattr(writer.os, "open", recording_open)
        writer.record_envelope(make_envelope())

        assert seen["flags"] & os.O_APPEND
        assert seen["flags"] & os.O_CREAT

    def test_no_fsync(self, audit_root, monkeypatch):
        """A durable-by-construction zone does not buy an fsync per record."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        def refuse(*args, **kwargs):
            raise AssertionError("the writer must not fsync")

        monkeypatch.setattr(writer.os, "fsync", refuse)
        assert writer.record_envelope(make_envelope()) is not None

    def test_missing_parent_directories_are_created(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        assert not audit_root.exists()

        writer.record_envelope(make_envelope())

        assert (audit_root / "alice").is_dir()

    def test_a_short_write_is_reported_rather_than_claimed(self, audit_root, monkeypatch):
        """A torn line is not a stored record; the caller learns so."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        real_write = os.write

        def short_write(fd, data):
            real_write(fd, data[:5])
            return 5

        monkeypatch.setattr(writer.os, "write", short_write)

        assert writer.record_envelope(make_envelope()) is None

    def test_a_torn_line_does_not_swallow_the_next_record(self, audit_root, monkeypatch):
        """One torn write costs one record, never the records after it.

        Without a terminator the next record is appended straight onto the
        fragment, so a single tear destroys a second, fully written record and
        the writer hands back a path for it.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        real_write = os.write
        torn = {"done": False}

        def short_first_write(fd, data):
            if not torn["done"]:
                torn["done"] = True
                return real_write(fd, data[:5])
            return real_write(fd, data)

        monkeypatch.setattr(writer.os, "write", short_first_write)

        assert writer.record_envelope(make_envelope(subject="torn")) is None
        second = writer.record_envelope(make_envelope(subject="intact"))

        assert second is not None
        lines = second.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[-1])["subject"] == "intact"

    def test_the_ledger_is_created_without_a_second_writer(self, audit_root, monkeypatch):
        """Group READ is the point of the mode; group write would let anyone
        in the shared audit group forge a record the ledger attributes to
        another uid."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        written = writer.record_envelope(make_envelope())

        current_umask = os.umask(0)
        os.umask(current_umask)
        assert stat.S_IMODE(written.stat().st_mode) == writer.LEDGER_FILE_MODE & ~current_umask
        assert not writer.LEDGER_FILE_MODE & (
            stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH
        )

    def test_the_ledger_file_mode(self):
        assert writer.LEDGER_FILE_MODE == 0o640

    def test_a_created_identity_directory_is_group_shared_and_setgid(self, audit_root, monkeypatch):
        """The fallback mkdir and the deploy-path provisioner create the SAME
        directory, so they must create it the same way: root-created 0o755
        would lock the dropped app user out of the ledger it exists to write,
        and no setgid would lose the host's group access to what a container
        wrote. ``mkdir(mode=...)`` is umask-masked, so the writer chmods.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        assert not audit_root.exists()

        writer.record_envelope(make_envelope())

        assert stat.S_IMODE((audit_root / "alice").stat().st_mode) == writer.LEDGER_DIR_MODE

    def test_a_chmod_that_is_refused_costs_the_mode_never_the_record(self, audit_root, monkeypatch):
        """Best-effort on the mode, never on the directory.

        A container whose app user may create a directory inside a host-owned
        mount but may not ``chmod`` it is exactly the host where this fallback
        path gets taken — and letting ``EPERM`` out would reach the
        never-raises boundary and drop the record there. A slightly narrow
        directory beats a dropped record.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        def refuse(*args, **kwargs):
            raise OSError(1, "EPERM")

        monkeypatch.setattr(writer.os, "chmod", refuse)

        written = writer.record_envelope(make_envelope())

        assert written is not None, "a refused chmod cost the record"
        assert written.is_file()
        assert (audit_root / "alice").is_dir()

    def test_the_directory_mode_is_the_deploy_paths_shared_mode(self):
        """Pinned rather than imported: the audit package must not depend on
        the deployment package, the way AUDIT_WRITER_ENV is pinned against the
        registry below."""
        from osprey.deployment.compose_generator import SHARED_CORPUS_DIR_MODE

        assert writer.LEDGER_DIR_MODE == SHARED_CORPUS_DIR_MODE == 0o2770


class TestRecordBound:
    """≤2 KB per line, with one documented exception."""

    def test_an_ordinary_record_is_well_inside_the_bound(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        written = writer.record_envelope(make_envelope(detail="channel=config"))

        assert len(written.read_bytes()) <= writer.MAX_RECORD_BYTES

    def test_an_oversize_detail_is_dropped_to_fit(self, audit_root, monkeypatch):
        """``detail`` is supplementary by contract, so it is what gives way.

        It takes long identifiers *and* a full ``detail`` to reach the bound —
        a maximal ``detail`` alone is only half of it — which is why the
        degrade is a step rather than a rule about one field.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        long_name = "n" * 256

        written = writer.record_envelope(
            make_envelope(
                actor=long_name,
                posture=long_name,
                session=long_name,
                role=long_name,
                detail="x" * 1024,
            )
        )

        (record,) = _records(written)
        assert record["detail"] == writer.DETAIL_DROPPED
        assert len(written.read_bytes()) <= writer.MAX_RECORD_BYTES
        assert record["subject"] == "connectors.epics.writes_enabled"

    def test_identifiers_are_never_sacrificed_to_the_bound(self, audit_root, monkeypatch):
        """A record whose identifiers were trimmed would name the wrong thing;
        over-budget-but-true beats in-budget-but-wrong."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        long_name = "n" * 256
        envelope = make_envelope(
            surface="http_config",
            actor=long_name,
            posture=long_name,
            subject=long_name,
            reason=long_name,
            role=long_name,
            detail="x" * 1024,
        )

        written = writer.record_envelope(envelope)

        (record,) = _records(written)
        assert record["subject"] == long_name
        assert record["reason"] == long_name

    def test_executor_source_is_written_whole(self, audit_root, monkeypatch):
        """The documented exception: on the executor surface the refused code
        IS the artifact, so the record keeps it and stays a single write()."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        code = "caput('PV', 1)\n" * 400
        assert len(code) > writer.MAX_RECORD_BYTES

        written = writer.record_envelope(
            make_envelope(surface=SURFACE_EXECUTOR, source=code, detail="tool=execute")
        )

        (record,) = _records(written)
        assert record["source"] == code
        assert record["detail"] == "tool=execute"

    def test_an_executor_record_drops_its_detail_before_its_atomicity(
        self, audit_root, monkeypatch
    ):
        """The executor exception buys the SOURCE, not the detail. An executor
        record whose source is small fits once the supplementary field gives
        way, so it keeps the interleave-proof append the module promises."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        long_name = "n" * 256
        source = "caput('PV', 1)"

        written = writer.record_envelope(
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

        (record,) = _records(written)
        assert record["source"] == source
        assert record["detail"] == writer.DETAIL_DROPPED
        assert len(written.read_bytes()) <= writer.MAX_RECORD_BYTES

    def test_a_truncated_source_still_says_so(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        written = writer.record_envelope(
            make_envelope(surface=SURFACE_EXECUTOR, source="c" * (MAX_SOURCE_CHARS + 10))
        )

        (record,) = _records(written)
        assert record["source_truncated"] is True
        assert len(record["source"]) == MAX_SOURCE_CHARS

    def test_an_oversize_record_is_still_one_line(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        written = writer.record_envelope(
            make_envelope(surface=SURFACE_EXECUTOR, source="c" * MAX_SOURCE_CHARS)
        )

        assert len(written.read_text().splitlines()) == 1

    def test_an_identifier_only_record_over_budget_is_written_and_flagged(
        self, audit_root, monkeypatch, caplog
    ):
        """Reachable with every bounded field at its maximum (~2.2 KB). The
        record is stored whole and the log says the append was oversize, so an
        operator learns of it without the ledger losing the record."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
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
            written = writer.record_envelope(envelope)

        (record,) = _records(written)
        assert record["subject"] == long_name
        assert record["detail"] == writer.DETAIL_DROPPED
        assert len(written.read_bytes()) > writer.MAX_RECORD_BYTES
        assert "exceeds" in caplog.text

    def test_the_bound_is_the_atomic_append_bound(self):
        assert writer.MAX_RECORD_BYTES == 2048


class TestSurfaceRouting:
    """The filename is derived from the surface, so it must be one component."""

    @pytest.mark.parametrize(
        "surface",
        ["../escape", "nested/name", "..", ".", "with\\sep", "with\x00nul"],
        ids=["parent", "separator", "dotdot", "dot", "backslash", "nul"],
    )
    def test_a_surface_that_is_not_a_path_component_is_quarantined(
        self, audit_root, monkeypatch, surface
    ):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        written = writer.record_envelope(make_envelope(surface=surface))

        assert written == audit_root / "alice" / f"{writer.FALLBACK_SURFACE}.jsonl"
        (record,) = _records(written)
        assert record["surface"] == surface

    def test_a_surface_too_long_for_a_file_name_is_shortened(self, audit_root, monkeypatch):
        """The envelope bounds a surface at 256 characters — one longer than a
        file name may be. Shortening the stem keeps the record; refusing the
        name would lose it, and the envelope still says which surface it was.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        surface = "s" * 256

        written = writer.record_envelope(make_envelope(surface=surface))

        assert len(written.name) <= 255
        assert written.name == f"{'s' * writer.MAX_LEDGER_STEM_BYTES}.jsonl"
        (record,) = _records(written)
        assert record["surface"] == surface

    def test_a_non_ascii_surface_is_shortened_by_bytes(self, audit_root, monkeypatch):
        """``NAME_MAX`` counts BYTES. A 200-character surface is a 400-byte
        name: APFS takes it, ext4 and overlayfs — what the containers actually
        mount — raise ENAMETOOLONG, and the never-raises boundary would swallow
        that and drop every record on the surface.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        surface = "é" * 200

        written = writer.record_envelope(make_envelope(surface=surface))

        assert len(written.name.encode("utf-8")) <= 255
        assert written.is_file()
        (record,) = _records(written)
        assert record["surface"] == surface


class TestNeverRaises:
    """Nothing the audit zone can do may reach the operation being recorded."""

    def test_an_unwritable_zone_degrades_to_none(self, audit_root, monkeypatch, tmp_path):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        monkeypatch.setattr(writer, "audit_dir", lambda: blocker)

        assert writer.record_envelope(make_envelope()) is None

    def test_a_broken_audit_dir_resolver_degrades_to_none(self, audit_root, monkeypatch):
        def explode():
            raise RuntimeError("no project root here")

        monkeypatch.setattr(writer, "audit_dir", explode)

        assert writer.record_envelope(make_envelope()) is None

    def test_a_failing_open_degrades_to_none(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        def explode(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(writer.os, "open", explode)

        assert writer.record_envelope(make_envelope()) is None

    def test_a_failing_write_degrades_to_none(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        def explode(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(writer.os, "write", explode)

        assert writer.record_envelope(make_envelope()) is None

    def test_the_descriptor_is_closed_even_when_the_write_fails(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")
        closed = []
        real_close = os.close

        def counting_close(fd):
            closed.append(fd)
            real_close(fd)

        def explode(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(writer.os, "close", counting_close)
        monkeypatch.setattr(writer.os, "write", explode)
        writer.record_envelope(make_envelope())

        assert len(closed) == 1

    def test_a_non_envelope_argument_degrades_to_none(self, audit_root):
        assert writer.record_envelope("not an envelope") is None


class TestServerLogCarriesTheEvent:
    """Logged before the durable write, so a broken zone downgrades rather
    than erases the trail — the invariant ``refusal_audit`` established."""

    def test_a_refusal_warns_even_when_the_write_fails(self, audit_root, monkeypatch, caplog):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        def explode(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(writer.os, "open", explode)
        with caplog.at_level("WARNING"):
            writer.record_envelope(make_envelope(subject="connectors.epics.writes_enabled"))

        assert "connectors.epics.writes_enabled" in caplog.text

    def test_an_allowed_record_does_not_warn(self, audit_root, monkeypatch, caplog):
        """The middleware records every admitted tool call; a warning apiece
        would drown the log the refusals need to stand out in."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "alice")

        with caplog.at_level("WARNING"):
            writer.record_envelope(make_envelope(decision=DECISION_ALLOWED, reason="allowed"))

        assert caplog.text == ""


class TestAuditDirSeam:
    """``audit_dir()`` is the one seam, and it is spelled by the workspace."""

    def test_it_is_the_workspace_audit_relpath_under_the_project_root(self, tmp_path, monkeypatch):
        from osprey.utils import workspace

        monkeypatch.setattr(workspace, "load_osprey_config", lambda: {}, raising=False)
        monkeypatch.setattr(workspace, "resolve_project_root", lambda cfg: tmp_path, raising=False)

        assert writer.audit_dir() == tmp_path / workspace.AUDIT_DIR_RELPATH

    def test_it_is_imported_lazily(self):
        """Kept inside the function so the writer stays importable from the
        MCP middleware and the HTTP layer without dragging the workspace
        resolver — and so tests have one seam instead of a project root."""
        source = Path(writer.__file__).read_text()
        assert "from osprey.utils.workspace import" in source
        module_scope_imports = [
            line for line in source.splitlines() if line.startswith("from osprey.utils.workspace")
        ]
        assert module_scope_imports == []


class TestMarkerSpelling:
    """The writer marker's spelling is a wire contract with the registry.

    ``registry/mcp.py`` reserved ``OSPREY_AUDIT_WRITER`` (post-merge removal
    from every server spec) before the writer that reads it existed. Neither
    can import the other — the registry must not depend on the audit package,
    and the writer must not drag the registry — so the copy is pinned here,
    mirroring ``TestMarkerSpellings`` in ``test_marker_nonpinnability.py``.
    """

    def test_the_writer_marker_matches_the_registry_reservation(self):
        from osprey.registry.mcp import AUDIT_WRITER_ENV as REGISTRY_AUDIT_WRITER_ENV

        assert writer.AUDIT_WRITER_ENV == REGISTRY_AUDIT_WRITER_ENV

    def test_the_marker_is_non_pinnable_from_a_server_spec(self):
        from osprey.registry.mcp import NON_PINNABLE_AUDIT_MARKERS

        assert writer.AUDIT_WRITER_ENV in NON_PINNABLE_AUDIT_MARKERS

    def test_the_documented_spelling(self):
        assert writer.AUDIT_WRITER_ENV == "OSPREY_AUDIT_WRITER"
