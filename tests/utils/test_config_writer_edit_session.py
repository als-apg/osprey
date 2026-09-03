"""``config_edit_session``: one ruamel load and one dump for a run of config edits.

Every writer in :mod:`osprey.utils.config_writer` round-trips the whole document
through ruamel's comment-preserving loader — load, mutate, dump. A build applies
a few dozen of them to each rendered ``config.yml``, and the load is what costs.
Inside a session the document is loaded once, every writer edits it in memory,
and it is dumped once: at the session's end, or earlier when something has to
read the file from disk.

What these tests pin, in order of importance:

* the bytes a session writes are the bytes the same edits wrote one at a time —
  comments, key order, indentation, quoting, all of it;
* the ordering contract with disk readers is explicit — ``flush_config_edits``
  is what makes a pending edit visible to a reader of the file's bytes, and a
  writer that bypasses the session is detected rather than overwritten.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from osprey.utils import config_writer
from osprey.utils.config_writer import (
    ConfigEditConflict,
    config_add_to_list,
    config_delete_field,
    config_edit_session,
    config_remove_from_list,
    config_replace_list,
    config_update_fields,
    flush_config_edits,
    load_config_document,
    save_config_document,
    update_yaml_file,
)

SAMPLE = """\
# ============================================================
# CLAUDE CODE
# ============================================================
claude_code:
  provider: 'anthropic'   # quoted on purpose
  default_model: haiku
  timeout: 300

# ============================================================
# WEB PANEL CONFIGURATION
# ============================================================
# Prose about panels.

web:
  panels:
    ariel:
      enabled: true
      url: http://localhost:8082
    # trailing panel docs

services:
  postgresql:
    path: ./services/postgresql
  openobserve:
    path: ./services/openobserve
    port: 5080          # Host port

# Services to deploy with `osprey up`
deployed_services:
  - postgresql
  - openobserve

scaffold:
  user_owned:
    - rules/facility

# ============================================================
# SAFETY CONTROLS
# ============================================================
approval:
  enabled: true
  tools:
    channel_write: always
"""


def _write(path: Path, text: str = SAMPLE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _recorded_edits(path: Path) -> list[Any]:
    """The operation sequence a build applies, in a build's shape.

    Overrides first (existing keys, new nested keys, a new root section), then
    a run of ownership registrations, a duplicate that must be a no-op, a
    removal, a whole-list replacement, a delete and the legacy nested-update
    writer. Returns what each call returned so the two paths can be compared
    on their results as well as on their bytes.
    """
    results: list[Any] = []
    results.append(
        config_update_fields(
            path,
            {
                "claude_code.default_model": "sonnet",
                "web.panels.ariel.enabled": False,
                "web.panels.events.url": "http://localhost:9091",
                "web.panels.events.enabled": True,
                "control_system.type": "mock",
                "control_system.writes_enabled": False,
                "approval.tools.channel_write": "never",
            },
        )
    )
    for name in ("rules/facility", "rules/facility-ops", "skills/orbit-check", "services/foo"):
        results.append(config_add_to_list(path, ["scaffold", "user_owned"], name))
    results.append(config_add_to_list(path, ["deployed_services"], "event_dispatcher"))
    results.append(config_remove_from_list(path, ["deployed_services"], "openobserve"))
    results.append(
        config_replace_list(
            path,
            ["modules", "web_terminals", "users"],
            [{"name": "alice", "index": 0}, {"name": "bob", "index": 1}],
        )
    )
    results.append(config_delete_field(path, "services.openobserve.port"))
    results.append(
        update_yaml_file(
            path,
            {"execution": {"execution_method": "container"}},
            create_backup=False,
            section_comments={"execution": "Execution backend"},
        )
    )
    return results


@pytest.fixture
def counters(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count ruamel round-trip loads and dumps the shared writer performs."""
    counts = {"load": 0, "dump": 0}
    real_load = config_writer._yaml.load
    real_dump = config_writer._yaml.dump

    def counting_load(stream: Any) -> Any:
        counts["load"] += 1
        return real_load(stream)

    def counting_dump(data: Any, stream: Any, **kwargs: Any) -> Any:
        counts["dump"] += 1
        return real_dump(data, stream, **kwargs)

    monkeypatch.setattr(config_writer._yaml, "load", counting_load)
    monkeypatch.setattr(config_writer._yaml, "dump", counting_dump)
    return counts


class TestByteIdentity:
    def test_a_session_writes_what_the_same_edits_wrote_one_at_a_time(self, tmp_path: Path):
        one_at_a_time = _write(tmp_path / "plain" / "config.yml")
        batched = _write(tmp_path / "batched" / "config.yml")

        plain_results = _recorded_edits(one_at_a_time)
        with config_edit_session(batched):
            batched_results = _recorded_edits(batched)

        assert batched.read_bytes() == one_at_a_time.read_bytes()
        assert batched_results == plain_results
        # And the edits landed: this is not two untouched copies agreeing.
        text = batched.read_text(encoding="utf-8")
        assert "default_model: sonnet" in text
        assert "# Host port" not in text  # deleted with its key
        assert "Execution backend" in text
        assert text != SAMPLE

    def test_the_file_is_untouched_until_the_session_flushes(self, tmp_path: Path):
        path = _write(tmp_path / "config.yml")

        with config_edit_session(path):
            config_update_fields(path, {"claude_code.timeout": 600})
            assert path.read_text(encoding="utf-8") == SAMPLE

        assert "timeout: 600" in path.read_text(encoding="utf-8")

    def test_a_missing_file_is_the_callers_problem_as_before(self, tmp_path: Path):
        path = tmp_path / "absent" / "config.yml"

        with config_edit_session(path):
            # Opening a session touches nothing: callers gate on `exists()`.
            assert not path.exists()
            with pytest.raises(FileNotFoundError):
                config_update_fields(path, {"a": 1})

        assert not path.exists()


class TestRoundTripCount:
    def test_one_at_a_time_round_trips_per_edit(self, tmp_path: Path, counters: dict[str, int]):
        path = _write(tmp_path / "config.yml")

        _recorded_edits(path)

        # Ten writers, one of which (the duplicate append) returns before saving.
        assert counters == {"load": 10, "dump": 9}

    def test_a_session_loads_once_and_dumps_once(self, tmp_path: Path, counters: dict[str, int]):
        path = _write(tmp_path / "config.yml")

        with config_edit_session(path):
            _recorded_edits(path)

        assert counters == {"load": 1, "dump": 1}

    def test_a_session_with_no_edits_writes_nothing(self, tmp_path: Path, counters: dict[str, int]):
        path = _write(tmp_path / "config.yml")
        before = path.stat().st_mtime_ns

        with config_edit_session(path):
            pass

        assert counters == {"load": 0, "dump": 0}
        assert path.stat().st_mtime_ns == before

    def test_a_nested_session_on_the_same_file_joins_the_outer_one(
        self, tmp_path: Path, counters: dict[str, int]
    ):
        path = _write(tmp_path / "config.yml")

        with config_edit_session(path):
            config_update_fields(path, {"claude_code.timeout": 1})
            with config_edit_session(path):
                config_update_fields(path, {"claude_code.timeout": 2})
            # The inner block ending is not the end of the edit.
            assert path.read_text(encoding="utf-8") == SAMPLE
            config_update_fields(path, {"claude_code.timeout": 3})

        assert "timeout: 3" in path.read_text(encoding="utf-8")
        assert counters == {"load": 1, "dump": 1}

    def test_sessions_on_different_files_are_independent(
        self, tmp_path: Path, counters: dict[str, int]
    ):
        a = _write(tmp_path / "a" / "config.yml")
        b = _write(tmp_path / "b" / "config.yml")

        with config_edit_session(a):
            config_update_fields(a, {"claude_code.timeout": 1})
            config_update_fields(b, {"claude_code.timeout": 2})  # outside any session
            assert "timeout: 2" in b.read_text(encoding="utf-8")
            assert "timeout: 300" in a.read_text(encoding="utf-8")

        assert "timeout: 1" in a.read_text(encoding="utf-8")
        assert counters == {"load": 2, "dump": 2}

    def test_the_session_is_keyed_on_the_file_not_its_spelling(
        self, tmp_path: Path, counters: dict[str, int]
    ):
        path = _write(tmp_path / "render" / "config.yml")
        spelled_differently = tmp_path / "render" / ".." / "render" / "config.yml"

        with config_edit_session(spelled_differently):
            config_update_fields(path, {"claude_code.timeout": 1})
            config_add_to_list(spelled_differently, ["deployed_services"], "x")

        assert counters == {"load": 1, "dump": 1}
        text = path.read_text(encoding="utf-8")
        assert "timeout: 1" in text
        assert "  - x" in text


class TestDiskReaders:
    """The ordering contract with anything that reads the file's bytes."""

    def test_flush_makes_pending_edits_visible_on_disk(
        self, tmp_path: Path, counters: dict[str, int]
    ):
        path = _write(tmp_path / "config.yml")

        with config_edit_session(path):
            config_update_fields(path, {"claude_code.timeout": 600})
            flush_config_edits(path)
            assert "timeout: 600" in path.read_text(encoding="utf-8")
            # Nothing pending: a second flush is not a second dump.
            flush_config_edits(path)
            assert counters["dump"] == 1
            config_update_fields(path, {"claude_code.timeout": 900})

        assert "timeout: 900" in path.read_text(encoding="utf-8")
        assert counters == {"load": 1, "dump": 2}

    def test_flush_outside_a_session_is_a_no_op(self, tmp_path: Path):
        path = _write(tmp_path / "config.yml")
        before = path.stat().st_mtime_ns

        flush_config_edits(path)
        flush_config_edits(tmp_path / "never-existed.yml")

        assert path.stat().st_mtime_ns == before

    def test_a_flushed_session_reloads_after_someone_else_writes_the_file(
        self, tmp_path: Path, counters: dict[str, int]
    ):
        path = _write(tmp_path / "config.yml")

        with config_edit_session(path):
            config_update_fields(path, {"claude_code.timeout": 600})
            flush_config_edits(path)
            # An injector with its own YAML instance rewrites the file in place.
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace("deployed_services:\n", "deployed_services:\n  - injected\n"),
                encoding="utf-8",
            )
            config_add_to_list(path, ["scaffold", "user_owned"], "rules/new")

        final = path.read_text(encoding="utf-8")
        assert "  - injected" in final  # the outside write survived
        assert "rules/new" in final  # and so did ours
        assert "timeout: 600" in final
        assert counters == {"load": 2, "dump": 2}

    def test_an_outside_write_under_pending_edits_is_a_conflict_not_a_clobber(self, tmp_path: Path):
        path = _write(tmp_path / "config.yml")
        outside = SAMPLE.replace("timeout: 300", "timeout: 42")

        with pytest.raises(ConfigEditConflict, match=str(path)):
            with config_edit_session(path):
                config_update_fields(path, {"claude_code.default_model": "sonnet"})
                path.write_text(outside, encoding="utf-8")  # nobody flushed first
                config_add_to_list(path, ["scaffold", "user_owned"], "rules/new")

        # The file is whatever the outside writer left; the pending edit is
        # dropped rather than written over it, at the failing call and at exit.
        assert path.read_text(encoding="utf-8") == outside

    def test_flushing_over_an_outside_write_is_the_same_conflict(self, tmp_path: Path):
        path = _write(tmp_path / "config.yml")
        outside = SAMPLE.replace("timeout: 300", "timeout: 42")

        with pytest.raises(ConfigEditConflict):
            with config_edit_session(path):
                config_update_fields(path, {"claude_code.default_model": "sonnet"})
                path.write_text(outside, encoding="utf-8")

        assert path.read_text(encoding="utf-8") == outside


class TestFailureInsideTheBlock:
    def test_edits_before_an_exception_still_land(self, tmp_path: Path):
        """Parity with one-at-a-time writes, where each call had already hit disk."""
        path = _write(tmp_path / "config.yml")

        with pytest.raises(RuntimeError, match="later step"):
            with config_edit_session(path):
                config_update_fields(path, {"claude_code.timeout": 600})
                raise RuntimeError("later step failed")

        assert "timeout: 600" in path.read_text(encoding="utf-8")

    def test_a_session_is_closed_after_an_exception(self, tmp_path: Path, counters: dict[str, int]):
        path = _write(tmp_path / "config.yml")

        with pytest.raises(RuntimeError):
            with config_edit_session(path):
                raise RuntimeError

        config_update_fields(path, {"claude_code.timeout": 1})
        assert "timeout: 1" in path.read_text(encoding="utf-8")  # immediate, no session
        assert counters == {"load": 1, "dump": 1}


def test_a_session_survives_the_file_being_replaced_wholesale(tmp_path: Path):
    """Copying a fresh render over the file is an outside write like any other."""
    path = _write(tmp_path / "config.yml")
    fresh = _write(tmp_path / "fresh.yml", SAMPLE.replace("haiku", "opus"))

    with config_edit_session(path):
        config_update_fields(path, {"claude_code.timeout": 1})
        flush_config_edits(path)
        shutil.copy2(fresh, path)
        config_update_fields(path, {"claude_code.timeout": 2})

    text = path.read_text(encoding="utf-8")
    assert "default_model: opus" in text
    assert "timeout: 2" in text


class TestDocumentApi:
    """``load_config_document`` / ``save_config_document``: the pair a writer that
    edits the document itself (the service injectors, the ``auto`` resolvers)
    uses instead of a loader of its own, so that it joins the session too."""

    def test_outside_a_session_the_pair_is_one_round_trip(
        self, tmp_path: Path, counters: dict[str, int]
    ):
        path = _write(tmp_path / "config.yml")

        doc = load_config_document(path)
        doc["claude_code"]["timeout"] = 7
        save_config_document(path, doc)

        text = path.read_text(encoding="utf-8")
        assert "timeout: 7" in text
        assert "# quoted on purpose" in text  # comments survive, as with every writer
        assert counters == {"load": 1, "dump": 1}

    def test_inside_a_session_the_document_is_shared_and_the_write_waits(
        self, tmp_path: Path, counters: dict[str, int]
    ):
        path = _write(tmp_path / "config.yml")

        with config_edit_session(path):
            doc = load_config_document(path)
            doc["claude_code"]["timeout"] = 7
            save_config_document(path, doc)
            assert path.read_text(encoding="utf-8") == SAMPLE  # nothing on disk yet
            # The dotted-key writers edit the very same object.
            config_update_fields(path, {"claude_code.default_model": "sonnet"})
            assert load_config_document(path) is doc
            assert doc["claude_code"]["default_model"] == "sonnet"

        text = path.read_text(encoding="utf-8")
        assert "timeout: 7" in text
        assert "default_model: sonnet" in text
        assert counters == {"load": 1, "dump": 1}

    def test_the_pair_writes_in_the_same_style_as_every_other_writer(self, tmp_path: Path):
        """Block sequences indented under their key, no line wrapping: one style
        for the whole file, whichever writer touched it last."""
        path = _write(tmp_path / "config.yml")
        long_value = "x" * 200

        doc = load_config_document(path)
        doc["deployed_services"].append("added")
        doc["claude_code"]["note"] = long_value
        save_config_document(path, doc)

        text = path.read_text(encoding="utf-8")
        assert "\n  - added\n" in text
        assert f"note: {long_value}\n" in text
