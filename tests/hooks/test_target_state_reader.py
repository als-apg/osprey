"""Tests for the stdlib-only control-context reader ``osprey_target_state``.

This module is imported by the hooks rather than invoked as a script, so it is
tested by direct import — the ``osprey_hook_log`` precedent.

The whole contract under test is "never surprise the caller": every failure mode
must arrive as the same baseline-fallback marker, nothing must ever raise, and a
record this reader accepts must be one ``osprey_connectors.control_context``
would accept too. The tests lean on two seams instead of on a real deployment:
``resolve_state_dir`` (pointed at ``tmp_path``) and ``_is_process_alive`` (so a
server report can be made live or dead without spawning anything).
"""

from __future__ import annotations

import json

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as reader

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Point the reader at an empty temp state directory."""
    directory = tmp_path / "control_target"
    directory.mkdir()
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: str(directory))
    return directory


@pytest.fixture
def unstamped(monkeypatch):
    """A process the web terminal did not stamp with an agent-data root."""
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)


@pytest.fixture
def alive_everything(monkeypatch):
    """Treat every PID as alive unless a test says otherwise."""
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: True)


def write_record(directory, **overrides):
    """Write a well-formed control-context record, with overrides applied."""
    record = {
        "schema": 1,
        "owner": {"kind": "web_terminal", "pid": 4242, "port": 8080},
        "target": "va",
        "generation": 3,
        "posture": {},
        "last_switch": None,
    }
    record.update(overrides)
    path = directory / reader.RECORD_FILENAME
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def write_report(directory, server_pid, **overrides):
    """Write one controls server's report, with overrides applied."""
    report = {
        "server_pid": server_pid,
        "session": None,
        "applied_target": "va",
        "applied_generation": 3,
        "targets": {
            "live": {"label": "ALS storage ring", "endpoint": "epics://", "real_machine": True},
            "va": {
                "label": "Virtual accelerator",
                "endpoint": "pva://vasrv",
                "real_machine": False,
            },
        },
    }
    report.update(overrides)
    path = directory / f"server_{server_pid}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_the_records_target_and_generation(state_dir):
    write_record(state_dir)

    result = reader.read_target()

    assert not reader.is_baseline(result)
    assert result["fallback"] is None
    assert result["reason"] is None
    assert result["target"] == "va"
    assert result["generation"] == 3


def test_result_always_carries_all_four_contract_keys(state_dir):
    write_record(state_dir)
    resolved = reader.read_target()

    for result in (resolved, reader.baseline_result()):
        assert set(result) == {"target", "generation", "fallback", "reason"}


def test_one_record_answers_every_reader_in_the_deployment(state_dir):
    """No pid ancestry, no per-session selection: one file, one answer."""
    write_record(state_dir, target="live", generation=9)

    assert reader.read_target()["target"] == "live"
    assert reader.read_record()["generation"] == 9


def test_read_record_carries_only_what_a_hook_answers_from(state_dir):
    """Ownership is the owner's business and convergence is nobody's here."""
    write_record(state_dir, posture={"live": "sandbox"})

    record = reader.read_record()

    assert set(record) == {"target", "generation", "posture"}


def test_generation_zero_is_a_real_generation(state_dir):
    write_record(state_dir, generation=0)

    assert reader.read_target()["generation"] == 0


# ---------------------------------------------------------------------------
# degraded records
# ---------------------------------------------------------------------------


def test_absent_record_yields_baseline_marker(state_dir):
    result = reader.read_target()

    assert reader.is_baseline(result)
    assert result["reason"] == reader.REASON_NO_STATE
    assert result["target"] is None
    assert result["generation"] is None


def test_missing_state_directory_yields_baseline_marker(tmp_path, monkeypatch):
    missing = tmp_path / "nowhere" / "control_target"
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: str(missing))

    assert reader.read_target()["reason"] == reader.REASON_NO_STATE


def test_unresolvable_repo_root_yields_baseline_marker(monkeypatch):
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: None)

    assert reader.is_baseline(reader.read_target())
    assert reader.read_record() is None


def test_corrupt_json_is_unreadable(state_dir):
    (state_dir / reader.RECORD_FILENAME).write_text("{not json", encoding="utf-8")

    assert reader.read_target()["reason"] == reader.REASON_UNREADABLE
    assert reader.read_record() is None


def test_non_dict_payload_is_unreadable(state_dir):
    (state_dir / reader.RECORD_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")

    assert reader.read_target()["reason"] == reader.REASON_UNREADABLE


@pytest.mark.parametrize("schema", [2, 0, "1", None, True])
def test_a_schema_this_reader_does_not_know_is_no_record(state_dir, schema):
    """Hooks and record are regenerated together; there is no migration path."""
    write_record(state_dir, schema=schema)

    assert reader.read_record() is None
    assert reader.read_target()["reason"] == reader.REASON_UNREADABLE


def test_a_record_without_a_schema_is_no_record(state_dir):
    (state_dir / reader.RECORD_FILENAME).write_text(
        json.dumps({"target": "va", "generation": 1}), encoding="utf-8"
    )

    assert reader.read_record() is None


@pytest.mark.parametrize("target", ["ring", "", None, "VA", 7])
def test_a_target_outside_the_vocabulary_is_no_record(state_dir, target):
    write_record(state_dir, target=target)

    assert reader.read_record() is None
    assert reader.read_target()["reason"] == reader.REASON_UNREADABLE


@pytest.mark.parametrize("generation", [-1, "3", None, True, 1.5])
def test_an_unusable_generation_is_no_record(state_dir, generation):
    """The identity trio is all-or-nothing: a pinned write needs the number."""
    write_record(state_dir, generation=generation)

    assert reader.read_record() is None


def test_unknown_extra_fields_are_tolerated(state_dir):
    write_record(state_dir, unexpected={"from": "a later version"})

    assert reader.read_target()["target"] == "va"


def test_parse_record_accepts_text_and_bytes():
    payload = {"schema": 1, "target": "live", "generation": 2}

    assert reader.parse_record(json.dumps(payload))["target"] == "live"
    assert reader.parse_record(json.dumps(payload).encode())["target"] == "live"
    assert reader.parse_record(b"\xff\xfe not utf-8") is None


def test_an_exploding_state_dir_still_returns_the_marker(monkeypatch):
    def boom(hook_input=None):
        raise RuntimeError("state dir resolution blew up")

    monkeypatch.setattr(reader, "resolve_state_dir", boom)

    assert reader.is_baseline(reader.read_target())
    assert reader.read_record() is None
    assert reader.read_target_view() is None
    assert reader.recorded_posture() == {}


def test_is_baseline_tolerates_non_dict_input():
    for value in (None, "baseline", 0, []):
        assert reader.is_baseline(value) is True


def test_selected_target_tolerates_any_shape():
    assert reader.selected_target({"target": " va "}) == "va"
    assert reader.selected_target({"target": ""}) is None
    assert reader.selected_target({}) is None
    assert reader.selected_target(None) is None


# ---------------------------------------------------------------------------
# how a target is spoken of
# ---------------------------------------------------------------------------


def test_view_folds_the_live_servers_metadata_onto_the_record(state_dir, alive_everything):
    write_record(state_dir)
    write_report(state_dir, 5150)

    view = reader.read_target_view()

    assert view["target"] == "va"
    assert reader.target_metadata(view, "va") == {
        "label": "Virtual accelerator",
        "endpoint": "pva://vasrv",
        "real_machine": False,
    }
    assert reader.target_metadata(view, "live")["real_machine"] is True


def test_view_without_a_record_is_no_view(state_dir, alive_everything):
    write_report(state_dir, 5150)

    assert reader.read_target_view() is None


def test_a_dead_servers_report_does_not_speak(state_dir, monkeypatch):
    write_record(state_dir)
    write_report(state_dir, 5151)
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: False)

    view = reader.read_target_view()

    assert view["targets"] == {}
    assert reader.target_metadata(view, "va") is None


def test_a_dead_servers_report_is_never_deleted(state_dir, monkeypatch):
    write_record(state_dir)
    path = write_report(state_dir, 5152)
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: False)

    reader.read_target_view()

    assert path.exists()


def test_a_live_report_answers_past_a_dead_one(state_dir, monkeypatch):
    write_record(state_dir)
    write_report(state_dir, 100, targets={"va": {"label": "dead", "real_machine": False}})
    write_report(state_dir, 200, targets={"va": {"label": "live", "real_machine": False}})
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: int(pid) == 200)

    view = reader.read_target_view()

    assert reader.target_metadata(view, "va")["label"] == "live"


def test_a_corrupt_report_does_not_hide_a_good_sibling(state_dir, alive_everything):
    write_record(state_dir)
    (state_dir / "server_100.json").write_text("{not json", encoding="utf-8")
    write_report(state_dir, 200)

    assert reader.target_metadata(reader.read_target_view(), "va")["endpoint"] == "pva://vasrv"


def test_a_report_with_no_metadata_yet_is_passed_over(state_dir, alive_everything):
    write_record(state_dir)
    write_report(state_dir, 100, targets={})
    write_report(state_dir, 200)

    assert reader.target_metadata(reader.read_target_view(), "va") is not None


def test_unrelated_files_in_the_directory_are_ignored(state_dir, alive_everything):
    write_record(state_dir)
    (state_dir / "server_notapid.json").write_text("{}", encoding="utf-8")
    (state_dir / "write_approval_abc.json").write_text("{}", encoding="utf-8")
    (state_dir / "notes.txt").write_text("hello", encoding="utf-8")

    view = reader.read_target_view()

    assert view["targets"] == {}


def test_target_metadata_degrades_rather_than_raising():
    assert reader.target_metadata(None, "va") is None
    assert reader.target_metadata({"targets": "nope"}, "va") is None
    assert reader.target_metadata({"targets": {"va": "nope"}}, "va") is None
    assert reader.target_metadata({"target": "va"}, "va") is None


# ---------------------------------------------------------------------------
# recorded posture
# ---------------------------------------------------------------------------


def test_recorded_posture_is_the_records_narrowings(state_dir):
    write_record(state_dir, posture={"live": "sandbox"})

    assert reader.recorded_posture() == {"live": "sandbox"}
    assert reader.target_posture("live") == reader.POSTURE_SANDBOX
    assert reader.target_posture("va") is None


def test_the_narrowing_is_not_addressed_to_one_session(state_dir, monkeypatch):
    """It is the deployment's: no session key selects it and none can miss it."""
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "someone-else")
    write_record(state_dir, posture={"va": "sandbox"})

    assert reader.target_sandboxed(None, "va") is True


def test_a_bare_sandbox_string_narrows_every_target(state_dir):
    write_record(state_dir, posture="sandbox")

    assert reader.recorded_posture() == dict.fromkeys(reader.CONTROL_TARGETS, "sandbox")


@pytest.mark.parametrize("value", ["writes", "", "off", 1, None, [], {"va": "writes"}])
def test_nothing_but_sandbox_survives_the_posture_filter(state_dir, value):
    """The writes posture is the absence of an entry; nothing here can widen."""
    write_record(state_dir, posture=value)

    assert reader.recorded_posture() == {}


def test_a_non_string_posture_key_is_dropped(state_dir):
    (state_dir / reader.RECORD_FILENAME).write_text(
        '{"schema": 1, "target": "va", "generation": 1, "posture": {"7": "sandbox"}}',
        encoding="utf-8",
    )

    assert reader.recorded_posture() == {"7": "sandbox"}


def test_no_record_reads_as_no_narrowing(state_dir):
    assert reader.recorded_posture() == {}
    assert reader.target_posture("live") is None


def test_a_targetless_lookup_takes_the_most_restrictive_entry(state_dir):
    write_record(state_dir, posture={"live": "sandbox"})

    assert reader.target_posture(None) == reader.POSTURE_SANDBOX
    assert reader.target_sandboxed(None, None) is True


def test_a_targetless_lookup_on_an_unnarrowed_record_is_permissive(state_dir):
    write_record(state_dir)

    assert reader.target_posture(None) is None
    assert reader.target_sandboxed(None, None) is False


# ---------------------------------------------------------------------------
# posture_unknown: fail-closed before the record exists
# ---------------------------------------------------------------------------


def test_posture_unknown_when_unstamped_and_no_record(state_dir, unstamped):
    """A bare ``claude`` on a deployment nobody has started yet is refused."""
    assert reader.posture_unknown() is True


def test_posture_known_once_the_record_exists(state_dir, unstamped):
    write_record(state_dir)

    assert reader.posture_unknown() is False


def test_posture_unknown_when_the_record_cannot_be_read(state_dir, unstamped):
    (state_dir / reader.RECORD_FILENAME).write_text("{not json", encoding="utf-8")

    assert reader.posture_unknown() is True


def test_a_stamped_root_is_never_unknown(state_dir, monkeypatch, tmp_path):
    """The stamp IS the evidence that this reader is looking where writes land."""
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))

    assert reader.posture_unknown() is False


def test_posture_unknown_does_not_ask_for_a_session_key(state_dir, unstamped, monkeypatch):
    """The session key indexes nothing; it cannot make a refusal appear."""
    monkeypatch.delenv(reader.POSTURE_SESSION_ENV_VAR, raising=False)
    assert reader.posture_unknown() is True

    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "web-1")
    assert reader.posture_unknown() is True


def test_session_key_is_the_audit_id(monkeypatch):
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "  web-1  ")
    assert reader.session_key() == "web-1"

    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "   ")
    assert reader.session_key() is None


# ---------------------------------------------------------------------------
# effective_writes_for
# ---------------------------------------------------------------------------

#: A deployment that arms writes on both machines it can reach.
_ARMED = {
    "type": "virtual_accelerator",
    "writes_enabled": True,
    "connector": {"virtual_accelerator": {"address": "x"}, "epics": {"address": "y"}},
}


def test_effective_writes_needs_the_deployment_ceiling(state_dir):
    write_record(state_dir)

    assert reader.effective_writes_for(None, _ARMED, "va") is True
    assert reader.effective_writes_for(None, {"type": "mock"}, "va") is False


def test_a_narrowing_refuses_an_armed_target(state_dir):
    write_record(state_dir, posture={"va": "sandbox"})

    assert reader.effective_writes_for(None, _ARMED, "va") is False
    assert reader.effective_writes_for(None, _ARMED, "live") is True


def test_a_readonly_run_refuses_an_armed_unnarrowed_target(state_dir, monkeypatch):
    write_record(state_dir)
    monkeypatch.setenv(reader.EXECUTION_MODE_ENV_VAR, reader.SANDBOX_MODE)

    assert reader.is_readonly_run() is True
    assert reader.effective_writes_for(None, _ARMED, "va") is False


def test_an_unidentified_target_is_answered_by_every_reachable_one(state_dir):
    """Stricter than the roster's union on purpose: a gate cannot guess."""
    write_record(state_dir, posture={"live": "sandbox"})

    assert reader.effective_writes_for(None, _ARMED, None) is False


# ---------------------------------------------------------------------------
# path contract
# ---------------------------------------------------------------------------


def test_paths_anchor_on_repo_root_and_agent_data(monkeypatch, tmp_path):
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(reader, "get_repo_root", lambda hook_input=None: str(tmp_path))

    expected = tmp_path / reader._AGENT_DATA_BASE_DIR / "control_target"
    assert reader.resolve_state_dir() == str(expected)
    assert reader.record_path() == str(expected / "control_context.json")
    assert reader.STATE_DIR_NAME == "control_target"
    assert reader.RECORD_FILENAME == "control_context.json"
    assert reader.REPORT_FILE_GLOB == "server_*.json"


def test_the_stamped_root_wins_over_the_derivation(monkeypatch, tmp_path):
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path / "stamped"))
    monkeypatch.setattr(reader, "get_repo_root", lambda hook_input=None: str(tmp_path))

    assert reader.agent_data_root() == str(tmp_path / "stamped")


def test_resolve_state_dir_returns_none_when_repo_root_is_empty(monkeypatch):
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(reader, "get_repo_root", lambda hook_input=None: "")

    assert reader.resolve_state_dir() is None
    assert reader.record_path() is None


def test_module_imports_no_third_party_dependencies():
    """Hooks run outside the venv: every guaranteed path must be stdlib only."""
    import ast
    from pathlib import Path

    source = Path(reader.__file__).read_text()
    tree = ast.parse(source)

    # Only unguarded, module-level imports count: the one osprey import is
    # deliberately inside a try/except with a literal fallback.
    top_level_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top_level_roots.add(node.module.split(".")[0])

    # `osprey_hook_log` is the sibling helper; everything else must be stdlib.
    assert top_level_roots <= {
        "__future__",
        "json",
        "os",
        "sys",
        "osprey_hook_log",
    }, top_level_roots
