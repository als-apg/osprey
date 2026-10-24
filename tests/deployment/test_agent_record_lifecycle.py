"""Lifecycle tests for agent-record, the service every deployment runs.

- The project image is built for it even when the profile deploys no other
  service that runs that image, and ``_start_stack`` starts a stack with no
  ``deployed_services`` and no web terminals.
- Removing a user's volumes first removes the agent-record container, which
  mounts every user's transcripts volume.
- The image entrypoint joins the spool's and the record dir's groups and never
  chowns either.
- A host-sessions deployment built before its first host session is told once
  that those sessions are recorded only after a rebuild.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV, claude_project_dir
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import (
    AGENT_RECORD_TEMPLATE_DIR,
    resolve_user_projects_volume_name,
    resolve_user_volume_names,
)
from osprey.deployment.web_terminals import lifecycle
from tests.deployment.web_terminals.test_lifecycle import _config as _web_config
from tests.deployment.web_terminals.test_lifecycle import _write_config

ENTRYPOINT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "project"
    / "entrypoint.sh"
)
SPOOL_VAR = "OSPREY_AGENT_RECORD_SPOOL_ROOT"
RECORD_VAR = "OSPREY_AGENT_RECORD_DIR"


# ── The project image and the start ─────────────────────────────────────────


def test_a_profile_with_no_project_image_service_builds_the_image_for_agent_record() -> None:
    config: dict[str, Any] = {"project_name": "hello", "deployed_services": []}

    assert AGENT_RECORD_TEMPLATE_DIR in container_lifecycle._deployed_project_image_services(config)
    assert container_lifecycle._project_image_build_target(config, {}) == "hello:local"


def test_an_image_override_takes_the_agent_record_build_away() -> None:
    config: dict[str, Any] = {"project_name": "hello", "deployed_services": []}

    env = {"OSPREY_WORKER_IMAGE": "registry.example.org/osprey:1"}
    assert container_lifecycle._project_image_build_target(config, env) is None


class _Reached(Exception):
    """Raised by the first step after the removed early return."""


def test_start_stack_starts_a_stack_with_no_services_and_no_terminals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _reached(_config: dict) -> None:
        raise _Reached

    monkeypatch.setattr(container_lifecycle, "_check_shared_disk_preflight", _reached)
    config: dict[str, Any] = {"project_name": "hello", "deployed_services": []}

    with pytest.raises(_Reached):
        container_lifecycle._start_stack(config, [], tmp_path)


# ── Removing a user's volumes ───────────────────────────────────────────────


@pytest.fixture
def runtime_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def _run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _run)
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(lifecycle, "verify_runtime_is_running", lambda config=None: (True, ""))
    return calls


def test_purging_a_user_removes_agent_record_before_the_transcripts_volume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_calls: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _web_config(["alice", "bob"])
    config_path = _write_config(tmp_path, config)
    claude_vol, agent_vol = resolve_user_volume_names(config, "alice")
    projects_vol = resolve_user_projects_volume_name(config, "alice")

    lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=True)

    removals = [
        c for c in runtime_calls if "demo-project-agent-record" in c or c[1:3] == ["volume", "rm"]
    ]
    assert removals == [
        ["docker", "rm", "-f", "demo-project-agent-record"],
        ["docker", "volume", "rm", claude_vol],
        ["docker", "volume", "rm", projects_vol],
        ["docker", "volume", "rm", agent_vol],
    ]
    out = " ".join(capsys.readouterr().out.split())
    assert "agent-record container was removed" in out
    assert "Run `osprey build`, then `osprey up`" in out


def test_retaining_a_users_volumes_leaves_agent_record_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_calls: list[list[str]]
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _web_config(["alice", "bob"]))

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    assert not [c for c in runtime_calls if "demo-project-agent-record" in c]


def test_a_missing_agent_record_container_is_not_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def _run(argv, **_kwargs):
        missing = argv[1] == "rm"
        return subprocess.CompletedProcess(argv, returncode=1 if missing else 0, stdout="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _run)

    lifecycle._apply_volume_policy(
        "docker",
        ["demo-project_alice-claude-projects"],
        archive=False,
        purge=True,
        repo_root=tmp_path,
        record_container="demo-project-agent-record",
    )

    assert "agent-record" not in capsys.readouterr().out


# ── The entrypoint ──────────────────────────────────────────────────────────


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    assert match, name
    return match.group(0)


def test_the_entrypoint_joins_both_directories_groups() -> None:
    body = _function(ENTRYPOINT.read_text(encoding="utf-8"), "join_mounted_groups")

    for var in (SPOOL_VAR, RECORD_VAR):
        assert f'join_mounted_group {var} "${{{var}:-}}"' in body
        assert f'probe_mounted_dir {var} "${{{var}:-}}" write' in body


def test_the_hand_back_prunes_the_spool_root_and_the_record_dir(tmp_path: Path) -> None:
    """Run the real function with ``find`` stubbed: both binds are pruned, so
    root never chowns the group-shared spool tree or the record dir."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    state = tmp_path / "var"
    spool = state / "spool"
    record = state / "agent_record"
    for directory in (spool / ".otlp", record):
        directory.mkdir(parents=True)
    args_file = tmp_path / "find-args"
    script = "\n".join(
        [
            f'find() {{ printf "%s\\n" "$*" > "{args_file}"; }}',
            "log() { :; }",
            _function(text, "hand_back_state_zone"),
            "hand_back_state_zone",
        ]
    )

    subprocess.run(
        ["sh", "-c", script],
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "STATE_DIR": str(state),
            SPOOL_VAR: str(spool),
            RECORD_VAR: str(record),
        },
    )

    args = args_file.read_text(encoding="utf-8")
    assert f"-path {spool} -prune -o" in args, args
    assert f"-path {record} -prune -o" in args, args
    assert "! -user osprey -exec chown osprey:osprey" in args, args


def test_the_hand_back_comment_names_both_exceptions() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    comment = text[: text.index("hand_back_state_zone() {")]

    for var in (SPOOL_VAR, RECORD_VAR):
        assert f"#   * {var}" in comment, var


# ── Host sessions not yet recorded ──────────────────────────────────────────


def _rendered(tmp_path: Path, mounts: list[Any]) -> list[str]:
    relative = f"./build/services/{AGENT_RECORD_TEMPLATE_DIR}/docker-compose.yml"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump({"services": {"agent-record": {"volumes": mounts}}}), encoding="utf-8"
    )
    return [relative]


@pytest.fixture
def facts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(tmp_path / "claude-home"))
    said: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_report_fact", said.append)
    return said


def _host_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "project_name": "hello",
        "project_root": str(tmp_path),
        "deployment": {"host_identity": "thorsten"},
    }


def test_a_build_without_the_host_projects_dir_is_reported_once(
    tmp_path: Path, facts: list[str]
) -> None:
    compose_files = _rendered(tmp_path, ["./var/spool:/app/hello/var/spool"])

    container_lifecycle._report_unmounted_host_transcripts(
        _host_config(tmp_path), compose_files, tmp_path
    )

    assert len(facts) == 1
    assert "host sessions" in facts[0]
    assert "after the first host session" in facts[0]
    assert "`osprey build`" in facts[0]


def test_a_build_that_mounted_the_host_projects_dir_says_nothing(
    tmp_path: Path, facts: list[str]
) -> None:
    host_dir = claude_project_dir(tmp_path / "build")
    compose_files = _rendered(
        tmp_path, [{"type": "bind", "source": host_dir.as_posix(), "target": "/data/x"}]
    )

    container_lifecycle._report_unmounted_host_transcripts(
        _host_config(tmp_path), compose_files, tmp_path
    )

    assert facts == []


def test_no_host_identity_says_nothing(tmp_path: Path, facts: list[str]) -> None:
    compose_files = _rendered(tmp_path, [])

    container_lifecycle._report_unmounted_host_transcripts(
        {"project_name": "hello", "project_root": str(tmp_path)}, compose_files, tmp_path
    )

    assert facts == []


def test_a_build_with_no_agent_record_render_says_nothing(tmp_path: Path, facts: list[str]) -> None:
    """A build from before agent-record rendered no bind to miss, so the host
    directory's absence at build time is not the reason it is unmounted."""
    container_lifecycle._report_unmounted_host_transcripts(
        _host_config(tmp_path), ["build/services/archive/docker-compose.yml"], tmp_path
    )

    assert facts == []
