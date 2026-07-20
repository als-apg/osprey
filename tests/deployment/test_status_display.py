"""Tests for the per-user web terminal section of ``osprey.deployment.status_display.show_status``.

Mocks the container runtime entirely: ``subprocess.run`` is patched to return canned
``docker ps -a --format json`` output for the first call and canned ``docker volume ls``
output for the second, so these tests never touch a real container runtime.
"""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from osprey.deployment import status_display


class _FakeConfigBuilder:
    """Stand-in for ConfigBuilder: show_status only ever reads ``.raw_config``."""

    def __init__(self, config_path):
        self.raw_config = _CONFIGS[config_path]


_CONFIGS: dict[str, dict] = {}


def _register_config(config_path, config):
    _CONFIGS[config_path] = config


def _base_config(*, enabled=True, users=None):
    return {
        "project_name": "demo-project",
        "facility": {"prefix": "dls"},
        "modules": {
            "web_terminals": {
                "enabled": enabled,
                "users": users if users is not None else [],
            }
        },
    }


def _ps_container(name, state, labels=None):
    return {
        "Names": [name],
        "Labels": labels or {},
        "State": state,
        "Ports": [],
        "Image": "img",
    }


def _ps_stdout(*containers):
    """Serialize containers as a JSON array (Podman ``ps`` format).

    Avoids the newline-separated-JSON-objects (Docker format) branch, which
    only round-trips through json.loads correctly for 2+ lines — a single
    bare JSON object also parses as valid JSON (a dict, not a list) on the
    first `json.loads(result.stdout)` attempt, which is exactly the ambiguity
    that format detection exists to route around.
    """
    return json.dumps(list(containers))


@pytest.fixture(autouse=True)
def _patch_config_builder(monkeypatch):
    monkeypatch.setattr(status_display, "ConfigBuilder", _FakeConfigBuilder)
    yield
    _CONFIGS.clear()


@pytest.fixture
def runtime_calls(monkeypatch):
    """Patch get_runtime_command/get_ps_command and subprocess.run.

    First subprocess.run call (ps) returns `ps_stdout`; the second (volume ls)
    returns `volume_stdout`. Both are set on the returned mutable dict before
    show_status() is invoked.
    """
    calls = {"argvs": [], "ps_stdout": "", "volume_stdout": ""}

    monkeypatch.setattr(
        status_display,
        "get_ps_command",
        lambda config, all_containers=False: ["docker", "ps", "-a", "--format", "json"],
    )
    monkeypatch.setattr(status_display, "get_runtime_command", lambda config=None: ["docker"])

    class _Result:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.returncode = returncode
            self.stderr = ""

    def _fake_run(cmd, capture_output=True, text=True, timeout=10):
        calls["argvs"].append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            return _Result(calls["ps_stdout"])
        if cmd[:2] == ["docker", "volume"]:
            return _Result(calls["volume_stdout"])
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(status_display.subprocess, "run", _fake_run)
    return calls


def test_per_user_section_reports_running_and_present(runtime_calls):
    config = _base_config(users=["alice", "bob"])
    _register_config("cfg.yml", config)

    claude_vol_alice, agent_vol_alice = status_display.resolve_user_volume_names(config, "alice")
    claude_vol_bob, agent_vol_bob = status_display.resolve_user_volume_names(config, "bob")

    runtime_calls["ps_stdout"] = _ps_stdout(
        _ps_container("dls-web-alice", "running"),
        _ps_container("dls-web-bob", "exited"),
    )
    runtime_calls["volume_stdout"] = "\n".join(
        [claude_vol_alice, agent_vol_alice]  # bob's volumes absent
    )

    console = Console(record=True, width=200)
    status_display.show_status("cfg.yml", console=console)
    output = console.export_text()

    assert "Web Terminal Users" in output
    assert "alice" in output and "bob" in output
    assert "Running" in output
    assert "Stopped" in output
    assert claude_vol_alice in output
    assert agent_vol_bob in output
    assert "missing" in output  # bob's volumes reported missing

    # ps queried once (all_containers=True), volumes queried once
    ps_calls = [c for c in runtime_calls["argvs"] if c[:2] == ["docker", "ps"]]
    volume_calls = [c for c in runtime_calls["argvs"] if c[:2] == ["docker", "volume"]]
    assert len(ps_calls) == 1
    assert len(volume_calls) == 1


def test_per_user_section_reports_not_created(runtime_calls):
    config = _base_config(users=["carol"])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = ""  # no containers at all
    runtime_calls["volume_stdout"] = ""  # no volumes at all

    console = Console(record=True, width=200)
    status_display.show_status("cfg.yml", console=console)
    output = console.export_text()

    assert "Web Terminal Users" in output
    assert "Not created" in output
    assert "missing" in output


def test_per_user_section_absent_when_disabled(runtime_calls):
    config = _base_config(enabled=False, users=["alice"])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = ""
    runtime_calls["volume_stdout"] = ""

    console = Console(record=True, width=200)
    status_display.show_status("cfg.yml", console=console)
    output = console.export_text()

    assert "Web Terminal Users" not in output
    # volume ls should never be issued when the section doesn't render
    volume_calls = [c for c in runtime_calls["argvs"] if c[:2] == ["docker", "volume"]]
    assert len(volume_calls) == 0


def test_per_user_section_absent_when_no_users(runtime_calls):
    config = _base_config(enabled=True, users=[])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = ""
    runtime_calls["volume_stdout"] = ""

    console = Console(record=True, width=200)
    status_display.show_status("cfg.yml", console=console)
    output = console.export_text()

    assert "Web Terminal Users" not in output


def test_per_user_section_handles_object_form_users(runtime_calls):
    """modules.web_terminals.users entries may be {"name": ..., "index": ...} objects."""
    config = _base_config(users=[{"name": "dana", "index": 0}])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = _ps_stdout(_ps_container("dls-web-dana", "running"))
    claude_vol, agent_vol = status_display.resolve_user_volume_names(config, "dana")
    runtime_calls["volume_stdout"] = "\n".join([claude_vol, agent_vol])

    console = Console(record=True, width=200)
    status_display.show_status("cfg.yml", console=console)
    output = console.export_text()

    assert "dana" in output
    assert "Running" in output
    assert claude_vol in output


def test_existing_services_table_still_renders(runtime_calls):
    """The per-user section must not interfere with the existing services table."""
    config = _base_config(users=["alice"])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = _ps_stdout(
        _ps_container(
            "demo-project-service", "running", labels={"osprey.project.name": "demo-project"}
        )
    )
    runtime_calls["volume_stdout"] = ""

    console = Console(record=True, width=200)
    status_display.show_status("cfg.yml", console=console)
    output = console.export_text()

    assert "Service Status" in output
    assert "demo-project-service" in output
    assert "Web Terminal Users" in output
