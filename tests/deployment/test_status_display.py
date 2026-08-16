"""Tests for the per-user web terminal section of ``osprey.deployment.status_display.show_status``.

Mocks the container runtime entirely: ``subprocess.run`` is patched to return canned
``docker ps -a --format json`` output for the first call and canned ``docker volume ls``
output for the second, so these tests never touch a real container runtime.
"""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from osprey.cli import styles
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.styles import osprey_theme
from osprey.deployment import status_display

_CONFIGS: dict[str, dict] = {}


def _fake_load_project_config(config_path, **kwargs):
    """Stand-in for the shared project-config loader keyed by the path passed in."""
    return _CONFIGS[config_path]


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
def _patch_config_loader(monkeypatch):
    monkeypatch.setattr(status_display, "load_project_config", _fake_load_project_config)
    yield
    _CONFIGS.clear()


@pytest.fixture(autouse=True)
def rendered(monkeypatch):
    """Capture BOTH renderer streams into one wide recording console.

    ``show_status`` prints through :mod:`osprey.cli.output`, which resolves its
    console per call -- report/note/section through the installed reporter,
    warnings and failures through ``styles.err_console``. Both are pointed here
    so an assertion reads what an operator sees on either stream. Wide, because
    a wrapped sentence defeats a substring assertion.
    """
    console = Console(record=True, width=200, theme=osprey_theme)

    class _Recording(PhaseReporter):
        # Untyped on purpose: rich's ``Console`` is Any to this repo's mypy
        # settings, so an annotated override trips ``no-any-return``.
        def out(self):
            return console

    previous = install_reporter(_Recording(color=False))
    monkeypatch.setattr(styles, "err_console", console)
    yield console
    install_reporter(previous)


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


def test_per_user_section_reports_running_and_present(runtime_calls, rendered):
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

    status_display.show_status("cfg.yml")
    output = rendered.export_text()

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


def test_per_user_section_reports_not_created(runtime_calls, rendered):
    config = _base_config(users=["carol"])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = ""  # no containers at all
    runtime_calls["volume_stdout"] = ""  # no volumes at all

    status_display.show_status("cfg.yml")
    output = rendered.export_text()

    assert "Web Terminal Users" in output
    assert "Not created" in output
    assert "missing" in output


def test_per_user_section_absent_when_disabled(runtime_calls, rendered):
    config = _base_config(enabled=False, users=["alice"])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = ""
    runtime_calls["volume_stdout"] = ""

    status_display.show_status("cfg.yml")
    output = rendered.export_text()

    assert "Web Terminal Users" not in output
    # volume ls should never be issued when the section doesn't render
    volume_calls = [c for c in runtime_calls["argvs"] if c[:2] == ["docker", "volume"]]
    assert len(volume_calls) == 0


def test_per_user_section_absent_when_no_users(runtime_calls, rendered):
    config = _base_config(enabled=True, users=[])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = ""
    runtime_calls["volume_stdout"] = ""

    status_display.show_status("cfg.yml")
    output = rendered.export_text()

    assert "Web Terminal Users" not in output


def test_per_user_section_handles_object_form_users(runtime_calls, rendered):
    """modules.web_terminals.users entries may be {"name": ..., "index": ...} objects."""
    config = _base_config(users=[{"name": "dana", "index": 0}])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = _ps_stdout(_ps_container("dls-web-dana", "running"))
    claude_vol, agent_vol = status_display.resolve_user_volume_names(config, "dana")
    runtime_calls["volume_stdout"] = "\n".join([claude_vol, agent_vol])

    status_display.show_status("cfg.yml")
    output = rendered.export_text()

    assert "dana" in output
    assert "Running" in output
    assert claude_vol in output


def test_existing_services_table_still_renders(runtime_calls, rendered):
    """The per-user section must not interfere with the existing services table."""
    config = _base_config(users=["alice"])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = _ps_stdout(
        _ps_container(
            "demo-project-service", "running", labels={"osprey.project.name": "demo-project"}
        )
    )
    runtime_calls["volume_stdout"] = ""

    status_display.show_status("cfg.yml")
    output = rendered.export_text()

    assert "Service Status" in output
    assert "demo-project-service" in output
    assert "Web Terminal Users" in output


def test_table_cells_carry_no_markup_tags(runtime_calls, rendered):
    """Cells are pre-styled ``Text``, never markup strings.

    ``output.table`` prints with ``markup=False`` and Rich propagates that into
    every nested renderable, so a cell built as ``"[success]● Running[/success]"``
    would print its own tags AND count them toward the column width. This is the
    guard that nobody reintroduces one.
    """
    config = _base_config(users=["alice"])
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = _ps_stdout(
        _ps_container("dls-web-alice", "running", labels={"osprey.project.name": "demo-project"})
    )
    runtime_calls["volume_stdout"] = ""

    status_display.show_status("cfg.yml")
    output = rendered.export_text()

    assert "● Running" in output
    assert "missing" in output  # the other pre-styled cell builder
    for tag in ("[success]", "[error]", "[warning]", "[dim]"):
        assert tag not in output


def test_project_containers_match_the_normalized_project_name(runtime_calls, rendered):
    """A project whose name compose normalized still owns its own containers.

    Labels are stamped with ``resolve_project_name(config)`` — the normalized
    form — so status must resolve the same way. Comparing the raw
    ``project_name`` instead files the project's own containers under "Other
    Osprey Containers".
    """
    config = _base_config(enabled=False)
    config["project_name"] = "Demo_Project!"
    _register_config("cfg.yml", config)

    runtime_calls["ps_stdout"] = _ps_stdout(
        _ps_container(
            "demo_project-service", "running", labels={"osprey.project.name": "demo_project"}
        )
    )

    status_display.show_status("cfg.yml")
    output = rendered.export_text()

    assert "demo_project-service" in output
    assert "Other Osprey Containers" not in output


# ---------------------------------------------------------------------------
# Agent section: the provider spec is read from the render, resolved from .env
# ---------------------------------------------------------------------------


def test_agent_section_expands_provider_placeholders_from_the_repo_env(
    tmp_path, monkeypatch, rendered
):
    """A ``${VAR}`` in the render's provider block resolves from ``<repo>/.env``.

    The render is disposable and holds no secrets; the deployment keeps them at
    its root, which is where ``_auth_availability`` already looks. Resolving
    the spec against the render instead would report a facility gateway as the
    literal ``${FACILITY_GATEWAY_URL}`` — a status line no operator can act on.
    """
    monkeypatch.delenv("FACILITY_GATEWAY_URL", raising=False)
    monkeypatch.setattr(status_display, "_artifact_drift", lambda *a, **k: None)

    repo_root = tmp_path / "facility"
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "config.yml").write_text(
        "api:\n"
        "  providers:\n"
        "    cborg:\n"
        "      base_url: ${FACILITY_GATEWAY_URL}\n"
        "claude_code:\n"
        "  provider: cborg\n",
        encoding="utf-8",
    )
    (repo_root / ".env").write_text("FACILITY_GATEWAY_URL=https://gw.test/v1\n", encoding="utf-8")

    status_display._print_agent_section(
        repo_root,
        build_dir,
        {"claude_code": {"provider": "cborg"}},
        show_agents=False,
    )
    output = rendered.export_text()

    # A section row, so the variable and its resolved value are two columns
    # rather than one ``KEY = value`` sentence.
    assert "ANTHROPIC_BASE_URL" in output
    assert "https://gw.test" in output
    assert "${FACILITY_GATEWAY_URL}" not in output
