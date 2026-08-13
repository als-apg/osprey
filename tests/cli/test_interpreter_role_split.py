"""Two interpreters, two rules: runtime is derived, only agent code reads config.

OSPREY launches two different kinds of Python and they must not be resolved the
same way:

* **OSPREY-runtime** — MCP server commands in ``.mcp.json``, framework hook
  commands in ``settings.json``, and the registry's ``{current_python_env}``
  substitution. Derived once and frozen into the generated artifact, so
  regenerating is what updates it.
* **Agent-authored code** — resolved by ``resolve_agent_interpreter`` on every
  call, so it tracks a project venv that appears after the build.

Neither reads config, and both prefer the project's own ``.venv``: for runtime
because ``osprey build`` installs ``osprey`` into it and a command pointing
inside the build output survives the framework tree moving, and for agent code
because the packages an operator installed are the packages agent code must be
able to import. They part company on *when* they resolve, not on what they read.

Config used to reach the runtime half through ``build_claude_code_context``'s
``config.execution.python_env_path or sys.executable``. That single ``or`` was
the last place a config file could name an OSPREY-runtime interpreter, and it
made every generated ``.mcp.json`` a snapshot of one machine: move the project,
mount it into a container, or rebuild the venv elsewhere and every MCP server
failed to launch.

These tests pin the split by pointing config at a decoy interpreter and proving
the derivation wins anyway. A test that only checked the default case would pass
with the bug fully intact, because the fallback and the correct answer are the
same value whenever config is silent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager

# Nonexistent, and obviously not the interpreter running these tests. Every
# assertion below reads "the generating interpreter won, not this".
DECOY = "/opt/facility/analysis-env/bin/python"

# The two ways a config can name an interpreter: the declaration the build
# records for provenance, and the retired key still sitting in already-deployed
# projects. Neither may reach a runtime launch site, together or apart.
DECOY_EXECUTION_BLOCKS: dict[str, dict] = {
    "environment-declaration": {
        "environment": {"python": DECOY, "packages": [], "inherit_exclude": []},
    },
    "retired-python-env-path": {"python_env_path": DECOY},
    "both-at-once": {
        "environment": {"python": DECOY, "packages": [], "inherit_exclude": []},
        "python_env_path": DECOY,
    },
}


def _regenerated_project(output_dir: Path, name: str, execution_overrides: dict) -> Path:
    """Build a project, force *execution_overrides* into it, and regenerate.

    Regeneration is the path that matters: it is the one that reads config.yml
    back off disk, so it is where a configured interpreter would get its chance
    to reach ``.mcp.json`` and the hooks block.
    """
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=name,
        output_dir=output_dir,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )

    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("execution", {}).update(execution_overrides)
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")

    manager.regenerate_claude_code(project_dir)
    return project_dir


def _framework_hook_commands(project_dir: Path) -> list[str]:
    """Return the framework ``osprey_*.py`` hook commands from settings.json."""
    settings = json.loads((project_dir / ".claude" / "settings.json").read_text(encoding="utf-8"))
    return [
        hook["command"]
        for event in ("PreToolUse", "PostToolUse")
        for entry in settings["hooks"].get(event, [])
        for hook in entry["hooks"]
        if "/.claude/hooks/osprey_" in hook["command"]
    ]


def _mcp_server_commands(project_dir: Path) -> list[str]:
    """Return every non-empty ``command`` from the generated .mcp.json."""
    mcp = json.loads((project_dir / ".mcp.json").read_text(encoding="utf-8"))
    return [server["command"] for server in mcp["mcpServers"].values() if server.get("command")]


def test_decoy_is_not_the_running_interpreter():
    """Anti-vacuous guard: the decoy must be a value that can actually fail."""
    assert DECOY != sys.executable


def _settings_commands(project_dir: Path) -> list[tuple[str, str]]:
    """Every ``(where, command)`` settings.json asks the shell to run."""
    settings = json.loads((project_dir / ".claude" / "settings.json").read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    status_line = settings.get("statusLine") or {}
    if status_line.get("command"):
        found.append(("statusLine", status_line["command"]))
    for event, entries in (settings.get("hooks") or {}).items():
        for entry in entries:
            for hook in entry.get("hooks", []):
                if hook.get("command"):
                    found.append((event, hook["command"]))
    return found


def test_no_settings_command_launches_a_bare_interpreter(tmp_path):
    """EVERY command settings.json runs must name a resolved interpreter.

    Enumerated from the rendered file rather than from a list of known sites,
    because the list is exactly what failed: two rounds of this fix each swept
    the sites someone thought of, and `statusLine` — the only launch site
    outside the `hooks` block — was missed by both. A bare `python` dies on a
    python3-only host, and it dies where nobody looks: the status line simply
    does not render.
    """
    project_dir = _regenerated_project(tmp_path, "bare-interpreter-scan", {})

    commands = _settings_commands(project_dir)
    assert commands, "expected settings.json to declare at least one command"

    offenders = [
        f"{where}: {command}"
        for where, command in commands
        if command.split()[0].strip('"') in {"python", "python3"}
    ]
    assert offenders == [], (
        "settings.json commands must not launch a bare interpreter:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("variant", sorted(DECOY_EXECUTION_BLOCKS))
class TestConfigCannotNameARuntimeInterpreter:
    """With config pointing elsewhere, every runtime launch site still derives."""

    @pytest.fixture
    def project_dir(self, tmp_path, variant) -> Path:
        return _regenerated_project(
            tmp_path, f"role-split-{variant}", DECOY_EXECUTION_BLOCKS[variant]
        )

    def test_mcp_server_commands_use_the_generating_interpreter(self, project_dir):
        commands = _mcp_server_commands(project_dir)

        assert commands, "expected at least one launched MCP server in .mcp.json"
        assert set(commands) == {sys.executable}

    def test_framework_hook_commands_use_the_generating_interpreter(self, project_dir):
        commands = _framework_hook_commands(project_dir)

        assert commands, "expected framework osprey_*.py hook commands after regen"
        for command in commands:
            assert command.startswith(f'"{sys.executable}" '), (
                f"hook must launch via the generating interpreter, got: {command}"
            )

    def test_registry_placeholder_substitutes_the_generating_interpreter(
        self, project_dir, variant
    ):
        """The third consumer of the key: ``{current_python_env}`` substitution.

        No bundled server definition uses the placeholder today, so this drives
        the substitution seam directly — it is still a live code path that any
        facility-defined external server can reach.
        """
        from osprey.registry.mcp import _resolve_placeholder

        manager = TemplateManager()
        config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project_dir, config
        )

        resolved = _resolve_placeholder("{current_python_env} -m osprey.mcp_server.custom", ctx)

        assert resolved == f"{sys.executable} -m osprey.mcp_server.custom"

    def test_the_decoy_reaches_no_generated_artifact(self, project_dir):
        """Whole-file sweep: the configured path appears in no rendered artifact.

        config.yml itself is excluded — recording the declaration there is task
        3.5's deliberate provenance, and is not a launch site.
        """
        artifacts = [
            project_dir / ".mcp.json",
            project_dir / ".claude" / "settings.json",
        ]

        for artifact in artifacts:
            assert DECOY not in artifact.read_text(encoding="utf-8"), (
                f"configured interpreter leaked into {artifact.name}"
            )


class TestDerivationHoldsWithoutAnyDeclaration:
    """The default case still has to work — it just cannot prove the rule alone."""

    @pytest.fixture(scope="class")
    def project_dir(self, tmp_path_factory) -> Path:
        return _regenerated_project(
            tmp_path_factory.mktemp("role-split-plain"), "role-split-plain", {}
        )

    def test_mcp_server_commands_use_the_generating_interpreter(self, project_dir):
        assert set(_mcp_server_commands(project_dir)) == {sys.executable}

    def test_framework_hook_commands_use_the_generating_interpreter(self, project_dir):
        commands = _framework_hook_commands(project_dir)

        assert commands
        assert all(c.startswith(f'"{sys.executable}" ') for c in commands)


def _fake_project_venv(project_dir: Path) -> Path:
    """Give *project_dir* a venv-shaped interpreter and return its path.

    ``osprey build`` installs ``osprey`` into the real thing; nothing here
    executes it, so a stub is enough to exercise the derivation.
    """
    venv_python = project_dir / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    venv_python.chmod(0o755)
    return venv_python


class TestAProjectVenvWinsOverTheGeneratingInterpreter:
    """A built project's commands point inside the build output, not at the builder.

    ``osprey build`` installs ``osprey`` into the project venv precisely so the
    project stops depending on the tree that built it. A command naming the
    framework's own interpreter breaks as soon as that framework checkout is
    moved or deleted — and it is the *builder's* path, so it is wrong on every
    machine but one.
    """

    @pytest.fixture(scope="class")
    def project_dir(self, tmp_path_factory) -> Path:
        output_dir = tmp_path_factory.mktemp("role-split-with-venv")
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="role-split-with-venv",
            output_dir=output_dir,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        _fake_project_venv(project_dir)
        manager.regenerate_claude_code(project_dir)
        return project_dir

    def test_mcp_commands_point_at_the_project_venv(self, project_dir):
        expected = str(project_dir / ".venv" / "bin" / "python")

        assert set(_mcp_server_commands(project_dir)) == {expected}

    def test_mcp_commands_stay_inside_the_project(self, project_dir):
        """The contract tests/integration/test_build_boot.py enforces end to end."""
        for command in _mcp_server_commands(project_dir):
            assert command.startswith(f"{project_dir}/"), (
                f"runtime command escapes the project dir: {command}"
            )

    def test_a_config_decoy_still_loses_to_the_project_venv(self, tmp_path):
        """Config loses to the derivation whether or not a project venv exists."""
        project_dir = _regenerated_project(tmp_path, "role-split-venv-decoy", {})
        venv_python = _fake_project_venv(project_dir)

        config_path = project_dir / "config.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config.setdefault("execution", {})["python_env_path"] = DECOY
        config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
        TemplateManager().regenerate_claude_code(project_dir)

        assert set(_mcp_server_commands(project_dir)) == {str(venv_python)}


class TestAStagedRenderNamesTheVenvItWillRunFrom:
    """A render written somewhere other than where it will run from.

    ``osprey build`` renders into ``<repo>/build/.tmp/<repo>`` and swaps that
    tree into ``<repo>/build`` by rename, but the venv is created at
    ``<repo>/build/.venv`` — its final path, because a virtual environment
    records its own absolute location — and only moves into the staged tree as
    part of the swap. So at the moment the Claude Code artifacts are rendered
    there is no ``.venv`` beside them, and the interpreter every MCP server
    launches with has to be named from the output zone instead.

    The render also carries a ``project_root`` override, because the repo root
    is the project root under this layout and it is one level above the tree
    being written. That override says nothing about *which machine* the render
    is for, which is why the venv's home is stated separately.
    """

    @staticmethod
    def _staged_repo(root: Path, name: str) -> tuple[Path, Path]:
        """Render a project into a staging tree; return (output zone, stage)."""
        manager = TemplateManager()
        output_zone = root / "build"
        stage = manager.create_project(
            project_name=name,
            output_dir=output_zone / ".tmp",
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        return output_zone, stage

    def test_staged_render_points_at_the_output_zone_venv(self, tmp_path):
        output_zone, stage = self._staged_repo(tmp_path, "staged-render")
        venv_python = _fake_project_venv(output_zone)
        assert not (stage / ".venv").exists(), "the staged tree must have no venv of its own"

        TemplateManager().regenerate_claude_code(
            stage,
            project_root_override=str(tmp_path),
            runtime_venv_dir=output_zone,
        )

        assert set(_mcp_server_commands(stage)) == {str(venv_python)}

    def test_a_staged_render_for_a_container_still_uses_the_generating_interpreter(self, tmp_path):
        """``--runtime-root`` names no local venv, so the fallback still applies."""
        output_zone, stage = self._staged_repo(tmp_path, "staged-container")
        venv_python = _fake_project_venv(output_zone)

        TemplateManager().regenerate_claude_code(
            stage,
            project_root_override="/app/staged-container",
        )

        assert set(_mcp_server_commands(stage)) == {sys.executable}
        assert str(venv_python) not in (stage / ".mcp.json").read_text(encoding="utf-8")

    def test_a_staged_render_without_a_venv_falls_back(self, tmp_path):
        """``--skip-deps`` builds no venv; the generating interpreter has osprey."""
        output_zone, stage = self._staged_repo(tmp_path, "staged-no-venv")
        assert not (output_zone / ".venv").exists()

        TemplateManager().regenerate_claude_code(
            stage,
            project_root_override=str(tmp_path),
            runtime_venv_dir=output_zone,
        )

        assert set(_mcp_server_commands(stage)) == {sys.executable}


class TestRenderingForAContainerUsesTheGeneratingInterpreter:
    """With a runtime root elsewhere, the host's venv is not the right answer.

    A container mounts the project at its own path, so the host ``.venv`` is
    neither what will exist at run time nor probeable from the host. Baking it in
    is the failure mode that broke every MCP server in the container.
    """

    def test_project_root_override_falls_back_to_the_generating_interpreter(self, tmp_path):
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="role-split-container",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        venv_python = _fake_project_venv(project_dir)

        config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
        ctx = claude_code.build_claude_code_context(
            manager.template_root,
            manager.jinja_env,
            project_dir,
            config,
            project_root_override="/app/role-split-container",
        )

        assert ctx["current_python_env"] == sys.executable
        assert ctx["current_python_env"] != str(venv_python)


class TestAgentCodeResolvesAtRunTimeNotGenerationTime:
    """The half of the split that a generated artifact cannot capture.

    Both halves derive rather than read config, and both prefer a project venv —
    but the runtime value is frozen into ``.mcp.json`` when the artifact is
    rendered, while agent code re-resolves on every call. A venv created after
    the build is therefore picked up by agent code and not by the generated
    artifact, which is exactly why ``osprey build`` exists.
    """

    @pytest.fixture(scope="class")
    def project_then_venv(self, tmp_path_factory) -> tuple[Path, Path]:
        output_dir = tmp_path_factory.mktemp("role-split-late-venv")
        project_dir = _regenerated_project(output_dir, "role-split-late-venv", {})
        return project_dir, _fake_project_venv(project_dir)

    def test_agent_code_picks_up_the_late_venv(self, project_then_venv):
        from osprey.mcp_server.python_executor.executor import resolve_agent_interpreter

        project_dir, venv_python = project_then_venv

        assert resolve_agent_interpreter(project_root=project_dir) == venv_python

    def test_the_already_rendered_artifact_still_names_the_generating_interpreter(
        self, project_then_venv
    ):
        project_dir, _ = project_then_venv

        assert set(_mcp_server_commands(project_dir)) == {sys.executable}

    def test_regenerating_adopts_the_late_venv(self, tmp_path):
        """Own project: regenerating is what re-freezes the runtime value."""
        project_dir = _regenerated_project(tmp_path, "role-split-late-venv-regen", {})
        assert set(_mcp_server_commands(project_dir)) == {sys.executable}

        venv_python = _fake_project_venv(project_dir)
        TemplateManager().regenerate_claude_code(project_dir)

        assert set(_mcp_server_commands(project_dir)) == {str(venv_python)}
