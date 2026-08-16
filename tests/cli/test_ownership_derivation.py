"""Symmetric ownership derivation (FR-7).

The build registers exactly what the conventions copied as
``scaffold.user_owned`` — one rule for every artifact class, skills and
services owned as whole directories — regen skips and prune never unlinks an
owned path, a uniform notice announces any framework-selected artifact a
profile file overrides, and a persona exclusion restores the framework render.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_persistence import (
    _apply_conventions,
    _register_convention_artifacts,
)
from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.errors import BuildProfileError


def _write(path: Path, content: str = "# artifact\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    root = tmp_path / "my-profile"
    root.mkdir()
    _write(root / "profile.yml", "name: my-profile\n")
    return root


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    root = tmp_path / "my-project"
    root.mkdir()
    return root


def _user_owned(project_path: Path) -> list[str]:
    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    return config["scaffold"]["user_owned"]


# ── is_user_owned: one rule for every artifact class ─────────────────


def test_catalog_artifact_owned_by_canonical_name():
    ctx = {"user_owned": ["rules/facility"]}
    assert claude_code.is_user_owned(".claude/rules/facility.md", ctx)


def test_agents_and_skills_have_no_hard_exclusion():
    """Agents and skills follow the same ownership rule as everything else."""
    ctx = {"user_owned": ["agents/orbit-writer", "skills/orbit-check"]}
    assert claude_code.is_user_owned(".claude/agents/orbit-writer.md", ctx)
    assert claude_code.is_user_owned(".claude/skills/orbit-check/SKILL.md", ctx)


def test_directory_entry_owns_every_file_beneath_it():
    """Whole-directory ownership is shape, not special-casing — non-.md too."""
    ctx = {"user_owned": ["skills/orbit-check", "services/beam-logger"]}
    assert claude_code.is_user_owned(".claude/skills/orbit-check/scripts/run.py", ctx)
    assert claude_code.is_user_owned("services/beam-logger/docker-compose.yml", ctx)


def test_unowned_paths_stay_framework_managed():
    ctx = {"user_owned": ["rules/facility-ops"]}
    assert not claude_code.is_user_owned(".claude/rules/safety.md", ctx)
    assert not claude_code.is_user_owned(".claude/skills/other-skill/SKILL.md", ctx)


def test_empty_user_owned_owns_nothing():
    assert not claude_code.is_user_owned(".claude/rules/facility.md", {"user_owned": []})


# ── Registration: derived from the applied set, per class ────────────


def test_every_ownable_class_registers(profile_dir: Path, project_path: Path):
    _write(project_path / "config.yml", yaml.safe_dump({"scaffold": {"user_owned": []}}))
    _write(profile_dir / "rules" / "facility-ops.md")
    _write(profile_dir / "agents" / "orbit-writer.md")
    _write(profile_dir / "commands" / "scan.md")
    _write(profile_dir / "output-styles" / "terse.md")
    _write(profile_dir / "skills" / "orbit-check" / "SKILL.md")
    _write(profile_dir / "skills" / "orbit-check" / "scripts" / "run.py", "print()\n")
    _write(profile_dir / "services" / "beam-logger" / "docker-compose.yml", "services: {}\n")
    _write(profile_dir / "mcp_servers" / "finder" / "server.py", "# srv\n")
    _write(profile_dir / "web-terminal-context" / "alice" / "notes.md")
    _write(profile_dir / "project" / "docs" / "runbook.md")

    applied = _apply_conventions(profile_dir, project_path, ["alice"])
    registered = _register_convention_artifacts(project_path, applied)

    assert set(_user_owned(project_path)) == {
        "rules/facility-ops",
        "agents/orbit-writer",
        "commands/scan",
        "output-styles/terse",
        "skills/orbit-check",
        "services/beam-logger",
    }
    # MCP servers register through claude_code.servers, per-user context is
    # roster-derived, and a mirror file landing outside .claude/ (docs/ here)
    # has no render that could contest it.
    assert registered == 6


def test_mirror_file_landing_under_claude_is_owned(profile_dir: Path, project_path: Path):
    """Ownership follows the destination: a mirror file in .claude/ is owned too.

    Nothing else protects it — no convention dir and no framework output claim
    .claude/probes/, so without ownership the prune branches could unlink it.
    """
    _write(project_path / "config.yml", yaml.safe_dump({"scaffold": {"user_owned": []}}))
    _write(profile_dir / "project" / ".claude" / "probes" / "custom.py", "print()\n")

    applied = _apply_conventions(profile_dir, project_path)
    _register_convention_artifacts(project_path, applied)

    assert _user_owned(project_path) == ["probes/custom.py"]
    assert claude_code.is_user_owned(
        ".claude/probes/custom.py", {"user_owned": ["probes/custom.py"]}
    )


def test_mirror_may_not_write_the_hooks_subtree(profile_dir: Path, project_path: Path):
    """`.claude/hooks/` has a channel now, so the mirror is not a second writer.

    Uniform with `.claude/rules/` and every other convention destination: two
    writers on one artifact give it two names — `project/.claude/hooks/x.py`
    and `hooks/x.py` — and a persona excluding one would silently leave the
    other standing.
    """
    _write(project_path / "config.yml", yaml.safe_dump({"scaffold": {"user_owned": []}}))
    _write(profile_dir / "project" / ".claude" / "hooks" / "custom.py", "print()\n")

    with pytest.raises(BuildProfileError) as excinfo:
        _apply_conventions(profile_dir, project_path)

    message = str(excinfo.value)
    assert "project/.claude/hooks/custom.py" in message
    assert "`hooks/`" in message
    # The migration a profile that predates the channel has to perform, stated
    # as two paths it can act on rather than as a channel name to interpret.
    assert "Move it: project/.claude/hooks/custom.py → hooks/custom.py" in message


# ── Shadow notices ───────────────────────────────────────────────────


def test_shadowed_file_gets_a_uniform_notice(
    profile_dir: Path, project_path: Path, caplog: pytest.LogCaptureFixture
):
    _write(project_path / ".claude" / "rules" / "safety.md", "# framework\n")
    _write(profile_dir / "rules" / "safety.md", "# profile\n")

    # Renderer migration (disposition row 35): the shadow notice stayed on the
    # logger and dropped to DEBUG, so `-v` is what surfaces it.
    with caplog.at_level(logging.DEBUG):
        _apply_conventions(profile_dir, project_path)

    assert "overrides framework rule 'rules/safety'" in caplog.text
    assert (project_path / ".claude" / "rules" / "safety.md").read_text() == "# profile\n"


def test_shadowed_directory_gets_a_uniform_notice(
    profile_dir: Path, project_path: Path, caplog: pytest.LogCaptureFixture
):
    _write(project_path / ".claude" / "skills" / "orbit-check" / "SKILL.md", "# framework\n")
    _write(profile_dir / "skills" / "orbit-check" / "SKILL.md", "# profile\n")

    # Renderer migration (disposition row 35): the shadow notice stayed on the
    # logger and dropped to DEBUG, so `-v` is what surfaces it.
    with caplog.at_level(logging.DEBUG):
        _apply_conventions(profile_dir, project_path)

    assert "overrides framework skill 'skills/orbit-check'" in caplog.text


def test_unshadowed_copies_get_no_notice(
    profile_dir: Path, project_path: Path, caplog: pytest.LogCaptureFixture
):
    _write(profile_dir / "rules" / "facility-ops.md")

    # DEBUG, not INFO: row 35 moved the notice down a level, and capturing above
    # it would pass here no matter what the build did.
    with caplog.at_level(logging.DEBUG):
        _apply_conventions(profile_dir, project_path)

    assert "overrides framework" not in caplog.text


# ── Persona exclusion: post-exclude ownership ────────────────────────


def test_excluded_shadow_restores_the_framework_render(
    profile_dir: Path, project_path: Path, caplog: pytest.LogCaptureFixture
):
    _write(project_path / ".claude" / "rules" / "safety.md", "# framework\n")
    _write(profile_dir / "rules" / "safety.md", "# profile\n")
    _write(profile_dir / "rules" / "facility-ops.md")

    # DEBUG, not INFO: row 35 moved the notice down a level, and capturing above
    # it would pass here no matter what the build did.
    with caplog.at_level(logging.DEBUG):
        applied = _apply_conventions(profile_dir, project_path, excluded={"rules/safety"})

    assert (project_path / ".claude" / "rules" / "safety.md").read_text() == "# framework\n"
    assert "overrides framework" not in caplog.text
    # Post-exclude: the excluded artifact is neither applied nor owned.
    assert applied.owned_artifacts == ["rules/facility-ops"]
    assert applied.by_category.get("rules") == 1


def test_excluded_directory_artifact_is_skipped(profile_dir: Path, project_path: Path):
    _write(profile_dir / "skills" / "orbit-check" / "SKILL.md")

    applied = _apply_conventions(profile_dir, project_path, excluded={"skills/orbit-check"})

    assert not (project_path / ".claude" / "skills" / "orbit-check").exists()
    assert applied.owned_artifacts == []


def test_excluded_entries_use_the_directory_spelling(profile_dir: Path, project_path: Path):
    """`exclude:` records name the convention DIRECTORY — "output-styles/terse", hyphenated."""
    _write(profile_dir / "output-styles" / "terse.md")

    applied = _apply_conventions(profile_dir, project_path, excluded={"output-styles/terse"})

    assert not (project_path / ".claude" / "output-styles" / "terse.md").exists()
    assert applied.owned_artifacts == []


def test_excluding_a_namespaced_command_leaves_its_same_stem_sibling(
    profile_dir: Path, project_path: Path
):
    """Markdown categories nest; the exclude name keeps the namespace.

    A basename-derived name would collapse `commands/osprey/scan` and
    `commands/ops/scan` into one exclude key, silently dropping both.
    """
    _write(profile_dir / "commands" / "osprey" / "scan.md")
    _write(profile_dir / "commands" / "ops" / "scan.md")

    applied = _apply_conventions(profile_dir, project_path, excluded={"commands/osprey/scan"})

    assert not (project_path / ".claude" / "commands" / "osprey" / "scan.md").exists()
    assert (project_path / ".claude" / "commands" / "ops" / "scan.md").is_file()
    # The surviving sibling is owned under the same (namespaced) name.
    assert applied.owned_artifacts == ["commands/ops/scan"]


def test_excluded_mirror_entry_keeps_its_nested_path(profile_dir: Path, project_path: Path):
    """A `project/` mirror exclusion names the full nested path, not one segment."""
    _write(profile_dir / "project" / "docs" / "runbook.md")
    _write(profile_dir / "project" / "docs" / "oncall.md")

    applied = _apply_conventions(profile_dir, project_path, excluded={"project/docs/runbook.md"})

    assert not (project_path / "docs" / "runbook.md").exists()
    assert (project_path / "docs" / "oncall.md").is_file()
    assert applied.by_category.get("project") == 1


# ── Regen and prune honor derived ownership ──────────────────────────


def _rendered_project(tmp_path: Path, name: str) -> Path:
    return TemplateManager().create_project(
        project_name=name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )


def _shadow_framework_artifacts(project: Path, profile: Path) -> tuple[str, str]:
    """Shadow one framework agent and one framework skill; return their names."""
    agents = sorted((project / ".claude" / "agents").glob("*.md"))
    assert agents, "rendered project should carry framework agents"
    agent_name = agents[0].stem
    _write(profile / "agents" / f"{agent_name}.md", "# profile agent\n")

    skills = sorted(p.name for p in (project / ".claude" / "skills").iterdir() if p.is_dir())
    assert skills, "rendered project should carry framework skills"
    skill_name = skills[0]
    _write(profile / "skills" / skill_name / "SKILL.md", "# profile skill\n")
    _write(profile / "skills" / skill_name / "helper.py", "print('profile')\n")
    return agent_name, skill_name


def _regen(project: Path, allowed_outputs: set[str] | None = None) -> None:
    manager = TemplateManager()
    config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project, config
    )
    claude_code.create_claude_code_integration(
        manager.template_root, manager.jinja_env, project, ctx, allowed_outputs
    )


def test_owned_artifacts_survive_regen(tmp_path: Path):
    """One artifact of each class claimed → a full re-render leaves it alone."""
    project = _rendered_project(tmp_path, "own-regen")
    profile = tmp_path / "profile"
    _write(profile / "rules" / "safety.md", "# profile safety\n")
    agent_name, skill_name = _shadow_framework_artifacts(project, profile)

    applied = _apply_conventions(profile, project)
    _register_convention_artifacts(project, applied)

    _regen(project)

    claude = project / ".claude"
    assert (claude / "rules" / "safety.md").read_text() == "# profile safety\n"
    assert (claude / "agents" / f"{agent_name}.md").read_text() == "# profile agent\n"
    assert (claude / "skills" / skill_name / "SKILL.md").read_text() == "# profile skill\n"
    assert (claude / "skills" / skill_name / "helper.py").is_file()


def test_prune_never_unlinks_owned_artifacts(tmp_path: Path):
    """An empty manifest prunes framework files but never owned ones."""
    project = _rendered_project(tmp_path, "own-prune")
    profile = tmp_path / "profile"
    _write(profile / "rules" / "safety.md", "# profile safety\n")
    agent_name, skill_name = _shadow_framework_artifacts(project, profile)

    applied = _apply_conventions(profile, project)
    _register_convention_artifacts(project, applied)

    pruned_probe = project / ".claude" / "rules" / "timezone.md"
    assert pruned_probe.is_file(), "probe for the prune pass should exist pre-regen"

    _regen(project, allowed_outputs=set())

    assert not pruned_probe.exists(), "empty manifest should prune unowned files"
    claude = project / ".claude"
    assert (claude / "rules" / "safety.md").read_text() == "# profile safety\n"
    assert (claude / "agents" / f"{agent_name}.md").read_text() == "# profile agent\n"
    assert (claude / "skills" / skill_name / "SKILL.md").read_text() == "# profile skill\n"
    assert (claude / "skills" / skill_name / "helper.py").is_file()
