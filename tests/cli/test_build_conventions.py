"""Applying a profile's convention directories at build time.

The mapping table itself is pinned in ``test_profile_conventions.py``; this
module pins what the build *does* with it — one case per category, the
``project/`` mirror's reserved-path rejection, the roster-derived per-user
context, and the ownership registration that follows a copy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_persistence import (
    _apply_conventions,
    _register_convention_artifacts,
    _resolve_context_roster,
)
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
    """A bare project directory — the conventions only need a root to land in."""
    root = tmp_path / "my-project"
    root.mkdir()
    return root


def _write_config(project_path: Path, config: dict) -> Path:
    return _write(project_path / "config.yml", yaml.safe_dump(config, sort_keys=False))


# ── One case per category ────────────────────────────────────────────


def test_markdown_categories_copy_file_by_file(profile_dir: Path, project_path: Path):
    _write(profile_dir / "rules" / "facility-ops.md")
    _write(profile_dir / "agents" / "orbit-writer.md")
    _write(profile_dir / "commands" / "ops" / "restart.md")
    _write(profile_dir / "output-styles" / "terse.md")

    applied = _apply_conventions(profile_dir, project_path)

    assert (project_path / ".claude/rules/facility-ops.md").is_file()
    assert (project_path / ".claude/agents/orbit-writer.md").is_file()
    assert (project_path / ".claude/commands/ops/restart.md").is_file()
    assert (project_path / ".claude/output-styles/terse.md").is_file()
    assert applied.copied == 4


def test_hooks_copy_file_by_file_and_keep_their_mode(profile_dir: Path, project_path: Path):
    """A hook is a script the agent runs, so its executable bit has to survive."""
    hook = _write(profile_dir / "hooks" / "facility_guard.py", "#!/usr/bin/env python3\n")
    hook.chmod(0o755)

    applied = _apply_conventions(profile_dir, project_path)

    landed = project_path / ".claude/hooks/facility_guard.py"
    assert landed.is_file()
    assert landed.stat().st_mode & 0o111
    assert applied.by_category == {"hooks": 1}
    assert applied.owned_artifacts == ["hooks/facility_guard.py"]


def test_skills_copy_as_whole_directories(profile_dir: Path, project_path: Path):
    _write(profile_dir / "skills" / "orbit-check" / "SKILL.md")
    _write(profile_dir / "skills" / "orbit-check" / "scripts" / "check.py", "print('hi')\n")

    applied = _apply_conventions(profile_dir, project_path)

    assert (project_path / ".claude/skills/orbit-check/SKILL.md").is_file()
    assert (project_path / ".claude/skills/orbit-check/scripts/check.py").is_file()
    # One copy, not one per file — the skill is claimed as a unit.
    assert applied.by_category == {"skills": 1}


def test_services_and_mcp_servers_copy_as_whole_directories(profile_dir: Path, project_path: Path):
    _write(profile_dir / "services" / "archiver" / "docker-compose.yml", "services: {}\n")
    _write(profile_dir / "mcp_servers" / "matlab" / "server.py", "x = 1\n")

    _apply_conventions(profile_dir, project_path)

    assert (project_path / "services/archiver/docker-compose.yml").is_file()
    assert (project_path / "_mcp_servers/matlab/server.py").is_file()


def test_project_mirror_lands_verbatim_on_the_root(profile_dir: Path, project_path: Path):
    _write(profile_dir / "project" / "docs" / "runbook.md")
    _write(profile_dir / "project" / "Makefile", "all:\n\techo hi\n")
    _write(profile_dir / "project" / ".gitignore", "_agent_data/\n")

    applied = _apply_conventions(profile_dir, project_path)

    assert (project_path / "docs/runbook.md").is_file()
    assert (project_path / "Makefile").read_text(encoding="utf-8").startswith("all:")
    assert (project_path / ".gitignore").is_file()
    assert applied.by_category == {"project": 3}


def test_directory_copy_replaces_a_stale_destination(profile_dir: Path, project_path: Path):
    _write(profile_dir / "skills" / "orbit-check" / "SKILL.md", "# new\n")
    _write(project_path / ".claude/skills/orbit-check/SKILL.md", "# stale\n")
    _write(project_path / ".claude/skills/orbit-check/gone.md", "# removed upstream\n")

    _apply_conventions(profile_dir, project_path)

    skill_dir = project_path / ".claude/skills/orbit-check"
    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == "# new\n"
    assert not (skill_dir / "gone.md").exists()


def test_nothing_to_apply_is_not_an_error(profile_dir: Path, project_path: Path):
    applied = _apply_conventions(profile_dir, project_path)
    assert applied.copied == 0
    assert applied.by_category == {}


# ── project/ mirror reserved paths ───────────────────────────────────


def test_reserved_mirror_path_fails_naming_its_channel(profile_dir: Path, project_path: Path):
    _write(profile_dir / "project" / "config.yml", "facility: {}\n")

    with pytest.raises(BuildProfileError) as excinfo:
        _apply_conventions(profile_dir, project_path)

    message = str(excinfo.value)
    assert "project/config.yml" in message
    assert "`config:` block" in message


def test_reserved_mirror_path_rejects_before_copying_anything(
    profile_dir: Path, project_path: Path
):
    """Validation is up front: a bad mirror leaves the project untouched."""
    _write(profile_dir / "rules" / "facility-ops.md")
    _write(profile_dir / "project" / ".mcp.json", "{}\n")

    with pytest.raises(BuildProfileError):
        _apply_conventions(profile_dir, project_path)

    assert not (project_path / ".claude/rules/facility-ops.md").exists()


def test_misshapen_convention_source_fails_before_copying(profile_dir: Path, project_path: Path):
    _write(profile_dir / "rules" / "facility-ops.md")
    _write(profile_dir / "skills" / "loose.md")

    with pytest.raises(BuildProfileError, match="one directory per skill"):
        _apply_conventions(profile_dir, project_path)

    assert not (project_path / ".claude/rules/facility-ops.md").exists()


def test_symlink_escaping_the_profile_fails_the_build(
    profile_dir: Path, project_path: Path, tmp_path: Path
):
    """A profile artifact must never smuggle content from outside the profile."""
    outside = _write(tmp_path / "outside" / "safety.md")
    _write(profile_dir / "rules" / "facility-ops.md")
    (profile_dir / "rules" / "safety.md").symlink_to(outside)

    with pytest.raises(BuildProfileError, match="self-contained"):
        _apply_conventions(profile_dir, project_path)

    assert not (project_path / ".claude/rules/facility-ops.md").exists()


def test_symlinked_destination_escaping_the_project_fails_the_build(
    profile_dir: Path, project_path: Path, tmp_path: Path
):
    """A pre-existing symlink at a destination path must not redirect the copy."""
    _write(profile_dir / "rules" / "facility-ops.md")
    outside = tmp_path / "outside-rules"
    outside.mkdir()
    (project_path / ".claude").mkdir()
    (project_path / ".claude" / "rules").symlink_to(outside)

    with pytest.raises(BuildProfileError, match="escapes the project root"):
        _apply_conventions(profile_dir, project_path)

    assert not (outside / "facility-ops.md").exists()


def test_profile_dir_inside_the_osprey_package_is_refused(project_path: Path):
    """Package directories are never profile roots, regardless of invocation."""
    import osprey.profiles.presets as presets_pkg

    presets_dir = Path(presets_pkg.__file__).parent

    with pytest.raises(BuildProfileError, match="inside the installed osprey package"):
        _apply_conventions(presets_dir, project_path, ["alice"])

    assert not (presets_dir / "web-terminal-context").exists()


def test_known_root_entries_derive_from_the_profile():
    """Every profile-relative path the profile declares exempts its root entry."""
    from osprey.cli.build_persistence import _profile_known_root_entries
    from osprey.cli.build_profile_model import BuildProfile
    from osprey.cli.build_profile_schema import DispatchConfig, EnvConfig, ServiceDef

    profile = BuildProfile(
        name="p",
        data="facility-data/channels",
        env=EnvConfig(file="secrets/.env.production"),
        dispatch=DispatchConfig(triggers="dispatch/triggers.yml"),
        services={"beam-logger": ServiceDef(template="service-templates/beam-logger")},
    )

    known = _profile_known_root_entries(profile, Path("/somewhere/my-profile.yml"))

    assert known == [
        "my-profile.yml",
        "facility-data",
        "secrets",
        "dispatch",
        "service-templates",
    ]


def test_known_root_entries_skip_paths_above_the_profile():
    from osprey.cli.build_persistence import _profile_known_root_entries
    from osprey.cli.build_profile_model import BuildProfile

    profile = BuildProfile(name="p", data="../data")

    assert _profile_known_root_entries(profile, None) == []


def test_unknown_root_entry_warns_as_a_candidate_typo(
    profile_dir: Path, project_path: Path, caplog: pytest.LogCaptureFixture
):
    """`rule/` for `rules/` is otherwise silent — nothing reads it."""
    _write(profile_dir / "rule" / "facility-ops.md")

    with caplog.at_level(logging.WARNING):
        _apply_conventions(profile_dir, project_path)

    assert "unrecognized" in caplog.text
    assert "rule" in caplog.text


def test_known_extra_root_entries_do_not_warn(
    profile_dir: Path, project_path: Path, caplog: pytest.LogCaptureFixture
):
    """A profile's own data: directory and profile filename are not typos."""
    (profile_dir / "facility-data").mkdir()
    _write(profile_dir / "my-profile.yml", "name: my-profile\n")

    with caplog.at_level(logging.WARNING):
        _apply_conventions(
            profile_dir, project_path, extra_known=["facility-data", "my-profile.yml"]
        )

    assert "unrecognized" not in caplog.text


# ── Roster-derived per-user context ──────────────────────────────────


def test_roster_users_get_their_context_copied(profile_dir: Path, project_path: Path):
    _write(profile_dir / "web-terminal-context" / "alice" / "extra.md")

    applied = _apply_conventions(profile_dir, project_path, ["alice"])

    assert (project_path / "docker/web-terminal-context/alice/extra.md").is_file()
    assert applied.by_category == {"web-terminal-context": 1}


def test_empty_roster_skips_per_user_context(profile_dir: Path, project_path: Path):
    """A persona build disables web terminals, so it resolves no roster at all."""
    _write(profile_dir / "web-terminal-context" / "alice" / "extra.md")

    applied = _apply_conventions(profile_dir, project_path, [])

    assert not (project_path / "docker/web-terminal-context").exists()
    assert applied.copied == 0


def test_missing_user_directory_is_seeded_in_the_profile(
    profile_dir: Path, project_path: Path, caplog: pytest.LogCaptureFixture
):
    _write(profile_dir / "web-terminal-context" / "alice" / "extra.md")

    with caplog.at_level(logging.INFO):
        applied = _apply_conventions(profile_dir, project_path, ["alice", "bob"])

    assert applied.seeded_users == ["bob"]
    seeded = profile_dir / "web-terminal-context" / "bob"
    assert seeded.is_dir()
    # A .gitkeep so the seeded directory survives a commit of the profile.
    assert (seeded / ".gitkeep").is_file()
    assert "web-terminal-context/bob" in caplog.text


def test_departed_user_warns_and_is_skipped(
    profile_dir: Path, project_path: Path, caplog: pytest.LogCaptureFixture
):
    _write(profile_dir / "web-terminal-context" / "alice" / "extra.md")
    _write(profile_dir / "web-terminal-context" / "departed" / "extra.md")

    with caplog.at_level(logging.WARNING):
        applied = _apply_conventions(profile_dir, project_path, ["alice"])

    assert applied.departed_users == ["departed"]
    assert not (project_path / "docker/web-terminal-context/departed").exists()
    assert "departed" in caplog.text
    # Warn and skip — the profile's own directory is never removed.
    assert (profile_dir / "web-terminal-context" / "departed" / "extra.md").is_file()


# ── Roster resolution from the built project's config ────────────────


def test_roster_resolves_from_the_project_config(project_path: Path):
    _write_config(
        project_path,
        {
            "modules": {
                "web_terminals": {
                    "enabled": True,
                    "users": [{"name": "alice", "index": 0}, "bob"],
                }
            }
        },
    )
    assert _resolve_context_roster(project_path) == ["alice", "bob"]


def test_disabled_web_terminals_resolve_no_roster(project_path: Path):
    _write_config(
        project_path,
        {"modules": {"web_terminals": {"enabled": False, "users": [{"name": "alice"}]}}},
    )
    assert _resolve_context_roster(project_path) == []


@pytest.mark.parametrize(
    "config",
    [{}, {"modules": {}}, {"modules": {"web_terminals": {"enabled": True}}}],
)
def test_absent_roster_resolves_empty(project_path: Path, config: dict):
    _write_config(project_path, config)
    assert _resolve_context_roster(project_path) == []


def test_missing_config_resolves_empty(project_path: Path):
    assert _resolve_context_roster(project_path) == []


# ── Ownership registration ───────────────────────────────────────────


def test_applied_artifacts_register_as_user_owned(profile_dir: Path, project_path: Path):
    _write_config(project_path, {"scaffold": {"user_owned": []}})
    _write(profile_dir / "rules" / "facility-ops.md")
    _write(profile_dir / "skills" / "orbit-check" / "SKILL.md")
    _write(profile_dir / "services" / "beam-logger" / "docker-compose.yml", "services: {}\n")
    _write(profile_dir / "project" / "docs" / "runbook.md")

    applied = _apply_conventions(profile_dir, project_path)
    registered = _register_convention_artifacts(project_path, applied)

    user_owned = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))[
        "scaffold"
    ]["user_owned"]
    assert "rules/facility-ops" in user_owned
    # Skills and services own as whole directories — their shape.
    assert "skills/orbit-check" in user_owned
    assert "services/beam-logger" in user_owned
    # The mirror is an escape hatch outside the ownership system.
    assert registered == len(user_owned)
    assert not any("runbook" in name for name in user_owned)


def test_registration_without_config_is_a_no_op(profile_dir: Path, project_path: Path):
    _write(profile_dir / "rules" / "facility-ops.md")
    applied = _apply_conventions(profile_dir, project_path)
    assert _register_convention_artifacts(project_path, applied) == 0


def test_registration_never_claims_framework_artifacts(profile_dir: Path, tmp_path: Path):
    """One profile rule must not claim the framework rules beside it.

    Registration works from what the conventions copied, never from scanning
    the destination directory — a rendered project already carries framework
    rules in `.claude/rules/`, and those stay framework-owned.
    """
    from osprey.cli.templates.manager import TemplateManager

    project_path = TemplateManager().create_project(
        project_name="reg-ownership",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    assert (project_path / ".claude" / "rules" / "safety.md").is_file(), (
        "rendered project should carry framework rules"
    )

    def user_owned() -> set[str]:
        config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
        return set(config["scaffold"]["user_owned"])

    owned_before = user_owned()
    _write(profile_dir / "rules" / "facility-ops.md")
    applied = _apply_conventions(profile_dir, project_path)
    _register_convention_artifacts(project_path, applied)

    assert applied.owned_artifacts == ["rules/facility-ops"]
    # Exactly the profile's artifact was claimed — the framework rules beside
    # it (safety.md and friends) stay framework-owned.
    assert user_owned() - owned_before == {"rules/facility-ops"}
    assert "rules/safety" not in user_owned()
