"""Tests for ``osprey build --force`` preserving user-owned project state.

``--force`` exists so the documented update loop (and the staleness
advisory's remedy) can re-render a project in place. That is only safe if it
never destroys what the user owns: ``.env`` (secrets, service tokens matching
live docker volumes), ``_agent_data/`` (agent workspace), and ``.git`` (the
project's own history). Everything framework-rendered is fair game.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from osprey.build.manifest import MANIFEST_FILENAME
from osprey.cli.build_cmd import _clear_rendered_project_dir, _copy_env_file
from osprey.cli.templates.scaffolding import _derive_env_from_profile
from osprey.utils.dotenv import merge_env_preserving_existing

# ---------------------------------------------------------------------------
# _clear_rendered_project_dir
# ---------------------------------------------------------------------------


def _scaffold_project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.yml").write_text("services: {}\n")
    (project / "README.md").write_text("readme\n")
    (project / "services").mkdir()
    (project / "services" / "docker-compose.yml").write_text("services: {}\n")
    (project / ".env").write_text("CBORG_API_KEY=secret\n")
    (project / "_agent_data").mkdir()
    (project / "_agent_data" / "notebook.md").write_text("operator notes\n")
    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return project


def test_clear_removes_rendered_entries(tmp_path):
    project = _scaffold_project(tmp_path)
    _clear_rendered_project_dir(project)
    assert not (project / "config.yml").exists()
    assert not (project / "README.md").exists()
    assert not (project / "services").exists()


def test_clear_preserves_user_owned_entries(tmp_path):
    project = _scaffold_project(tmp_path)
    _clear_rendered_project_dir(project)
    assert (project / ".env").read_text() == "CBORG_API_KEY=secret\n"
    assert (project / "_agent_data" / "notebook.md").read_text() == "operator notes\n"
    assert (project / ".git" / "HEAD").exists()


def test_clear_reports_what_it_preserved(tmp_path):
    project = _scaffold_project(tmp_path)
    preserved = _clear_rendered_project_dir(project)
    assert set(preserved) == {".env", "_agent_data", ".git"}


def _write_manifest(project: Path, **extra) -> None:
    (project / MANIFEST_FILENAME).write_text(json.dumps({"project_name": "proj", **extra}))


def test_clear_warns_when_secrets_were_never_synced_to_the_profile(tmp_path, caplog):
    """A recorded ``False`` is surfaced before the manifest carrying it is wiped."""
    project = _scaffold_project(tmp_path)
    _write_manifest(project, secrets_synced_to_profile=False)

    with caplog.at_level(logging.WARNING):
        _clear_rendered_project_dir(project)

    assert "only copy" in caplog.text
    assert str(project / ".env") in caplog.text


def test_clear_is_silent_when_the_manifest_does_not_say(tmp_path, caplog):
    """``None`` is unknown — a legacy manifest must not read as a problem."""
    project = _scaffold_project(tmp_path)
    _write_manifest(project)

    with caplog.at_level(logging.WARNING):
        _clear_rendered_project_dir(project)

    assert "only copy" not in caplog.text


def test_clear_is_silent_when_the_secrets_are_in_the_profile(tmp_path, caplog):
    project = _scaffold_project(tmp_path)
    _write_manifest(project, secrets_synced_to_profile=True)

    with caplog.at_level(logging.WARNING):
        _clear_rendered_project_dir(project)

    assert "only copy" not in caplog.text


def test_clear_on_project_without_user_state(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.yml").write_text("services: {}\n")
    preserved = _clear_rendered_project_dir(project)
    assert preserved == []
    assert not (project / "config.yml").exists()
    assert project.exists()


# ---------------------------------------------------------------------------
# merge_env_preserving_existing
# ---------------------------------------------------------------------------


def test_merge_existing_value_wins():
    rendered = "API_KEY=fresh-default\nTZ=UTC\n"
    existing = "API_KEY=user-secret\n"
    merged = merge_env_preserving_existing(rendered, existing)
    assert "API_KEY=user-secret" in merged
    assert "fresh-default" not in merged
    assert "TZ=UTC" in merged


def test_merge_appends_existing_only_keys():
    rendered = "API_KEY=fresh\n"
    existing = "API_KEY=user\nZO_ROOT_USER_PASSWORD=pinned-in-volume\n"
    merged = merge_env_preserving_existing(rendered, existing)
    assert "ZO_ROOT_USER_PASSWORD=pinned-in-volume" in merged


def test_merge_keeps_rendered_comments_and_structure():
    rendered = "# Provider keys\nAPI_KEY=fresh\n\n# Paths\nPROJECT_ROOT=/x\n"
    existing = "API_KEY=user\n"
    merged = merge_env_preserving_existing(rendered, existing)
    assert "# Provider keys" in merged
    assert "# Paths" in merged
    assert merged.index("API_KEY=user") < merged.index("PROJECT_ROOT=/x")


def test_merge_ignores_comments_in_existing():
    rendered = "A=1\n"
    existing = "# just a comment\nA=2\n"
    merged = merge_env_preserving_existing(rendered, existing)
    assert "A=2" in merged
    assert merged.count("just a comment") == 0


# ---------------------------------------------------------------------------
# .env writers must merge, not clobber
# ---------------------------------------------------------------------------


def test_copy_env_file_merges_into_existing(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "prod.env").write_text("API_KEY=template-default\nNEW_VAR=from-template\n")
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".env").write_text("API_KEY=user-secret\nUSER_ONLY=kept\n")

    _copy_env_file(profile_dir, project, "prod.env")

    content = (project / ".env").read_text()
    assert "API_KEY=user-secret" in content
    assert "NEW_VAR=from-template" in content
    assert "USER_ONLY=kept" in content
    assert "template-default" not in content


def test_copy_env_file_plain_copy_when_no_existing(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "prod.env").write_text("API_KEY=template-default\n")
    project = tmp_path / "proj"
    project.mkdir()

    _copy_env_file(profile_dir, project, "prod.env")

    assert (project / ".env").read_text() == "API_KEY=template-default\n"


def _jinja_env_for(tmp_path: Path, template_body: str):
    import jinja2

    template_root = tmp_path / "templates"
    (template_root / "project").mkdir(parents=True)
    (template_root / "project" / "env.j2").write_text(template_body)
    return jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_root)))


def test_derived_env_takes_the_profile_secret_over_the_render(tmp_path):
    """The profile ``.env`` is where a facility secret lives, so it wins."""
    jinja_env = _jinja_env_for(tmp_path, "CBORG_API_KEY={{ env.CBORG_API_KEY }}\nTZ={{ env.TZ }}\n")
    project = tmp_path / "proj"
    project.mkdir()

    _derive_env_from_profile(
        jinja_env,
        project,
        {"env": {"CBORG_API_KEY": "shell-value", "TZ": "UTC"}},
        "CBORG_API_KEY=profile-secret\n",
        "",
    )

    content = (project / ".env").read_text()
    assert "CBORG_API_KEY=profile-secret" in content
    assert "TZ=UTC" in content
    assert "shell-value" not in content


def test_derived_env_keeps_runtime_written_keys_and_drops_the_rest(tmp_path):
    """A rebuild carries no project state the profile cannot regenerate.

    The exception is the enumerated runtime writers: a minted service token is
    pinned to the container already trusting it, so it survives verbatim even
    though nothing else in the old file does.
    """
    jinja_env = _jinja_env_for(tmp_path, "CBORG_API_KEY={{ env.CBORG_API_KEY }}\n")
    project = tmp_path / "proj"
    project.mkdir()

    _derive_env_from_profile(
        jinja_env,
        project,
        {"env": {"CBORG_API_KEY": "shell-value"}},
        "CBORG_API_KEY=profile-secret\n",
        "EVENT_DISPATCHER_TOKEN=tok\nHAND_EDITED=gone\n",
    )

    content = (project / ".env").read_text()
    assert "EVENT_DISPATCHER_TOKEN=tok" in content
    assert "HAND_EDITED" not in content


def test_derived_env_is_a_plain_render_without_a_profile(tmp_path):
    jinja_env = _jinja_env_for(tmp_path, "CBORG_API_KEY={{ env.CBORG_API_KEY }}\n")
    project = tmp_path / "proj"
    project.mkdir()

    _derive_env_from_profile(jinja_env, project, {"env": {"CBORG_API_KEY": "shell-value"}}, "", "")

    assert (project / ".env").read_text().startswith("CBORG_API_KEY=shell-value")
