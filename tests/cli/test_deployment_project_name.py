"""A deployment's tracked name: ``project_name:`` in ``profile.yml``.

Every host-visible name — the compose project, volumes, image tags, each persona
render — derives from the profile's ``project_name``, never from the folder the
checkout happens to sit in. These tests pin the four places that holds:

* the profile model accepts only compose's own spelling of the name;
* the persona catalog stores each render's SOURCE (``build_profile``) and the
  build derives ``project``/``project_path`` through one helper;
* ``osprey build`` refuses a profile with no name, or one still spelling the
  derived catalog keys, with the exact fix;
* no resource name in ``src/osprey`` is spelled from the checkout's folder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_deploy import build_lint_view
from osprey.cli.build_profile_emit import persona_catalog_layer
from osprey.cli.build_profile_model import (
    BuildProfile,
    persona_render_key_errors,
    project_name_errors,
)
from osprey.cli.derived_keys import derived_key_errors
from osprey.deployment.web_terminals.persona_naming import (
    derived_persona_catalog,
    persona_project,
)
from osprey.errors import BuildProfileError

CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

_CATALOG = {
    "modules.web_terminals": {
        "enabled": True,
        "personas": {
            "readonly": {"build_profile": "personas/readonly.yml"},
            "external": {"project_path": "/srv/renders/external"},
        },
    }
}


def _build(runner: CliRunner, cwd: Path):
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        return runner.invoke(build, CI_FLAGS)
    finally:
        os.chdir(previous)


def _rewrite_profile(repo: Path, old: str, new: str) -> None:
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert old in text, f"the exemplar no longer spells {old!r}"
    profile.write_text(text.replace(old, new, 1), encoding="utf-8")


class TestTheProfileField:
    """``BuildProfile.validate`` holds ``project_name`` to compose's spelling."""

    def test_a_normalized_name_is_accepted(self):
        assert project_name_errors("demo-facility_2") == []

    def test_an_absent_name_is_left_to_the_build(self):
        """A bundled preset carries none; the build is what refuses a repo without one."""
        assert project_name_errors(None) == []

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_name_is_refused(self, value, tmp_path):
        with pytest.raises(BuildProfileError, match="must be a non-empty name"):
            BuildProfile(name="Demo", project_name=value).validate(tmp_path)

    @pytest.mark.parametrize(
        ("value", "normalized"),
        [("My.Repo", "myrepo"), ("Demo Facility", "demofacility"), ("-demo-", "demo")],
    )
    def test_a_name_compose_would_rewrite_is_refused_naming_its_spelling(
        self, value, normalized, tmp_path
    ):
        with pytest.raises(BuildProfileError) as excinfo:
            BuildProfile(name="Demo", project_name=value).validate(tmp_path)
        assert f"`project_name: {normalized}`" in str(excinfo.value)

    def test_a_name_with_nothing_usable_is_refused(self):
        (error,) = project_name_errors("...")
        assert "no character a compose project name may use" in error

    def test_the_config_altitude_spelling_names_the_top_level_field(self):
        (error,) = derived_key_errors({"project_name": "demo"})
        assert "the top-level `project_name:` field sets it" in error


class TestThePersonaCatalog:
    """The catalog stores each render's source; the build writes where it lands."""

    def test_the_emitted_layer_carries_only_the_source(self):
        layer = persona_catalog_layer(["readonly", "admin"])
        assert layer == {
            "config": {
                "modules.web_terminals.personas.readonly.build_profile": "personas/readonly.yml",
                "modules.web_terminals.personas.admin.build_profile": "personas/admin.yml",
            }
        }

    def test_one_spelling_names_the_render_and_its_directory(self):
        assert persona_project("als-exemplar", "readonly") == (
            "als-exemplar-readonly",
            "build/als-exemplar-readonly",
        )

    def test_derived_keys_cover_build_profile_entries_only(self):
        assert derived_persona_catalog(_CATALOG, "demo") == {
            "modules.web_terminals.personas.readonly.project": "demo-readonly",
            "modules.web_terminals.personas.readonly.project_path": "build/demo-readonly",
        }

    def test_a_render_with_the_module_off_derives_nothing(self):
        """A persona's own render inherits the catalog with the module disabled."""
        persona_view = {**_CATALOG, "modules.web_terminals.enabled": False}
        assert derived_persona_catalog(persona_view, "demo") == {}

    @pytest.mark.parametrize("key", ["project", "project_path"])
    def test_a_spelled_render_key_is_refused_for_a_built_persona(self, key):
        config = {**_CATALOG, f"modules.web_terminals.personas.readonly.{key}": "x"}
        (error,) = persona_render_key_errors(config)
        assert f"modules.web_terminals.personas.readonly.{key}" in error
        assert "the build derives them from project_name" in error
        assert "Delete these lines from profile.yml" in error

    def test_an_external_render_keeps_its_explicit_path(self):
        assert persona_render_key_errors(_CATALOG) == []

    def test_the_lint_view_carries_the_derived_keys_and_the_root_name(self):
        view = build_lint_view(None, _CATALOG, "demo")
        assert view["project_name"] == "demo"
        assert view["modules.web_terminals.personas.readonly.project"] == "demo-readonly"
        assert "modules.web_terminals.personas.external.project" not in view

    def test_the_lint_view_derives_nothing_without_a_name(self):
        assert build_lint_view(None, _CATALOG, None) == _CATALOG


class TestTheBuild:
    """``osprey build`` names everything from the profile, and says how to migrate."""

    def test_the_folder_name_does_not_name_the_deployment(self, lifecycle_repo_factory, tmp_path):
        repo = lifecycle_repo_factory(tmp_path / "My.Other Checkout")

        result = _build(CliRunner(), repo)

        assert result.exit_code == 0, result.output
        config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
        assert config["project_name"] == "als-exemplar"
        assert (repo / "build" / "als-exemplar-readonly" / "config.yml").is_file()
        personas = config["modules"]["web_terminals"]["personas"]
        assert personas["readonly"]["project"] == "als-exemplar-readonly"
        assert personas["readonly"]["project_path"] == "build/als-exemplar-readonly"

    def test_a_host_overlay_renames_the_instance(self, lifecycle_repo):
        (lifecycle_repo / "profiles").mkdir(exist_ok=True)
        (lifecycle_repo / "profiles" / "scratch.yml").write_text(
            "project_name: als-exemplar-scratch\n", encoding="utf-8"
        )
        (lifecycle_repo / ".env.variant").write_text(
            "OSPREY_PROFILE_VARIANT=scratch\n", encoding="utf-8"
        )

        result = _build(CliRunner(), lifecycle_repo)

        assert result.exit_code == 0, result.output
        config = yaml.safe_load(
            (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8")
        )
        assert config["project_name"] == "als-exemplar-scratch"
        personas = config["modules"]["web_terminals"]["personas"]
        assert personas["admin"]["project"] == "als-exemplar-scratch-admin"

    def test_a_profile_without_a_name_is_refused_with_the_line_to_add(
        self, lifecycle_repo_factory, tmp_path, caplog
    ):
        repo = lifecycle_repo_factory(tmp_path / "Legacy.Checkout")
        _rewrite_profile(repo, "project_name: als-exemplar\n", "")

        result = _build(CliRunner(), repo)

        assert result.exit_code != 0
        assert "states no project_name" in caplog.text
        assert "    project_name: legacycheckout\n" in caplog.text
        assert "existing volumes keep their names" in caplog.text
        assert not (repo / "build").exists()

    def test_a_catalog_still_spelling_the_render_is_refused(self, lifecycle_repo, caplog):
        _rewrite_profile(
            lifecycle_repo,
            "      readonly:\n        build_profile: personas/readonly.yml\n",
            "      readonly:\n        project: als-exemplar-readonly\n"
            "        project_path: build/als-exemplar-readonly\n"
            "        build_profile: personas/readonly.yml\n",
        )

        result = _build(CliRunner(), lifecycle_repo)

        assert result.exit_code != 0
        assert (
            "modules.web_terminals.personas.readonly.project, "
            "modules.web_terminals.personas.readonly.project_path are rendered by the build"
        ) in caplog.text
        assert "Delete these lines from profile.yml" in caplog.text


#: Where a folder name may still be read: as a display label or a documented
#: fallback for a config no build rendered — never as a resource name.
_FOLDER_NAME_ALLOWED: dict[str, int] = {
    # The migration message proposes the normalized folder name.
    "cli/build_cmd.py": 1,
    # Display labels that fall back to the folder when the profile has no `name:`:
    # phase titles, the CI pipeline title, `profile expand`'s reference header.
    "cli/deploy_cmd.py": 1,
    "cli/deploy_scaffold.py": 1,
    "cli/deploy_scaffold_templates.py": 2,
    # `osprey facility check` / `import mml` read a profile with no
    # `project_name:` under the folder name: it folds the zero-source identity
    # and leafs a scratch render, and names no host resource.
    "cli/facility_cmd.py": 1,
    "cli/profile_expand.py": 1,
}

_FOLDER_NAME_RE = re.compile(r"repo_root\.name|-\{delta\.stem\}")


def test_no_resource_name_is_spelled_from_the_checkouts_folder():
    """``persona_project`` and ``project_name`` are the only spellings.

    A new ``repo_root.name`` or ``-{delta.stem}`` in ``src/osprey`` is a name
    two clones of one deployment would disagree on. Add it to the allow-list
    only for a display label or a documented config-less fallback.
    """
    src = Path(__file__).resolve().parents[2] / "src" / "osprey"
    found: dict[str, int] = {}
    for path in sorted(src.rglob("*.py")):
        hits = len(_FOLDER_NAME_RE.findall(path.read_text(encoding="utf-8")))
        if hits:
            found[path.relative_to(src).as_posix()] = hits
    unexpected = {
        name: count for name, count in found.items() if count > _FOLDER_NAME_ALLOWED.get(name, 0)
    }
    assert unexpected == {}, (
        "folder-derived names outside the allow-list (derive them from project_name "
        f"through persona_project / resolve_project_name instead): {unexpected}"
    )
