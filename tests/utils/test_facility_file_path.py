"""``resolve_facility_file_path``: the rule for ``ariel.vocabulary.path``.

The build states the vocabulary a facility ships as the render's copy of it, a
path inside the render's facility tree. That tree's name is one the project
root does not hold, so the value resolves beside ``config.yml`` wherever the
config is: a host render, a login's render, a container. Any other value is
the profile's own file and follows the project-root rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.facility.settings import VOCABULARY_SOURCE
from osprey.repo_layout import entry, render_path
from osprey.utils.config_paths import (
    render_facility_relative,
    resolve_config_relative_path,
    resolve_facility_file_path,
)
from osprey_connectors.workspace import BUILD_DIR_NAME

RENDER_FACILITY = render_path("facility")


@pytest.mark.parametrize(
    "config_dir",
    [f"repo/{BUILD_DIR_NAME}", f"repo/{BUILD_DIR_NAME}/demo-logbook", "app"],
    ids=["host render", "login render", "container"],
)
def test_the_path_the_build_states_resolves_beside_the_config(tmp_path: Path, config_dir: str):
    render = tmp_path / config_dir

    assert resolve_facility_file_path(VOCABULARY_SOURCE, render) == render / VOCABULARY_SOURCE


def test_the_host_render_is_where_the_project_root_rule_leaves_the_render(tmp_path: Path):
    """The source tree is not spelled as the render spells it, so the root holds no such file."""
    render = tmp_path / "repo" / BUILD_DIR_NAME

    assert entry("facility").path != RENDER_FACILITY
    assert resolve_config_relative_path(VOCABULARY_SOURCE, render) == (
        tmp_path / "repo" / VOCABULARY_SOURCE
    )


def test_a_path_outside_the_render_facility_tree_is_the_projects_own(tmp_path: Path):
    render = tmp_path / "repo" / BUILD_DIR_NAME

    assert resolve_facility_file_path("ops/vocabulary.yml", render) == (
        tmp_path / "repo" / "ops" / "vocabulary.yml"
    )


def test_an_absolute_path_is_itself(tmp_path: Path):
    stated = tmp_path / RENDER_FACILITY / "ariel" / "vocabulary.yml"

    assert resolve_facility_file_path(str(stated), tmp_path / "repo" / BUILD_DIR_NAME) == stated


@pytest.mark.parametrize(
    ("value", "inside"),
    [
        (f"{RENDER_FACILITY}/ariel/vocabulary.yml", "ariel/vocabulary.yml"),
        (f"./{RENDER_FACILITY}/ariel/other.yml", "ariel/other.yml"),
        (RENDER_FACILITY, None),
        (f"{RENDER_FACILITY}/../ops/vocabulary.yml", None),
        (f"{entry('facility').path}/ariel/vocabulary.yml", None),
        (f"/{RENDER_FACILITY}/ariel/vocabulary.yml", None),
        (f"~/{RENDER_FACILITY}/ariel/vocabulary.yml", None),
    ],
)
def test_only_a_relative_path_inside_the_tree_is_in_the_tree(value: str, inside: str | None):
    found = render_facility_relative(value)

    assert (found.as_posix() if found is not None else None) == inside
