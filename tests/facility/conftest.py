"""Real builds shared by the facility tests.

``built_control_assistant`` is the one real ``osprey build --skip-deps`` of the
control-assistant preset that every demo test reads. It is session-scoped, so a
run builds it once per process; a module that uses it carries
``xdist_group("built_control_assistant")``, so under ``--dist loadgroup``
every such module lands on one worker and the build runs once per run.

``build_project`` builds a small project whose ``data/facility/`` a test writes.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from tests.facility._synthetic_trees import write_tree


def init_project(parent: Path, preset: str, name: str) -> Path:
    """Run ``osprey init`` for a preset into ``parent/name``.

    Args:
        parent: The directory the repo is created in.
        preset: The preset name.
        name: The repo's directory name, which is the project name.

    Returns:
        The repo.
    """
    from osprey.cli.init_cmd import init

    repo = parent / name
    result = CliRunner().invoke(init, [str(repo), "--preset", preset, "--no-git"])
    assert result.exit_code == 0, result.output
    return repo


def run_build(repo: Path) -> Result:
    """Run ``osprey build --skip-deps`` on a repo.

    Args:
        repo: The repo.

    Returns:
        The CliRunner result.
    """
    from osprey.cli.build_cmd import build

    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps"])


@dataclass
class BuiltProject:
    """A built repo and what ``render_facility_outputs`` wrote in each render.

    Attributes:
        repo: The repo.
        written: Each render root, to the files ``render_facility_outputs``
            wrote there.
    """

    repo: Path
    written: dict[Path, list[Path]] = field(default_factory=dict)

    @property
    def build_dir(self) -> Path:
        """The deployment's own render root."""
        from osprey.utils.workspace import BUILD_DIR_NAME

        return self.repo / BUILD_DIR_NAME

    @property
    def facility_dir(self) -> Path:
        """The repo's ``data/facility`` directory."""
        return self.repo / "data" / "facility"

    @cached_property
    def facility_raw(self) -> bytes:
        """The bytes of ``build/facility.json``."""
        from osprey.facility.render import FACILITY_FILE

        return (self.build_dir / FACILITY_FILE).read_bytes()

    @cached_property
    def facility(self) -> dict[str, Any]:
        """``build/facility.json``, parsed."""
        document: dict[str, Any] = json.loads(self.facility_raw)
        return document


@pytest.fixture(scope="session")
def built_control_assistant(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BuiltProject]:
    """The control-assistant preset, initialised and built once per process.

    Tests read it and never write to it.
    """
    from osprey.facility import render

    repo = init_project(tmp_path_factory.mktemp("built-ca"), "control-assistant", "demo")
    built = BuiltProject(repo)
    real = render.render_facility_outputs

    def spy(render_dir: Path, doc: Any, rendered_config: Any) -> list[Path]:
        written = real(render_dir, doc, rendered_config)
        built.written[render_dir] = list(written)
        return written

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(render, "render_facility_outputs", spy)
        result = run_build(repo)
    assert result.exit_code == 0, result.output
    yield built


@pytest.fixture
def build_project(
    tmp_path: Path,
) -> Callable[..., tuple[BuiltProject, Result]]:
    """Build a hello-world repo whose ``data/facility/`` holds the given tree.

    The returned callable takes the tree (``None`` for no ``data/facility/`` at
    all) and the repo name, and returns the project and the build's result.
    """

    def build(tree: Mapping[str, Any] | None, name: str = "demo") -> tuple[BuiltProject, Result]:
        repo = init_project(tmp_path, "hello-world", name)
        facility_dir = repo / "data" / "facility"
        shutil.rmtree(facility_dir)
        if tree is not None:
            write_tree(facility_dir, tree)
        return BuiltProject(repo), run_build(repo)

    return build
