"""Real builds shared across test directories, used by ``tests/conftest.py``.

``built_control_assistant`` is the one real ``osprey build --skip-deps`` of the
control-assistant preset that the facility and build tests read. It is
session-scoped, so a run builds it once per process; a module that uses it
carries ``xdist_group("built_control_assistant")``, so under ``--dist
loadgroup`` every such module lands on one worker and the build runs once per
run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from click.testing import CliRunner, Result


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


class RenderOutputs(NamedTuple):
    """What ``render_facility_outputs`` wrote in one render.

    Attributes:
        render_dir: The render root it was handed. A render is staged before it
            moves into ``build/``, so this is the staging path.
        files: Each file it wrote, by its path relative to ``render_dir``, to
            the bytes it wrote.
    """

    render_dir: Path
    files: dict[str, bytes]


@dataclass
class BuiltProject:
    """A built repo and what ``render_facility_outputs`` wrote in each render.

    Attributes:
        repo: The repo.
        outputs: One entry per render, in render order. The bytes are taken as
            they are written, before the render moves into ``build/``.
    """

    repo: Path
    outputs: list[RenderOutputs] = field(default_factory=list)

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

    def spy(render_dir: Path, doc: Any, rendered_config: Any, facility_dir: Path) -> list[Path]:
        written = real(render_dir, doc, rendered_config, facility_dir)
        built.outputs.append(
            RenderOutputs(
                render_dir,
                {path.relative_to(render_dir).as_posix(): path.read_bytes() for path in written},
            )
        )
        return written

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(render, "render_facility_outputs", spy)
        result = run_build(repo)
    assert result.exit_code == 0, result.output
    yield built
