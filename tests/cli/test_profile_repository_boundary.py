"""Every input of a repository's profile resolves under the repository root.

A repository builds from a clone or is refused before anything is installed:
``osprey validate``, ``osprey facility validate`` and ``osprey build`` refuse a
profile whose ``extends`` file, ``data:`` tree or any other input resolves
outside the repository, one line per input naming the key, the path and the
remedy. A repository that passes builds images that carry every input the
stamp folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import pytest
from click.testing import CliRunner

from osprey.cli.build_profile_document import _read_profile_document
from osprey.cli.build_profile_merge import (
    inputs_outside_repository,
    resolve_profile_document,
)
from osprey.cli.main import cli
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.workspace import BUILD_DIR_NAME, PROFILE_FILENAME
from tests._builds import init_project

pytestmark = [pytest.mark.slow]

#: The repo's directory name.
REPO = "demo"

#: The sentence every refusal line ends with.
REMEDY = "Move it into the repository."


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A freshly initialised repository, inside ``tmp_path/parent``."""
    return init_project(tmp_path / "parent", "hello-world", REPO)


def _prepend(repo: Path, line: str) -> None:
    profile = repo / PROFILE_FILENAME
    profile.write_text(f"{line}\n{profile.read_text(encoding='utf-8')}", encoding="utf-8")


def _set_data(repo: Path, spelling: str) -> None:
    profile = repo / PROFILE_FILENAME
    text = profile.read_text(encoding="utf-8")
    assert "\ndata: data\n" in text
    profile.write_text(text.replace("\ndata: data\n", f"\ndata: {spelling}\n"), encoding="utf-8")


def _extends_outside(repo: Path) -> Path:
    """Make the repo's profile extend a base beside the repository; return the base."""
    base = repo.parent / "site-base.yml"
    base.write_text("name: Site\n", encoding="utf-8")
    _prepend(repo, "extends: ../site-base.yml")
    return base


def _data_outside(repo: Path) -> Path:
    """Move the data tree beside the repository and name it ``../data``; return it."""
    outside = repo.parent / "data"
    (repo / "data").rename(outside)
    _set_data(repo, "../data")
    return outside


def _outside(repo: Path) -> tuple[Any, ...]:
    profile = repo / PROFILE_FILENAME
    document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)
    return inputs_outside_repository(profile, document)


def test_an_extends_base_beside_the_repository_is_outside(repo: Path) -> None:
    base = _extends_outside(repo)

    outside = _outside(repo)

    assert [entry.path.resolve() for entry in outside] == [base.resolve()]


def test_an_extends_base_inside_the_repository_is_not(repo: Path) -> None:
    (repo / "site-base.yml").write_text("name: Site\n", encoding="utf-8")
    _prepend(repo, "extends: site-base.yml")

    assert _outside(repo) == ()


def test_a_data_tree_beside_the_repository_is_outside(repo: Path) -> None:
    data = _data_outside(repo)

    outside = _outside(repo)

    assert [(entry.label, entry.path.resolve()) for entry in outside] == [("data", data.resolve())]


def test_a_data_symlink_that_resolves_outside_is_outside(repo: Path) -> None:
    data = repo.parent / "shared-data"
    (repo / "data").rename(data)
    (repo / "data").symlink_to(data, target_is_directory=True)

    outside = _outside(repo)

    assert [(entry.label, entry.path.resolve()) for entry in outside] == [("data", data.resolve())]


def test_the_loader_refuses_an_outside_extends_naming_the_key_path_and_remedy(
    repo: Path,
) -> None:
    from osprey.cli.build_profile_load import load_profile

    base = _extends_outside(repo)

    with pytest.raises(BuildProfileError) as raised:
        load_profile(repo / PROFILE_FILENAME)

    message = str(raised.value)
    assert message == f"extends: {base.resolve().as_posix()} is outside the repository. {REMEDY}"


def test_validate_refuses_an_outside_data_tree(repo: Path) -> None:
    data = _data_outside(repo)

    result = CliRunner().invoke(cli, ["validate", "--repo", str(repo)])

    assert result.exit_code != 0, result.output
    assert (
        f"data: ../data resolves to {data.resolve().as_posix()}, outside the repository. {REMEDY}"
    ) in result.output


def test_facility_validate_refuses_an_outside_extends(repo: Path) -> None:
    base = _extends_outside(repo)

    result = CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])

    assert result.exit_code == 1, result.output
    assert base.resolve().as_posix() in result.output


@pytest.mark.parametrize("make_outside", [_extends_outside, _data_outside])
def test_build_refuses_an_outside_input_before_the_venv(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    make_outside: Any,
) -> None:
    """The one build here without ``--skip-deps``: the venv step is a spy that must not run."""
    from osprey.cli import build_cmd

    ran: list[Any] = []

    def venv_spy(*args: Any, **kwargs: Any) -> NoReturn:
        ran.append(args)
        raise AssertionError("the project venv was created before the boundary refusal")

    monkeypatch.setattr(build_cmd, "_create_project_venv", venv_spy)
    outside = make_outside(repo)

    result = CliRunner().invoke(build_cmd.build, ["--repo", str(repo), "--skip-lifecycle"])

    assert result.exit_code != 0, result.output
    assert ran == []
    assert outside.resolve().as_posix() in caplog.text
    assert not (repo / BUILD_DIR_NAME / "config.yml").exists()
