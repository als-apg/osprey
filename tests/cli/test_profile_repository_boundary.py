"""Every input of a repository's profile resolves under the repository root.

A repository builds from a clone or is refused before anything is installed:
``osprey validate``, ``osprey facility validate`` and ``osprey build`` refuse a
profile whose ``extends`` file, ``data:`` tree or any other input resolves
outside the repository, one line per input naming the key, the path and the
remedy. A repository that passes builds images that carry every input the
stamp folds.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import pytest
from click.testing import CliRunner

from osprey.cli.build_injectors import _locate_pkg_services
from osprey.cli.build_profile_document import _read_profile_document
from osprey.cli.build_profile_merge import (
    OutsideInput,
    bundled_service_template,
    inputs_outside_repository,
    profile_inputs,
    resolve_profile_document,
)
from osprey.cli.main import cli
from osprey.utils.workspace import PROFILE_FILENAME
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.workspace import BUILD_DIR_NAME
from tests._builds import init_project, run_build

pytestmark = [pytest.mark.slow]

#: The repo's directory name.
REPO = "demo"

#: The sentence every refusal line ends with.
REMEDY = "Move it into the repository."

#: A file only the outside directory holds; a copy of that directory carries it.
MARKER = "MARKER"


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


def _set_template(repo: Path, spelling: str) -> None:
    """Declare one service, ``svc``, whose template is *spelling*."""
    profile = repo / PROFILE_FILENAME
    text = profile.read_text(encoding="utf-8")
    assert "\nservices: {}\n" in text
    profile.write_text(
        text.replace("\nservices: {}\n", f"\nservices:\n  svc:\n    template: {spelling}\n"),
        encoding="utf-8",
    )


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


def _template_outside(repo: Path) -> Path:
    """Put a service template beside the repository and name it ``../svc``; return it."""
    template = repo.parent / "svc"
    template.mkdir()
    (template / "docker-compose.yml.j2").write_text("services: {}\n", encoding="utf-8")
    _set_template(repo, "../svc")
    return template


def _outside(repo: Path) -> tuple[OutsideInput, ...]:
    profile = repo / PROFILE_FILENAME
    document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)
    return inputs_outside_repository(document)


def test_an_extends_base_beside_the_repository_is_outside(repo: Path) -> None:
    base = _extends_outside(repo)

    outside = _outside(repo)

    assert [found.entry.path.resolve() for found in outside] == [base.resolve()]


def test_an_extends_base_inside_the_repository_is_not(repo: Path) -> None:
    (repo / "site-base.yml").write_text("name: Site\n", encoding="utf-8")
    _prepend(repo, "extends: site-base.yml")

    assert _outside(repo) == ()


def test_a_data_tree_beside_the_repository_is_outside(repo: Path) -> None:
    data = _data_outside(repo)

    outside = _outside(repo)

    assert [(found.entry.label, found.entry.path.resolve()) for found in outside] == [
        ("data", data.resolve())
    ]


def test_a_data_symlink_that_resolves_outside_is_outside(repo: Path) -> None:
    data = repo.parent / "shared-data"
    (repo / "data").rename(data)
    (repo / "data").symlink_to(data, target_is_directory=True)

    outside = _outside(repo)

    assert [(found.entry.label, found.entry.path.resolve()) for found in outside] == [
        ("data", data.resolve())
    ]


def test_a_nested_directory_link_that_resolves_outside_is_refused(repo: Path) -> None:
    from osprey.cli.build_profile_load import load_profile

    elsewhere = repo.parent / "elsewhere"
    elsewhere.mkdir()
    (repo / "data" / "sub").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(BuildProfileError) as raised:
        load_profile(repo / PROFILE_FILENAME)

    assert str(raised.value) == (
        f"data: data/sub resolves to {elsewhere.resolve().as_posix()}, "
        f"outside the repository. {REMEDY}"
    )


def test_a_nested_file_link_that_resolves_outside_is_refused(repo: Path) -> None:
    from osprey.cli.build_profile_load import load_profile

    target = repo.parent / "x.yml"
    target.write_text("{}\n", encoding="utf-8")
    (repo / "data" / "x.yml").symlink_to(target)

    with pytest.raises(BuildProfileError) as raised:
        load_profile(repo / PROFILE_FILENAME)

    assert str(raised.value) == (
        f"data: data/x.yml resolves to {target.resolve().as_posix()}, "
        f"outside the repository. {REMEDY}"
    )


def test_a_nested_link_inside_the_repository_is_admitted(repo: Path) -> None:
    other = repo / "other"
    other.mkdir()
    (repo / "data" / "sub").symlink_to(other, target_is_directory=True)

    assert _outside(repo) == ()


def test_a_link_under_the_runtime_subtree_is_not_walked(repo: Path) -> None:
    from osprey.utils.workspace import RUNTIME_DATA_DIR_NAME

    elsewhere = repo.parent / "elsewhere"
    elsewhere.mkdir()
    runtime = repo / "data" / RUNTIME_DATA_DIR_NAME
    runtime.mkdir(exist_ok=True)
    (runtime / "k").symlink_to(elsewhere, target_is_directory=True)

    assert _outside(repo) == ()


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


def test_validate_refuses_an_outside_service_template(repo: Path) -> None:
    template = _template_outside(repo)

    result = CliRunner().invoke(cli, ["validate", "--repo", str(repo)])

    assert result.exit_code != 0, result.output
    assert (
        f"services.svc.template: ../svc resolves to {template.resolve().as_posix()}, "
        f"outside the repository. {REMEDY}"
    ) in result.output


def test_facility_validate_refuses_an_outside_extends(repo: Path) -> None:
    base = _extends_outside(repo)

    result = CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])

    assert result.exit_code == 1, result.output
    assert base.resolve().as_posix() in result.output


@pytest.mark.parametrize("make_outside", [_extends_outside, _data_outside, _template_outside])
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


def test_a_repository_built_from_a_clone_images_every_input_its_stamp_folds(
    repo: Path,
) -> None:
    from osprey.cli.build_cmd import _image_build_inputs

    result = run_build(repo)
    assert result.exit_code == 0, result.output

    profile = repo / PROFILE_FILENAME
    document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)
    folded = {entry.path.resolve() for entry in profile_inputs(document)}
    imaged = {(repo.resolve() / path) for path in _image_build_inputs(repo)}

    assert folded
    assert folded <= imaged


def test_an_in_repo_service_template_builds_and_is_imaged(repo: Path) -> None:
    from osprey.cli.build_cmd import _image_build_inputs

    template = repo / "svc"
    template.mkdir()
    (template / "docker-compose.yml.j2").write_text("services: {}\n", encoding="utf-8")
    _set_template(repo, "svc")

    result = run_build(repo)

    assert result.exit_code == 0, result.output
    imaged = {(repo.resolve() / path) for path in _image_build_inputs(repo)}
    assert template.resolve() in imaged


def _shipped_services() -> Path:
    """The directory the framework's service templates ship in."""
    return _locate_pkg_services().resolve()


def _abs_path_spelling(outside: Path) -> str:
    return f"osprey.{outside.as_posix()}"


def _parent_escape_spelling(outside: Path) -> str:
    return f"osprey.a/../../{os.path.relpath(outside, _shipped_services().parent)}"


def _dotted_spelling(_outside_dir: Path) -> str:
    return "osprey../x"


@pytest.mark.parametrize("spell", [_abs_path_spelling, _parent_escape_spelling, _dotted_spelling])
def test_a_template_that_is_not_a_plain_name_copies_nothing_from_outside(
    repo: Path, tmp_path: Path, spell: Callable[[Path], str]
) -> None:
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    (outside / MARKER).write_text("outside\n", encoding="utf-8")
    (outside / "docker-compose.yml.j2").write_text("services: {}\n", encoding="utf-8")
    spelling = spell(outside)
    _set_template(repo, spelling)

    assert bundled_service_template(spelling) is None

    validated = CliRunner().invoke(cli, ["validate", "--repo", str(repo)])
    built = run_build(repo)

    assert validated.exit_code != 0, validated.output
    assert built.exit_code != 0, built.output
    assert list((repo / BUILD_DIR_NAME).rglob(MARKER)) == []


def test_every_shipped_template_name_is_bundled() -> None:
    names = sorted(
        compose.parent.name for compose in _shipped_services().glob("*/docker-compose.yml.j2")
    )

    assert names
    assert {name: bundled_service_template(f"osprey.{name}") for name in names} == {
        name: name for name in names
    }
