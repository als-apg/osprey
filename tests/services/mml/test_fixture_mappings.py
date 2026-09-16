"""Every synthetic MML fixture carries a hand-filled mapping that passes the check.

Each fixture directory under ``tests/fixtures/mml/`` holds a reviewed
``mapping.yaml`` beside its export. These tests import the export into a
scratch deployment repo exactly as a user would, place the committed mapping
over it, and require ``osprey mml map --check --no-derived`` to pass -- so a
later change to the mapping schema or its semantic rules cannot silently
orphan a fixture that downstream chain tests rely on.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"

#: How each fixture is imported: the files copied into the repo (the first of
#: them is the import input) and the extra ``mml import`` arguments.
IMPORTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "tango": (("export.json",), ("--system", "RING")),
    "dualkey": (("export.json",), ("--system", "STOR")),
    "casedup": (("export.json",), ("--system", "MAIN")),
    "wrapped": (("export.json",), ("--system", "INJ")),
    "dialect": (("export.json",), ()),
    "paired": (("quokka.ring.ao.json", "quokka.ring.ad.json"), ()),
    "mat": (("quokka_booster.mat",), ()),
}


def _import(root: Path, name: str) -> None:
    files, extra = IMPORTS[name]
    for filename in files:
        shutil.copy(FIXTURES / name / filename, root / filename)
    result = CliRunner().invoke(cli, ["mml", "import", files[0], *extra], catch_exceptions=False)
    assert result.exit_code == 0, result.output


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "deploy"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def test_every_fixture_directory_is_mapped() -> None:
    directories = {path.name for path in FIXTURES.iterdir() if path.is_dir()}
    directories.discard("__pycache__")
    assert directories == set(IMPORTS)


@pytest.mark.parametrize("name", sorted(IMPORTS))
def test_committed_mapping_passes_check_without_derived(repo: Path, name: str) -> None:
    mapping = FIXTURES / name / "mapping.yaml"
    assert mapping.is_file(), f"{name} has no committed mapping.yaml"
    _import(repo, name)
    shutil.copy(mapping, repo / "data" / "mml" / "mapping.yaml")

    result = CliRunner().invoke(
        cli, ["mml", "map", "--check", "--no-derived"], catch_exceptions=False
    )

    assert "Traceback" not in result.output
    assert result.exit_code == 0, result.output
    assert "passes the check" in result.output
