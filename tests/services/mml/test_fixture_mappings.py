"""Every synthetic MML fixture carries a hand-filled mapping that passes the check.

Each fixture directory under ``tests/fixtures/mml/`` holds a reviewed
``mapping.yaml`` beside its export. These tests import the export into a
scratch deployment repo exactly as a user would, place the committed mapping
over it, and require ``osprey mml map --check --no-derived`` to pass -- so a
later change to the mapping schema or its semantic rules cannot silently
orphan a fixture that downstream chain tests rely on.

The two real-facility exports also pin how much judgment they ask of their
reviewer, so a detection change that quietly stops pending something -- or
starts pending more -- shows up here rather than in the committed answers.

A 2.0 export asks for one more kind of answer: the families its virtual
accelerator leaves open. Those fixtures are discovered from the exports
themselves rather than listed, so a 2.0 tree committed later is held to the
same rule the day it lands.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"

#: The suffixes an export's siblings carry. ``mml import`` finds each of them
#: beside the file it is handed, so a sibling is copied into the repo but never
#: named on the command line.
SIBLINGS = (".ad.json", ".va.json", ".response.json", ".lattice.mat")

#: How each fixture is imported: the files copied into the repo (every one that
#: is not a :data:`SIBLINGS` sibling is an import input) and the extra
#: ``mml import`` arguments.
IMPORTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "tango": (("export.json",), ("--system", "RING")),
    "dualkey": (("export.json",), ("--system", "STOR")),
    "casedup": (("export.json",), ("--system", "MAIN")),
    "wrapped": (("export.json",), ("--system", "INJ")),
    "dialect": (("export.json",), ()),
    "paired": (("quokka.ring.ao.json", "quokka.ring.ad.json"), ()),
    "mat": (("quokka_booster.mat",), ()),
    "nsls2": (
        (
            "nsls2.storagering.ao.json",
            "nsls2.storagering.ad.json",
            "nsls2.storagering.va.json",
            "nsls2.storagering.response.json",
            "nsls2.storagering.lattice.mat",
            "nsls2.ltb.ao.json",
            "nsls2.ltb.ad.json",
            "nsls2.ltb.va.json",
            "nsls2.ltb.response.json",
            "nsls2.ltb.lattice.mat",
        ),
        (),
    ),
    "spear3": (
        (
            "spear3.storagering.ao.json",
            "spear3.storagering.ad.json",
            "spear3.storagering.va.json",
            "spear3.storagering.response.json",
            "spear3.storagering.lattice.mat",
        ),
        (),
    ),
    "synthetic": (
        (
            "quokka.sr.ao.json",
            "quokka.sr.ad.json",
            "quokka.sr.va.json",
            "quokka.sr.response.json",
            "quokka.sr.lattice.mat",
        ),
        (),
    ),
}

#: The fixture trees a 2.0 export commits, discovered rather than listed: the
#: virtual accelerator of an export lives in its ``*.va.json`` sibling, so a
#: directory that carries one is a tree whose mapping must decide about it.
TWO_ZERO_TREES = tuple(
    sorted(
        directory.name
        for directory in FIXTURES.iterdir()
        if directory.is_dir() and any(directory.glob("*.va.json"))
    )
)


def _import(root: Path, name: str) -> None:
    files, extra = IMPORTS[name]
    for filename in files:
        shutil.copy(FIXTURES / name / filename, root / filename)
    inputs = [name for name in files if not name.endswith(SIBLINGS)]
    result = CliRunner().invoke(cli, ["mml", "import", *inputs, *extra], catch_exceptions=False)
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


def test_a_two_zero_export_is_committed() -> None:
    # The case below is parametrised over the discovery, so a directory that
    # stopped carrying a ``*.va.json`` would empty it silently rather than fail.
    assert TWO_ZERO_TREES, "no fixture export carries a *.va.json sibling"


@pytest.mark.parametrize("name", TWO_ZERO_TREES)
def test_a_two_zero_mapping_answers_every_virtual_accelerator_slot(name: str) -> None:
    """A committed 2.0 mapping decides the families its export left open.

    ``map --check`` refuses a null answer, so this says nothing the check does
    not -- except which fixture owes the answer, and that the block is there to
    answer at all, which is what the chain tests run unattended on. A tree
    whose export leaves nothing open carries an empty set of slots, which is
    an answered block too.
    """
    document = yaml.safe_load((FIXTURES / name / "mapping.yaml").read_text(encoding="utf-8"))

    block = document.get("virtual_accelerator")
    assert block is not None, f"{name} exports a virtual accelerator its mapping says nothing about"
    assert block["families"], f"{name} decides about no family of its export"
    for family, body in block["families"].items():
        slot = body.get("slot")
        if slot is not None:
            assert slot["answer"] is not None, f"{name}: {family}.{slot['kind']} is unanswered"


def _unanswered_slots(document: dict) -> int:
    """Return how many judgment slots a freshly written skeleton left null."""
    total = 0
    for family in document.get("judgments", {}).values():
        for field in family.get("rows_beyond_devices", {}).values():
            total += sum(1 for answer in field.values() if answer is None)
        total += sum(1 for answer in family.get("unbound_devices", {}).values() if answer is None)
        total += 1 if family.get("shared_pvs", "answered") is None else 0
    return total


@pytest.mark.parametrize(("name", "slots"), [("nsls2", 13), ("spear3", 17)])
def test_real_export_asks_for_a_fixed_number_of_judgments(
    repo: Path, name: str, slots: int
) -> None:
    _import(repo, name)

    result = CliRunner().invoke(cli, ["mml", "map", "--init", "--force"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    document = yaml.safe_load((repo / "data" / "mml" / "mapping.yaml").read_text(encoding="utf-8"))
    assert _unanswered_slots(document) == slots
