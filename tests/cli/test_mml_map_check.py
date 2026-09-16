"""``osprey mml map --check``: every rejection a reviewed mapping can hit, end to end.

Each case starts from a committed mapping that passes the check, applies one
mutation, and runs the real command under ``CliRunner``. A rejection must exit
non-zero and name the offending key as the head of a ``<key>: <message>``
line; an acceptance must exit zero. The mutations cover the null slots, the
PN_LOCAL and case-fold collisions, the section order, the directions table and
its agreement with the vote, the class/branch hierarchy, and ``--no-derived``.
"""

from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"

#: How each fixture used here is imported: its export file and the extra
#: ``mml import`` arguments.
IMPORTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "wrapped": ("export.json", ("--system", "INJ")),
    "casedup": ("export.json", ("--system", "MAIN")),
    "dialect": ("export.json", ()),
}

#: Value for :func:`mutate` that removes the addressed key.
DELETE = object()

_SEGMENT = re.compile(r"\[([^\]]+)\]|([^.\[\]]+)")


def _segments(dotted_key: str) -> list[str]:
    """Split ``a.b[c.d].e`` into ``["a", "b", "c.d", "e"]``.

    Brackets address a key that itself holds a dot, as every ``directions``
    key (``<family>.<field>``) does.
    """
    parts = [bracketed or plain for bracketed, plain in _SEGMENT.findall(dotted_key)]
    assert parts, f"empty key {dotted_key!r}"
    return parts


def mutate(mapping: dict, dotted_key: str, value: Any) -> dict:
    """Return a copy of ``mapping`` with ``dotted_key`` set to ``value``.

    Intermediate dicts are created when absent (so a new ``branches:`` block
    can be added in one call); ``DELETE`` removes the key instead.
    """
    document = copy.deepcopy(mapping)
    *parents, last = _segments(dotted_key)
    node = document
    for part in parents:
        node = node.setdefault(part, {})
        assert isinstance(node, dict), f"{dotted_key!r}: {part!r} is not a mapping"
    if value is DELETE:
        assert last in node, f"{dotted_key!r} is not in the mapping"
        del node[last]
    else:
        node[last] = copy.deepcopy(value)
    return document


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "deploy"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def _import(root: Path, name: str) -> dict:
    """Import fixture ``name`` into ``root`` and return its committed mapping."""
    filename, extra = IMPORTS[name]
    shutil.copy(FIXTURES / name / filename, root / filename)
    result = CliRunner().invoke(cli, ["mml", "import", filename, *extra], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return yaml.safe_load((FIXTURES / name / "mapping.yaml").read_text(encoding="utf-8"))


def _check(root: Path, document: dict, *args: str):
    (root / "data" / "mml" / "mapping.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    result = CliRunner().invoke(cli, ["mml", "map", "--check", *args], catch_exceptions=False)
    assert "Traceback" not in result.output
    return result


def _problem_lines(output: str, key: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip().startswith(f"{key}: ")]


def _assert_rejected(result, key: str, message: str | None = None) -> None:
    assert result.exit_code != 0, result.output
    lines = _problem_lines(result.output, key)
    assert lines, f"no line names {key!r}:\n{result.output}"
    if message is not None:
        assert any(message in line for line in lines), result.output


_STATED_READ = {"direction": "read", "provenance": "stated", "override": False}

#: (fixture, [(dotted key, value), ...], offending key, message fragment)
REJECTIONS: dict[str, tuple[str, list[tuple[str, Any]], str, str]] = {
    "null-direction": (
        "wrapped",
        [("directions[QM.Monitor].direction", None)],
        "directions.QM.Monitor.direction",
        "must not be null",
    ),
    "pn-local-illegal-system-name": (
        "wrapped",
        [("systems.INJ.name", "IN J"), ("section_order", ["IN J"])],
        "systems.INJ.name",
        "not a valid PN_LOCAL token",
    ),
    "sirius-case-fold-without-rename": (
        "casedup",
        [("families.bpmx.rename", DELETE)],
        "families.bpmx",
        "set rename",
    ),
    "section-order-not-a-permutation": (
        "wrapped",
        [("section_order", ["INJ", "INJ"])],
        "section_order[1]",
        "appears more than once",
    ),
    "section-order-missing-a-system": (
        "wrapped",
        [("section_order", [])],
        "section_order",
        "is missing the system names",
    ),
    "direction-key-names-absent-field": (
        "wrapped",
        [("directions[QM.Nope]", _STATED_READ)],
        "directions.QM.Nope",
        "absent from 'QM' in ao.json",
    ),
    "shared-new-class-two-branches": (
        "casedup",
        [("families.bpmx.branch", "Corrector")],
        "families.bpmx.branch",
        "already has the branch",
    ),
    "declared-branch-undeclared-parent": (
        "wrapped",
        [("branches.Pulsed", {"parent": "Kicker", "description": None})],
        "branches.Pulsed.parent",
        "names neither a packaged class nor a declared branch",
    ),
    "system-names-differing-by-case": (
        "dialect",
        [("systems.BOOST.name", "ring"), ("section_order", ["RING", "ring"])],
        "systems.BOOST.name",
        "folds to the same lower-case name",
    ),
    "null-facility-token": (
        "wrapped",
        [("facility.token", None)],
        "facility.token",
        "must not be null",
    ),
    "stated-direction-against-vote": (
        "wrapped",
        [("directions[QM.Monitor].direction", "write")],
        "directions.QM.Monitor",
        "disagrees with the vote",
    ),
    "family-absent-from-ao": (
        "wrapped",
        [
            (
                "families.NOPE",
                {
                    "aliases": ["NOPE"],
                    "description": "Not in the export.",
                    "provenance": "stated",
                    "channels": 0,
                    "fields": {},
                },
            )
        ],
        "families.NOPE",
        "absent from ao.json",
    ),
    "branch-nowhere": (
        "wrapped",
        [("families.QM.branch", "Nowhere")],
        "families.QM.branch",
        "names neither a packaged class nor a declared branch",
    ),
    "class-is-root": (
        "wrapped",
        [("families.QM.class", "AcceleratorDevice")],
        "families.QM.class",
        "root class",
    ),
    "declared-branch-named-like-packaged-class": (
        "wrapped",
        [("branches.Magnet", {"parent": "AcceleratorDevice", "description": None})],
        "branches.Magnet",
        "named like a packaged class",
    ),
    "declared-branch-cycle": (
        "wrapped",
        [
            ("branches.A", {"parent": "B", "description": None}),
            ("branches.B", {"parent": "A", "description": None}),
        ],
        "branches.A.parent",
        "cycle",
    ),
    "packaged-class-wrong-branch": (
        "wrapped",
        [("families.QM.class", "HCorrector"), ("families.QM.branch", "Vacuum")],
        "families.QM.branch",
        "packaged class 'HCorrector' has the parent 'Corrector'",
    ),
    "channel-bearing-group-without-directions-key": (
        "wrapped",
        [("directions[QM.Setpoint]", DELETE)],
        "directions.QM.Setpoint",
        "no directions entry",
    ),
}


class TestHelpers:
    def test_mutate_addresses_dotted_directions_keys(self) -> None:
        base = {"directions": {"QM.On": {"direction": "read"}}}

        changed = mutate(base, "directions[QM.On].direction", "write")

        assert changed["directions"]["QM.On"]["direction"] == "write"
        assert base["directions"]["QM.On"]["direction"] == "read"

    def test_mutate_deletes_and_creates(self) -> None:
        base = {"directions": {"QM.On": {}}, "families": {}}

        assert mutate(base, "directions[QM.On]", DELETE)["directions"] == {}
        assert mutate(base, "branches.A.parent", "B")["branches"] == {"A": {"parent": "B"}}


class TestCommittedBase:
    @pytest.mark.parametrize("name", sorted(IMPORTS))
    def test_unmutated_mapping_passes(self, repo: Path, name: str) -> None:
        result = _check(repo, _import(repo, name), "--no-derived")

        assert result.exit_code == 0, result.output
        assert "passes the check" in result.output


class TestRejections:
    @pytest.mark.parametrize("case", sorted(REJECTIONS))
    def test_mutation_is_rejected_naming_the_key(self, repo: Path, case: str) -> None:
        name, edits, key, message = REJECTIONS[case]
        document = _import(repo, name)
        for dotted_key, value in edits:
            document = mutate(document, dotted_key, value)

        result = _check(repo, document)

        _assert_rejected(result, key, message)

    def test_cycle_names_both_branches(self, repo: Path) -> None:
        document = _import(repo, "wrapped")
        document = mutate(document, "branches.A", {"parent": "B", "description": None})
        document = mutate(document, "branches.B", {"parent": "A", "description": None})

        result = _check(repo, document)

        _assert_rejected(result, "branches.A.parent", "cycle")
        _assert_rejected(result, "branches.B.parent", "cycle")


class TestAcceptances:
    def test_zero_channel_family_without_class_or_branch_passes(self, repo: Path) -> None:
        document = _import(repo, "wrapped")
        gun = document["families"]["GUN"]
        assert gun["channels"] == 0
        assert "class" not in gun and "branch" not in gun

        result = _check(repo, document)

        assert result.exit_code == 0, result.output
        assert not _problem_lines(result.output, "families.GUN")

    def test_stated_direction_on_undecided_vote_passes_without_override(self, repo: Path) -> None:
        # QM.On carries no MemberOf tags, so the vote is undecided; either
        # stated direction is accepted without override.
        document = _import(repo, "wrapped")
        assert document["directions"]["QM.On"]["direction"] == "read"
        document = mutate(document, "directions[QM.On].direction", "write")
        assert document["directions"]["QM.On"]["override"] is False

        result = _check(repo, document)

        assert result.exit_code == 0, result.output
        assert not _problem_lines(result.output, "directions.QM.On")


class TestNoDerived:
    @pytest.mark.parametrize(
        ("dotted_key", "key"),
        [
            ("families.QM.fields.Monitor.provenance", "families.QM.fields.Monitor.provenance"),
            ("directions[QM.Monitor].provenance", "directions.QM.Monitor.provenance"),
        ],
        ids=["derived-description", "derived-direction"],
    )
    def test_one_derived_slot_fails_only_under_no_derived(
        self, repo: Path, dotted_key: str, key: str
    ) -> None:
        document = mutate(_import(repo, "wrapped"), dotted_key, "derived")

        plain = _check(repo, document)
        assert plain.exit_code == 0, plain.output

        strict = _check(repo, document, "--no-derived")
        _assert_rejected(strict, key, "is derived")
        assert len([ln for ln in strict.output.splitlines() if ": is derived" in ln]) == 1
