"""``osprey mml map --check``: every rejection a reviewed mapping can hit, end to end.

Each case starts from a committed mapping that passes the check, applies one
mutation, and runs the real command under ``CliRunner``. A rejection must exit
non-zero and name the offending key as the head of a ``<key>: <message>``
line; an acceptance must exit zero. The mutations cover the null slots, the
PN_LOCAL and case-fold collisions, the section order, the directions table and
its agreement with the vote, the class/branch hierarchy, and ``--no-derived``.

The ``virtual_accelerator:`` block is checked on the one committed 2.0 export,
``synthetic``, which has no reviewed mapping of its own: those cases start from
the skeleton ``map --init`` writes with every other slot answered, so a problem
count is a statement about the block alone. They cover what the parser refuses
by name, what the ring refuses, the three ways the tree and the block can
disagree -- no block, a block naming another system, and a deck that was never
imported -- and, over the same deck with its cavity taken off, the two ways a
document can miss an open ``values:`` question.

The ``judgments:`` block is checked on the real two-system NSLS-II export,
whose committed mapping answers thirteen slots. Those cases pin more than a
key: each says how many problems the whole run may report, so "exactly one
problem" is an assertion rather than a hope, and one of them forbids any
``directions.`` line, because an answer the export refuses mints no field and
must not also be reported as a missing directions entry.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli
from tests.cli.test_mml_map import SYNTHETIC, SYNTHETIC_SYSTEM, _fill

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"

#: How each fixture used here is imported: the export files, relative to
#: :data:`FIXTURES`, and the extra ``mml import`` arguments. A multi-file
#: export lists every ``ao.json``; ``import`` finds the ``ad.json`` siblings.
IMPORTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "wrapped": (("wrapped/export.json",), ("--system", "INJ")),
    "casedup": (("casedup/export.json",), ("--system", "MAIN")),
    "dialect": (("dialect/export.json",), ()),
    "nsls2": (("nsls2/nsls2.storagering.ao.json", "nsls2/nsls2.ltb.ao.json"), ()),
}

#: Value for :func:`mutate` that removes the addressed key.
DELETE = object()

_SEGMENT = re.compile(r"\[([^\]]+)\]|([^.\[\]]+)")


def _segments(dotted_key: str) -> list[Any]:
    """Split ``a.b[c.d].e`` into ``["a", "b", "c.d", "e"]``.

    Brackets address a key that itself holds a dot, as every ``directions``
    key (``<family>.<field>``) does, and keep it a string however it reads. A
    bare all-digit segment is the YAML integer key a device ordinal and a
    supply group's lowest ordinal are written as, so it converts.
    """
    parts = [
        bracketed if bracketed else int(plain) if plain.isdigit() else plain
        for bracketed, plain in _SEGMENT.findall(dotted_key)
    ]
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
    sources, extra = IMPORTS[name]
    return _import_files(root, name, [FIXTURES / source for source in sources], extra)


def _import_files(root: Path, name: str, paths: list[Path], extra: tuple[str, ...]) -> dict:
    """Import ``paths`` into ``root`` and return fixture ``name``'s committed mapping."""
    inputs = [str(path) for path in paths]
    result = CliRunner().invoke(cli, ["mml", "import", *inputs, *extra], catch_exceptions=False)
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


_PROBLEM_COUNT = re.compile(r"^Error: (\d+) problems? in ", re.MULTILINE)


def _problem_count(output: str) -> int:
    """How many problems the run reported, read off the line it exits with."""
    match = _PROBLEM_COUNT.search(output)
    assert match is not None, f"no problem count in:\n{output}"
    return int(match.group(1))


def _lines_starting(output: str, prefix: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip().startswith(prefix)]


def _judgment_block(root: Path, dotted: str) -> list[str]:
    """The ``PROFILE.md`` lines under one ``<system>.<family>`` judgment heading."""
    lines = (root / "data" / "mml" / "PROFILE.md").read_text(encoding="utf-8").splitlines()
    heads = [index for index, line in enumerate(lines) if line.startswith(f"- `{dotted}`, ")]
    assert len(heads) == 1, f"{dotted!r} heads {len(heads)} blocks, not one"
    start = heads[0] + 1
    end = next((i for i in range(start, len(lines)) if not lines[i].startswith("  ")), len(lines))
    return lines[start:end]


#: The LTB ``BEND`` fields whose channel lists the two-system case shortens.
_LTB_BEND_FIELDS = ("Monitor", "Setpoint", "OnControl", "Fault")


def _import_short_ltb_bend(root: Path, work: Path) -> dict:
    """Import nsls2 with LTB ``BEND`` one device short of its channels.

    Every ``BEND`` channel list of the LTB export loses its last entry, so
    device 4 of that system carries nothing and the family pends
    ``unbound_devices.4`` there. StorageRing's own sixty-device ``BEND`` is
    copied untouched, so what an answer may reach is a fact about the export
    rather than about the edit.

    Args:
        root: The deployment repo to import into.
        work: A directory to build the edited export in.

    Returns:
        The committed nsls2 mapping, which answers every other slot.
    """
    exports = work / "exports"
    exports.mkdir()
    # The decks travel with the export: a 2.0 sibling is checked against the
    # ring it was sampled over, so leaving them behind refuses the whole check.
    for source in sorted((FIXTURES / "nsls2").glob("nsls2.*")):
        shutil.copy(source, exports / source.name)
    short = exports / "nsls2.ltb.ao.json"
    export = json.loads(short.read_text(encoding="utf-8"))
    for field in _LTB_BEND_FIELDS:
        export["BEND"][field]["ChannelNames"] = export["BEND"][field]["ChannelNames"][:-1]
    short.write_text(json.dumps(export), encoding="utf-8")
    return _import_files(root, "nsls2", [exports / "nsls2.storagering.ao.json", short], ())


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


#: The nsls2 ``DCCT`` row the committed mapping answers ``{field: Lifetime}``.
LIFETIME_ROW = "judgments.DCCT.rows_beyond_devices.Monitor[SR:C03-BI{DCCT:1}Lifetime-I]"


class JudgmentCase(NamedTuple):
    """One mutation of a ``judgments:`` block and what ``--check`` must say.

    Attributes:
        fixture: The fixture whose committed mapping the case starts from.
        edits: :func:`mutate` arguments, applied in order.
        key: The document path the refusal names.
        message: A fragment of that line's message.
        problems: How many problems the whole run reports. Pinning it is the
            point: a judgment refusal must not drag a second problem in.
        forbidden: A line prefix no reported problem may start with.
    """

    fixture: str
    edits: list[tuple[str, Any]]
    key: str
    message: str
    problems: int
    forbidden: str | None = None


#: The judgment answers a reviewed mapping can get wrong, one mutation each.
JUDGMENTS: dict[str, JudgmentCase] = {
    "answer-set-back-to-null": JudgmentCase(
        "nsls2",
        [("judgments.BEND.shared_pvs", None)],
        "judgments.BEND.shared_pvs",
        "must not be null",
        problems=1,
    ),
    "field-name-the-family-already-carries": JudgmentCase(
        "nsls2",
        [
            (LIFETIME_ROW, {"field": "Monitor"}),
            ("families.DCCT.fields.Lifetime", DELETE),
            ("directions[DCCT.Lifetime]", DELETE),
        ],
        LIFETIME_ROW,
        "the field name 'Monitor' is a key DCCT already carries",
        problems=1,
        forbidden="directions.",
    ),
    "pending-ordinal-with-no-slot": JudgmentCase(
        "nsls2",
        [("judgments.TUNE.unbound_devices.3", DELETE)],
        "judgments.TUNE.unbound_devices.3",
        "is pending in StorageRing and has no answer",
        problems=1,
    ),
    "family-absent-from-the-export": JudgmentCase(
        "nsls2",
        [("judgments.NOPE", {"shared_pvs": "keep_all"})],
        "judgments.NOPE",
        "names a family absent from ao.json",
        problems=1,
    ),
    "owner-outside-its-supply-group": JudgmentCase(
        "wrapped",
        [("judgments.QM.shared_pvs", {}), ("judgments.QM.shared_pvs.2", 9)],
        "judgments.QM.shared_pvs.2",
        "device 9 is not a member of supply group 2",
        problems=1,
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

    def test_mutate_addresses_an_ordinal_as_the_integer_key_yaml_writes(self) -> None:
        base = {"judgments": {"TUNE": {"unbound_devices": {3: "drop"}}}}

        changed = mutate(base, "judgments.TUNE.unbound_devices.3", "keep")

        assert changed["judgments"]["TUNE"]["unbound_devices"] == {3: "keep"}
        assert mutate(base, "judgments.TUNE.unbound_devices.3", DELETE)["judgments"]["TUNE"] == {
            "unbound_devices": {}
        }

    def test_mutate_walks_through_an_integer_parent(self) -> None:
        base = {"judgments": {"QM": {"shared_pvs": {2: {}}}}}

        changed = mutate(base, "judgments.QM.shared_pvs.2.owner", 3)

        assert changed["judgments"]["QM"]["shared_pvs"] == {2: {"owner": 3}}

    def test_mutate_keeps_a_bracketed_digit_segment_a_string(self) -> None:
        # Only a bare segment is an ordinal; a signal in brackets stays itself.
        changed = mutate({}, "judgments.DCCT.rows_beyond_devices.Monitor[7]", "drop")

        assert changed["judgments"]["DCCT"]["rows_beyond_devices"]["Monitor"] == {"7": "drop"}


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


class TestJudgmentRejections:
    """The judgment answers the export or the mapping refuses, end to end."""

    @pytest.mark.parametrize("case", sorted(JUDGMENTS))
    def test_mutation_is_rejected_naming_the_key_and_nothing_else(
        self, repo: Path, case: str
    ) -> None:
        spec = JUDGMENTS[case]
        document = _import(repo, spec.fixture)
        for dotted_key, value in spec.edits:
            document = mutate(document, dotted_key, value)

        result = _check(repo, document)

        _assert_rejected(result, spec.key, spec.message)
        assert _problem_count(result.output) == spec.problems, result.output
        if spec.forbidden is not None:
            assert not _lines_starting(result.output, spec.forbidden), result.output

    def test_a_created_field_asks_for_its_entry_and_its_direction_once_each(
        self, repo: Path
    ) -> None:
        """A ``field:`` answer the export takes stands; its two entries are named once each."""
        document = _import(repo, "nsls2")
        document = mutate(document, "families.DCCT.fields.Lifetime", DELETE)
        document = mutate(document, "directions[DCCT.Lifetime]", DELETE)

        result = _check(repo, document)

        assert _problem_count(result.output) == 2, result.output
        created = _problem_lines(result.output, LIFETIME_ROW)
        assert len(created) == 1, result.output
        assert "creates the field 'Lifetime' of DCCT in StorageRing" in created[0]
        assert len(_problem_lines(result.output, "directions.DCCT.Lifetime")) == 1, result.output


class TestTwoSystems:
    """A judgment is pending, and answered, per system."""

    def test_an_ordinal_pending_in_one_system_is_answered_for_that_system(
        self, repo: Path, tmp_path: Path
    ) -> None:
        document = _import_short_ltb_bend(repo, tmp_path)
        storage = "\n".join(_judgment_block(repo, "StorageRing.BEND"))
        ltb = "\n".join(_judgment_block(repo, "LTB.BEND"))
        assert "unbound device" not in storage, storage
        assert "unbound device ordinal 4 `[1, 4]`" in ltb, ltb

        unanswered = _check(repo, document, "--no-derived")
        _assert_rejected(unanswered, "judgments.BEND.unbound_devices.4", "is pending in LTB")

        answered = _check(
            repo, mutate(document, "judgments.BEND.unbound_devices.4", "drop"), "--no-derived"
        )

        assert answered.exit_code == 0, answered.output
        assert "passes the check" in answered.output


class TestAcceptances:
    def test_an_owner_answer_naming_a_member_of_its_group_passes(self, repo: Path) -> None:
        # wrapped's one supply group is QM's, keyed by its lowest ordinal 2;
        # an owner that strands the group's other member is the reviewer's call.
        document = mutate(_import(repo, "wrapped"), "judgments.QM.shared_pvs", {})
        document = mutate(document, "judgments.QM.shared_pvs.2", 3)

        result = _check(repo, document)

        assert result.exit_code == 0, result.output
        assert not _lines_starting(result.output, "judgments."), result.output

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


#: Where an open slot's answer lives, per family of the synthetic block.
IDGAP_ANSWER = "virtual_accelerator.families.IDGAP.slot.answer"
SEPTUM_ANSWER = "virtual_accelerator.families.SEPTUM.slot.answer"


@pytest.fixture
def va_base(repo: Path) -> dict:
    """The synthetic 2.0 skeleton with every slot answered, block included.

    ``synthetic`` commits no reviewed mapping, so the base is what ``--init``
    writes: the block as the export proposes it, and every slot outside it
    filled, so each case below reports only what its own mutation caused.
    """
    result = CliRunner().invoke(cli, ["mml", "import", str(SYNTHETIC)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    init = CliRunner().invoke(cli, ["mml", "map", "--init"], catch_exceptions=False)
    assert init.exit_code == 0, init.output
    document = yaml.safe_load((repo / "data" / "mml" / "mapping.yaml").read_text(encoding="utf-8"))
    return _fill(document)


@pytest.fixture
def va_no_cavity(repo: Path) -> dict:
    """The same skeleton over a deck with no cavity, so the block asks a quantity.

    A facility whose Middle Layer holds the radio frequency saves no cavity,
    and the block the rules then write carries the one open ``values:``
    question there is -- the voltage of the cavity emit builds for it. The
    cavity is the deck's last element, so no position the export states moves.
    """
    import at

    from osprey.services.mml.loaders.mat import load_lattice

    result = CliRunner().invoke(cli, ["mml", "import", str(SYNTHETIC)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    deck = repo / "data" / "mml" / "lattice" / f"{SYNTHETIC_SYSTEM}.mat"
    ring = load_lattice(deck)
    assert isinstance(ring[-1], at.RFCavity), "the synthetic deck no longer ends in its cavity"
    del ring[-1]
    at.save_mat(ring, str(deck), mat_key="THERING")
    init = CliRunner().invoke(cli, ["mml", "map", "--init"], catch_exceptions=False)
    assert init.exit_code == 0, init.output
    document = yaml.safe_load((repo / "data" / "mml" / "mapping.yaml").read_text(encoding="utf-8"))
    document = _fill(document)
    assert document["virtual_accelerator"]["families"]["RF"]["values"]["voltage"]["answer"]
    return document


class TestVirtualAccelerator:
    """The ``virtual_accelerator`` block, end to end on the 2.0 tree."""

    def test_the_answered_block_passes(self, repo: Path, va_base: dict) -> None:
        result = _check(repo, va_base)

        assert result.exit_code == 0, result.output
        assert "passes the check" in result.output

    @pytest.mark.parametrize("key", [IDGAP_ANSWER, SEPTUM_ANSWER], ids=["escape-hatch", "attype"])
    def test_an_unanswered_slot_is_named_and_is_the_only_problem(
        self, repo: Path, va_base: dict, key: str
    ) -> None:
        document = mutate(va_base, key, None)

        result = _check(repo, document)

        _assert_rejected(result, key, "must not be null")
        assert _problem_count(result.output) == 1, result.output

    @pytest.mark.parametrize(
        ("key", "shown"),
        [
            (IDGAP_ANSWER, "must be latch, ignore_hook or null"),
            (SEPTUM_ANSWER, "must be latch, strength:<PolynomB|PolynomA>[<i>]"),
        ],
        ids=["escape-hatch", "attype"],
    )
    def test_a_word_outside_the_vocabulary_is_refused_by_name(
        self, repo: Path, va_base: dict, key: str, shown: str
    ) -> None:
        # The vocabulary of every slot kind is closed: the refusal names the
        # words that are in it rather than carrying an unknown one through.
        document = mutate(va_base, key, "sideways")

        result = _check(repo, document)

        _assert_rejected(result, key, shown)
        assert "got 'sideways'" in result.output
        assert "is not a valid mapping document" in result.output

    def test_an_answer_the_ring_refuses_names_the_element_and_the_field(
        self, repo: Path, va_base: dict
    ) -> None:
        # SEPTUM's element is a drift, which carries no multipole at all, so
        # the answer is refused against the deck rather than the vocabulary.
        document = mutate(va_base, SEPTUM_ANSWER, "strength:PolynomB[9]")

        result = _check(repo, document)

        _assert_rejected(
            result,
            SEPTUM_ANSWER,
            f"element DR (DriftPass) of SEPTUM in {SYNTHETIC_SYSTEM} takes no PolynomB[9]",
        )
        assert _problem_count(result.output) == 1, result.output

    def test_a_mapping_without_a_block_is_asked_once_to_init(
        self, repo: Path, va_base: dict
    ) -> None:
        # One problem for the whole export, not one per family: the mapping
        # decides nothing about a virtual accelerator the tree carries.
        document = mutate(va_base, "virtual_accelerator", DELETE)

        result = _check(repo, document)

        _assert_rejected(result, "virtual_accelerator", "run osprey mml map --init")
        assert _problem_count(result.output) == 1, result.output
        assert not _lines_starting(result.output, "virtual_accelerator.")

    def test_a_block_naming_another_system_is_one_problem_about_the_system(
        self, repo: Path, va_base: dict
    ) -> None:
        document = mutate(va_base, "virtual_accelerator.system", "LTB")

        result = _check(repo, document)

        _assert_rejected(
            result,
            "virtual_accelerator.system",
            f"names 'LTB', and the export carries a virtual accelerator for {SYNTHETIC_SYSTEM!r}",
        )
        assert _problem_count(result.output) == 1, result.output

    def test_a_null_system_is_named(self, repo: Path, va_base: dict) -> None:
        document = mutate(va_base, "virtual_accelerator.system", None)

        result = _check(repo, document)

        _assert_rejected(result, "virtual_accelerator.system", "must not be null")
        assert _problem_count(result.output) == 1, result.output

    def test_a_deck_outside_the_tree_is_one_problem_and_judges_no_answer(
        self, repo: Path, va_base: dict
    ) -> None:
        # The deck is what an answer binding an element is held to, so without
        # it even an answer the ring would refuse is left unjudged.
        document = mutate(va_base, SEPTUM_ANSWER, "strength:PolynomB[9]")
        (repo / "data" / "mml" / "lattice" / f"{SYNTHETIC_SYSTEM}.mat").unlink()

        result = _check(repo, document)

        _assert_rejected(
            result,
            "virtual_accelerator",
            f"the deck {SYNTHETIC_SYSTEM} was sampled over is not in the tree",
        )
        assert _problem_count(result.output) == 1, result.output
        assert not _problem_lines(result.output, SEPTUM_ANSWER), result.output

    def test_the_answered_quantity_passes(self, repo: Path, va_no_cavity: dict) -> None:
        result = _check(repo, va_no_cavity)

        assert result.exit_code == 0, result.output
        assert "passes the check" in result.output

    def test_a_dropped_quantity_is_named_as_unanswered(
        self, repo: Path, va_no_cavity: dict
    ) -> None:
        # Deleting the question is not answering it: emit needs the number,
        # so the check has to be the command that asks for it.
        document = mutate(va_no_cavity, "virtual_accelerator.families.RF.values.voltage", DELETE)

        result = _check(repo, document)

        _assert_rejected(
            result, "virtual_accelerator.families.RF.values.voltage", "is not answered"
        )
        assert _problem_count(result.output) == 1, result.output

    def test_a_misspelt_quantity_is_named_on_both_counts(
        self, repo: Path, va_no_cavity: dict
    ) -> None:
        # Renaming the key leaves the real question unanswered and adds one
        # the rules never asked, and a reviewer needs to be told both.
        document = copy.deepcopy(va_no_cavity)
        values = document["virtual_accelerator"]["families"]["RF"]["values"]
        values["voltages"] = values.pop("voltage")

        result = _check(repo, document)

        _assert_rejected(
            result, "virtual_accelerator.families.RF.values.voltage", "is not answered"
        )
        _assert_rejected(
            result,
            "virtual_accelerator.families.RF.values.voltages",
            f"answers no open question of RF in {SYNTHETIC_SYSTEM}",
        )
        assert _problem_count(result.output) == 2, result.output


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
