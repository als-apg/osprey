"""``osprey mml map``: the command surface around the mapping services.

Pins what the services cannot: that exactly one of ``--init``/``--check`` runs,
that ``--init`` writes ``data/mml/mapping.yaml`` from the imported export and
never overwrites a reviewed file without ``--force``, and that ``--check``
names every problem as ``<key>: <message>``, reports the derived counts as a
warning, and exits non-zero on any problem -- including a file that is not a
structurally valid mapping.

The virtual-accelerator block a 2.0 export adds is written by appending text
rather than by re-dumping the document, so its lanes here compare bytes: the
block ``--force-va`` writes over an existing one is the same bytes, and the
prefix a reviewer edited keeps its own. The one committed 2.0 export is
``synthetic``; every other fixture is a 1.0 tree, which must say so in one
line and otherwise leave the file alone.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.services.mml.mapping import parse_mapping
from osprey.services.mml.mapping.branches import packaged_classes
from osprey.services.mml.mapping.schema import VA_KINDS, VA_SLOT_KINDS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"

#: The 2.0 export these lanes build on: its ``ao.json`` pulls in the ``ad``,
#: ``va`` and ``response`` siblings and the deck it was sampled over. It is
#: the one whose families are enumerated below; the real 2.0 fixtures
#: (``spear3``, ``nsls2``) are driven by the chain lanes.
SYNTHETIC = FIXTURES / "synthetic" / "quokka.sr.ao.json"

#: The system ``synthetic`` imports as, and what its block must decide about.
SYNTHETIC_SYSTEM = "SR"

#: Families in the synthetic block, and the open slot each of the two the
#: export leaves undecided carries (FR3).
SYNTHETIC_FAMILIES = 17
SYNTHETIC_OPEN_SLOTS = {"IDGAP": "escape_hatch", "SEPTUM": "attype"}

#: A valid answer per slot kind, for :func:`_fill`. Keyed by every kind the
#: schema closes the vocabulary to, so a new kind fails here rather than
#: silently leaving a slot null.
VA_SLOT_ANSWERS = {"attype": "latch", "shared_field": "latch", "escape_hatch": "latch"}

#: What :func:`_fill` answers each open quantity with, keyed by what the
#: quantity is. A cavity built from an export that states no voltage is run at
#: a few megavolts, which is what a storage ring of this size runs at; the
#: orbit does not depend on it.
VA_VALUE_ANSWERS = {"voltage": 3.0e6}

#: Every committed 1.0 fixture: the export files, relative to :data:`FIXTURES`,
#: the extra ``mml import`` arguments, and the system its "no 2.0 export" line
#: names. Each of these directories also holds a reviewed ``mapping.yaml``.
ONE_ZERO_IMPORTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    "casedup": (("casedup/export.json",), ("--system", "MAIN"), "MAIN"),
    "dialect": (("dialect/export.json",), (), "BOOST, RING"),
    "dualkey": (("dualkey/export.json",), ("--system", "STOR"), "STOR"),
    "mat": (("mat/quokka_booster.mat",), (), "BOOSTER"),
    "paired": (("paired/quokka.ring.ao.json",), (), "RING"),
    "tango": (("tango/export.json",), ("--system", "RING"), "RING"),
    "wrapped": (("wrapped/export.json",), ("--system", "INJ"), "INJ"),
}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with the ``wrapped`` fixture imported as system ``INJ``."""
    root = tmp_path / "deploy"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    shutil.copy(FIXTURES / "wrapped" / "export.json", root / "flat.json")
    result = CliRunner().invoke(
        cli, ["mml", "import", "flat.json", "--system", "INJ"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    return root


def _deploy(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make ``root`` a deployment repo (a ``profile.yml`` marker) and the cwd."""
    root.mkdir(parents=True)
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def _import(*inputs: Path, extra: tuple[str, ...] = ()) -> None:
    result = CliRunner().invoke(
        cli, ["mml", "import", *[str(path) for path in inputs], *extra], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output


@pytest.fixture
def va_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with the 2.0 ``synthetic`` export imported as ``SR``.

    The deck lands in ``data/mml/lattice/SR.mat`` beside ``va.json``, which is
    what makes this a tree ``map`` writes a virtual-accelerator block for.
    """
    root = _deploy(tmp_path / "deploy20", monkeypatch)
    _import(SYNTHETIC)
    assert (root / "data" / "mml" / "lattice" / f"{SYNTHETIC_SYSTEM}.mat").is_file()
    return root


def _map(*args: str):
    return CliRunner().invoke(cli, ["mml", "map", *args], catch_exceptions=False)


def _mapping_path(repo: Path) -> Path:
    return repo / "data" / "mml" / "mapping.yaml"


def _load(repo: Path) -> dict:
    return yaml.safe_load(_mapping_path(repo).read_text(encoding="utf-8"))


def _write(repo: Path, document: dict) -> None:
    _mapping_path(repo).write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _fill(document: dict) -> dict:
    """Fill every unfilled slot of a skeleton so it passes the check.

    The virtual-accelerator block is answered too -- each slot by its kind and
    each open quantity by what it is -- so a 2.0 skeleton reaches the same
    place a 1.0 one does. Imported by
    ``test_mml_map_check.py`` and ``test_mml_emit.py``, whose lanes start from a
    document whose every other slot is already settled.
    """
    packaged = packaged_classes()
    document["facility"]["token"] = "quokka"
    # The fixture carries no AD, so facility and system prose start null.
    for entry in (document["facility"], *document["systems"].values()):
        if entry["description"] is None:
            entry["description"] = "Filled by the reviewer."
            entry["provenance"] = "stated"
    for family in document["families"].values():
        if "class" not in family:
            continue
        cls = family["class"]
        if cls in packaged:
            family["branch"] = None
        else:
            family["branch"] = sorted(packaged)[0]
    for slot in document["directions"].values():
        if slot["direction"] is None:
            slot["direction"] = "read"
    for family in document.get("judgments", {}).values():
        for field in family.get("rows_beyond_devices", {}).values():
            for signal in list(field):
                if field[signal] is None:
                    field[signal] = "drop"
        unbound = family.get("unbound_devices", {})
        for ordinal in list(unbound):
            if unbound[ordinal] is None:
                unbound[ordinal] = "drop"
        if "shared_pvs" in family and family["shared_pvs"] is None:
            family["shared_pvs"] = "keep_all"
    block = document.get("virtual_accelerator")
    if block is not None:
        for family in block["families"].values():
            slot = family.get("slot")
            if slot is not None and slot.get("answer") is None:
                slot["answer"] = VA_SLOT_ANSWERS[slot["kind"]]
            for name, value in family.get("values", {}).items():
                if value.get("answer") is None:
                    value["answer"] = VA_VALUE_ANSWERS[name]
    return document


class TestVerbSelection:
    def test_neither_verb_is_a_usage_error(self, repo: Path) -> None:
        result = _map()

        assert result.exit_code == 2
        assert "--init" in result.output and "--check" in result.output

    def test_both_verbs_is_a_usage_error(self, repo: Path) -> None:
        result = _map("--init", "--check")

        assert result.exit_code == 2
        assert not _mapping_path(repo).exists()

    def test_force_without_init_is_a_usage_error(self, repo: Path) -> None:
        assert _map("--init").exit_code == 0

        result = _map("--check", "--force")

        assert result.exit_code == 2

    def test_no_derived_without_check_is_a_usage_error(self, repo: Path) -> None:
        result = _map("--init", "--no-derived")

        assert result.exit_code == 2
        assert not _mapping_path(repo).exists()


class TestInit:
    def test_writes_a_skeleton_that_parses(self, repo: Path) -> None:
        result = _map("--init")

        assert result.exit_code == 0, result.output
        assert "Traceback" not in result.output
        document = _load(repo)
        mapping = parse_mapping(document)
        assert "INJ" in mapping.systems
        assert {"BPM", "QM", "GUN"} <= set(mapping.families)
        assert "mapping.yaml" in result.output

    def test_skeleton_is_readable_block_yaml(self, repo: Path) -> None:
        _map("--init")

        text = _mapping_path(repo).read_text(encoding="utf-8")
        assert text.startswith("facility:")
        assert "{" not in text.split("\n", 1)[0]
        assert "directions:" in text

    def test_closing_line_counts_the_pending_judgments(self, repo: Path) -> None:
        result = _map("--init")

        assert result.exit_code == 0, result.output
        # The fixture's QM shares IJ:QM2:RB between devices 2 and 3: one slot.
        assert _load(repo)["judgments"] == {"QM": {"shared_pvs": None}}
        assert "1 judgment to answer" in result.output

    def test_closing_line_omits_judgments_when_none_pend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "plain"
        root.mkdir()
        (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
        monkeypatch.chdir(root)
        shutil.copy(FIXTURES / "tango" / "export.json", root / "flat.json")
        imported = CliRunner().invoke(
            cli, ["mml", "import", "flat.json", "--system", "RING"], catch_exceptions=False
        )
        assert imported.exit_code == 0, imported.output

        result = _map("--init")

        assert result.exit_code == 0, result.output
        assert "judgment" not in result.output
        assert "judgments" not in yaml.safe_load(
            (root / "data" / "mml" / "mapping.yaml").read_text(encoding="utf-8")
        )

    def test_refuses_to_overwrite_without_force(self, repo: Path) -> None:
        assert _map("--init").exit_code == 0
        _mapping_path(repo).write_text("reviewed: true\n", encoding="utf-8")

        result = _map("--init")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "--force" in result.output
        assert _mapping_path(repo).read_text(encoding="utf-8") == "reviewed: true\n"

    def test_force_overwrites(self, repo: Path) -> None:
        assert _map("--init").exit_code == 0
        _mapping_path(repo).write_text("reviewed: true\n", encoding="utf-8")

        result = _map("--init", "--force")

        assert result.exit_code == 0, result.output
        assert "facility" in _load(repo)

    def test_without_an_import_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
        monkeypatch.chdir(root)

        result = _map("--init")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "mml import" in result.output
        assert not (root / "data" / "mml" / "mapping.yaml").exists()


class TestCheck:
    def test_fresh_skeleton_names_each_unfilled_slot(self, repo: Path) -> None:
        _map("--init")

        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        lines = result.output.splitlines()
        assert any(line.strip().startswith("facility.token: ") for line in lines), result.output
        # The On/OnControl pair has no MemberOf tags, so its vote is undecided.
        assert any(line.strip().startswith("directions.QM.On.direction: ") for line in lines)

    def test_filled_mapping_passes_and_warns_derived_counts(self, repo: Path) -> None:
        _map("--init")
        _write(repo, _fill(_load(repo)))

        result = _map("--check")

        assert result.exit_code == 0, result.output
        assert "Traceback" not in result.output
        warning = next(line for line in result.output.splitlines() if "derived" in line.lower())
        # BPM.X, QM.Setpoint/Monitor/On/OnControl; GUN.Monitor has no channel.
        assert "5 directions" in warning
        assert "3 family" in warning and "0 facility" in warning

    def test_no_derived_reports_every_derived_slot(self, repo: Path) -> None:
        _map("--init")
        _write(repo, _fill(_load(repo)))

        result = _map("--check", "--no-derived")

        assert result.exit_code != 0
        assert "directions.BPM.X.provenance: is derived" in result.output

    def test_structural_error_is_named_by_key(self, repo: Path) -> None:
        _map("--init")
        document = _load(repo)
        document["directions"]["BPM.X"]["direction"] = "sideways"
        _write(repo, document)

        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "directions.BPM.X" in result.output

    def test_invalid_yaml_is_refused(self, repo: Path) -> None:
        _map("--init")
        _mapping_path(repo).write_text("facility: [unclosed\n", encoding="utf-8")

        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "mapping.yaml" in result.output

    def test_non_mapping_document_is_refused(self, repo: Path) -> None:
        _map("--init")
        _mapping_path(repo).write_text("- a\n- b\n", encoding="utf-8")

        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output

    def test_missing_mapping_points_at_init(self, repo: Path) -> None:
        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "--init" in result.output


def _answers_comments(text: str) -> list[str]:
    """The ``# answers:`` comment following each ``answer: null`` line."""
    lines = text.splitlines()
    nulls = [index for index, line in enumerate(lines) if line.strip() == "answer: null"]
    comments = []
    for index in nulls:
        assert index + 1 < len(lines), f"nothing follows the answer on line {index}"
        following = lines[index + 1].strip()
        assert following.startswith("# answers: "), following
        comments.append(following)
    return comments


class TestInitVirtualAccelerator:
    """``--init`` on the one committed 2.0 tree: the block, and replacing it."""

    def test_an_answer_is_known_for_every_slot_kind(self) -> None:
        # _fill answers a slot by its kind, so a kind it does not know would
        # leave a null slot and turn a passing lane into a puzzle.
        assert set(VA_SLOT_ANSWERS) == VA_SLOT_KINDS

    def test_block_decides_every_family_and_leaves_the_open_slots_null(self, va_repo: Path) -> None:
        result = _map("--init")

        assert result.exit_code == 0, result.output
        assert "Traceback" not in result.output
        document = _load(va_repo)
        block = document["virtual_accelerator"]
        assert block["system"] == SYNTHETIC_SYSTEM
        assert len(block["families"]) == SYNTHETIC_FAMILIES
        open_slots = {
            name: family["slot"]
            for name, family in block["families"].items()
            if family.get("slot") is not None
        }
        assert {name: slot["kind"] for name, slot in open_slots.items()} == SYNTHETIC_OPEN_SLOTS
        assert [slot["answer"] for slot in open_slots.values()] == [None, None]
        assert parse_mapping(document).virtual_accelerator is not None

    def test_closing_line_counts_the_open_slots(self, va_repo: Path) -> None:
        result = _map("--init")

        assert result.exit_code == 0, result.output
        assert f"{len(SYNTHETIC_OPEN_SLOTS)} virtual-accelerator slots" in result.output

    def test_each_open_slot_is_followed_by_the_words_it_accepts(self, va_repo: Path) -> None:
        _map("--init")

        comments = _answers_comments(_mapping_path(va_repo).read_text(encoding="utf-8"))
        assert len(comments) == len(SYNTHETIC_OPEN_SLOTS)
        escape_hatch = next(line for line in comments if "ignore_hook" in line)
        assert escape_hatch == "# answers: latch, ignore_hook"
        attype = next(line for line in comments if line != escape_hatch)
        assert "latch" in attype
        assert all(kind in attype for kind in VA_KINDS), attype

    def test_a_second_init_refuses_and_names_force_va(self, va_repo: Path) -> None:
        assert _map("--init").exit_code == 0
        before = _mapping_path(va_repo).read_bytes()

        result = _map("--init")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "--force-va" in result.output
        assert _mapping_path(va_repo).read_bytes() == before

    def test_force_va_rewrites_the_block_byte_identically(self, va_repo: Path) -> None:
        assert _map("--init").exit_code == 0
        before = _mapping_path(va_repo).read_bytes()

        result = _map("--init", "--force-va")

        assert result.exit_code == 0, result.output
        assert _mapping_path(va_repo).read_bytes() == before
        assert f"{SYNTHETIC_FAMILIES} families" in result.output
        assert f"{len(SYNTHETIC_OPEN_SLOTS)} null slots" in result.output

    def test_force_va_keeps_a_reviewed_prefix_and_refreshes_the_block(self, va_repo: Path) -> None:
        assert _map("--init").exit_code == 0
        path = _mapping_path(va_repo)
        written = path.read_text(encoding="utf-8")
        # One edit in the prefix (the facility's provenance) and one in the
        # block (an answer), so the run has both a prefix to keep and a block
        # to throw away.
        reviewed = written.replace("provenance: derived", "provenance: stated", 1)
        assert reviewed != written
        path.write_text(reviewed.replace("answer: null", "answer: latch", 1), encoding="utf-8")

        result = _map("--init", "--force-va")

        assert result.exit_code == 0, result.output
        head = reviewed[: reviewed.index("virtual_accelerator:")]
        rewritten = path.read_text(encoding="utf-8")
        assert rewritten.startswith(head)
        assert rewritten == reviewed

    def test_deleting_the_block_and_re_initialising_reproduces_it(self, va_repo: Path) -> None:
        assert _map("--init").exit_code == 0
        path = _mapping_path(va_repo)
        written = path.read_text(encoding="utf-8")
        prefix = written[: written.index("virtual_accelerator:")]
        path.write_text(prefix, encoding="utf-8")

        result = _map("--init")

        assert result.exit_code == 0, result.output
        rewritten = path.read_text(encoding="utf-8")
        assert rewritten.startswith(prefix)
        assert rewritten == written

    def test_no_deck_in_the_tree_writes_no_block(self, va_repo: Path) -> None:
        # va.json states a virtual accelerator, but the deck it was sampled
        # over was never filed: the skeleton is written without a block.
        lattice_dir = va_repo / "data" / "mml" / "lattice"
        (lattice_dir / f"{SYNTHETIC_SYSTEM}.mat").unlink()

        result = _map("--init")

        assert result.exit_code == 0, result.output
        assert "Traceback" not in result.output
        assert (
            f"no lattice deck for {SYNTHETIC_SYSTEM} in {lattice_dir}; VA block not written"
            in result.output
        )
        assert "virtual-accelerator slot" not in result.output
        assert "virtual_accelerator" not in _mapping_path(va_repo).read_text(encoding="utf-8")


class TestInitOnAOneZeroTree:
    """A tree with no 2.0 export says so, and leaves the reviewed file alone."""

    @pytest.fixture
    def one_zero(
        self, request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Path, str]:
        """Import one 1.0 fixture and lay its reviewed mapping in the tree."""
        name = request.param
        sources, extra, named = ONE_ZERO_IMPORTS[name]
        root = _deploy(tmp_path / "deploy10", monkeypatch)
        _import(*(FIXTURES / source for source in sources), extra=extra)
        shutil.copy(FIXTURES / name / "mapping.yaml", _mapping_path(root))
        return root, named

    @pytest.mark.parametrize("one_zero", sorted(ONE_ZERO_IMPORTS), indirect=True)
    def test_init_says_so_keeps_the_file_and_still_wants_force(
        self, one_zero: tuple[Path, str]
    ) -> None:
        root, named = one_zero
        before = _mapping_path(root).read_bytes()

        result = _map("--init")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert f"no 2.0 export for {named}; VA block not written" in result.output
        assert "--force" in result.output
        assert "--force-va" not in result.output
        assert _mapping_path(root).read_bytes() == before

    @pytest.mark.parametrize("one_zero", sorted(ONE_ZERO_IMPORTS), indirect=True)
    def test_the_reviewed_mapping_passes_the_check(self, one_zero: tuple[Path, str]) -> None:
        root, _ = one_zero

        result = _map("--check", "--no-derived")

        assert result.exit_code == 0, result.output
        assert "passes the check" in result.output
        assert not [
            line for line in result.output.splitlines() if line.startswith("virtual_accelerator")
        ]
