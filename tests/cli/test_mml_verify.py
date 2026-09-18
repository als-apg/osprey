"""``osprey mml verify`` at the command line: the verb, the report, the refusals.

The comparison itself is pinned in ``tests/services/mml/test_va_verify.py``,
which calls :func:`osprey.services.mml.va.verify.verify` with structured inputs
and reads the :class:`VerifyReport` it returns. Nothing here reaches for that
object. Every lane below runs the whole chain a reviewer runs -- ``import`` →
``map --init`` → the reviewed mapping → ``map --check`` → ``emit`` → ``verify``
-- and then asserts only what the command actually puts in front of a person:
its exit code, the lines it prints, and the bytes of ``data/mml/VA-REPORT.md``.

That surface carries obligations of its own:

- **the verb and the path** -- the command is ``osprey mml verify``, the report
  lands at ``data/mml/VA-REPORT.md``, and the two refusals plus the closing
  line hold the deployment to ``emit`` → ``verify`` → ``osprey build``: a tree
  ``emit`` has not written is refused by the name of every file it is missing,
  and a run that succeeds sends the reader to ``osprey build`` next;
- **the report's shape** -- its six sections in the order the install skill
  walks a reviewer through them, a Verdict whose counts and percentage are the
  ones the command computed and printed, an orbit-response table that accounts
  for every compared entry, and rows left out named by the ``[sector, device]``
  pair they were aligned on rather than by their position in the file;
- **determinism** -- a second run over an unchanged tree rewrites the report
  byte for byte, prints the same lines, and touches nothing else under
  ``data/``.

Every one of those runs against each committed 2.0 export, discovered from the
fixtures rather than listed, so a tree committed later is covered the day it
lands; one guard lane fails loudly if the discovery ever collapses to nothing.

**The synthetic ring reproduces its own exported matrix, and a pass here is
plumbing rather than physics.** The fixture's matrix is measured about the same
6D closed orbit the served ring boots on, so a corrector's path-length change is
paid for in energy on both sides and neither column picks up a constant the
other lacks. Measured about the 4D orbit instead it would: three orders below
the response on a real ring, a quarter of it on a four-cell toy with a huge
momentum compaction. So the whole matrix is asked to pass and no entry is
allowed to be an outlier -- which says every binding, calibration and row
pairing is right, and says nothing about any machine, because both sides of the
comparison came off one deck.
"""

from __future__ import annotations

import math
import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli

# The files the command itself refuses a tree for, so a file added to that set
# gets its own refusal lane below without a name being typed here.
from osprey.cli.mml_cmd import _VA_SERVED
from osprey.services.mml.va.verify import REPORT_FILENAME
from tests.cli.test_mml_emit import FIXTURES, TWO_ZERO_TREES

pytest.importorskip("lume_pyat")
pytest.importorskip("linkml_runtime")

#: A 1.0 export: no lattice, no calibrations and no response matrix, so no
#: virtual accelerator for this verb to hold anything against.
ONE_ZERO = FIXTURES / "paired"

#: The report, relative to the deployment repo: where the install skill sends a
#: reviewer between ``emit`` and ``osprey build``.
REPORT = Path("data") / "mml" / REPORT_FILENAME

#: What ``emit`` must have written for a model to be built over the tree, as a
#: person reads the path. Each is removed on its own below, so the refusal
#: names the file that is missing rather than the first one the command looks
#: for.
SERVED = tuple((Path("data") / name).as_posix() for name in _VA_SERVED)

#: The report's sections, in the order a reviewer reads them.
SECTIONS = (
    "## Verdict",
    "## Export",
    "## Orbit response",
    "## Rows not compared",
    "## Widened bands",
    "## Nominals the model does not maintain",
)

_VERDICT = re.compile(r"^(\d+) of (\d+) entries \(([\d.]+) %\) are inside the band")
_SPOKEN = re.compile(r"^(\d+) of (\d+) response entries \(([\d.]+) %\) are inside the band")
_SIGN = re.compile(r"^Sign agrees on (\d+) of the (\d+) entries above the floor")
_DEVICE_ROW = re.compile(r"^\[\d+, \d+\]$")
_COUNT = re.compile(r"^(\d+) \(")


# ===================================================================
# Running the verb
# ===================================================================


@dataclass(frozen=True)
class Tree:
    """One committed 2.0 export, installed and verified.

    Attributes:
        name: The fixture directory's name.
        root: The deployment repo the chain ran in.
        spoken: Everything ``verify`` printed.
        report: ``VA-REPORT.md`` as the run left it.
    """

    name: str
    root: Path
    spoken: str
    report: str


def _run(root: Path, *args: str):
    """Run one verb of the chain against ``root``, as a reviewer runs it."""
    return CliRunner().invoke(cli, ["mml", *args, "--repo", str(root)], catch_exceptions=False)


def _must(root: Path, *args: str) -> str:
    """Run one verb and refuse anything but a clean exit."""
    result = _run(root, *args)
    assert "Traceback" not in result.output
    assert result.exit_code == 0, f"osprey mml {' '.join(args)}:\n{result.output}"
    return result.output


def _install(root: Path, fixture: Path) -> Tree:
    """Install one fixture export end to end and verify what it emitted.

    The reviewed mapping is copied over the skeleton ``map --init`` writes, the
    way a facility's own review leaves it, so the virtual-accelerator block
    carries answers rather than nulls.
    """
    exports = sorted(fixture.glob("*.ao.json"))
    assert exports, f"{fixture.name} commits a virtual accelerator but no export"
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    _must(root, "import", *(str(path) for path in exports))
    _must(root, "map", "--init")
    shutil.copy(fixture / "mapping.yaml", root / "data" / "mml" / "mapping.yaml")
    _must(root, "map", "--check")
    _must(root, "emit")
    spoken = _must(root, "verify")
    return Tree(
        name=fixture.name,
        root=root,
        spoken=spoken,
        report=(root / REPORT).read_text(encoding="utf-8"),
    )


@pytest.fixture(scope="module")
def trees(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Tree]:
    """A per-fixture installed tree, built once and shared by every lane.

    Module scoped: the chain writes a whole deployment and boots a ring, and
    every assertion below reads the same tree rather than paying for it again.
    """
    built: dict[str, Tree] = {}

    def build(name: str) -> Tree:
        if name not in built:
            built[name] = _install(tmp_path_factory.mktemp(f"verify-{name}"), FIXTURES / name)
        return built[name]

    return build


@pytest.fixture(scope="module", params=TWO_ZERO_TREES)
def tree(request: pytest.FixtureRequest, trees: Callable[[str], Tree]) -> Tree:
    """Every committed 2.0 fixture in turn, installed and verified."""
    return trees(request.param)


def _only(name: str) -> pytest.MarkDecorator:
    """Run a lane only where the fixture it names commits a 2.0 export."""
    return pytest.mark.skipif(
        name not in TWO_ZERO_TREES, reason=f"{name} commits no 2.0 export yet"
    )


# ===================================================================
# Reading the report
# ===================================================================


def _section(text: str, heading: str) -> list[str]:
    """The lines under one ``##`` heading, its own subsections included."""
    lines = text.splitlines()
    assert heading in lines, f"{heading} is not in the report"
    start = lines.index(heading)
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return lines[start + 1 : end]


def _subsections(lines: Sequence[str]) -> dict[str, list[str]]:
    """Each ``###`` subsection of a section, keyed by its heading."""
    found: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        if line.startswith("### "):
            current = line.removeprefix("### ")
            found[current] = []
        elif current is not None:
            found[current].append(line)
    return found


def _tables(lines: Iterable[str]) -> list[list[list[str]]]:
    """Every markdown table in a block of lines, as its data rows' cells."""
    tables: list[list[list[str]]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("|"):
            current.append(line)
            continue
        if current:
            tables.append(_cells(current))
            current = []
    if current:
        tables.append(_cells(current))
    return tables


def _cells(rows: Sequence[str]) -> list[list[str]]:
    """The data rows of one table, past its header and rule."""
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows[2:]]


def _one_table(text: str, heading: str) -> list[list[str]]:
    """The first table of one section: the one the section is named for."""
    tables = _tables(_section(text, heading))
    assert tables, f"{heading} states no table"
    return tables[0]


def _verdict(text: str) -> tuple[int, int, float]:
    """The Verdict's counts and percentage: passed, compared, per cent."""
    return _counts(_VERDICT, _section(text, "## Verdict"))


def _spoken_verdict(spoken: str) -> tuple[int, int, float]:
    """The same three numbers, as the command printed them."""
    return _counts(_SPOKEN, spoken.splitlines())


def _counts(pattern: re.Pattern[str], lines: Iterable[str]) -> tuple[int, int, float]:
    for line in lines:
        found = pattern.match(line)
        if found:
            return int(found[1]), int(found[2]), float(found[3])
    raise AssertionError(f"no line matches {pattern.pattern!r}")


def _sign(text: str) -> tuple[int, int]:
    """How many entries agree in sign, and how many sit above the floor."""
    for line in _section(text, "## Verdict"):
        found = _SIGN.match(line)
        if found:
            return int(found[1]), int(found[2])
    raise AssertionError("the Verdict asserts no sign agreement")


def _blocks(text: str) -> dict[str, list[str]]:
    """The orbit-response summary, keyed by ``monitors ← actuators``."""
    return {f"{row[0]} ← {row[1]}": row for row in _one_table(text, "## Orbit response")}


def _inside(row: Sequence[str]) -> int:
    """How many entries of one block landed inside their band."""
    found = _COUNT.match(row[3])
    assert found, f"{row[3]!r} does not count the entries inside the band"
    return int(found[1])


def _agreement(row: Sequence[str]) -> tuple[int, int]:
    """One block's sign agreement, as the table spells it."""
    agreed, _, checked = row[4].partition("/")
    return int(agreed), int(checked)


def _median_ratio(row: Sequence[str]) -> float:
    """One block's median magnitude ratio; ``nan`` where it states none."""
    return math.nan if row[5] == "—" else float(row[5])


def _tree_bytes(root: Path, *, except_for: Path) -> dict[str, bytes]:
    """Every file under ``data/`` bar one, keyed by its path relative to root."""
    data = root / "data"
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(data.rglob("*"))
        if path.is_file() and path.relative_to(root) != except_for
    }


# ===================================================================
# The verb, the report path and the order of the chain
# ===================================================================


class TestTheCommandTheInstallSkillNames:
    """The name a facility types, the file it is sent to, and what comes next."""

    def test_a_two_zero_export_is_committed(self) -> None:
        # Every parametrised lane below is driven by this discovery, so a
        # fixture directory that stopped carrying a *.va.json sibling would
        # empty them all silently rather than fail.
        assert TWO_ZERO_TREES, "no fixture export carries a *.va.json sibling"

    def test_the_command_names_the_files_a_served_model_boots_from(self) -> None:
        # The refusal lane below is parametrised over this set, so an empty one
        # would leave a tree emit never wrote unasked about.
        assert "data/simulation/va_bindings.json" in SERVED

    def test_the_verb_is_mml_verify(self) -> None:
        result = CliRunner().invoke(cli, ["mml", "--help"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        assert re.search(r"^\s*verify\s+\S", result.output, re.MULTILINE), result.output

    def test_it_writes_the_report_where_the_reviewer_is_sent(self, tree: Tree) -> None:
        assert (tree.root / REPORT).is_file()
        assert str(tree.root / REPORT) in tree.spoken

    def test_it_sends_the_reader_to_osprey_build_next(self, tree: Tree) -> None:
        # The step after verify, and the reason the report exists: the emitted
        # tree is copied into the deployment only once a person has read it.
        assert "Read it, then run osprey build" in tree.spoken

    @pytest.mark.parametrize("missing", SERVED)
    def test_a_tree_emit_has_not_written_is_refused_by_the_missing_files_name(
        self, tree: Tree, tmp_path: Path, missing: str
    ) -> None:
        root = tmp_path / "unemitted"
        shutil.copytree(tree.root, root)
        (root / missing).unlink()
        (root / REPORT).unlink()

        result = _run(root, "verify")

        assert result.exit_code == 1, result.output
        assert missing in result.output
        assert "run osprey mml emit first" in result.output
        assert not (root / REPORT).is_file()

    def test_a_one_zero_tree_is_refused_with_the_files_a_two_zero_export_files(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "one-zero"
        root.mkdir()
        (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
        _must(root, "import", str(ONE_ZERO / "quokka.ring.ao.json"))
        _must(root, "map", "--init")
        shutil.copy(ONE_ZERO / "mapping.yaml", root / "data" / "mml" / "mapping.yaml")

        result = _run(root, "verify")

        assert result.exit_code == 1, result.output
        assert "va.json" in result.output
        assert "response.json" in result.output
        assert "mml_export 2.0" in result.output
        assert not (root / REPORT).is_file()


# ===================================================================
# What the report says, on every tree that carries a virtual accelerator
# ===================================================================


class TestTheReportOfEveryCommittedTree:
    """``VA-REPORT.md``: its sections, its arithmetic and its row names."""

    def test_it_carries_every_section_in_the_order_a_reviewer_reads_them(self, tree: Tree) -> None:
        lines = tree.report.splitlines()
        positions = [lines.index(heading) for heading in SECTIONS if heading in lines]

        assert len(positions) == len(SECTIONS), f"missing {set(SECTIONS) - set(lines)}"
        assert positions == sorted(positions), "the sections are out of order"

    def test_the_verdict_states_the_ratio_the_command_computed(self, tree: Tree) -> None:
        passed, compared, percent = _verdict(tree.report)

        assert compared, "not one entry of the exported matrix was compared"
        assert passed <= compared
        assert percent == pytest.approx(round(passed / compared * 100, 1), abs=0.05)
        assert _spoken_verdict(tree.spoken) == (passed, compared, percent)

    def test_the_orbit_response_table_accounts_for_every_compared_entry(self, tree: Tree) -> None:
        passed, compared, _ = _verdict(tree.report)
        rows = _one_table(tree.report, "## Orbit response")

        assert rows
        assert sum(int(row[2]) for row in rows) == compared
        assert sum(_inside(row) for row in rows) == passed

    def test_the_verdicts_sign_agreement_is_the_sum_of_the_blocks(self, tree: Tree) -> None:
        agreed, checked = _sign(tree.report)
        rows = _one_table(tree.report, "## Orbit response")
        blocks = [_agreement(row) for row in rows]

        assert sum(one for one, _ in blocks) == agreed
        assert sum(total for _, total in blocks) == checked

    def test_a_row_that_was_not_compared_is_named_by_its_sector_and_device(
        self, tree: Tree
    ) -> None:
        # Rows are paired with the judged DeviceList by the sector and device
        # they name, never by where they sit in the file, so a row left out is
        # named the same way and a reviewer can find the magnet it belongs to.
        lines = _section(tree.report, "## Rows not compared")
        tables = _tables(lines)
        if not tables:
            assert "Every row of every block reached a binding" in "\n".join(lines)
            return

        for row in tables[0]:
            assert _DEVICE_ROW.match(row[3]), f"{row[3]!r} is not a [sector, device] pair"

    def test_a_second_run_rewrites_the_report_byte_for_byte_and_touches_nothing_else(
        self, tree: Tree
    ) -> None:
        # Nothing the comparison decides may depend on when it was run: a
        # reviewer who repeats it has to get the same verdict to act on.
        before = _tree_bytes(tree.root, except_for=REPORT)

        spoken = _must(tree.root, "verify")

        assert spoken == tree.spoken
        assert (tree.root / REPORT).read_text(encoding="utf-8") == tree.report
        assert _tree_bytes(tree.root, except_for=REPORT) == before


# ===================================================================
# The synthetic ring against its own exported matrix
# ===================================================================


@_only("synthetic")
class TestTheSyntheticRingAgainstItsOwnMatrix:
    """A four-cell toy whose matrix was measured about the orbit the model runs on.

    Both sides are the same deck, so what the comparison proves is the chain
    between them -- the bindings, the device order, the units -- and nothing
    about any machine. A pass here is plumbing, not physics.
    """

    @pytest.fixture(scope="class")
    def report(self, trees: Callable[[str], Tree]) -> str:
        return trees("synthetic").report

    def test_the_vertical_correctors_move_the_vertical_monitors_as_the_file_says(
        self, report: str
    ) -> None:
        row = _blocks(report)["BPMy ← VC"]
        agreed, checked = _agreement(row)

        assert _inside(row) == int(row[2]), row
        assert checked and agreed == checked
        assert _median_ratio(row) == pytest.approx(1.0, abs=0.01)

    def test_no_entry_of_any_block_is_an_outlier(self, report: str) -> None:
        """A wrong calibration scales a column, a mispaired row swaps two entries
        of it, an orbit solved about another point adds one number to every entry
        alike. Any of them would put a row in one of these tables.
        """
        listed = _subsections(_section(report, "## Orbit response"))

        assert not [heading for heading in listed if "outliers against" in heading]

    def test_the_whole_matrix_passes_the_criterion(self, report: str) -> None:
        """Every entry of every block, and the count says every one was reached.

        The toy pairs four monitors of each plane with three horizontal and four
        vertical correctors, twice over, so the whole matrix is 56 entries. A
        chain that compared one entry and passed it would satisfy the ratio
        alone.
        """
        passed, compared, _ = _verdict(report)

        assert compared == 56
        assert passed == compared


# ===================================================================
# Success criterion 2 — the two re-exported facility matrices
# ===================================================================


class TestTheReExportedFacilityMatrices:
    """The named exports the feature is judged on, each held to its own bar.

    A model-derived matrix and a matrix measured on the machine are read
    differently: the first has to reproduce nearly every entry, the second only
    has to agree on which way the beam moves and on the scale of the bulk. Each
    lane skips until its facility commits a 2.0 export, and runs the day one
    lands without a name being typed again.
    """

    @_only("nsls2")
    def test_the_nsls2_matrix_is_reproduced_entry_by_entry(
        self, trees: Callable[[str], Tree]
    ) -> None:
        report = trees("nsls2").report
        passed, compared, _ = _verdict(report)
        agreed, checked = _sign(report)

        assert compared
        assert passed / compared >= 0.95, f"{passed} of {compared}"
        assert checked and agreed == checked

    @_only("spear3")
    def test_the_spear3_matrix_agrees_on_sign_and_on_scale(
        self, trees: Callable[[str], Tree]
    ) -> None:
        report = trees("spear3").report
        agreed, checked = _sign(report)
        ratios = {
            name: _median_ratio(row)
            for name, row in _blocks(report).items()
            if math.isfinite(_median_ratio(row))
        }

        assert checked and agreed == checked
        assert ratios
        for name, ratio in ratios.items():
            assert 0.8 <= ratio <= 1.25, f"{name} median ratio {ratio}"
