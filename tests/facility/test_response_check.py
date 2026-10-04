"""The response check ``osprey facility validate`` runs per kept response export.

The tree cases import a fixture tree's exports into a fresh ``data/facility/``
under the tree's mapping, widen the limits records the build names, and hold
each model's kept ``imported/mml/<model>.response.json`` against the model.
The figure cases build entries by hand, and the parity case pins the figures
to ``osprey.services.mml.va.verify`` on spear3.
"""

from __future__ import annotations

import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from osprey.facility import response_check
from osprey.facility.build import build_facility
from osprey.facility.layers.mml.importer import LAYER_DIR, import_mml
from osprey.facility.layers.mml.mapping import MAPPING_FILE
from osprey.facility.response_check import (
    Block,
    Entry,
    LeftOut,
    ModelCheck,
    banded,
    check_responses,
    compare,
    figures,
    judge,
    report,
)
from tests.facility.test_mml_layer_seed_once import OUTSIDE, WIDENED, _widen

at = pytest.importorskip("at")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

#: Each tree's exports, by the stem every file of one export is named after.
TREES: dict[str, tuple[str, ...]] = {
    "spear3": ("spear3.storagering",),
    "nsls2": ("nsls2.storagering", "nsls2.ltb"),
    "synthetic": ("quokka.sr",),
}

SPEAR3_LINE = (
    "response check StorageRing: measured BPMx/HCM median ratio 0.922 (pass at 0.8 to 1.25)"
)
NSLS2_LINES = [
    "response check LTB: model - judged blocks 0 (pass at 0)",
    "response check StorageRing: model BPMx/HCM inside band 1.000 (pass at 0.99)",
]
NSLS2_LTB_LEFT_OUT = (
    "response check LTB: left out 24 rows (24 unwired, 0 no width, 0 unsolved, 0 table calibration)"
)
SYNTHETIC_LINE = "response check SR: model BPMx/HC inside band 1.000 (pass at 0.99)"

#: The synthetic tree's one setpoint whose nominal lies outside its stated band,
#: with the edge that holds it.
SYNTHETIC_WIDENED: dict[str, tuple[str, float]] = {OUTSIDE: ("max_value", 2.0)}


def _repo(root: Path, tree: str) -> Path:
    """A repo whose ``data/facility/`` is one fixture tree, imported and building clean."""
    facility = root / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / tree / MAPPING_FILE, target)
    import_mml([FIXTURES / tree / f"{stem}.ao.json" for stem in TREES[tree]], facility)
    _widen(facility, WIDENED.get(tree, {}))
    if tree == "synthetic":
        _widen(facility, SYNTHETIC_WIDENED)
    (root / "profile.yml").write_text("name: scratch\ndata: data\n", encoding="utf-8")
    return root


def _facility(repo: Path) -> Path:
    return repo / "data" / "facility"


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


def _validate(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Result:
    """Run ``osprey facility validate`` with the profile render left out.

    The response check runs on the facility file, before the render; the
    render of a whole profile is held by the verb's own cases.
    """
    from osprey.cli import build_cmd

    monkeypatch.setattr(build_cmd, "_render_project", lambda *args, **kwargs: None)
    return CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])


@pytest.fixture(scope="module")
def spear3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _repo(tmp_path_factory.mktemp("spear3"), "spear3")


@pytest.fixture(scope="module")
def nsls2(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _repo(tmp_path_factory.mktemp("nsls2"), "nsls2")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _repo(tmp_path_factory.mktemp("synthetic"), "synthetic")


@pytest.fixture(scope="module")
def spear3_blocks(spear3: Path) -> tuple[Block, ...]:
    facility = _facility(spear3)
    document = build_facility(facility, project_name="scratch")
    model = next(model for model in document["models"] if model["name"] == "StorageRing")
    return compare(facility, document, model)


# --- the verb -----------------------------------------------------------------------


def test_a_measured_export_prints_one_line_and_exits_0(
    spear3: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _snapshot(spear3)

    result = _validate(spear3, monkeypatch)

    assert result.exit_code == 0, result.output
    assert (result.stdout, result.stderr) == ("", SPEAR3_LINE + "\n")
    assert _snapshot(spear3) == before


def test_a_model_derived_export_prints_one_line_and_exits_0(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The export's one monitor device whose status flag is down is not counted."""
    before = _snapshot(synthetic)

    result = _validate(synthetic, monkeypatch)

    assert result.exit_code == 0, result.output
    assert (result.stdout, result.stderr) == ("", SYNTHETIC_LINE + "\n")
    assert _snapshot(synthetic) == before


def test_a_model_derived_export_passes_block_by_block(nsls2: Path) -> None:
    facility = _facility(nsls2)
    document = build_facility(facility, project_name="scratch")

    checks = check_responses(facility, document)

    assert [check.line for check in checks] == NSLS2_LINES
    assert all(check.passed for check in checks)


def test_the_rows_left_out_are_named_on_a_second_line(nsls2: Path) -> None:
    """The LTB export's monitor families are wired to no device; the StorageRing export's all are."""
    facility = _facility(nsls2)
    document = build_facility(facility, project_name="scratch")
    stream = io.StringIO()

    passed = report(check_responses(facility, document), stream)

    assert passed
    assert stream.getvalue().splitlines() == [
        NSLS2_LINES[0],
        NSLS2_LTB_LEFT_OUT,
        NSLS2_LINES[1],
    ]


def test_one_judged_block_scaled_by_a_tenth_exits_1(
    nsls2: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "nsls2"
    shutil.copytree(nsls2, repo)
    path = _facility(repo) / LAYER_DIR / "StorageRing.response.json"
    response = json.loads(path.read_text(encoding="utf-8"))
    block = response["blocks"][0]
    assert (block["monitor"]["family"], block["actuator"]["family"]) == ("BPMx", "HCM")
    assert block["origin"] == "model"
    block["data"] = [[1.10 * value for value in row] for row in block["data"]]
    path.write_text(json.dumps(response), encoding="utf-8")
    before = _snapshot(repo)

    result = _validate(repo, monkeypatch)

    assert result.exit_code == 1, result.output
    assert (result.stdout, result.stderr) == (
        "",
        NSLS2_LINES[0]
        + "\n"
        + NSLS2_LTB_LEFT_OUT
        + "\n"
        + "response check StorageRing: model BPMx/HCM inside band 0.022 (fail at 0.99)\n",
    )
    assert _snapshot(repo) == before


def test_a_tree_without_a_kept_export_checks_nothing(spear3: Path, tmp_path: Path) -> None:
    repo = tmp_path / "spear3"
    shutil.copytree(spear3, repo)
    facility = _facility(repo)
    (facility / LAYER_DIR / "StorageRing.response.json").unlink()

    assert check_responses(facility, build_facility(facility, project_name="scratch")) == []


def test_an_unreadable_export_exits_1_without_a_traceback(
    spear3: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "spear3"
    shutil.copytree(spear3, repo)
    (_facility(repo) / LAYER_DIR / "StorageRing.response.json").write_text("{", encoding="utf-8")

    result = _validate(repo, monkeypatch)

    assert result.exit_code == 1
    assert "The response check cannot run." in result.stderr
    assert "Traceback" not in result.output


# --- parity with the comparison it re-expresses -------------------------------------


def test_the_bands_are_the_ones_mml_verify_draws() -> None:
    from osprey.services.mml.va import verify

    assert response_check.TOLERANCE_FRACTION == verify.TOLERANCE_FRACTION == 0.05
    assert response_check.FLOOR_FRACTION == verify.FLOOR_FRACTION == 0.1
    assert response_check.POLARITY_SHARE == verify.POLARITY_SHARE == 0.5


def test_the_figures_are_mml_verifys_on_spear3(spear3_blocks: tuple[Block, ...]) -> None:
    """Each block's entries, banded and weighed by ``verify.py``, give the same figures."""
    from dataclasses import asdict

    from osprey.services.mml.va import verify

    def theirs(entry: Entry) -> Any:
        return verify.Entry(
            monitor_address=entry.monitor_address,
            actuator_address=entry.actuator_address,
            monitor_device=entry.monitor_device,
            actuator_device=entry.actuator_device,
            file_value=entry.file_value,
            model_value=entry.model_value,
            tolerance=0.0,
            floor=0.0,
        )

    floor = verify.FLOOR_FRACTION * verify._rms(
        entry.file_value for block in spear3_blocks for entry in block.entries
    )
    assert len(spear3_blocks) == 4
    for block in spear3_blocks:
        draft = verify._Draft(
            block={
                "monitor": {"family": block.monitor_family},
                "actuator": {"family": block.actuator_family},
                "origin": block.origin,
            },
            entries=tuple(theirs(entry) for entry in block.entries),
            dropped=(),
            swept=(),
            monitor_plane="",
            actuator_plane="",
            unjudged=None if block.judged else "cross-plane",
        )
        report = verify._banded(draft, floor)

        assert block.entries, block.name
        assert [asdict(entry) for entry in block.entries] == [
            asdict(entry) for entry in report.entries
        ], block.name
        for mine, their in zip(block.entries, report.entries, strict=True):
            assert (mine.passed, mine.above_floor, mine.sign_agrees) == (
                their.passed,
                their.above_floor,
                their.sign_agrees,
            )
        assert block.reversed_columns == tuple(column.device for column in report.polarity)
        for mine_figures, their_figures in (
            (block.counts, report.counts),
            (block.full, report.full),
        ):
            assert asdict(mine_figures) == pytest.approx(asdict(their_figures), nan_ok=True)
            assert (mine_figures.agreed is None) == (not block.judged), block.name


def test_spear3_holds_the_counts_mml_verify_reports(spear3_blocks: tuple[Block, ...]) -> None:
    """The orbit-response table ``osprey mml verify`` writes for the same export.

    A cross-plane block's model entries are the deck's coupling at the noise
    level, so an unjudged block counts no sign agreement: its ``agreed`` is
    ``None``. Every other figure is pinned for every block.
    """

    def row(block: Block) -> tuple[Any, ...]:
        full = block.full
        return (
            block.judged,
            full.compared,
            full.passed,
            full.agreed,
            full.checked,
            len(block.reversed_columns),
        )

    table = {block.name: row(block) for block in spear3_blocks}
    assert table == {
        "BPMx/HCM": (True, 3306, 795, 3064, 3231, 3),
        "BPMx/VCM": (False, 3192, 387, None, 11, 0),
        "BPMy/HCM": (False, 3306, 753, None, 0, 0),
        "BPMy/VCM": (True, 3192, 1358, 3007, 3007, 0),
    }
    assert {block.origin for block in spear3_blocks} == {"measured"}
    ratios = {block.name: block.full.median_ratio for block in spear3_blocks}
    assert ratios["BPMx/HCM"] == pytest.approx(0.921625, rel=1e-5)
    assert ratios["BPMy/VCM"] == pytest.approx(0.944868, rel=1e-5)


# --- the bars -----------------------------------------------------------------------


def _block(
    origin: str,
    pairs: list[tuple[float, float]],
    *,
    judged: bool = True,
    columns: int = 1,
    name: str = "BPMx",
) -> Block:
    """A banded block of ``(file, model)`` entries, dealt over ``columns`` columns."""
    entries = tuple(
        Entry(
            monitor_address=f"M{index}",
            actuator_address=f"A{index % columns}",
            monitor_device=index // columns + 1,
            actuator_device=index % columns + 1,
            file_value=file_value,
            model_value=model_value,
        )
        for index, (file_value, model_value) in enumerate(pairs)
    )
    block = Block(name, "HCM", origin, entries, judged)
    floor = response_check.FLOOR_FRACTION * math.sqrt(
        sum(file_value**2 for file_value, _ in pairs) / len(pairs)
    )
    return banded(block, floor)


def _line(*blocks: Block) -> str:
    return ModelCheck("SR", judge(blocks)).line


def test_a_check_that_left_nothing_out_prints_one_line() -> None:
    check = ModelCheck("SR", judge([_block("model", [(1.0, 1.0)] * 4)]))

    assert check.lines == [check.line]


def test_the_left_out_line_counts_each_reason() -> None:
    left_out = LeftOut(unwired=3, no_width=1) + LeftOut(unsolved=1)
    check = ModelCheck("SR", judge([_block("model", [(1.0, 1.0)] * 4)]), left_out)

    assert check.lines == [
        "response check SR: model BPMx/HCM inside band 1.000 (pass at 0.99)",
        "response check SR: left out 5 rows "
        "(3 unwired, 1 no width, 1 unsolved, 0 table calibration)",
    ]


def test_an_entry_is_banded_at_five_per_cent_of_itself_or_of_the_floor() -> None:
    block = _block("model", [(1.0, 1.04), (1.0, 1.06), (0.01, 0.012), (0.01, 0.02)])
    floor = 0.1 * math.sqrt((1.0 + 1.0 + 1e-4 + 1e-4) / 4)

    assert [entry.floor for entry in block.entries] == pytest.approx([floor] * 4)
    assert [entry.tolerance for entry in block.entries] == pytest.approx(
        [0.05, 0.05, 0.05 * floor, 0.05 * floor]
    )
    assert [entry.passed for entry in block.entries] == [True, False, True, False]
    assert [entry.above_floor for entry in block.entries] == [True, True, False, False]


def test_a_model_derived_block_needs_ninety_nine_per_cent_inside_the_band() -> None:
    inside = [(1.0, 1.0)] * 99
    assert _line(_block("model", [*inside, (1.0, 2.0)])) == (
        "response check SR: model BPMx/HCM inside band 0.990 (pass at 0.99)"
    )
    assert _line(_block("model", [*inside[:97], (1.0, 2.0), (1.0, 2.0)])) == (
        "response check SR: model BPMx/HCM inside band 0.980 (fail at 0.99)"
    )


def test_a_measured_block_is_read_through_its_median_ratio_and_its_sign() -> None:
    assert _line(_block("measured", [(1.0, 0.9)] * 20)) == (
        "response check SR: measured BPMx/HCM median ratio 0.900 (pass at 0.8 to 1.25)"
    )
    assert _line(_block("measured", [(1.0, 0.79)] * 20)) == (
        "response check SR: measured BPMx/HCM median ratio 0.790 (fail at 0.8 to 1.25)"
    )
    assert _line(_block("measured", [(1.0, 1.26)] * 20)) == (
        "response check SR: measured BPMx/HCM median ratio 1.260 (fail at 0.8 to 1.25)"
    )
    flipped = [(1.0, 1.0)] * 18 + [(1.0, -1.0)] * 2
    assert _line(_block("measured", flipped)) == (
        "response check SR: measured BPMx/HCM sign agreement 0.900 (fail at 0.95)"
    )


def test_a_judged_block_needs_one_entry_above_the_floor() -> None:
    assert _line(_block("model", [(0.0, 0.0)] * 4)) == (
        "response check SR: model BPMx/HCM above floor 0 (fail at 1)"
    )
    assert _line(_block("measured", [(0.0, 0.0)] * 4)) == (
        "response check SR: measured BPMx/HCM above floor 0 (fail at 1)"
    )


def test_a_failing_block_is_named_ahead_of_a_passing_one() -> None:
    good = _block("model", [(1.0, 1.0)] * 10, name="BPMy")
    bad = _block("model", [(1.0, 2.0)] * 10)

    assert _line(good, bad) == "response check SR: model BPMx/HCM inside band 0.000 (fail at 0.99)"
    assert _line(bad, good) == "response check SR: model BPMx/HCM inside band 0.000 (fail at 0.99)"


def test_a_cross_plane_block_is_judged_by_nothing() -> None:
    cross = _block("model", [(1.0, 2.0)] * 10, judged=False)
    good = _block("model", [(1.0, 1.0)] * 10, name="BPMy")

    assert (
        _line(cross, good) == "response check SR: model BPMy/HCM inside band 1.000 (pass at 0.99)"
    )
    assert ModelCheck("SR", judge([cross], "model")).line == (
        "response check SR: model - judged blocks 0 (pass at 0)"
    )


def test_a_cross_plane_block_counts_no_sign_agreement() -> None:
    """Above-floor entries whose signs all agree count only where the block is judged."""
    pairs = [(1.0, 1.0)] * 10
    cross = _block("measured", pairs, judged=False)
    judged = _block("measured", pairs)

    for weighed in (cross.counts, cross.full, figures(cross.entries, judged=False)):
        assert weighed.checked == 10
        assert weighed.agreed is None
        assert math.isnan(weighed.sign_ratio)
    for weighed in (judged.counts, judged.full, figures(judged.entries)):
        assert weighed.checked == 10
        assert weighed.agreed == weighed.checked
        assert weighed.sign_ratio == 1.0


def test_a_reversed_column_is_set_aside() -> None:
    """Two columns of four entries; every entry of the second runs backwards."""
    pairs = [(1.0, 1.0), (1.0, -1.0)] * 4
    block = _block("measured", pairs, columns=2)

    assert block.reversed_columns == (2,)
    assert block.counts.compared == 4
    assert _line(block) == (
        "response check SR: measured BPMx/HCM median ratio 1.000 (pass at 0.8 to 1.25)"
    )


def test_a_column_half_flipped_is_not_reversed() -> None:
    block = _block("measured", [(1.0, 1.0), (1.0, -1.0)], columns=1)

    assert block.reversed_columns == ()
    assert block.counts.sign_ratio == 0.5


# --- the solve ----------------------------------------------------------------------


def test_a_corrector_whose_sweep_has_no_solve_leaves_its_column_out(
    spear3: Path, spear3_blocks: tuple[Block, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first sweep's solve is refused; every other column is still judged."""
    from lume_pyat.exceptions import OrbitSolveError
    from lume_pyat.simulator import PyATSimulator

    original = PyATSimulator.solve
    calls = {"count": 0}

    def solve(self: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OrbitSolveError("closed orbit solve raised AtError: lost")
        return original(self)

    monkeypatch.setattr(PyATSimulator, "solve", solve)
    facility = _facility(spear3)
    document = build_facility(facility, project_name="scratch")
    model = next(model for model in document["models"] if model["name"] == "StorageRing")

    blocks = compare(facility, document, model)

    def measured(block: Block, unsolved: str = "") -> list[tuple[str, str, float]]:
        return [
            (entry.monitor_address, entry.actuator_address, entry.model_value)
            for entry in block.entries
            if entry.actuator_address != unsolved
        ]

    unsolved = spear3_blocks[0].entries[0].actuator_address
    assert [block.name for block in blocks] == [block.name for block in spear3_blocks]
    for block, whole in zip(blocks, spear3_blocks, strict=True):
        assert measured(block) == measured(whole, unsolved), block.name
    assert len(measured(blocks[0])) < len(measured(spear3_blocks[0]))
    assert ModelCheck("StorageRing", judge(blocks)).line.startswith(
        "response check StorageRing: measured BPMx/HCM median ratio "
    )
    assert sum((block.left_out for block in blocks), LeftOut()) == LeftOut(unsolved=2)


def test_a_monitor_with_a_table_calibration_is_left_out_and_counted(synthetic: Path) -> None:
    """The reading is carried into physics by one slope, which only a straight line has."""
    facility = _facility(synthetic)
    document = build_facility(facility, project_name="scratch")
    model = next(model for model in document["models"] if model["name"] == "SR")
    address = "QK:BPMx:1:CUR:RB"
    record = next(record for record in model["wiring"] if record["address"] == address)
    record["calibration"] = {
        "curve": {"table": {"grid": [-1000.0, 0.0, 1000.0], "values": [-1.0, 0.0, 1.0]}},
        "inverse": {"table": {"grid": [-1.0, 0.0, 1.0], "values": [-1000.0, 0.0, 1000.0]}},
        "energy_scaling": "none",
    }

    blocks = compare(facility, document, model)

    assert [block.name for block in blocks if block.monitor_family == "BPMx"] == [
        "BPMx/HC",
        "BPMx/VC",
    ]
    assert all(entry.monitor_address != address for block in blocks for entry in block.entries)
    assert sum((block.left_out for block in blocks), LeftOut()) == LeftOut(table_calibration=2)
    assert ModelCheck("SR", judge(blocks), LeftOut(table_calibration=2)).lines[1] == (
        "response check SR: left out 2 rows "
        "(0 unwired, 0 no width, 0 unsolved, 2 table calibration)"
    )
