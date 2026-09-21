"""``osprey mml verify``: the export's response matrix, re-measured on the model.

The whole verb runs here on the one committed 2.0 export, through the chain a
reviewer runs -- ``import``, ``map --init``, the answers, ``emit`` -- so every
assertion is made against a tree this repository can actually produce rather
than against a hand-written stand-in of one.

What the lanes pin:

- **the criterion** -- the per-entry band and its floor, which is a share of
  the whole matrix's scale rather than of one column's, both as arithmetic on
  a hand-built entry and as the number every entry of the real comparison was
  actually given;
- **which blocks carry a verdict** -- the in-plane ones, read off the emitted
  bindings rather than off any family's name, with the cross-plane blocks
  compared and printed and pooled into nothing;
- **which columns it is made of** -- a corrector whose column the file has
  running backwards is named on its own and left out of what its block counts,
  while a scatter of flipped entries is not;
- **alignment** -- an entry names the addresses its two device rows name, and
  a row no judged device sits at is reported instead of compared, so a whole
  column can never land on the wrong magnet;
- **the ``Status`` filter** -- the response file's own zero row is dropped and
  named, while the ``Status`` the AO carries is reported and never applied;
- **the report** -- every section the install skill sends a reviewer to
  ``data/mml/VA-REPORT.md`` for, including the two things the emit lane had to
  decide quietly: the band a nominal widened and the nominals the model only
  seeds.

**The synthetic ring reproduces its own exported matrix entry by entry.** The
fixture measures its matrix about the *6D* closed orbit, which is the orbit the
served model runs on with its cavity, so a corrector's path-length change is
paid for the same way on both sides. Measured about the 4D orbit instead, every
monitor of a column would move by one extra constant -- three orders below the
response on a real ring, a quarter of it on a four-cell toy with a huge momentum
compaction. The lane that holds the two together therefore asks each entry to
come back within the export's own rounding floor, which is what says every
binding, calibration and row pairing is right.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli

# The command's own reading of a tree, so the structured assertions below and
# the report file on disk come from one run over one tree.
from osprey.cli.mml_cmd import _parse_mapping_file, _read_import, _va_export, _va_lane
from osprey.services.mml.emit.context import build_context
from osprey.services.mml.emit.va import emit_channel_limits, emit_machine
from osprey.services.mml.va.verify import (
    FLOOR_FRACTION,
    REPORT_FILENAME,
    TOLERANCE_FRACTION,
    Entry,
    VerifyReport,
    _polarity,
    model_channels,
    render_report,
    verify,
)
from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.manifest.classify import (
    PARTITION_PYAT_COUPLED,
    READBACK_SUBFIELD,
    SETPOINT_SUBFIELD,
)
from tests.cli.test_mml_map import _fill

pytest.importorskip("lume_pyat")
pytest.importorskip("linkml_runtime")

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"

#: The one committed 2.0 export: the only tree in this repository that carries
#: a lattice, per-family calibrations and a response matrix together.
SYNTHETIC = FIXTURES / "synthetic" / "quokka.sr.ao.json"

#: A 1.0 export, which carries none of them.
PAIRED = FIXTURES / "paired"


def _run(root: Path, *args: str):
    """Run one verb of the chain in ``root``, as a reviewer runs it."""
    here = Path.cwd()
    os.chdir(root)
    try:
        return CliRunner().invoke(cli, ["mml", *args], catch_exceptions=False)
    finally:
        os.chdir(here)


def _answer_mapping(root: Path) -> None:
    """Fill every slot of the skeleton the way the map tests fill one."""
    path = root / "data" / "mml" / "mapping.yaml"
    document = _fill(yaml.safe_load(path.read_text(encoding="utf-8")))
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")


@pytest.fixture(scope="module")
def emitted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A deployment repo with the synthetic export imported, answered and emitted.

    Module scoped: the chain writes the whole deployment, and every lane below
    reads the same tree rather than paying for it again.
    """
    root = tmp_path_factory.mktemp("va-verify")
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    assert _run(root, "import", str(SYNTHETIC)).exit_code == 0
    assert _run(root, "map", "--init").exit_code == 0
    _answer_mapping(root)
    result = _run(root, "emit")
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture(scope="module")
def inputs(emitted: Path) -> dict[str, Any]:
    """Everything :func:`verify` is called with, read off the emitted tree."""
    out_dir = emitted / "data" / "mml"
    ao, ad = _read_import(out_dir)
    mapping_path = out_dir / "mapping.yaml"
    mapping = _parse_mapping_file(mapping_path)
    export = _va_export(out_dir, ao, ad, mapping)
    assert export is not None and export.pending is not None
    lane = _va_lane(ao, mapping, export.pending)
    ctx = build_context(out_dir / "ao.json", mapping_path, ao)
    data = emitted / "data"
    bindings = load_bindings(data / "simulation" / "va_bindings.json")
    _machine, seeds = emit_machine(
        lane.verdicts, lane.views, lane.judged_va, mapping, ctx, lane.element_bindings
    )
    _limits, bands = emit_channel_limits(
        json.loads((data / "channel_limits.json").read_text(encoding="utf-8")),
        bindings.bindings,
        (),
        ctx,
        views=lane.views,
        system=lane.system,
    )
    return {
        "data": data,
        "system": lane.system,
        "response": json.loads((out_dir / "response.json").read_text(encoding="utf-8"))[
            lane.system
        ],
        "views": lane.views,
        "verdicts": lane.verdicts,
        "judged_va": lane.judged_va,
        "seeds": seeds,
        "bands": bands,
        "bindings": bindings,
        "monitors": lane.monitors,
    }


@pytest.fixture(scope="module")
def result(inputs: dict[str, Any]) -> VerifyReport:
    """The comparison itself, run once for every lane that reads it."""
    return _verify(inputs)


def _verify(inputs: dict[str, Any], response: dict | None = None) -> VerifyReport:
    return verify(
        inputs["data"],
        system=inputs["system"],
        response=inputs["response"] if response is None else response,
        views=inputs["views"],
        verdicts=inputs["verdicts"],
        judged_va=inputs["judged_va"],
        seeds=inputs["seeds"],
        bands=inputs["bands"],
        monitors=inputs["monitors"],
    )


def _block(result: VerifyReport, monitors: str, actuators: str):
    """One block of the comparison, named the way the export names it."""
    found = [
        block
        for block in result.blocks
        if block.monitor_family == monitors and block.actuator_family == actuators
    ]
    assert len(found) == 1, f"{monitors} <- {actuators} is not one block of {result.blocks}"
    return found[0]


def _entry(file_value: float, model_value: float, floor: float) -> Entry:
    """One entry banded the way the comparison bands it."""
    return Entry(
        monitor_address="QK:BPMx:1:CUR:RB",
        actuator_address="QK:HC:1:CUR:SP",
        monitor_device=1,
        actuator_device=1,
        file_value=file_value,
        model_value=model_value,
        tolerance=TOLERANCE_FRACTION * max(abs(file_value), floor),
        floor=floor,
    )


def _compared(report: VerifyReport) -> set[tuple[str, str, float, float]]:
    """Every entry the comparison made, as what it paired and what it found."""
    return {
        (entry.monitor_address, entry.actuator_address, entry.file_value, entry.model_value)
        for block in report.blocks
        for entry in block.entries
    }


class TestCriterion:
    """``|R_model - R_file| <= 0.05 * max(|R_file|, 0.1 * rms(matrix))``."""

    def test_an_entry_inside_its_band_passes(self) -> None:
        entry = _entry(file_value=1.0, model_value=1.04, floor=0.0)

        assert entry.tolerance == pytest.approx(0.05)
        assert entry.passed

    def test_an_entry_outside_its_band_fails(self) -> None:
        assert not _entry(file_value=1.0, model_value=1.06, floor=0.0).passed

    def test_the_floor_bands_a_near_zero_entry_by_the_matrixs_scale(self) -> None:
        """A cross-plane zero is asked to stay small, not to be exact.

        Without the floor its tolerance would be zero and the last bit of a
        solve would fail it; with the floor it is held to a share of what the
        whole exported matrix is worth.
        """
        entry = _entry(file_value=0.0, model_value=0.004, floor=0.1)

        assert entry.tolerance == pytest.approx(0.005)
        assert entry.passed
        assert not _entry(file_value=0.0, model_value=0.006, floor=0.1).passed

    def test_the_sign_is_asserted_only_above_the_floor(self) -> None:
        below = _entry(file_value=0.05, model_value=-0.05, floor=0.1)
        above = _entry(file_value=0.5, model_value=-0.5, floor=0.1)

        assert not below.above_floor
        assert below.sign_agrees is None
        assert above.sign_agrees is False
        assert _entry(file_value=0.5, model_value=0.5, floor=0.1).sign_agrees is True

    def test_every_compared_entry_was_given_the_band_the_formula_states(
        self, result: VerifyReport
    ) -> None:
        """The floor is the whole matrix's rms, recomputed here from the file alone.

        Every entry of every block, judged or not, is weighed against the one
        number: an entry's band says how far the model may sit from the file,
        and which block the entry happens to belong to is no part of that.
        """
        assert result.compared
        compared = [entry for block in result.blocks for entry in block.entries]
        rms = math.sqrt(sum(entry.file_value**2 for entry in compared) / len(compared))

        for entry in compared:
            assert entry.floor == pytest.approx(FLOOR_FRACTION * rms)
            assert entry.tolerance == pytest.approx(
                TOLERANCE_FRACTION * max(abs(entry.file_value), entry.floor)
            )

    def test_a_column_the_file_states_as_zero_is_banded_by_the_rest_of_the_matrix(
        self, inputs: dict[str, Any]
    ) -> None:
        """The case the per-column floor got wrong, on a real comparison.

        A model-derived export states its cross-plane blocks as exactly zero.
        Weighed against its own column that block has no scale at all, so the
        band is zero wide and the model's last solved bit -- 1e-13 of a metre
        per radian -- is a miss on every entry of it. Weighed against the
        matrix it belongs to, the same entry is asked only to stay small,
        which is what a column of zeros says.
        """
        response = copy.deepcopy(inputs["response"])
        zeroed = next(
            body
            for body in response["blocks"]
            if body["monitor"]["family"] == "BPMx" and body["actuator"]["family"] == "VC"
        )
        zeroed["data"] = [[0.0] * len(row) for row in zeroed["data"]]

        report = _verify(inputs, response)
        block = _block(report, "BPMx", "VC")

        assert block.entries
        assert {entry.file_value for entry in block.entries} == {0.0}
        floor = block.entries[0].floor
        assert floor > 0.0
        assert floor == pytest.approx(
            FLOOR_FRACTION
            * math.sqrt(
                sum(entry.file_value**2 for other in report.blocks for entry in other.entries)
                / sum(other.compared for other in report.blocks)
            )
        )
        assert _entry(file_value=0.0, model_value=1e-13, floor=floor).passed
        assert not _entry(file_value=0.0, model_value=1e-13, floor=0.0).passed

    def test_an_entry_against_a_zero_band_is_ranked_rather_than_divided_by(self) -> None:
        """A file stating zero everywhere it was compared gives the matrix no scale.

        Its floor is zero and so is every band in it, and the outlier tables
        rank by how many times its band an entry missed by -- so the one place
        the criterion has no denominator is the one the report reads first.
        """
        assert _entry(file_value=0.0, model_value=1e-12, floor=0.0).excess == math.inf
        assert _entry(file_value=0.0, model_value=0.0, floor=0.0).excess == 0.0
        assert _entry(file_value=1.0, model_value=1.1, floor=0.0).excess == pytest.approx(2.0)

    def test_the_outlier_table_lists_an_entry_against_a_zero_band(
        self, result: VerifyReport
    ) -> None:
        first, *rest = result.blocks

        text = render_report(
            replace(result, blocks=(replace(first, entries=(_entry(0.0, 1e-09, 0.0),)), *rest)),
            provenance="",
        )

        assert f"{first.monitor_family} outliers against {first.actuator_family}" in text


class TestInPlaneJudging:
    """A block carries a verdict where its two sides work in one plane.

    The export pairs every monitor family with every corrector family, so half
    the blocks of a two-plane machine hold one plane's monitors against the
    other plane's correctors. What sits there is whatever the deck couples the
    planes by -- exactly zero in a model-derived file, the measurement's own
    noise in a measured one -- and neither is a statement about the bindings,
    the calibrations or the device order that this comparison exists to check.
    So those blocks are compared and printed like any other and pool into
    nothing.

    Which plane each side works in is read off the emitted bindings: a monitor
    states the axis it reads, a kick states the component of its attribute it
    writes. Never off the family's name -- a facility calls its horizontal
    correctors whatever it likes, and ``tests/va/test_facility_seam.py`` is the
    standing gate on that habit.
    """

    def test_the_blocks_that_pair_one_plane_with_itself_are_judged(
        self, result: VerifyReport
    ) -> None:
        for monitors, actuators in (("BPMx", "HC"), ("BPMy", "VC")):
            block = _block(result, monitors, actuators)
            assert block.compared
            assert block.judged
            assert block.unjudged is None

    def test_a_cross_plane_block_is_compared_and_reported_rather_than_judged(
        self, result: VerifyReport
    ) -> None:
        for monitors, actuators in (("BPMx", "VC"), ("BPMy", "HC")):
            block = _block(result, monitors, actuators)
            assert block.compared, f"{monitors} <- {actuators} was not compared"
            assert not block.judged
            assert block.unjudged == "cross-plane"

    def test_the_totals_pool_the_judged_blocks_alone(self, result: VerifyReport) -> None:
        """The verdict counts what the criterion means something about."""
        judged = [_block(result, "BPMx", "HC"), _block(result, "BPMy", "VC")]

        assert result.judged == tuple(judged)
        assert result.compared == sum(block.compared for block in judged)
        assert result.passed == sum(block.passed for block in judged)
        assert result.signed == (
            sum(block.signed[0] for block in judged),
            sum(block.signed[1] for block in judged),
        )
        assert result.compared < sum(block.compared for block in result.blocks)

    def test_the_plane_is_read_off_the_bindings_and_not_off_the_family_name(
        self, inputs: dict[str, Any], result: VerifyReport
    ) -> None:
        """The pairing the judgment made, re-derived from the emitted file.

        A monitor binding's ``attribute`` is the axis it reads and a kick's
        ``index`` is the component of ``KickAngle`` it writes, which are the
        two numbers the rule is allowed to consult.
        """
        axes = {
            binding.family: binding.attribute
            for binding in inputs["bindings"].bindings
            if binding.kind == "monitor"
        }
        kicks = {
            binding.family: ("x", "y")[binding.index]
            for binding in inputs["bindings"].bindings
            if binding.kind == "kick"
        }

        assert axes == {"BPMx": "x", "BPMy": "y"}
        assert kicks == {"HC": "x", "VC": "y"}
        for block in result.blocks:
            assert block.monitor_plane == axes[block.monitor_family]
            assert block.actuator_plane == kicks[block.actuator_family]
            assert block.judged == (block.monitor_plane == block.actuator_plane)

    def test_a_family_the_bindings_place_in_no_plane_is_reported_by_name(
        self, inputs: dict[str, Any]
    ) -> None:
        """A block this cannot place is refused, never judged on one side's plane.

        The file names the families its matrix was measured against, and a
        family that drives no corrector has no plane to be judged in -- so the
        block says which family it could not place rather than borrowing the
        monitor's plane and passing itself off as in-plane.
        """
        response = copy.deepcopy(inputs["response"])
        for body in response["blocks"]:
            if body["actuator"]["family"] == "HC":
                body["actuator"]["family"] = "QF"

        block = _block(_verify(inputs, response), "BPMx", "QF")

        assert not block.judged
        assert block.unjudged == "the bindings place 'QF' in no plane"
        assert block.actuator_plane == ""

    def test_a_family_the_bindings_place_in_two_planes_is_reported_by_name(
        self, inputs: dict[str, Any]
    ) -> None:
        """One family, two planes: nothing to hold either side of a block to."""
        from osprey.services.mml.va.verify import _family_plane

        kicks = [binding for binding in inputs["bindings"].bindings if binding.family == "HC"]
        assert len(kicks) > 1

        plane, reason = _family_plane("HC", {0: kicks[0], 1: replace(kicks[1], index=1)})

        assert plane == ""
        assert reason == "the bindings place 'HC' in x and y"
        assert _family_plane("HC", {0: kicks[0]}) == ("x", None)


class TestPolarityOutliers:
    """A corrector the two sides disagree about the direction of is named.

    Inside a judged block the sign is the one thing a lattice cannot argue
    with: a corrector pushes the beam one way or the other. A scatter of
    flipped entries is the comparison finding something in the ring, but a
    column flipped -- above the floor, on more than half of every entry of it
    that was compared -- at the size the rest of the matrix is worth, is the
    file and the model disagreeing about that device --
    a cable, a convention, a sign in a database somebody typed in 2006. Nothing
    in a deck reproduces it and no tolerance should absorb it, so the column is
    reported on its own and left out of what the block counts, and the bar the
    block is held to is then about the correctors whose direction both sides
    agree on.
    """

    @staticmethod
    def _flipped(
        response: dict, monitors: str, actuators: str, *, column: int, rows: set[int] | None = None
    ) -> dict:
        """The same document with one column's sign turned over in the file."""
        block = next(
            body
            for body in response["blocks"]
            if body["monitor"]["family"] == monitors and body["actuator"]["family"] == actuators
        )
        for index, row in enumerate(block["data"]):
            if rows is None or index in rows:
                row[column] = -row[column]
        return response

    def test_the_export_this_runs_on_has_no_column_running_backwards(
        self, result: VerifyReport
    ) -> None:
        """The baseline the lanes below move away from."""
        assert result.polarity == ()
        for block in result.blocks:
            assert block.polarity == ()
            assert block.counted == block.entries

    def test_a_column_flipped_over_its_whole_length_is_named(self, inputs: dict[str, Any]) -> None:
        response = self._flipped(copy.deepcopy(inputs["response"]), "BPMx", "HC", column=0)

        block = _block(_verify(inputs, response), "BPMx", "HC")

        assert len(block.polarity) == 1
        (flipped,) = block.polarity
        assert flipped.family == "HC"
        assert flipped.address == "QK:HC:1:CUR:SP"
        assert flipped.device == 1
        assert flipped.checked and flipped.flipped == flipped.checked
        assert flipped.median_ratio == pytest.approx(1.0, abs=0.01)

    def test_the_named_column_is_left_out_of_what_the_block_and_the_verdict_count(
        self, inputs: dict[str, Any], result: VerifyReport
    ) -> None:
        """The row keeps the block's own numbers; the counts drop the column."""
        response = self._flipped(copy.deepcopy(inputs["response"]), "BPMx", "HC", column=0)
        before = _block(result, "BPMx", "HC")
        column = sum(1 for entry in before.entries if entry.actuator_device == 1)

        report = _verify(inputs, response)
        block = _block(report, "BPMx", "HC")

        assert column
        assert block.compared == before.compared
        assert {entry.actuator_device for entry in block.counted} == {2, 3, 4}
        assert block.counts.compared == block.compared - column
        assert block.counts.signed[1] == block.counts.signed[0]
        assert block.signed[1] < block.signed[0]
        assert report.compared == block.counts.compared + _block(report, "BPMy", "VC").compared
        assert report.signed[1] == report.signed[0]

    def test_a_column_flipped_on_fewer_than_half_its_entries_is_not_an_outlier(
        self, inputs: dict[str, Any]
    ) -> None:
        """One entry of a column is the comparison working, not a reversed device."""
        response = self._flipped(
            copy.deepcopy(inputs["response"]), "BPMx", "HC", column=1, rows={0}
        )

        block = _block(_verify(inputs, response), "BPMx", "HC")

        assert block.signed[1] < block.signed[0], "the flip did not reach the comparison"
        assert block.polarity == ()
        assert block.counted == block.entries

    @staticmethod
    def _column(signs: Sequence[int], *, floor: float = 1.0) -> list[Entry]:
        """One corrector's column: ``+1`` agrees, ``-1`` is flipped, ``0`` is small."""
        entries = []
        for index, sign in enumerate(signs):
            size = 10.0 if sign else 0.1 * floor
            entries.append(
                Entry(
                    monitor_device=index + 1,
                    monitor_address=f"BPM:{index + 1}",
                    actuator_device=7,
                    actuator_address="COR:7",
                    file_value=size,
                    model_value=size if sign >= 0 else -size,
                    tolerance=0.05 * max(size, floor),
                    floor=floor,
                )
            )
        return entries

    def test_one_flipped_entry_among_small_ones_does_not_retire_the_column(self) -> None:
        """The column is mostly too small to have a sign, so it is no polarity fact.

        Held against the entries above the floor alone, this column reads one
        of one and takes its nine small entries out of the block with it. The
        rule counts against the whole column, so it stays in.
        """
        assert _polarity(self._column([-1] + [0] * 9), "HC") == ()

    def test_a_column_mostly_sizable_and_mostly_flipped_is_named(self) -> None:
        """What a reversed cable looks like: 56 of a column of 57, as SPEAR3's are."""
        (column,) = _polarity(self._column([-1] * 56 + [0]), "HC")

        assert (column.flipped, column.checked) == (56, 57)
        assert column.median_ratio == pytest.approx(1.0)

    def test_a_column_flipped_above_the_floor_but_not_over_half_of_itself_is_kept(self) -> None:
        """Half the column sizable and every sizable entry flipped is still a tie."""
        assert _polarity(self._column([-1] * 28 + [0] * 29), "HC") == ()

    def test_a_cross_plane_column_has_no_polarity_to_be_wrong_about(
        self, inputs: dict[str, Any]
    ) -> None:
        """A block nothing is judged on states no polarity outlier either.

        What sits in a cross-plane block is the deck's own coupling, and a sign
        there is not a claim about which way a corrector pushes the beam.
        """
        response = self._flipped(copy.deepcopy(inputs["response"]), "BPMx", "VC", column=0)

        block = _block(_verify(inputs, response), "BPMx", "VC")

        assert not block.judged
        assert block.polarity == ()

    def test_the_report_names_the_column_and_restates_the_block_without_it(
        self, inputs: dict[str, Any]
    ) -> None:
        """The page carries the channel to go and look at, and the bar without it."""
        response = self._flipped(copy.deepcopy(inputs["response"]), "BPMx", "HC", column=0)
        report = _verify(inputs, response)
        block = _block(report, "BPMx", "HC")
        (flipped,) = block.polarity

        text = render_report(report, provenance="the synthetic export")

        assert "## Polarity outliers" in text
        assert f"| BPMx ← HC | HC | QK:HC:1:CUR:SP | {flipped.flipped}/{flipped.checked} |" in text
        assert "| yes, less 1 polarity column |" in text
        assert "Left out, the judged blocks stand at:" in text
        assert f"| BPMx | HC | {block.counts.compared} |" in text
        assert "are named under Polarity outliers" in text

    def test_the_report_says_when_every_corrector_agrees(self, result: VerifyReport) -> None:
        text = render_report(result, provenance="the synthetic export")

        assert "Every corrector moves the beam the way the file says it did." in text
        assert "Left out, the judged blocks stand at:" not in text


class TestAlignment:
    """Rows are paired by their ``DeviceList`` key, never by position."""

    def test_an_entry_names_the_addresses_of_the_two_devices_it_pairs(
        self, result: VerifyReport
    ) -> None:
        for entry in _block(result, "BPMx", "HC").entries:
            assert entry.monitor_address == f"QK:BPMx:{entry.monitor_device}:CUR:RB"
            assert entry.actuator_address == f"QK:HC:{entry.actuator_device}:CUR:SP"

    def test_every_device_of_both_families_is_compared(self, result: VerifyReport) -> None:
        block = _block(result, "BPMx", "HC")

        assert {entry.monitor_device for entry in block.entries} == {1, 2, 3, 4}
        assert {entry.actuator_device for entry in block.entries} == {1, 2, 3, 4}

    def test_a_row_no_judged_device_sits_at_is_reported_rather_than_compared(
        self, inputs: dict[str, Any]
    ) -> None:
        """A row the judged family does not carry is a gap, not a shift.

        Pairing the two lists positionally would quietly move the rest of the
        column up one device and still pass on a periodic ring.
        """
        response = json.loads(json.dumps(inputs["response"]))
        response["blocks"][0]["monitor"]["device_list"][1] = [9, 9]

        block = _block(_verify(inputs, response), "BPMx", "HC")

        assert {entry.monitor_device for entry in block.entries} == {1, 3, 4}
        assert [row.row for row in block.dropped if row.side == "monitor"] == ["[9, 9]"]
        assert "no device of the judged BPMx" in block.dropped[0].reason

    def test_permuting_the_files_own_row_order_changes_no_entry(
        self, inputs: dict[str, Any], result: VerifyReport
    ) -> None:
        """The file may list its rows in any order; the pairing is by key.

        Every block's monitor ``DeviceList``, its ``Status`` and the matrix
        rows that belong to them are permuted together by one fixed
        permutation. Nothing about the measurement changed, so nothing about
        the comparison may: an implementation that indexed either list by
        position would pair a different monitor with each row and land on a
        different set of entries.
        """
        order = [3, 1, 0, 2]
        response = json.loads(json.dumps(inputs["response"]))
        for block in response["blocks"]:
            monitor = block["monitor"]
            assert len(monitor["device_list"]) == len(order)
            monitor["device_list"] = [monitor["device_list"][index] for index in order]
            monitor["status"] = [monitor["status"][index] for index in order]
            block["data"] = [block["data"][index] for index in order]

        permuted = _verify(inputs, response)

        assert _compared(permuted)
        assert _compared(permuted) == _compared(result)

    def test_a_matrix_that_is_not_the_shape_of_its_two_device_lists_is_refused(
        self, inputs: dict[str, Any]
    ) -> None:
        """A matrix written against another pair of lists places no entry.

        An entry is read at ``data[monitor row][actuator row]``, and an index
        past the data states no number -- so a block a column short would
        compare the part that overlaps and report the rest as rows the
        bindings never reached, which is not what happened to them.
        """
        response = copy.deepcopy(inputs["response"])
        response["blocks"][0]["data"] = [row[:-1] for row in response["blocks"][0]["data"]]

        report = _verify(inputs, response)
        refused = _block(report, "BPMx", "HC")

        assert refused.compared == 0
        assert refused.dropped
        for row in refused.dropped:
            assert "a 4 by 3 matrix" in row.reason
            assert "4 devices of 'BPMx'" in row.reason
            assert "4 of 'HC'" in row.reason
        assert _block(report, "BPMy", "HC").compared

    def test_a_corrector_two_devices_share_is_dropped_by_name_on_both_rows(
        self, inputs: dict[str, Any]
    ) -> None:
        """A supply feeding magnets in series measures nothing one column says.

        The model moves the whole string with that one knob and the response
        file's column is one magnet trimmed on its own, so the two are not the
        same measurement. Taking the first magnet would compare them anyway
        and say nothing, while the rest of the string reported that the
        bindings drive nothing there -- which is not what happened to it.
        """
        views = copy.deepcopy(inputs["views"])
        field = next(view for view in views if view.raw_name == "HC").fields["Setpoint"]
        key = field.keys[0]
        slots = list(field.slots(key))
        slots[1] = slots[0]
        field.body[key] = slots

        report = verify(
            inputs["data"],
            system=inputs["system"],
            response=inputs["response"],
            views=views,
            verdicts=inputs["verdicts"],
            judged_va=inputs["judged_va"],
            seeds=inputs["seeds"],
            bands=inputs["bands"],
        )
        block = _block(report, "BPMx", "HC")

        assert {entry.actuator_device for entry in block.entries} == {3, 4}
        reasons = [row.reason for row in block.dropped if row.side == "actuator"]
        assert len(reasons) == 2
        for reason in reasons:
            assert "QK:HC:1:CUR:SP feeds this magnet in series with 1 other of 'HC'" in reason
            assert "the model moves the whole string at once" in reason
            assert "the file measured one magnet" in reason

    def test_a_block_with_no_monitor_row_and_no_matrix_is_named_rather_than_passed(
        self, inputs: dict[str, Any]
    ) -> None:
        """Zero rows against zero devices agree, and say nothing happened.

        Such a block compares nothing and drops no monitor row, so a reader is
        left to work out from an empty comparison that there was never
        anything in it. The actuator rows carry the reason instead.
        """
        response = copy.deepcopy(inputs["response"])
        response["blocks"][0]["monitor"]["device_list"] = []
        response["blocks"][0]["monitor"]["status"] = []
        response["blocks"][0]["data"] = []

        block = _block(_verify(inputs, response), "BPMx", "HC")

        assert block.compared == 0
        assert block.dropped
        for row in block.dropped:
            assert "the block states no matrix" in row.reason
            assert "0 devices of 'BPMx'" in row.reason


class TestStatusFilter:
    """The response file's ``Status`` is applied; the AO's is reported."""

    def test_a_row_the_response_file_marks_status_0_is_never_compared(
        self, result: VerifyReport
    ) -> None:
        """The synthetic export's third vertical monitor was out of the measurement."""
        for monitors in ("BPMy",):
            for actuators in ("HC", "VC"):
                block = _block(result, monitors, actuators)
                assert 3 not in {entry.monitor_device for entry in block.entries}

    def test_the_dropped_row_says_which_status_dropped_it(self, result: VerifyReport) -> None:
        dropped = [row for block in result.blocks for row in block.dropped]

        assert dropped
        for row in dropped:
            assert row.reason == "the response file marks the row Status 0"
            assert row.row == "[3, 1]"
            assert row.family == "BPMy"

    def test_the_export_this_runs_on_marks_no_device_out_of_service(
        self, result: VerifyReport
    ) -> None:
        """The baseline the lane below moves away from."""
        assert result.ao_status_off == ()

    @pytest.fixture(scope="class")
    def ao_off(self, inputs: dict[str, Any]) -> VerifyReport:
        """The same comparison, with the AO marking one corrector out of service."""
        views = copy.deepcopy(inputs["views"])
        for view in views:
            if view.raw_name == "HC":
                view.body["Status"] = [[1], [0], [1], [1]]
        return verify(
            inputs["data"],
            system=inputs["system"],
            response=inputs["response"],
            views=views,
            verdicts=inputs["verdicts"],
            judged_va=inputs["judged_va"],
            seeds=inputs["seeds"],
            bands=inputs["bands"],
        )

    def test_a_device_the_ao_marks_off_is_reported_and_still_compared(
        self, ao_off: VerifyReport
    ) -> None:
        """The AO says what the control system thinks today; the matrix is older.

        Applying it would silently shrink an exported matrix by whatever a
        facility happened to have switched off on the morning of the install.
        """
        assert ao_off.ao_status_off == ("HC: 1 of 4 devices",)
        block = _block(ao_off, "BPMx", "HC")
        assert {entry.actuator_device for entry in block.entries} == {1, 2, 3, 4}
        for row in block.dropped:
            assert "Status 0" in row.reason

    def test_the_report_says_the_ao_status_was_read_and_never_applied(
        self, ao_off: VerifyReport
    ) -> None:
        """The one place the report tells an operator the device is still in.

        Without it the ``HC: 1 of 4 devices`` line reads as a count of what was
        excluded, which is the opposite of what happened.
        """
        text = render_report(ao_off, provenance="the synthetic export")

        assert "read for information and never applied" in text
        assert "- HC: 1 of 4 devices" in text


class TestServedNamespace:
    """An emitted tree carries no manifest, so the bindings are the channel set."""

    def test_every_bound_address_is_served_in_the_partition_the_model_drives(
        self, inputs: dict[str, Any]
    ) -> None:
        channels = model_channels(inputs["bindings"])
        subfields = {channel["address"]: channel["subfield"] for channel in channels}

        assert {channel["partition"] for channel in channels} == {PARTITION_PYAT_COUPLED}
        for binding in inputs["bindings"].bindings:
            if binding.kind == "monitor":
                assert subfields[binding.setpoint_address] in {"X", "Y"}
                continue
            assert subfields[binding.setpoint_address] == SETPOINT_SUBFIELD
            if binding.readback_address:
                assert subfields[binding.readback_address] == READBACK_SUBFIELD

    def test_an_address_is_named_once(self, inputs: dict[str, Any]) -> None:
        channels = model_channels(inputs["bindings"])

        assert len({channel["address"] for channel in channels}) == len(channels)


class TestTheModelAgainstTheExport:
    """What the comparison found, and what shape the difference has."""

    def test_the_model_reproduces_the_vertical_response_of_the_vertical_correctors(
        self, result: VerifyReport
    ) -> None:
        block = _block(result, "BPMy", "VC")

        assert block.compared
        assert block.pass_ratio == 1.0
        assert block.median_ratio == pytest.approx(1.0, abs=0.01)

    def test_every_column_reproduces_the_export_entry_by_entry(self, result: VerifyReport) -> None:
        """The file and the model are one deck measured about one orbit.

        A wrong calibration scales a column, a wrong element moves one entry of
        it, a mispaired row swaps two, and an orbit solved about a different
        point adds one constant to every entry alike. None of those survives
        here, so what the residual measures is the chain between the export and
        the served model and nothing else.

        The bound is the export's own rounding floor rather than a tolerance:
        the matrix is written to six significant digits, so two sides that
        agree exactly still differ in the last one. Anything looser stops
        catching the faults above in every block.
        """
        for block in result.blocks:
            columns: dict[int, list[Entry]] = {}
            for entry in block.entries:
                columns.setdefault(entry.actuator_device, []).append(entry)
            for device, entries in columns.items():
                residuals = [entry.model_value - entry.file_value for entry in entries]
                scale = math.sqrt(sum(entry.file_value**2 for entry in entries) / len(entries))
                assert max(abs(residual) for residual in residuals) <= 1e-4 * scale, (
                    f"{block.monitor_family} <- {block.actuator_family} device {device} "
                    f"residuals {residuals} are not the column's own numbers"
                )

    def test_the_sign_agrees_wherever_the_energy_term_does_not_dominate(
        self, result: VerifyReport
    ) -> None:
        checked, agreed = _block(result, "BPMy", "VC").signed

        assert checked
        assert agreed == checked


class TestTheReport:
    """``VA-REPORT.md``: what a reviewer reads before ``osprey build``."""

    @pytest.fixture(scope="class")
    def text(self, emitted: Path) -> str:
        result = _run(emitted, "verify")
        assert result.exit_code == 0, result.output
        return (emitted / "data" / "mml" / REPORT_FILENAME).read_text(encoding="utf-8")

    def test_the_command_writes_it_where_the_install_skill_sends_the_reviewer(
        self, emitted: Path, text: str
    ) -> None:
        assert (emitted / "data" / "mml" / "VA-REPORT.md").is_file()
        assert text.startswith("# Virtual accelerator — Quokka SR")

    def test_it_carries_every_section(self, text: str) -> None:
        for heading in (
            "## Verdict",
            "## Export",
            "## Orbit response",
            "## Polarity outliers",
            "## Rows not compared",
            "## Widened bands",
            "## Trimmed conversions",
            "## Monitors served as markers",
            "## Nominals the model does not maintain",
        ):
            assert heading in text, heading

    def test_it_says_when_every_conversion_runs_one_way(self, text: str) -> None:
        assert "Every sampled conversion converts one way over its whole grid." in text

    def test_it_says_when_every_monitor_on_the_deck_addresses_its_own_reading(
        self, text: str
    ) -> None:
        assert "Every monitor-type element on the deck is read or uniquely named." in text

    def test_it_names_each_repeated_monitor_it_served_as_a_marker(
        self, result: VerifyReport
    ) -> None:
        """A reviewer reads which positions of the deck stopped reading."""
        from osprey.services.mml.va.elements import ServedMarker

        text = render_report(
            replace(
                result,
                markers=(
                    ServedMarker(name="GE", elements=180),
                    ServedMarker(name="GS", elements=180),
                ),
            ),
            provenance="the synthetic export",
        )

        assert "| GE | 180 |" in text
        assert "| GS | 180 |" in text
        assert "no family reads them" in text
        assert "reads no beam position anywhere" not in text

    def test_it_says_when_the_conversion_leaves_the_system_no_monitor(
        self, result: VerifyReport
    ) -> None:
        """A model with no monitor left measures no orbit and publishes none.

        The counts do not say it: reading it off them means knowing how many
        monitor-type elements the deck held to begin with.
        """
        from osprey.services.mml.va.elements import ServedMarker

        text = render_report(
            replace(
                result,
                markers=(ServedMarker(name="BPM", elements=7),),
                monitors=0,
            ),
            provenance="the synthetic export",
        )

        assert (
            f"The conversion leaves {result.system} with no monitor-type element at all, "
            "so the served system reads no beam position anywhere" in text
        )

    def test_it_names_each_conversion_that_kept_one_stretch_of_itself(
        self, result: VerifyReport
    ) -> None:
        """A reviewer reads the span beside the operating point it was kept for."""
        from osprey.services.mml.emit.va import CalibrationTrim

        text = render_report(
            replace(
                result,
                trims=(
                    CalibrationTrim(
                        family="QF",
                        address="QK:QF:1:CUR:SP",
                        curve="monitor_inverse",
                        working=-1.97,
                        kept=(-14.4, 2.51),
                        dropped=1,
                    ),
                ),
            ),
            provenance="the synthetic export",
        )

        assert "| QK:QF:1:CUR:SP | QF | monitor_inverse | -14.4 to 2.51 | -1.97 | 1 |" in text
        assert "one reading would answer two hardware values" in text

    def test_it_says_when_a_judged_block_holds_nothing_above_the_floor(
        self, result: VerifyReport
    ) -> None:
        """A block under the whole file's floor passes the band for free.

        The floor is the matrix's, so a judged block an order of magnitude
        smaller than the rest of the file can hold no entry above it. Every
        entry is then inside a band that is the floor, the sign and the ratio
        state nothing, and the row says as much only as a ``0/0`` and a dash.
        """
        judged = next(block for block in result.blocks if block.judged)
        sunk = replace(
            judged,
            entries=tuple(
                replace(entry, floor=1.0e9, tolerance=0.05 * 1.0e9) for entry in judged.entries
            ),
        )
        report = replace(
            result, blocks=(sunk, *(block for block in result.blocks if block is not judged))
        )
        assert not sunk.counts.checked

        text = render_report(report, provenance="the synthetic export")

        assert (
            f"No entry of {sunk.monitor_family} ← {sunk.actuator_family} sits above the "
            "matrix floor, so the block is held to the band alone" in text
        )

    def test_it_says_nothing_of_the_kind_while_a_judged_block_has_a_sign(
        self, text: str, result: VerifyReport
    ) -> None:
        assert any(block.counts.checked for block in result.judged)
        assert "sits above the matrix floor, so the block is held to the band alone" not in text

    def test_it_says_where_a_cavity_the_deck_does_not_carry_came_from(
        self, result: VerifyReport
    ) -> None:
        """A served ring with an element the facility's deck lacks says so.

        The emit lane builds one onto a deck that holds no cavity, because a
        machine that holds its radio frequency moves in dispersion the way
        only a cavity makes a model move. A reader holding the two files side
        by side finds the extra element here rather than wondering at it.

        Both frequencies are printed to the hertz, because the gap between
        them is the last few figures of a nine-figure number and it is the
        whole reason the built one is preferred to the stated one.
        """
        from osprey.services.mml.va.verdicts import BuiltCavity

        text = render_report(
            replace(
                result,
                cavity=BuiltCavity(
                    family="RF",
                    nominal_hz=499680000.0,
                    harmonic=1320,
                    voltage=3_000_000.0,
                    frequency_hz=499680594.88,
                ),
            ),
            provenance="the synthetic export",
        )

        assert (
            "| built cavity | RF at 499680595 Hz on harmonic 1320, 3e+06 V; "
            "the export states 499680000 Hz |" in text
        )
        assert "The deck carries no cavity of its own" in text
        assert "for the RF family to drive" in text

    def test_the_cavity_row_says_when_no_voltage_was_answered(self, result: VerifyReport) -> None:
        """The reviewer's one open value, read off the same row."""
        from osprey.services.mml.va.verdicts import BuiltCavity

        text = render_report(
            replace(
                result,
                cavity=BuiltCavity(family="RF", nominal_hz=499680000.0, harmonic=1320),
            ),
            provenance="the synthetic export",
        )

        assert (
            "| built cavity | RF at harmonic 1320, not yet built against a deck, "
            "an unanswered voltage; the export states 499680000 Hz |" in text
        )

    def test_it_says_nothing_about_a_cavity_where_the_deck_carries_its_own(self, text: str) -> None:
        assert "no cavity of its own" not in text

    def test_it_states_where_the_matrix_came_from_and_at_which_energy(self, text: str) -> None:
        assert "| origin | model |" in text
        assert "| file energy | 2 GeV |" in text
        assert "| deck energy | 2 GeV |" in text
        assert "| energy at nominal | 2 GeV |" in text

    def test_it_states_the_criterion_beside_the_pass_ratio(self, text: str) -> None:
        assert "|R_model - R_file| <= 0.05 * max(|R_file|, 0.1 * rms(matrix))" in text
        assert "root mean square of every entry of the file that was compared" in text
        assert "Sign agrees on" in text

    def test_it_says_which_blocks_the_verdict_is_made_of(self, text: str) -> None:
        """The reader has to know the counts are not the whole table.

        The toy's four blocks are two in-plane and two cross-plane, and the
        Verdict pools the first two, so the page names the other two and says
        what they hold.
        """
        assert "The counts above are the judged blocks alone" in text
        assert "BPMx ← VC (cross-plane)" in text
        assert "BPMy ← HC (cross-plane)" in text

    def test_the_block_table_says_whether_each_block_carries_a_verdict(self, text: str) -> None:
        judged = [
            line for line in text.splitlines() if line.startswith("| BPM") and "←" not in line
        ]

        assert [line.rsplit("|", 2)[1].strip() for line in judged] == [
            "yes",
            "reported only, cross-plane",
            "reported only, cross-plane",
            "yes",
        ]

    def test_it_says_what_a_model_derived_matrix_can_prove(self, text: str) -> None:
        """The toy's matrix came off its own deck, so a pass is plumbing.

        Nothing else on the page separates that from a machine the model agrees
        with, and a reviewer reads this before ``osprey build``.
        """
        assert "| origin | model |" in text
        assert "computed from a model rather than measured on the machine" in text
        assert "is not the machine agreeing with the model" in text

    def test_it_names_the_file_the_matrix_was_read_from(self, text: str) -> None:
        """Origin says measured or model; only this row says whether a file answered.

        A facility may keep a computed matrix in a file, so the two questions
        come apart, and a page stating only the origin leaves a reader to guess
        which deck a model matrix came off.
        """
        assert "| read from | no file: the model was measured |" in text

    def test_it_lists_no_outlier_when_the_model_reproduces_the_file(self, text: str) -> None:
        """The toy's matrix is its own deck's, so there is nothing to list."""
        assert "outliers against" not in text

    def test_it_lists_an_outlier_with_both_addresses(self, inputs: dict[str, Any]) -> None:
        """An entry the model does not reproduce is named from both sides.

        The reviewer reads the report to find out which channel pair disagrees,
        so an entry alone is no use to them: the row carries the monitor it was
        read on and the corrector it was measured against.
        """
        response = copy.deepcopy(inputs["response"])
        block = next(
            body
            for body in response["blocks"]
            if body["monitor"]["family"] == "BPMx" and body["actuator"]["family"] == "HC"
        )
        block["data"][0][0] = block["data"][0][0] * 2.0 + 1.0

        text = render_report(_verify(inputs, response), provenance="")

        assert "BPMx outliers against HC" in text
        assert "| QK:BPMx:1:CUR:RB | QK:HC:1:CUR:SP |" in text

    def test_it_lists_the_band_a_nominal_widened(self, text: str) -> None:
        """The synthetic horizontal corrector's nominal sits outside its ``Range``."""
        assert "| QK:HC:1:CUR:SP | HC |" in text
        assert "-1 to 1.5" in text

    def test_it_lists_the_nominals_the_model_only_seeds_with_their_at_field(
        self, text: str
    ) -> None:
        assert "| BDM | Setpoint | BEND |" in text

    def test_it_lists_a_nominal_nothing_was_seeded_from(self, text: str) -> None:
        assert "non-finite nominal" in text
        assert "not hardware" in text


class TestThePerDeviceSweep:
    """A response file states one sweep width per corrector, and each is used."""

    @staticmethod
    def _swept(
        inputs: dict[str, Any], response: dict, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[VerifyReport, dict[str, list[float]]]:
        """Run the comparison, recording every width every address was driven by."""
        from osprey.services.mml.va import verify as module

        sweep = module.orbit_response
        seen: dict[str, list[float]] = {}

        def recording(model, binding, delta, *, monitors):
            seen.setdefault(binding.setpoint_address, []).append(delta)
            return sweep(model, binding, delta, monitors=monitors)

        monkeypatch.setattr(module, "orbit_response", recording)
        return _verify(inputs, response), seen

    @staticmethod
    def _stated(response: dict, family: str, delta: Any) -> dict:
        """The same document with one family's blocks stating another sweep width."""
        for block in response["blocks"]:
            if block["actuator"]["family"] == family:
                block["actuator_delta"] = delta
        return response

    @staticmethod
    def _driven(seen: dict[str, list[float]], family: str) -> dict[str, list[float]]:
        """What one family's addresses were swept by, out of everything recorded."""
        return {address: widths for address, widths in seen.items() if f":{family}:" in address}

    def test_a_scalar_sweeps_every_device_by_it(
        self, inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The width a file states as one number is the width of every column."""
        report, seen = self._swept(inputs, copy.deepcopy(inputs["response"]), monkeypatch)

        assert report.compared
        assert self._driven(seen, "HC") == {
            f"QK:HC:{device}:CUR:SP": [1e-05] for device in (1, 2, 3, 4)
        }

    def test_each_device_is_swept_by_its_own_width(
        self, inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A machine trims the kick per corrector, so a column is measured by its own.

        Reading the vector as one number -- its first entry, or its mean --
        would measure three of these four columns at a width the file never
        used, and on a real export the widths differ by ten per cent.
        """
        widths = [1e-05, 2e-05, 3e-05, 4e-05]
        response = self._stated(copy.deepcopy(inputs["response"]), "HC", widths)

        report, seen = self._swept(inputs, response, monkeypatch)

        assert self._driven(seen, "HC") == {
            f"QK:HC:{device}:CUR:SP": [width]
            for device, width in zip((1, 2, 3, 4), widths, strict=True)
        }
        assert self._driven(seen, "VC") == {
            f"QK:VC:{device}:CUR:SP": [1e-05] for device in (1, 2, 3, 4)
        }
        assert _block(report, "BPMx", "HC").compared == 16

    def test_a_width_per_device_changes_no_entry_of_a_linear_model(
        self, inputs: dict[str, Any], result: VerifyReport
    ) -> None:
        """The response is per unit of actuator, so the width divides back out.

        Which is what makes the comparison a statement about the bindings and
        the calibrations rather than about the width the export measured at.
        """
        response = self._stated(
            copy.deepcopy(inputs["response"]), "HC", [1e-05, 2e-05, 3e-05, 4e-05]
        )

        block = _block(_verify(inputs, response), "BPMx", "HC")

        assert block.pass_ratio == 1.0
        measured = {
            (entry.monitor_device, entry.actuator_device): entry.model_value
            for entry in _block(result, "BPMx", "HC").entries
        }
        for entry in block.entries:
            assert entry.model_value == pytest.approx(
                measured[(entry.monitor_device, entry.actuator_device)], rel=1e-3
            )

    def test_one_sweep_serves_both_planes_of_one_corrector_family(
        self, inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The BPMx and BPMy blocks of one family are two readings of one pass.

        Their arms are the same two lattice states, and a solve per block would
        pay for every corrector of the machine twice.
        """
        response = self._stated(
            copy.deepcopy(inputs["response"]), "HC", [1e-05, 2e-05, 3e-05, 4e-05]
        )

        report, seen = self._swept(inputs, response, monkeypatch)

        assert _block(report, "BPMx", "HC").compared
        assert _block(report, "BPMy", "HC").compared
        assert sorted(seen) == [
            f"QK:{kind}:{device}:CUR:SP" for kind in ("HC", "VC") for device in (1, 2, 3, 4)
        ]
        assert [len(widths) for widths in seen.values()] == [1] * 8

    def test_a_vector_of_another_length_refuses_the_block_naming_both(
        self, inputs: dict[str, Any]
    ) -> None:
        """A vector that does not run with the ``DeviceList`` pairs nothing.

        Sweeping the devices it does cover would hand the reviewer a comparison
        over an alignment the file never states.
        """
        response = self._stated(copy.deepcopy(inputs["response"]), "HC", [1e-05, 2e-05])

        report = _verify(inputs, response)
        block = _block(report, "BPMx", "HC")

        assert block.compared == 0
        assert all("2 actuator_delta values" in row.reason for row in block.dropped)
        assert all("4 devices of 'HC'" in row.reason for row in block.dropped)
        assert _block(report, "BPMx", "VC").compared

    def test_a_device_the_file_states_no_width_for_is_dropped_with_its_status(
        self, inputs: dict[str, Any]
    ) -> None:
        """The export fills the width of a device its file does not hold with a non-number.

        That is the file's own answer about one corrector, so the column is
        reported and the rest of the matrix is still compared.
        """
        response = self._stated(
            copy.deepcopy(inputs["response"]), "HC", [1e-05, None, 3e-05, 4e-05]
        )

        block = _block(_verify(inputs, response), "BPMx", "HC")

        assert {entry.actuator_device for entry in block.entries} == {1, 3, 4}
        dropped = [row for row in block.dropped if row.side == "actuator"]
        assert [row.row for row in dropped] == ["[2, 1]"]
        assert "no actuator_delta for this device" in dropped[0].reason
        assert "Status 1" in dropped[0].reason

    def test_a_width_stated_as_a_one_column_row_is_read_like_any_other(
        self, inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MATLAB writes a column vector as one-column rows, and it is a vector.

        The ``Status`` beside it is already read that way. A width read
        straight out of ``[1e-05]`` is not a number, every column of the block
        would state no finite width, and the block would be dropped whole --
        which is the failure the per-device sweep was written to fix, in a
        shape this module already knows facilities use.
        """
        widths = [1e-05, 2e-05, 3e-05, 4e-05]
        response = self._stated(
            copy.deepcopy(inputs["response"]), "HC", [[width] for width in widths]
        )

        report, seen = self._swept(inputs, response, monkeypatch)

        assert self._driven(seen, "HC") == {
            f"QK:HC:{device}:CUR:SP": [width]
            for device, width in zip((1, 2, 3, 4), widths, strict=True)
        }
        assert _block(report, "BPMx", "HC").compared == 16
        assert not [row for row in _block(report, "BPMx", "HC").dropped if row.side == "actuator"]

    def test_the_block_table_states_the_range_it_swept_with(
        self, inputs: dict[str, Any], result: VerifyReport
    ) -> None:
        """A reviewer reads the width beside the ratio it was measured at."""
        response = self._stated(
            copy.deepcopy(inputs["response"]), "HC", [1e-05, 2e-05, 3e-05, 4e-05]
        )

        assert "| 1e-05 |" in render_report(result, provenance="")
        assert "| 1e-05 to 4e-05 |" in render_report(_verify(inputs, response), provenance="")


class TestASweepThatLosesTheOrbit:
    """One corrector the solver refuses is one column, not the whole matrix."""

    def test_the_device_is_named_in_the_report_and_the_rest_is_still_compared(
        self, inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real ring loses its closed orbit on some arm of some corrector.

        Letting that propagate costs the reviewer every other column and hands
        them a traceback instead of a sentence, so the refusal is kept against
        the device it belongs to and the column is reported as a row that was
        not compared.
        """
        from osprey.services.mml.va import verify as module

        swept = module.orbit_response
        refused = "QK:HC:2:CUR:SP"

        def losing_the_orbit(model, binding, delta, *, monitors):
            if binding.setpoint_address == refused:
                raise OrbitSolveError("the +delta/2 arm has no stable closed orbit")
            return swept(model, binding, delta, monitors=monitors)

        monkeypatch.setattr(module, "orbit_response", losing_the_orbit)

        report = _verify(inputs)

        block = _block(report, "BPMx", "HC")
        assert {entry.actuator_device for entry in block.entries} == {1, 3, 4}
        reasons = [row.reason for row in block.dropped if row.side == "actuator"]
        assert len(reasons) == 1
        assert refused in reasons[0]
        assert "no stable closed orbit" in reasons[0]
        assert _block(report, "BPMx", "VC").compared

    def test_the_command_writes_the_report_rather_than_raising(
        self, tmp_path: Path, emitted: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The verb's whole product is the report; a refused column is in it."""
        from osprey.services.mml.va import verify as module

        swept = module.orbit_response

        def losing_the_orbit(model, binding, delta, *, monitors):
            if binding.setpoint_address == "QK:HC:2:CUR:SP":
                raise OrbitSolveError("the +delta/2 arm has no stable closed orbit")
            return swept(model, binding, delta, monitors=monitors)

        monkeypatch.setattr(module, "orbit_response", losing_the_orbit)
        root = tmp_path / "lost-orbit"
        shutil.copytree(emitted, root)

        result = _run(root, "verify")

        assert result.exit_code == 0, result.output
        text = (root / "data" / "mml" / REPORT_FILENAME).read_text(encoding="utf-8")
        assert "no stable closed orbit" in text
        assert "QK:HC:2:CUR:SP" in text


class TestProvenance:
    """Origin, energy and timestamp are per-block facts and are reported as such."""

    def test_the_blocks_are_reported_in_the_order_the_document_writes_them(
        self, inputs: dict[str, Any]
    ) -> None:
        """A refused block keeps its place, so report and export read alike.

        The second block is made undrivable by taking its sweep width away.
        Collecting refusals first would move it to the head of every table and
        of ``blocks[0]``, which the Export section reads.
        """
        response = json.loads(json.dumps(inputs["response"]))
        response["blocks"][1]["actuator_delta"] = None
        written = [
            (block["monitor"]["family"], block["actuator"]["family"])
            for block in response["blocks"]
        ]

        report = _verify(inputs, response)

        assert [(block.monitor_family, block.actuator_family) for block in report.blocks] == written
        assert report.blocks[1].compared == 0
        assert all("actuator_delta" in row.reason for row in report.blocks[1].dropped)

    def test_the_export_table_names_the_blocks_that_disagree(self, result: VerifyReport) -> None:
        """A measured block beside a model-derived one is two provenances.

        Reporting either one as the export's would assert the wrong energy,
        the wrong origin and the wrong measurement date over half the entries.
        """
        first, second, *rest = result.blocks
        text = render_report(
            replace(
                result,
                blocks=(
                    replace(first, origin="measured", gev=3.0, timestamp="2019-04-01T07:00:00"),
                    replace(second, origin="model", gev=2.0, timestamp="2026-09-17T09:00:00"),
                    *rest,
                ),
            ),
            provenance="the synthetic export",
        )

        first_name = f"{first.monitor_family} \u2190 {first.actuator_family}"
        second_name = f"{second.monitor_family} \u2190 {second.actuator_family}"
        assert f"| origin | measured ({first_name}); model (" in text
        assert f"| file energy | 3 GeV ({first_name}); 2 GeV (" in text
        assert f"2019-04-01T07:00:00 ({first_name})" in text
        assert second_name in text
        assert f"{first_name} was measured at 3 GeV and the deck is built for 2 GeV" in text

    def test_the_export_table_states_a_field_once_where_every_block_agrees(
        self, result: VerifyReport
    ) -> None:
        """The common case stays one plain row, with no block names in it."""
        text = render_report(result, provenance="the synthetic export")

        assert "| origin | model |" in text
        assert "| file energy | 2 GeV |" in text
        assert (
            "The matrix was measured at 2 GeV, which is the energy the deck is built for." in text
        )


class TestRefusals:
    """``verify`` is asked for evidence, so it never reports nothing and exits zero."""

    def test_a_tree_that_was_never_emitted_is_refused_by_name(
        self, tmp_path: Path, emitted: Path
    ) -> None:
        root = tmp_path / "unemitted"
        shutil.copytree(emitted, root)
        (root / "data" / "simulation" / "va_bindings.json").unlink()

        result = _run(root, "verify")

        assert result.exit_code != 0
        assert "data/simulation/va_bindings.json" in result.output
        assert "osprey mml emit" in result.output

    def test_a_1_0_tree_is_refused_naming_the_files_a_2_0_export_files(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "one-oh"
        root.mkdir()
        (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
        for name in ("quokka.ring.ao.json", "quokka.ring.ad.json"):
            shutil.copy(PAIRED / name, root / name)
        assert _run(root, "import", "quokka.ring.ao.json").exit_code == 0
        assert _run(root, "map", "--init").exit_code == 0
        _answer_mapping(root)

        result = _run(root, "verify")

        assert result.exit_code != 0
        assert "va.json" in result.output
        assert "response.json" in result.output
        assert "mml_export 2.0" in result.output
