"""``osprey mml verify``: the export's response matrix, re-measured on the model.

The whole verb runs here on the one committed 2.0 export, through the chain a
reviewer runs -- ``import``, ``map --init``, the answers, ``emit`` -- so every
assertion is made against a tree this repository can actually produce rather
than against a hand-written stand-in of one.

What the lanes pin:

- **the criterion** -- the per-entry band and its floor, both as arithmetic on
  a hand-built entry and as the number every entry of the real comparison was
  actually given;
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
    _machine, seeds = emit_machine(lane.verdicts, lane.views, lane.judged_va, mapping, ctx)
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
    """``|R_model - R_file| <= 0.05 * max(|R_file|, 0.1 * rms(column))``."""

    def test_an_entry_inside_its_band_passes(self) -> None:
        entry = _entry(file_value=1.0, model_value=1.04, floor=0.0)

        assert entry.tolerance == pytest.approx(0.05)
        assert entry.passed

    def test_an_entry_outside_its_band_fails(self) -> None:
        assert not _entry(file_value=1.0, model_value=1.06, floor=0.0).passed

    def test_the_floor_bands_a_near_zero_entry_by_its_columns_scale(self) -> None:
        """A cross-plane zero is asked to stay small, not to be exact.

        Without the floor its tolerance would be zero and the last bit of a
        solve would fail it; with the floor it is held to a share of what the
        rest of its column is worth.
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
        """The floor is the column's own rms, recomputed here from the file alone."""
        assert result.compared
        for block in result.blocks:
            columns: dict[int, list[Entry]] = {}
            for entry in block.entries:
                columns.setdefault(entry.actuator_device, []).append(entry)
            for entries in columns.values():
                values = [entry.file_value for entry in entries]
                rms = math.sqrt(sum(value * value for value in values) / len(values))
                for entry in entries:
                    assert entry.floor == pytest.approx(FLOOR_FRACTION * rms)
                    assert entry.tolerance == pytest.approx(
                        TOLERANCE_FRACTION * max(abs(entry.file_value), entry.floor)
                    )


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
            "## Rows not compared",
            "## Widened bands",
            "## Nominals the model does not maintain",
        ):
            assert heading in text, heading

    def test_it_states_where_the_matrix_came_from_and_at_which_energy(self, text: str) -> None:
        assert "| origin | model |" in text
        assert "| file energy | 2 GeV |" in text
        assert "| deck energy | 2 GeV |" in text
        assert "| energy at nominal | 2 GeV |" in text

    def test_it_states_the_criterion_beside_the_pass_ratio(self, text: str) -> None:
        assert "|R_model - R_file| <= 0.05 * max(|R_file|, 0.1 * rms(column))" in text
        assert "Sign agrees on" in text

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
