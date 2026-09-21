"""The shipped MATLAB Middle Layer exporter and its README.

No MATLAB runs here. The script is held to its written contract instead: it is
pullable into a deployment, it encodes with the options that keep non-finite
values out of ``null``, it names the five files and ``_export`` keys the
importer reads, it saves the model ring before anything samples the machine,
it refuses a family rather than the whole export, and a document in exactly
that dialect imports with no flags.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from tests.cli.test_scaffold_ci import named_commands, unresolvable
from tests.templates.mml_export_contract import (
    EXPORTER_VERSION,
    VA_CALIBRATION_KEYS,
    VA_ENERGY_TABLE_KEYS,
    VA_FAMILY_KEYS,
    VA_LATTICE_KEYS,
    VA_MONITOR_KEYS,
    VA_NOMINAL_KEYS,
    VA_READOUT_KEYS,
    VA_SETPOINT_KEYS,
    VA_VOCABULARIES,
)

from osprey.cli.main import cli
from osprey.services.mml.judgments import _VA_READOUT_ROW_KEYS
from osprey.services.mml.loaders.json_any import load_json
from osprey.services.mml.normalize import normalize_family
from osprey.services.mml.systems import resolve_system

REPO_ROOT = Path(__file__).resolve().parents[2]
MML_DIR = REPO_ROOT / "src" / "osprey" / "templates" / "apps" / "control_assistant" / "data" / "mml"
EXPORTER = MML_DIR / "mml_export.m"
README = MML_DIR / "README.md"
PAIRED_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mml" / "paired" / "quokka.ring.ao.json"

PULL_LINE = "osprey scaffold pull control-assistant:data/mml/mml_export.m"

#: Every Middle Layer call that reaches into the live machine or the model.
#: getpvmodel and measbpmresp mutate the ring to reach a solvable state, so the
#: lattice save has to precede all of them.
SAMPLING_CALLS = (
    "hw2physics(",
    "physics2hw(",
    "getpvmodel(",
    "getbpmresp(",
    "measbpmresp(",
    "bend2gev(",
)

#: The helpers the export may call before the lattice is saved: naming a file
#: and reading a string out of the Accelerator Data reach neither the machine
#: nor the model.
PURE_HELPERS = ("local_text", "local_field", "local_filename")


def _code(source: str) -> str:
    """The source with its comments stripped, so prose cannot stand in for code."""
    return "\n".join(re.sub(r"%.*$", "", line) for line in source.splitlines())


def _arguments(text: str) -> list[str]:
    """One argument list, split on the commas that separate arguments."""
    args: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char == "," and depth == 0:
            args.append(current.strip())
            current = ""
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        current += char
    args.append(current.strip())
    return args


def _sampling_calls(source: str, name: str) -> list[list[str]]:
    """The argument list of every call of one Middle Layer conversion function.

    The lookbehind keeps a helper whose own name ends in the function's name
    (``local_sample_hw2physics``) out of the result.
    """
    pattern = rf"(?<![A-Za-z0-9_]){name}\(([^;]*?)\);"
    return [_arguments(args) for args in re.findall(pattern, _code(source))]


def _function_body(source: str, name: str) -> str:
    """The lines of one local function, up to the next function definition.

    A ``for`` or ``try`` at the top level of a MATLAB function closes with its
    own ``end`` in column zero, so only the next ``function`` line is a
    reliable boundary.
    """
    opening = re.search(rf"^function\s+.*\b{name}\(", source, re.M)
    assert opening is not None, f"{name} is not defined"
    rest = source[opening.end() :]
    following = re.search(r"^function\b", rest, re.M)
    return rest[: following.start()] if following else rest


@pytest.fixture
def exporter_source() -> str:
    return EXPORTER.read_text(encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo: a ``profile.yml`` marker and nothing else."""
    root = tmp_path / "deployment"
    root.mkdir()
    (root / "profile.yml").write_text("", encoding="utf-8")
    return root


def _pull(repo: Path, target: str):
    return CliRunner().invoke(cli, ["scaffold", "pull", target, "--repo", str(repo)])


# ---------------------------------------------------------------------------
# Pulling
# ---------------------------------------------------------------------------


def test_pulling_the_directory_lands_both_files(repo: Path) -> None:
    result = _pull(repo, "control-assistant:data/mml")

    assert result.exit_code == 0, result.output
    for source in (EXPORTER, README):
        landed = repo / "data" / "mml" / source.name
        assert landed.is_file(), result.output
        assert landed.read_bytes() == source.read_bytes()


def test_pulling_the_script_alone_lands_only_the_script(repo: Path) -> None:
    result = _pull(repo, "control-assistant:data/mml/mml_export.m")

    assert result.exit_code == 0, result.output
    assert (repo / "data" / "mml" / "mml_export.m").read_bytes() == EXPORTER.read_bytes()
    assert not (repo / "data" / "mml" / "README.md").exists()


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def test_readme_names_the_pull_line_and_it_resolves() -> None:
    text = README.read_text(encoding="utf-8")

    assert PULL_LINE in text
    scaffold_chains = {chain for chain in named_commands(text) if chain[:1] == ("scaffold",)}
    assert scaffold_chains
    assert [unresolvable(chain) for chain in scaffold_chains if unresolvable(chain)] == []


def test_readme_states_the_one_command_usage_and_the_five_files() -> None:
    text = README.read_text(encoding="utf-8")

    assert "```matlab\nmml_export\n```" in text
    for name in ("lattice.mat", "ao.json", "ad.json", "va.json", "response.json"):
        assert f"<machine>.<submachine>.{name}" in text, name
    assert "osprey mml import" in text


def test_readme_states_what_the_export_needs_of_the_middle_layer() -> None:
    """A run without the simulator model loaded refuses; a reader should know why."""
    text = README.read_text(encoding="utf-8")

    assert "THERING" in text, "the model ring has to be loaded for the export to run"
    assert "before the export samples anything" in text, "the lattice is saved first"


# ---------------------------------------------------------------------------
# Script contract
# ---------------------------------------------------------------------------


def test_script_reads_the_live_ao_and_ad(exporter_source: str) -> None:
    assert re.search(r"^function\s+files\s*=\s*mml_export\(", exporter_source, re.M)
    assert re.search(r"\bAO\s*=\s*getao\b", exporter_source)
    assert re.search(r"\bAD\s*=\s*getad\b", exporter_source)


def test_every_jsonencode_call_keeps_non_finite_out_of_null(exporter_source: str) -> None:
    calls = re.findall(r"jsonencode\(([^;]*)\)", exporter_source)

    assert calls, "the script must encode with jsonencode"
    for call in calls:
        assert re.search(r"'ConvertInfAndNaN'\s*,\s*false", call), call


def test_a_body_that_did_not_encode_as_an_object_is_refused_by_name(
    exporter_source: str,
) -> None:
    """The splice drops the body's opening brace, so a non-object body is not JSON."""
    body = _code(_function_body(exporter_source, "local_document"))

    assert "mml_export:document" in body
    assert "encoded(1) ~= '{'" in body
    assert body.index("mml_export:document") < body.index("encoded(2:end)")


def test_script_applies_the_normalisation_rules(exporter_source: str) -> None:
    assert "func2str(" in exporter_source
    assert "deblank(" in exporter_source
    assert "'$fn'" in exporter_source
    for spelling in ("'Inf'", "'-Inf'", "'NaN'"):
        assert spelling in exporter_source
    assert "'Handles'" in exporter_source


def test_script_writes_the_five_export_file_names(exporter_source: str) -> None:
    for suffix in (".ao.json", ".ad.json", ".lattice.mat", ".va.json", ".response.json"):
        assert f"'{suffix}'" in exporter_source, suffix


def test_script_returns_the_five_paths(exporter_source: str) -> None:
    returned = re.search(r"files\s*=\s*\{([^}]*)\}", exporter_source)

    assert returned is not None
    assert [name.strip() for name in returned.group(1).split(",")] == [
        "aoFile",
        "adFile",
        "latticeFile",
        "vaFile",
        "responseFile",
    ]


def test_export_block_names_version_matlab_machine_submachine_timestamp(
    exporter_source: str,
) -> None:
    block = re.search(r"export\s*=\s*struct\((.*?)\);", exporter_source, re.S)

    assert block is not None
    keys = re.findall(r"'(\w+)'\s*,", block.group(1))
    assert keys == ["exporter", "matlab", "machine", "submachine", "timestamp"]
    assert '"_export"' in exporter_source


def test_the_shipped_exporter_announces_the_current_version(exporter_source: str) -> None:
    version = re.search(r"EXPORTER_VERSION\s*=\s*'([^']+)'", exporter_source)

    assert version is not None
    assert version.group(1) == EXPORTER_VERSION


def test_an_older_export_keeps_its_own_version_token() -> None:
    """A document is read at the version it was written with, not the current one."""
    fixture = json.loads(PAIRED_FIXTURE.read_text(encoding="utf-8"))

    assert fixture["_export"]["exporter"] == "mml_export 1.0.0"


# ---------------------------------------------------------------------------
# Order: the lattice is the first thing written
# ---------------------------------------------------------------------------


def test_the_lattice_is_saved_as_thering_in_a_readable_format(exporter_source: str) -> None:
    body = _function_body(exporter_source, "local_save_lattice")
    save = re.search(r"save\([^)]*'THERING'[^)]*'-v7'[^)]*\)", body)
    declaration = re.search(r"^\s*global\s+THERING\b", body, re.M)

    assert save is not None, "THERING must be saved under its own name, in -v7"
    assert declaration is not None, "the saved name has to be in scope of the save"
    assert declaration.start() < save.start()


def test_nothing_samples_the_machine_before_the_lattice_is_saved(
    exporter_source: str,
) -> None:
    """getpvmodel and measbpmresp mutate the ring, so the save comes first.

    The invariant is the order the export runs in, so it is read off the main
    function's own body: what a local function is defined next to says nothing
    about when it is called.
    """
    body = _code(_function_body(exporter_source, "mml_export"))
    before = body[: body.index("local_save_lattice(")]

    for call in SAMPLING_CALLS:
        assert call not in before, call
    for helper in sorted(set(re.findall(r"\b(local_\w+)\(", before))):
        assert helper in PURE_HELPERS, helper
    for helper in PURE_HELPERS:
        helper_body = _code(_function_body(exporter_source, helper))
        for call in SAMPLING_CALLS:
            assert call not in helper_body, (helper, call)


def test_the_lattice_is_saved_before_any_file_the_export_writes(
    exporter_source: str,
) -> None:
    steps = ("local_save_lattice(", "local_write(aoFile", "local_va_export(")
    seen = [exporter_source.find(step) for step in steps]

    assert -1 not in seen, steps
    assert seen == sorted(seen)


def test_the_header_states_the_order_and_names_the_files(exporter_source: str) -> None:
    header = exporter_source.split("EXPORTER_VERSION", 1)[0].lower()

    for name in ("lattice.mat", "ao.json", "ad.json", "va.json", "response.json"):
        assert name in header, name
    assert "thering is saved before" in header
    assert "getpvmodel" in header
    assert "measbpmresp" in header


# ---------------------------------------------------------------------------
# A refusing family costs the export nothing but its own block
# ---------------------------------------------------------------------------


def test_every_family_is_sampled_inside_its_own_try_catch(exporter_source: str) -> None:
    body = _function_body(exporter_source, "local_va_export")

    assert re.search(r"^\s*for\b", body, re.M), "the families are walked one by one"
    assert re.search(r"^\s*try\b", body, re.M)
    assert re.search(r"^\s*catch\s+err\b", body, re.M)


def test_a_refused_family_is_recorded_with_the_matlab_message(
    exporter_source: str,
) -> None:
    body = _function_body(exporter_source, "local_va_export")

    assert re.search(r"'refused'[^\n]*err\.message", body)


def test_a_refused_response_matrix_is_recorded_too(exporter_source: str) -> None:
    body = _function_body(exporter_source, "local_va_export")
    catches = re.findall(r"^\s*catch\s+err\b", body, re.M)

    assert len(catches) >= 2, "the response matrix is guarded like a family"


# ---------------------------------------------------------------------------
# Calibration: sampled through the facility's own conversion functions
# ---------------------------------------------------------------------------


def test_the_calibration_sampler_takes_the_field_it_samples(exporter_source: str) -> None:
    signature = re.search(
        r"^function\s+\[?[\w, ]*\]?\s*=\s*local_sample_calibration\(([^)]*)\)",
        exporter_source,
        re.M,
    )

    assert signature is not None, "local_sample_calibration is not defined"
    assert _arguments(signature.group(1))[:4] == ["family", "field", "DeviceList", "AO"]


def test_every_calibration_sample_passes_one_column_of_device_values(
    exporter_source: str,
) -> None:
    """A facility conversion function takes a column of devices and refuses a matrix."""
    calls = _sampling_calls(exporter_source, "hw2physics") + _sampling_calls(
        exporter_source, "physics2hw"
    )

    assert calls, "the calibration is sampled through the Middle Layer's own conversion"
    for arguments in calls:
        assert re.fullmatch(r"\w+\(:,\s*\w+\)", arguments[2]), arguments


def test_a_calibration_is_sampled_at_the_energy_it_is_handed(exporter_source: str) -> None:
    """The energy is the caller's, so the same grid can be re-sampled at another one."""
    calls = _sampling_calls(exporter_source, "hw2physics") + _sampling_calls(
        exporter_source, "physics2hw"
    )

    assert calls
    for arguments in calls:
        assert len(arguments) == 5, arguments
        assert arguments[4] == "energy", arguments


def test_a_calibration_is_sampled_at_thirty_three_points(exporter_source: str) -> None:
    body = _code(_function_body(exporter_source, "local_grid_points"))

    assert re.search(r"=\s*33;", body), "the point count lives in one place"
    assert _code(exporter_source).count("33") == 1


def test_a_monitor_only_calibration_falls_back_to_ten_millimetres(
    exporter_source: str,
) -> None:
    body = _function_body(exporter_source, "local_monitor_grid")

    assert re.search(r"=\s*0\.010;", _code(body)), "a beam position span in metres"
    assert "10 mm" in body


def test_a_calibration_names_where_its_grid_came_from(exporter_source: str) -> None:
    assert set(re.findall(r"source\s*=\s*'(\w+)';", exporter_source)) == {
        "range",
        "fallback",
        "setpoint",
    }
    assert exporter_source.count("grid_source") >= 2


def test_a_linear_calibration_is_two_numbers_and_any_other_is_a_table(
    exporter_source: str,
) -> None:
    body = _function_body(exporter_source, "local_calibration")

    assert re.search(r"'kind',\s*'linear',\s*'gain',\s*\w+,\s*'offset'", body)
    assert re.search(r"'kind',\s*'table',\s*'grid',\s*\w+,\s*'values'", body)


def test_a_calibration_is_never_read_from_the_stored_conversion_parameters(
    exporter_source: str,
) -> None:
    """The same parameters are three different algorithms at three facilities."""
    assert "HW2PhysicsParams" not in exporter_source
    assert "Physics2HWParams" not in exporter_source


def test_a_calibration_records_its_conversion_function_by_name(
    exporter_source: str,
) -> None:
    body = _function_body(exporter_source, "local_fcn_name")

    assert "func2str(" in body
    assert "'file'" not in body, "the handle's file path belongs to the exporting machine"
    assert len(re.findall(r"\.fcn = local_fcn_name\(", exporter_source)) == 2


def test_a_refused_calibration_keeps_the_fields_already_sampled(
    exporter_source: str,
) -> None:
    body = _function_body(exporter_source, "local_sample_fields")

    assert body.count("sampled = struct()") == 1
    assert body.index("sampled = struct()") < body.index("catch err")
    assert re.search(r"refused\s*=\s*err\.message;", body)


def test_the_monitor_calibration_inverse_is_sampled_over_the_setpoint_image(
    exporter_source: str,
) -> None:
    """MML keeps the two directions as independent data, so the inverse is sampled."""
    body = _function_body(exporter_source, "local_sample_inverse")

    assert "sampled.Setpoint.values" in body
    assert "sampled.Monitor.values" in body
    assert "local_sample_physics2hw(" in _code(body)
    assert "hw2physics" not in _code(body)


def test_a_band_that_does_not_hold_its_anchor_is_stretched_rather_than_abandoned(
    exporter_source: str,
) -> None:
    """A conversion turns over outside the span the facility runs the device over.

    So a device whose stated band misses its anchor is sampled over that band
    widened to the anchor and no further, per device, and only a device with
    no band at all gets the wide symmetric span.
    """
    grid = _code(_function_body(exporter_source, "local_hardware_grid"))
    band = _code(_function_body(exporter_source, "local_finite_band"))

    assert "banded = local_finite_band(range, nDev);" in grid
    assert "low(banded) = min(range(banded, 1), nominal(banded));" in grid
    assert "high(banded) = max(range(banded, 2), nominal(banded));" in grid
    assert "max(2 * abs(nominal), 1)" in grid
    assert "isfinite(range)" in band
    assert "range(:, 1) < range(:, 2)" in band
    assert "local_range_holds_nominal" not in exporter_source


def test_a_stretched_band_is_not_indexed_on_a_field_that_has_none(
    exporter_source: str,
) -> None:
    """MATLAB checks the column subscript of an empty array even when no row is selected."""
    grid = _code(_function_body(exporter_source, "local_hardware_grid"))

    guard = grid.index("if any(banded)")
    assert guard < grid.index("low(banded) =")
    assert grid.index("high(banded) =") < grid.index("end", guard)
    assert grid.index("low = -max(") < guard, "every device starts on the symmetric span"


def test_the_grid_source_word_is_the_weakest_grid_the_field_used(
    exporter_source: str,
) -> None:
    """One word per field, and a field is only on its Range when every device had a band."""
    grid = _code(_function_body(exporter_source, "local_hardware_grid"))

    assert re.search(r"if all\(banded\)\s*\n\s*source = 'range';", grid)
    assert re.search(r"else\s*\n\s*source = 'fallback';", grid)


def test_a_linear_calibration_is_the_line_held_against_every_sample(
    exporter_source: str,
) -> None:
    """Two endpoints agreeing is not a straight line; every sample has to lie on it."""
    body = _code(_function_body(exporter_source, "local_calibration"))
    tolerance = _code(_function_body(exporter_source, "local_linear_tolerance"))

    assert "all(all(residual" in body
    assert "local_linear_tolerance()" in body
    assert re.search(r"=\s*1e-9;", tolerance)


def test_a_field_with_no_usable_sample_is_refused_by_name(exporter_source: str) -> None:
    """A row of non-finite samples is not a calibration, and must not read as one."""
    body = _code(_function_body(exporter_source, "local_sample_calibration"))
    guard = _code(_function_body(exporter_source, "local_require_finite"))

    assert body.index("local_require_finite(") < body.index("local_calibration(")
    assert "sum(isfinite(values), 2) < 2" in guard
    assert re.search(r"error\('mml_export:samples'", guard)
    assert "local_fcn_name(" in guard, "the message names the conversion that answered"
    assert "min(min(grid(bad, :)))" in guard, "and the grid it was asked over"


def test_the_monitor_inverse_is_held_to_the_same_finite_rule(exporter_source: str) -> None:
    body = _code(_function_body(exporter_source, "local_sample_inverse"))

    assert body.index("local_require_finite(") < body.index("local_calibration(")


def test_a_partly_non_finite_table_records_the_span_it_answers_over(
    exporter_source: str,
) -> None:
    """The gap is marked rather than written out as numbers the consumer trusts."""
    body = _code(_function_body(exporter_source, "local_calibration"))
    span = _code(_function_body(exporter_source, "local_finite_span"))

    assert "'finite_span', local_finite_span(grid, values)" in body
    assert "find(isfinite(values(r, :)))" in span


def test_a_nominal_the_middle_layer_could_not_give_is_recorded_not_re_anchored(
    exporter_source: str,
) -> None:
    """Sampling a magnet at +-1 A because its nominal came back NaN has to be visible."""
    body = _code(_function_body(exporter_source, "local_anchor"))

    assert set(re.findall(r"anchor\s*=\s*'(\w+)';", exporter_source)) == {
        "nominal",
        "range_midpoint",
        "zero",
    }
    assert "(range(usable, 1) + range(usable, 2)) / 2" in body
    assert "calibration.anchor = anchor;" in _code(exporter_source)


def test_a_field_with_no_band_is_anchored_without_indexing_its_empty_range(
    exporter_source: str,
) -> None:
    """MATLAB checks the column subscript of an empty array even when no row is selected.

    The Monitor field of a magnet family has neither a nominal nor a Range, so
    every device is missing and none is usable; the midpoint line must not run.
    """
    body = _code(_function_body(exporter_source, "local_anchor"))

    guard = body.index("if any(usable)")
    assert guard < body.index("range(usable, 1)")
    assert body.index("range(usable, 1)") < body.index("end", guard), "inside the guard"
    assert body.index("anchored(missing & ~usable) = 0;") > body.index("end", guard)


def test_a_conversion_that_answers_in_another_shape_is_refused_by_name(
    exporter_source: str,
) -> None:
    """An empty answer is a null assignment in MATLAB and would delete a grid point."""
    guard = _code(_function_body(exporter_source, "local_column"))

    for sampler in ("local_sample_hw2physics", "local_sample_physics2hw"):
        assert "local_column(" in _code(_function_body(exporter_source, sampler)), sampler
    assert "numel(column) ~= nDev" in guard
    assert re.search(r"error\('mml_export:shape'", guard)
    assert "column = column(:);" in guard


# ---------------------------------------------------------------------------
# Energy scaling: the same grid, sampled again at a moved energy
# ---------------------------------------------------------------------------


def test_energy_scaling_resamples_the_same_grid_two_percent_higher(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_energy_scaling"))
    step = _code(_function_body(exporter_source, "local_energy_step"))

    assert "local_sample_hw2physics(family, field, DeviceList, grid, energy * step)" in body
    assert re.search(r"=\s*1\.02;", step), "the step lives in one place"
    assert _code(exporter_source).count("1.02") == 1


def test_energy_scaling_weighs_both_samples_with_the_middle_layers_own_rigidity(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_energy_scaling"))

    assert "getbrho(energy)" in body
    assert "getbrho(energy * step)" in body
    assert "2.99792458" not in exporter_source, "the rigidity is never re-derived here"


def test_energy_scaling_says_why_the_rest_mass_is_mandatory(exporter_source: str) -> None:
    """The massless form misses by more than the tolerance, so it cannot decide this."""
    body = _function_body(exporter_source, "local_energy_scaling")

    assert "rest mass" in body
    assert "getbrho(" in _code(body), "the rigidity is the Middle Layer's own"


def test_energy_scaling_is_brho_within_one_part_in_a_million(exporter_source: str) -> None:
    body = _code(_function_body(exporter_source, "local_energy_scaling"))
    tolerance = _code(_function_body(exporter_source, "local_energy_tolerance"))

    assert "local_energy_tolerance()" in body
    assert re.search(r"scaling\s*=\s*'brho';", body)
    assert re.search(r"scaling\s*=\s*'none';", body)
    assert re.search(r"=\s*1e-6;", tolerance), "the tolerance lives in one place"
    assert _code(exporter_source).count("1e-6") == 1


def test_energy_scaling_records_the_worst_deviation_it_measured(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_energy_scaling"))
    fields = _code(_function_body(exporter_source, "local_sample_fields"))

    assert re.search(r"deviation\s*=\s*max\(deviations\(:\)\);", body)
    assert "energy_scaling" in fields
    assert "energy_deviation" in fields


def test_energy_scaling_compares_only_the_points_both_samples_reached(
    exporter_source: str,
) -> None:
    """A grid point outside a facility ramp table converts to NaN at either energy."""
    body = _code(_function_body(exporter_source, "local_energy_scaling"))

    assert "isfinite(reference) & isfinite(moved)" in body
    assert "deviations(scale == 0 | ~both) = 0;" in body
    assert "local_require_finite(" in body


def test_energy_scaling_states_that_a_conversion_ignoring_the_energy_is_none(
    exporter_source: str,
) -> None:
    """A gain and offset does not scale, which is what MML itself does."""
    body = _function_body(exporter_source, "local_energy_scaling")

    assert "reads as none" in body


def test_the_energy_table_and_the_scaling_refuse_an_unusable_model_energy(
    exporter_source: str,
) -> None:
    """getenergymodel answers with nothing when the deck carries no Energy field.

    An empty energy is not an error downstream: hw2physics and getbrho both
    substitute the machine energy for it, so the two samples of the scaling
    would be taken at the same energy and every family would read as carrying
    the rigidity.
    """
    guard = _function_body(exporter_source, "local_require_energy")
    code = _code(guard)

    assert "isscalar(energy)" in code
    assert "isfinite(energy)" in code
    assert re.search(r"energy\s*<=\s*0", code)
    assert "getenergymodel" in guard, "the refusal names the call that answered"

    scaling = _code(_function_body(exporter_source, "local_energy_scaling"))
    assert scaling.index("local_require_energy(") < scaling.index("energy * step")

    table = _code(_function_body(exporter_source, "local_energy_table"))
    assert table.index("local_require_energy(") < table.index("bend2gev(")


def test_a_field_held_as_a_struct_array_states_nothing_rather_than_throwing(
    exporter_source: str,
) -> None:
    """A struct array answers one value per element, which is no value to hand back.

    Without the check the read throws where no caller catches it, and the
    family collapses to a refusal with nothing else in its block.
    """
    body = _code(_function_body(exporter_source, "local_subfield"))
    names = _code(_function_body(exporter_source, "local_field_names"))

    assert "isscalar(AO.(family).(field))" in body
    assert body.index("isscalar(") < body.index("value = AO.(family).(field).(name)")
    assert "isscalar(value)" in names, "the same rule names the fields of a family"


def test_a_table_row_with_no_finite_sample_is_refused_by_name(
    exporter_source: str,
) -> None:
    """The named refusal surfaces, not a MATLAB index error three frames up."""
    body = _code(_function_body(exporter_source, "local_finite_span"))

    assert "isempty(finite)" in body
    assert "mml_export:samples" in body
    assert body.index("isempty(finite)") < body.index("finite(1)")


# ---------------------------------------------------------------------------
# The energy table: what the facility's own bend-to-energy conversion answers
# ---------------------------------------------------------------------------


def test_energy_table_candidates_are_bends_that_are_not_correctors(
    exporter_source: str,
) -> None:
    """NSLS-II's BEND family is a bend by its membership alone: its ATType is SEXT."""
    body = _code(_function_body(exporter_source, "local_energy_candidate"))

    assert "local_subfield(AO, family, 'AT', 'ATType')" in body
    assert "strcmpi(" in body, "the lattice type is matched case-insensitively"
    assert "local_member(memberOf, 'BEND')" in body
    assert "~local_member(memberOf, 'COR')" in body


def test_energy_table_candidates_read_the_member_list_case_insensitively(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_member"))

    assert "strcmpi(" in body


def test_energy_table_is_sampled_over_the_setpoint_grid_of_one_device_row(
    exporter_source: str,
) -> None:
    """A facility's conversion reads the ramp of the device row it is handed."""
    body = _code(_function_body(exporter_source, "local_energy_table"))

    assert "row = DeviceList(1, :);" in body
    assert "sampled.(field).grid(1, :)" in body
    assert "energyTable.device_row = row;" in body


def test_energy_table_converts_one_grid_point_per_call(exporter_source: str) -> None:
    """Handed a vector of currents with one device row, bend2gev answers for one."""
    body = _code(_function_body(exporter_source, "local_energy_table"))
    calls = _sampling_calls(exporter_source, "bend2gev")

    assert re.search(r"for k = 1:numel\(grid\)", body)
    assert calls, "the table is sampled through the facility's own conversion"
    for arguments in calls:
        assert len(arguments) == 5, arguments
        assert arguments[:2] == ["family", "field"], arguments
        assert arguments[2] in ("grid(k)", "I_nom"), arguments
        assert arguments[3] == "row", arguments


def test_energy_table_spells_out_the_hardware_units_of_every_conversion(
    exporter_source: str,
) -> None:
    """A facility copy defaults the units flag from getunits and converts twice."""
    calls = _sampling_calls(exporter_source, "bend2gev") + _sampling_calls(
        exporter_source, "gev2bend"
    )

    assert calls
    for arguments in calls:
        assert arguments[-1] == "'Hardware'", arguments


def test_energy_table_takes_the_nominal_current_from_the_facility_inverse(
    exporter_source: str,
) -> None:
    """NSLS-II's ring ships bend2gev and no inverse, so the read nominal stands in."""
    body = _code(_function_body(exporter_source, "local_nominal_current"))

    assert "exist('gev2bend', 'file') == 0" in body
    assert "gev2bend(family, field, energy, row, 'Hardware')" in body
    assert "local_nominal_of(nominals, field)" in body
    assert "getpvmodel" not in body, "the nominal is read from the export's own seam"


def test_energy_table_records_the_energy_at_the_nominal_current(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_energy_table"))

    assert "energyTable.I_nom = I_nom;" in body
    assert "bend2gev(family, field, I_nom, row, 'Hardware')" in body
    assert "energyAtNominal = local_column(" in body
    assert "energyTable.energy_at_nominal = energyAtNominal;" in body


def test_energy_table_is_written_exactly_as_it_was_sampled(exporter_source: str) -> None:
    """Whether a constant table is a knob at all is the consumer's call, not MATLAB's."""
    body = _code(_function_body(exporter_source, "local_energy_table"))

    assert "energyTable.values = values;" in body
    assert "diff(" not in body
    assert "unique(" not in body
    assert not re.search(r"values\s*==", body)
    assert "flat" not in body.lower()


def test_energy_table_refuses_a_conversion_that_answered_with_no_number(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_energy_table"))

    assert "~any(isfinite(values))" in body
    assert "mml_export:energy_table" in body


def test_energy_table_keeps_what_it_sampled_when_the_nominal_refuses(
    exporter_source: str,
) -> None:
    """A bend2gev or gev2bend failure costs the family its block, never the export."""
    body = _function_body(exporter_source, "local_energy_table")

    assert body.count("energyTable = struct()") == 1
    assert body.index("energyTable = struct()") < body.index("catch err")
    assert re.search(r"refused\s*=\s*err\.message;", body)
    assert body.index("energyTable.values = values;") < body.index("local_nominal_current(")


def test_the_energy_table_records_the_span_it_answers_over(exporter_source: str) -> None:
    """A ramp table that ends partway is the common case on a unipolar bend.

    Every other table of the export names the span it converts over, so a
    consumer never reads a spelled-out non-finite entry as a number.
    """
    body = _code(_function_body(exporter_source, "local_energy_table"))

    assert "energyTable.finite_span = local_finite_span(grid, values);" in body
    assert body.index("energyTable.values = values;") < body.index("finite_span")


def test_energy_table_refuses_a_non_finite_energy_at_its_nominal_current(
    exporter_source: str,
) -> None:
    """The ramp not covering the deck's own setting is refused, not written as "NaN".

    The check comes before the value reaches the block: a scalar spelled
    "NaN" in a slot every other scalar of the table carries a number in reads
    like a number of this machine all the way to the consumer that uses it.
    """
    body = _code(_function_body(exporter_source, "local_energy_table"))

    assert "~isfinite(energyAtNominal)" in body
    assert body.index("~isfinite(energyAtNominal)") < body.index("energyTable.energy_at_nominal =")
    guard = body[body.index("~isfinite(energyAtNominal)") :]
    assert re.search(r"error\('mml_export:energy_table'", guard)


# ---------------------------------------------------------------------------
# The nominals: what the Middle Layer says a family is set to
# ---------------------------------------------------------------------------


def test_the_nominal_is_the_setpoint_and_the_monitor_only_without_one(
    exporter_source: str,
) -> None:
    """A setpoint is the setting; a monitor-only family's reading is all it has."""
    body = _code(_function_body(exporter_source, "local_nominal_field"))

    assert "names = {'Setpoint', 'Monitor'};" in body
    assert body.index("'Setpoint'") < body.index("'Monitor'")
    assert "return" in body, "the first field the family carries is the one read"


def test_the_nominal_read_asks_for_hardware_units_and_for_the_units_statement(
    exporter_source: str,
) -> None:
    """getpvmodel hands back physics units for a conversion it does not know.

    Only its struct output says which units it answered in, and a physics
    number sizing a hardware grid is a wrong grid rather than a wrong label.
    """
    calls = _sampling_calls(exporter_source, "getpvmodel")
    body = _code(_function_body(exporter_source, "local_sample_nominal"))

    assert calls, "the nominal is read through the Middle Layer's own model read"
    for arguments in calls:
        assert arguments[3] == "'Hardware'", arguments
    assert [arguments[4] for arguments in calls] == ["'Struct'", "'Numeric'"]
    assert "units = local_text(local_field(answer, 'Units'));" in body
    assert "answer = local_field(answer, 'Data');" in body


def test_the_nominal_read_takes_the_whole_device_list_and_passes_no_time(
    exporter_source: str,
) -> None:
    """The model read converts at the model's own energy; a fourth number is a time."""
    calls = _sampling_calls(exporter_source, "getpvmodel")

    assert calls
    for arguments in calls:
        assert len(arguments) == 5, arguments
        assert arguments[:3] == ["family", "field", "DeviceList"], arguments


def test_a_nominal_answered_in_physics_units_is_refused_for_the_hardware_nominal(
    exporter_source: str,
) -> None:
    """The fact is recorded; only the number a hardware grid would be sized by goes."""
    body = _code(_function_body(exporter_source, "local_nominals"))

    assert "~strcmpi(units, 'Hardware')" in body
    assert re.search(r"error\('mml_export:nominals'", body)
    assert body.index("recorded.(field) = record;") < body.index("mml_export:nominals")
    assert body.index("mml_export:nominals") < body.index("nominals.(field) = values;")


def test_a_nominal_records_the_units_the_read_answered_in(exporter_source: str) -> None:
    body = _code(_function_body(exporter_source, "local_nominals"))

    assert "record.values = values;" in body
    assert "record.units = units;" in body


def test_a_nominal_records_the_at_block_the_read_went_through(
    exporter_source: str,
) -> None:
    """A field-level AT block is an override the simulator alone honours."""
    resolved = _code(_function_body(exporter_source, "local_at_of"))
    body = _code(_function_body(exporter_source, "local_nominals"))

    assert "local_subfield(AO, family, field, 'AT')" in resolved
    assert "local_field(AO.(family), 'AT')" in resolved
    assert resolved.index("local_subfield(") < resolved.index("local_field(AO.(family)")
    assert "record.at_type = local_text(local_field(at, 'ATType'));" in body
    assert "record.at_index = local_field(at, 'ATIndex');" in body


def test_a_fabricated_nominal_is_recorded_and_flagged_synthetic(
    exporter_source: str,
) -> None:
    """A hard-coded milliamp figure and a stub of ones are not readings of this ring."""
    body = _code(_function_body(exporter_source, "local_synthetic"))
    nominals = _code(_function_body(exporter_source, "local_nominals"))

    assert "strcmpi(family, 'DCCT')" in body
    for at_type in ("'Septum'", "'null'", "'Photon BPM'"):
        assert at_type in body, at_type
    assert "findcells(THERING, 'FamName', family)" in body
    assert "isempty(at)" in body, "a family with no AT block is answered from zeros"
    assert "all(~isfinite(values))" in body, "a lattice type the read does not know"
    assert "record.synthetic = local_synthetic(family, at, values);" in nominals
    assert nominals.index("record.values = values;") < nominals.index("record.synthetic")


def test_a_nominals_at_index_keeps_the_non_finite_spelling_of_the_export(
    exporter_source: str,
) -> None:
    """A device padded with fewer elements than its siblings keeps its "NaN" cells."""
    body = _code(_function_body(exporter_source, "local_nominals"))
    resolved = _code(_function_body(exporter_source, "local_at_of"))
    spelling = _code(_function_body(exporter_source, "local_nonfinite"))

    for source in (body, resolved):
        assert "isfinite" not in source, "the index is recorded as the family stores it"
        assert "isnan" not in source
    assert "out = 'NaN';" in spelling


def test_a_refused_nominal_read_costs_the_family_its_nominal_and_no_more(
    exporter_source: str,
) -> None:
    """One family's conversion refusing is a message in its block, not an error out."""
    body = _function_body(exporter_source, "local_nominals")

    assert body.count("recorded = struct()") == 1
    assert body.index("recorded = struct()") < body.index("catch err")
    assert body.index("nominals = struct()") < body.index("catch err")
    assert re.search(r"refused\s*=\s*err\.message;", body)


def test_a_family_with_no_devices_is_refused_before_the_nominal_read(
    exporter_source: str,
) -> None:
    """An empty device list reaches the model read as an index error three frames up."""
    body = _code(_function_body(exporter_source, "local_nominals"))

    assert "nDev == 0" in body
    assert re.search(r"error\('mml_export:devices'", body)
    assert body.index("mml_export:devices") < body.index("local_sample_nominal(")


def test_the_nominal_seam_hands_the_grids_one_column_per_field(
    exporter_source: str,
) -> None:
    """Every grid of the export is sized by this seam, read through one accessor."""
    signature = re.search(
        r"^function\s+\[([\w, ]*)\]\s*=\s*local_nominals\(([^)]*)\)",
        exporter_source,
        re.M,
    )
    accessor = _code(_function_body(exporter_source, "local_nominal_of"))

    assert signature is not None, "local_nominals is not defined"
    assert _arguments(signature.group(1)) == ["recorded", "nominals", "refused"]
    assert _arguments(signature.group(2)) == ["family", "AO", "DeviceList"]
    assert "nominals.(field) = values;" in _code(_function_body(exporter_source, "local_nominals"))
    assert "nominal = nominals.(field);" in accessor


def test_the_nominal_read_is_the_only_getpvmodel_call_the_export_makes(
    exporter_source: str,
) -> None:
    """The read mutates the ring, so it stays behind the one seam the save precedes."""
    body = _function_body(exporter_source, "local_sample_nominal")

    assert "getpvmodel(" in SAMPLING_CALLS
    assert _code(exporter_source).count("getpvmodel(") == _code(body).count("getpvmodel(")
    assert _code(body).count("getpvmodel(") == 2


# ---------------------------------------------------------------------------
# The lattice fingerprint: which ring the rest of the file belongs to
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_the_four_facts_a_consumer_can_recompute(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_lattice_fingerprint"))

    assert re.findall(r"\bfingerprint\.(\w+)\s*=", body) == list(VA_LATTICE_KEYS)


def test_the_element_count_is_the_whole_ring(exporter_source: str) -> None:
    """A ring-parameter element is an element: the Middle Layer indexes past it."""
    body = _code(_function_body(exporter_source, "local_lattice_fingerprint"))

    assert "fingerprint.elements = numel(THERING);" in body
    assert re.search(r"^\s*global\s+THERING\b", body, re.M)


def test_the_digest_is_the_family_names_in_ring_order(exporter_source: str) -> None:
    body = _code(_function_body(exporter_source, "local_lattice_fingerprint"))
    names = _code(_function_body(exporter_source, "local_famnames"))

    assert "local_sha256(strjoin(local_famnames(THERING), newline))" in body
    assert "for k = 1:numel(ring)" in names
    assert "local_field(ring{k}, 'FamName')" in names


def test_the_digest_is_taken_over_utf8_bytes_as_lower_case_hex(
    exporter_source: str,
) -> None:
    """The default encoding is the exporting machine's, and would make the digest
    depend on where the export ran."""
    body = _code(_function_body(exporter_source, "local_sha256"))

    assert "typecast(unicode2native(text, 'UTF-8'), 'int8')" in body
    assert "java.security.MessageDigest.getInstance('SHA-256')" in body
    assert "lower(" in body and "dec2hex(" in body


def test_the_digest_is_taken_over_the_family_name_verbatim(
    exporter_source: str,
) -> None:
    """The consumer hashes the name its own lattice file holds, so a rule applied
    on this side alone refuses a pair that is in fact the same ring."""
    body = _code(_function_body(exporter_source, "local_famnames"))
    header = exporter_source.split("EXPORTER_VERSION", 1)[0]

    assert "local_text(" not in body
    assert "strtrim" not in body
    assert "(1, :)" not in body
    assert "name = char(name);" in body
    assert "each name as the ring carries" in header


def test_a_family_name_of_more_than_one_row_is_refused_by_name(
    exporter_source: str,
) -> None:
    """A char matrix is no one name, and reducing it to a row is the rule that
    would make the digest disagree with the consumer's."""
    body = _code(_function_body(exporter_source, "local_famnames"))

    assert "size(name, 1) > 1" in body
    assert body.count("error('mml_export:lattice'") == 2


def test_an_element_with_no_family_name_is_refused_by_name(exporter_source: str) -> None:
    """A consumer binds an element by its name; one it cannot name it cannot bind."""
    body = _code(_function_body(exporter_source, "local_famnames"))

    assert "~ischar(name) || isempty(name)" in body
    assert re.search(r"error\('mml_export:lattice'", body)


def test_the_ring_parameter_elements_are_recorded_by_position(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_ringparam_indices"))

    assert "local_field(ring{k}, 'Class')" in body
    assert "'RingParam'" in body
    assert "indices(end+1) = k;" in body


def test_the_fingerprint_energy_is_the_one_every_family_is_sampled_at(
    exporter_source: str,
) -> None:
    """One read of the model energy, handed to the fingerprint and to every family."""
    body = _code(_function_body(exporter_source, "local_va_export"))
    fingerprint = _code(_function_body(exporter_source, "local_lattice_fingerprint"))

    assert body.count("getenergymodel") == 1
    assert "local_lattice_fingerprint(energy)" in body
    assert "local_va_family(family, AO, energy)" in body
    assert "fingerprint.energy_gev = energy;" in fingerprint


def test_an_unreadable_model_energy_refuses_the_fingerprint_by_name(
    exporter_source: str,
) -> None:
    require = _code(_function_body(exporter_source, "local_require_lattice_energy"))
    fingerprint = _code(_function_body(exporter_source, "local_lattice_fingerprint"))

    assert "~isscalar(energy) || ~isfinite(energy) || energy <= 0" in require
    assert re.search(r"error\('mml_export:lattice'", require)
    assert fingerprint.index("local_require_lattice_energy(energy)") < fingerprint.index(
        "fingerprint.elements"
    )


def test_the_fingerprint_is_written_beside_the_families_and_refuses_like_one(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_va_export"))

    assert "struct('lattice', local_lattice_fingerprint(energy))" in body
    assert "struct('lattice', struct('refused', err.message))" in body
    assert body.index("struct('lattice'") < body.index("va.families = struct();")


# ---------------------------------------------------------------------------
# The family block: the key set every reader of an export is held to
# ---------------------------------------------------------------------------


def test_the_family_block_carries_exactly_the_frozen_key_set(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_va_family"))

    assert set(re.findall(r"\bblock\.(\w+)\s*=", body)) == set(VA_FAMILY_KEYS)


def test_the_header_freezes_the_same_keys_the_export_writes(exporter_source: str) -> None:
    """The header is what a reader of an export reads the file by."""
    header = exporter_source.split("EXPORTER_VERSION", 1)[0]

    for key in (
        VA_LATTICE_KEYS
        + VA_FAMILY_KEYS
        + VA_SETPOINT_KEYS
        + VA_MONITOR_KEYS
        + VA_NOMINAL_KEYS
        + VA_ENERGY_TABLE_KEYS
        + VA_CALIBRATION_KEYS
        + VA_READOUT_KEYS
    ):
        assert key in header, key


def test_the_header_freezes_the_words_a_reader_switches_on(exporter_source: str) -> None:
    """A reader branches on these strings, so the header owes it the whole list."""
    header = exporter_source.split("EXPORTER_VERSION", 1)[0]
    code = _code(exporter_source)

    for field, words in VA_VOCABULARIES.items():
        assert field in header, field
        assert " | ".join(f'"{word}"' for word in words) in header, field
        for word in words:
            assert f"'{word}'" in code, word
    assert "monitor_inverse\n%   carries grid_source and fcn but no anchor" in header


def test_the_header_states_the_shape_of_a_family_it_could_not_read(
    exporter_source: str,
) -> None:
    """The last-resort catch writes refused alone; a reader that indexes
    device_list unconditionally raises on such a block instead of listing it."""
    header = exporter_source.split("EXPORTER_VERSION", 1)[0]
    flat = " ".join(header.replace("%", " ").split())

    assert "except a family the export could not read at all, whose block is refused alone" in flat


def test_the_header_tells_a_reader_how_one_row_is_written(exporter_source: str) -> None:
    """jsonencode flattens a single row, so a one-device family is one flat array."""
    header = exporter_source.split("EXPORTER_VERSION", 1)[0].lower()

    assert "one row per device" in header
    assert "flat" in header


def test_the_nominal_is_read_before_the_grids_it_sizes(exporter_source: str) -> None:
    """Read after them, every grid would be sized by zero instead of by a setting."""
    body = _code(_function_body(exporter_source, "local_va_family"))
    steps = (
        "local_nominals(family, AO, DeviceList)",
        "local_sample_fields(family, AO, nominals, energy)",
        "local_energy_table(family, DeviceList, sampled, nominals, energy)",
    )
    for step in steps:
        assert step in body, step

    assert [body.index(step) for step in steps] == sorted(body.index(s) for s in steps)


def test_the_block_carries_the_calibration_rather_than_the_samples(
    exporter_source: str,
) -> None:
    """A line reproduces its own grid and a table carries its own, so the 33 samples
    per device stay out of the file."""
    body = _code(_function_body(exporter_source, "local_va_family"))

    assert "struct('calibration', sampled.Setpoint.calibration)" in body
    assert "struct('calibration', sampled.Monitor.calibration)" in body
    for spelling in ("setpoint.grid", "setpoint.values", "monitor.grid", "monitor.values"):
        assert re.search(rf"\b{spelling}\s*=", body) is None, spelling


def test_the_energy_verdict_is_lifted_into_the_setpoint_block(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_va_family"))

    assert "isfield(sampled.Setpoint, 'energy_scaling')" in body
    assert "setpoint.energy_scaling = sampled.Setpoint.energy_scaling;" in body
    assert "setpoint.energy_deviation = sampled.Setpoint.energy_deviation;" in body


def test_the_monitor_inverse_is_lifted_into_the_monitor_block(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_va_family"))

    assert "isfield(sampled.Monitor, 'monitor_inverse')" in body
    assert "monitor.monitor_inverse = sampled.Monitor.monitor_inverse;" in body


def test_a_fact_that_was_never_read_is_absent_rather_than_empty(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_va_family"))

    assert "isfield(sampled, 'Setpoint')" in body
    assert "isfield(sampled, 'Monitor')" in body
    assert "~isempty(fieldnames(recorded))" in body
    assert "~isempty(fieldnames(energyTable))" in body
    assert "~isempty(refused)" in body


def test_the_energy_table_is_sampled_only_for_an_energy_candidate(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_va_family"))

    assert "block.energy_candidate = local_energy_candidate(AO, family);" in body
    assert "if block.energy_candidate" in body
    assert body.index("block.energy_candidate =") < body.index("local_energy_table(")


def test_the_block_records_every_refusal_of_the_steps_that_can_refuse(
    exporter_source: str,
) -> None:
    """A refusal stops no step after it, so each step that refused owes its own
    reason for the key it left out - and one reason met twice is carried once."""
    body = _code(_function_body(exporter_source, "local_va_family"))
    merged = _code(_function_body(exporter_source, "local_refusals"))

    assert "local_refusals({refusedNominal, refusedSample, refusedReadout, refusedTable})" in body
    assert "refusedTable = '';" in body, "a family that is no candidate refused no table"
    assert "isempty(message) || any(strcmp(kept, message))" in merged
    assert "strjoin(kept, '; ')" in merged


def test_the_fields_of_a_family_are_the_ones_carrying_a_mode(
    exporter_source: str,
) -> None:
    """Mode is the word a field is read through; the AT block carries none."""
    body = _code(_function_body(exporter_source, "local_field_names"))

    assert "fieldnames(body)" in body
    assert "isstruct(value)" in body
    assert "isfield(value, 'Mode')" in body


def test_the_device_list_is_the_order_every_row_of_the_block_is_in(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_va_family"))

    assert "DeviceList = local_device_list(AO, family);" in body
    assert "block.device_list = DeviceList;" in body


def test_the_nominal_block_carries_the_frozen_nominal_keys(exporter_source: str) -> None:
    body = _code(_function_body(exporter_source, "local_nominals"))

    for key in VA_NOMINAL_KEYS:
        assert f"record.{key} =" in body, key


def test_the_energy_table_carries_the_frozen_energy_table_keys(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_energy_table"))

    for key in VA_ENERGY_TABLE_KEYS:
        assert f"energyTable.{key} =" in body, key


def test_a_calibration_carries_the_frozen_calibration_keys(exporter_source: str) -> None:
    calibration = _code(_function_body(exporter_source, "local_calibration"))
    setpoint = _code(_function_body(exporter_source, "local_sample_calibration"))
    inverse = _code(_function_body(exporter_source, "local_sample_inverse"))

    for key in ("kind", "gain", "offset", "grid", "values", "finite_span"):
        assert f"'{key}'" in calibration, key
    for key in ("grid_source", "anchor", "fcn"):
        assert f"calibration.{key} =" in setpoint, key
    for key in ("grid_source", "fcn"):
        assert f"inverse.{key} =" in inverse, key


def test_the_readout_carries_the_frozen_readout_keys_and_no_defaults(
    exporter_source: str,
) -> None:
    """What is written is what the facility states, looked up the way it keeps it.

    The Middle Layer's own readers take these from the family's Monitor field
    first, from the family itself second and from the facility's physics data
    third, and answer a family that carries none with a default. A default says
    the facility stated something it did not, so a key with nothing behind it
    is left out instead.
    """
    body = _code(_function_body(exporter_source, "local_readout"))

    assert re.search(r"keys = \{(.+?)\};", body, re.S)
    for key in ("Gain", "Offset", "Roll", "Crunch"):
        assert f"'{key}'" in body, key
    for key in VA_READOUT_KEYS:
        assert f"'{key}'" in body, key
    assert "local_subfield(AO, family, 'Monitor'" in body
    assert body.index("local_subfield(AO, family, 'Monitor'") < body.index("local_field(stored")
    assert body.index("local_field(stored") < body.index("local_physdata(family")
    assert "continue" in body, "a key the family carries nothing for is absent, not a default"


def test_the_reader_realigns_the_readout_keys_the_contract_freezes() -> None:
    """The reader's leg of the triple, which it cannot import from here.

    Source cannot import a test module, so the reader re-types the four names
    it realigns a judged device order by. Held against the contract, a key
    renamed in the export cannot leave the reader realigning one nobody writes
    and quietly leaving the renamed one in the export's device order.
    """
    assert _VA_READOUT_ROW_KEYS == VA_READOUT_KEYS


def test_the_physics_data_is_read_and_a_facility_without_one_states_nothing(
    exporter_source: str,
) -> None:
    """The third lookup level, and the one that is allowed to be absent.

    A facility keeps its fitted gains and rolls on file until an operating mode
    copies them into the Accelerator Objects, so a session that has not run one
    states them only there. Reading the file is a read and never a write, and a
    facility with no such file answers with an error, which is the same fact as
    an empty field and never a refusal the block records.
    """
    body = _code(_function_body(exporter_source, "local_physdata"))

    assert "getphysdata(family, name, DeviceList)" in body
    assert "try" in body and "catch" in body
    assert body.count("value = [];") == 2, "an error there is absence, not a refusal"
    assert "setphysdata" not in body, "the physics data is read, never written"


def test_a_number_stated_once_for_the_family_is_written_for_every_device(
    exporter_source: str,
) -> None:
    """One row is what the whole family holds, which is what the export writes.

    The Middle Layer hands a stored row of one back for every device in the
    list, so a facility that states a single gain has stated it for each of its
    monitors. Writing that one number per device leaves a consumer one shape to
    read instead of two, and costs the family nothing when it states the other
    shape instead.
    """
    body = _code(_function_body(exporter_source, "local_readout_column"))

    assert "isscalar(value)" in body
    assert "ones(nDev, 1) * double(value)" in body
    assert body.index("isscalar(value)") < body.index("rows == nDev")
    assert "rows == nDev && numel(value) == nDev" in body


def test_a_readout_that_is_not_one_number_per_device_is_refused_by_name(
    exporter_source: str,
) -> None:
    """A shape that cannot be one per device, and a value that is not a number.

    The refusal names the place the number was found in, because the owner is
    asked to write these messages down rather than repair the Accelerator
    Objects, and a message naming a field the value did not come from sends
    them to the wrong one.
    """
    source = _function_body(exporter_source, "local_readout_column")
    body = _code(source)

    assert "~isnumeric(value)" in body, "a value of the right length can still be no number"
    assert body.count("error('mml_export:readout'") == 2
    assert "'%s holds %s values rather than numbers.'" in source
    assert "'%s holds %d by %d values for %d devices.'" in source
    assert source.count("', where,") == 2, "both refusals name where the number was found"


def test_one_refused_readout_key_costs_the_family_that_key_alone(
    exporter_source: str,
) -> None:
    """Three good numbers are not lost to a fourth the facility wrote badly.

    The export runs once, on a licensed host, and there is no second pass to
    recover what a single malformed field took with it.
    """
    body = _code(_function_body(exporter_source, "local_readout"))

    guard = body.index("try")
    assert body.index("for k = 1:size(keys, 1)") < guard, "the guard is inside the loop"
    assert "readout = struct();" in body
    assert body.index("readout = struct();") < guard, "a refusal never resets what was written"
    assert "refusals{end+1} = err.message;" in body
    assert "refused = local_refusals(refusals);" in body


def test_the_readout_sits_beside_the_conversion_it_calibrates(exporter_source: str) -> None:
    """A family with no monitor conversion has no monitor reading to correct."""
    body = _code(_function_body(exporter_source, "local_va_family"))

    assert "monitor.readout = readout;" in body
    assert body.index("monitor.readout = readout;") < body.index("block.Monitor = monitor;")
    assert "refusedReadout" in body, "a readout it could not read costs the family nothing else"


# ---------------------------------------------------------------------------
# The response matrix: the facility's own measurement, read once
# ---------------------------------------------------------------------------


def test_the_response_matrix_is_read_in_physics_units_with_no_energy_scaling(
    exporter_source: str,
) -> None:
    """Hardware units are a matrix per conversion, and a scaled matrix no longer
    matches the operating point its own correctors were at."""
    calls = _sampling_calls(exporter_source, "getbpmresp")

    assert calls == [["'Struct'", "'NoEnergyScaling'", "'Physics'"]]


def test_the_response_is_reached_one_way_or_the_other_and_never_twice(
    exporter_source: str,
) -> None:
    """Both walk the ring, so both stay behind the one seam the save precedes.

    A facility that names a file has it read and a facility that names none has
    the model measured, so the export holds one of each call and takes exactly
    one of them per run.
    """
    code = _code(exporter_source)
    body = _code(_function_body(exporter_source, "local_response"))
    model = _code(_function_body(exporter_source, "local_response_of_model"))

    assert "getbpmresp(" in SAMPLING_CALLS
    assert "measbpmresp(" in SAMPLING_CALLS
    assert code.count("getbpmresp(") == 1
    assert code.count("measbpmresp(") == 1
    assert body.count("getbpmresp(") == 1
    assert model.count("measbpmresp(") == 1
    assert "else" in body, "one branch or the other, never both"


def test_the_response_file_names_are_read_the_way_the_middle_layer_reads_them(
    exporter_source: str,
) -> None:
    """getrespmat reads AD.OpsData.RespFiles, which holds one name or a cell of them."""
    body = _code(_function_body(exporter_source, "local_response_files"))

    assert "local_field(AD, 'OpsData')" in body
    assert "local_field(ops, 'RespFiles')" in body
    assert "names = {char(names)};" in body, "one name is a name, not a cell of characters"
    assert "~isempty(name)" in body, "a blank matches no file"


def test_a_response_matrix_with_no_file_named_measures_the_model_instead(
    exporter_source: str,
) -> None:
    """Naming nothing is the one state the Middle Layer answers with a dialog box.

    So that state is never handed to the read. It ends where every other way of
    having no file already ends -- the Middle Layer's own measurement of the
    model, taken with no dialog and no display -- and the origin of each block
    says the matrix is the model's.

    A name the facility does have is left to the read, whether or not anything
    is there.
    """
    body = _code(_function_body(exporter_source, "local_response"))
    model = _code(_function_body(exporter_source, "local_response_of_model"))

    assert "isempty(local_response_files(AD))" in body
    assert body.index("isempty(local_response_files(AD))") < body.index("getbpmresp(")
    assert body.index("local_response_of_model()") < body.index("getbpmresp(")
    assert "error('mml_export:response'" not in model, "an empty list is measured, not refused"
    assert "exist(" not in body, "whether a named file is there is the read's own business"


def test_the_model_measurement_asks_for_no_dialog_no_archive_and_no_display(
    exporter_source: str,
) -> None:
    """An unattended export writes its own files and draws no window.

    The energy-scaling word is the file read's own: the model measurement
    builds its matrix in physics units directly and would read that word as the
    name of a family.
    """
    model = _code(_function_body(exporter_source, "local_response_of_model"))
    calls = _sampling_calls(exporter_source, "measbpmresp")

    assert calls == [["'Model'", "'Struct'", "'Physics'", "'NoArchive'", "'NoDisplay'"]]
    assert "'NoEnergyScaling'" not in model
    assert "file = '';" in model, "no file answered, and the read leaves it empty"


def test_a_response_answer_that_is_no_matrix_is_refused_by_name(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_response"))

    assert "isempty(S) || ~isstruct(S)" in body
    assert re.search(r"error\('mml_export:response'", body)


def test_a_response_block_names_the_two_families_it_was_measured_between(
    exporter_source: str,
) -> None:
    """A block is identified by its families, never by its place in the grid."""
    blocks = _code(_function_body(exporter_source, "local_response_blocks"))
    side = _code(_function_body(exporter_source, "local_response_side"))

    assert "local_response_block(S(m, a))" in blocks
    assert "side.family = local_text(local_field(body, 'FamilyName'));" in side
    assert "side.device_list = devices;" in side
    assert "'x'" not in blocks and "'y'" not in blocks, "the planes are the facility's own"


def test_a_response_block_records_the_matrix_and_its_operating_point(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_response_block"))
    side = _code(_function_body(exporter_source, "local_response_side"))

    for key in (
        "block.monitor =",
        "block.actuator =",
        "block.origin =",
        "block.timestamp =",
        "block.gev =",
        "block.units =",
        "block.units_string =",
        "block.modulation_method =",
        "block.actuator_delta =",
        "block.data =",
    ):
        assert key in body, key
    assert "side.mode = local_text(local_field(body, 'Mode'));" in side
    assert "side.data = local_response_rounded(" in side, "the point the matrix is a secant about"


def test_the_response_is_measured_unless_a_mode_says_the_model(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_response_origin"))

    assert "origin = 'measured';" in body
    assert "origin = 'model';" in body
    assert body.index("origin = 'measured';") < body.index("origin = 'model';")
    assert "strcmpi(monitorMode, {'Simulator', 'Model'})" in body
    assert "strcmpi(actuatorMode, {'Simulator', 'Model'})" in body


def test_a_response_side_with_no_status_counts_every_device_good(
    exporter_source: str,
) -> None:
    """A file written before the flag existed is not a file saying nothing is good."""
    body = _code(_function_body(exporter_source, "local_response_status"))

    assert "status = ones(nDev, 1);" in body
    assert "isscalar(status)" in body
    assert "status = status * ones(nDev, 1);" in body, "one flag covers the whole side"
    assert "numel(status) ~= nDev" in body
    assert re.search(r"error\('mml_export:response'", body)


def test_a_status_kept_as_true_and_false_is_still_a_status(exporter_source: str) -> None:
    """Read as no flags at all, a logical side's bad devices would come out good."""
    body = _code(_function_body(exporter_source, "local_response_status"))

    assert "~(isnumeric(status) || islogical(status))" in body
    assert "status = double(status(:));" in body


def test_the_operating_point_is_held_to_the_device_list(exporter_source: str) -> None:
    """A column of another length converts each row about another device's setting."""
    side = _code(_function_body(exporter_source, "local_response_side"))
    point = _code(_function_body(exporter_source, "local_response_point"))

    assert "side.data = local_response_rounded(local_response_point(body, name, nDev));" in side
    assert "isscalar(point) && isnan(point)" in point, "an absent point is not a wrong one"
    assert "numel(point) ~= nDev" in point
    assert re.search(r"error\('mml_export:response'", point)


def test_a_response_matrix_that_does_not_line_up_with_its_devices_is_refused(
    exporter_source: str,
) -> None:
    """The read fills a missing device with a row of nothing and then indexes down."""
    body = _code(_function_body(exporter_source, "local_response_matrix"))

    assert "size(data, 1) ~= nMonitor" in body
    assert "size(data, 2) ~= nActuator" in body
    assert re.search(r"error\('mml_export:response'", body)


def test_a_response_side_with_no_device_list_is_refused_by_name(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_response_side"))

    assert "isempty(devices) || ~isnumeric(devices)" in body
    assert body.index("isempty(devices)") < body.index("side.family")


def test_every_measured_response_number_carries_six_significant_digits(
    exporter_source: str,
) -> None:
    """The last digits of a measurement are noise, and a double's worth of them is
    megabytes of text the consumer compares nothing against."""
    rounded = _code(_function_body(exporter_source, "local_response_rounded"))
    digits = _code(_function_body(exporter_source, "local_response_digits"))
    block = _code(_function_body(exporter_source, "local_response_block"))
    side = _code(_function_body(exporter_source, "local_response_side"))

    assert "round(double(x), local_response_digits(), 'significant')" in rounded
    assert "n = 6;" in digits
    for key in ("block.gev", "block.actuator_delta", "block.data"):
        assert f"{key} = local_response_rounded(" in block, key
    identities = side[: side.index("side.data")]
    assert "local_response_rounded" not in identities, "a device list is not a measurement"


def test_a_non_finite_response_number_is_recorded_rather_than_refused(
    exporter_source: str,
) -> None:
    """A device the file does not hold is filled with values that are not numbers."""
    body = _code(_function_body(exporter_source, "local_response_number"))

    assert "value = NaN;" in body
    assert "isfinite" not in body


def test_the_response_timestamp_is_written_in_the_exports_own_spelling(
    exporter_source: str,
) -> None:
    body = _code(_function_body(exporter_source, "local_response_timestamp"))

    assert "datestr(datenum(value), 'yyyy-mm-ddTHH:MM:SS')" in body
    assert "catch" in body, "a stamp datenum refuses costs the matrix nothing"
    assert body.count("text = '';") >= 2


def test_an_absent_label_is_no_text_rather_than_an_index_error(
    exporter_source: str,
) -> None:
    """local_field answers an absent field with '', and reading row one of that errors.

    Every label of a stored response matrix - the units, the modulation
    method, the mode each side was read in - is a field a file may not carry.
    """
    body = _code(_function_body(exporter_source, "local_text"))

    assert "ischar(value) && ~isempty(value)" in body
    assert body.index("text = '';") < body.index("value(1, :)")


# ---------------------------------------------------------------------------
# Dialect: what the script writes imports with no flags
# ---------------------------------------------------------------------------

#: One family spelled exactly as ``mml_export.m`` encodes it: a handle as
#: ``{"$fn", "file"}``, an N-row matrix with non-finite entries as rows of mixed
#: numbers and strings, char-matrix rows deblanked, logicals already 0/1.
_EXPORTED_AO = """{"_export":{"exporter":"mml_export 1.0.0","matlab":"24.2.0.1 (R2024b)",\
"machine":"Quokka","submachine":"BOOSTER","timestamp":"2026-01-01T00:00:00"},\
"QF":{"FamilyName":"QF","MemberOf":["QF","Magnet"],"DeviceList":[[1,1],[1,2]],\
"Status":[1,0],"CommonNames":["qf-1","qf-2"],"Position":[1.5,2.5],\
"Setpoint":{"MemberOf":["QF","Setpoint"],"Mode":"Simulator","DataType":"Scalar",\
"ChannelNames":["QK:B:QF1:SP",""],"Units":"Hardware","HWUnits":"A","PhysicsUnits":"1/m^2",\
"Range":[["-Inf","Inf"],[0,"NaN"]],\
"HW2PhysicsFcn":{"$fn":"amp2k","file":"/mml/quokka/amp2k.m"}}}}"""

_EXPORTED_AD = """{"_export":{"exporter":"mml_export 1.0.0","matlab":"24.2.0.1 (R2024b)",\
"machine":"Quokka","submachine":"BOOSTER","timestamp":"2026-01-01T00:00:00"},\
"Machine":"Quokka","SubMachine":"BOOSTER","Energy":"Inf"}"""


def test_the_exported_dialect_imports_with_no_flags(tmp_path: Path) -> None:
    ao_path = tmp_path / "quokka.booster.ao.json"
    ao_path.write_text(_EXPORTED_AO, encoding="utf-8")
    (tmp_path / "quokka.booster.ad.json").write_text(_EXPORTED_AD, encoding="utf-8")

    loaded = load_json(ao_path)

    assert loaded.system_keyed is False
    assert "_export" not in loaded.ao
    assert loaded.export["submachine"] == "BOOSTER"
    assert loaded.ad["SubMachine"] == "BOOSTER"
    assert resolve_system(loaded, None) == "BOOSTER"


def test_the_exported_dialect_is_already_canonical(tmp_path: Path) -> None:
    """Normalising the exporter's output changes nothing but blank slots."""
    ao_path = tmp_path / "quokka.booster.ao.json"
    ao_path.write_text(_EXPORTED_AO, encoding="utf-8")

    body = load_json(ao_path).ao["QF"]
    normalized = normalize_family(body)

    setpoint = normalized["Setpoint"]
    assert setpoint["HW2PhysicsFcn"] == {"$fn": "amp2k", "file": "/mml/quokka/amp2k.m"}
    assert setpoint["Range"] == [["-Inf", "Inf"], [0, "NaN"]]
    assert setpoint["ChannelNames"] == ["QK:B:QF1:SP", None]
    assert {k: v for k, v in normalized.items() if k != "Setpoint"} == {
        k: v for k, v in body.items() if k != "Setpoint"
    }
