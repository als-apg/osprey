"""``pyaml_measure``'s arguments resolve against the render the server runs in.

Each case renders a small facility with two measured models (``SR`` periodic,
``LINE`` single-pass) and resolves a request against it, the way the tool does
before any script is built: from the directory holding the rendered
``config.yml``, whether that is a host render one zone down in ``build/`` or a
container project that is its own render.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from osprey.facility.render import render_facility_outputs
from osprey.mcp_server.python_executor.tools._pyaml_measure_args import (
    KINDS,
    MeasureArgs,
    resolve_measure_args,
)
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors, with_rf
from tests.mcp_server.conftest import assert_error

pytest.importorskip("pyaml")

#: The measurement file SR is given: every kind its instruments allow.
SR_KINDS = ["orm", "dispersion", "trm"]


def _tree() -> dict[str, Any]:
    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"]["kinds"] = list(SR_KINDS)
    tree["measurement/SR.yaml"] |= {"corrector_delta": 1.0e-5, "frequency_delta": 100.0}
    return tree


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One render of the two measured models, shared by every case."""
    root = tmp_path_factory.mktemp("pyaml-measure-args")
    document, facility = built_document(root / "tree", _tree())
    render = root / "render"
    render.mkdir()
    render_facility_outputs(render, document, {}, facility)
    return render


def _host(rendered: Path, tmp_path: Path) -> Path:
    """A host layout: ``<repo>/profile.yml`` with the render at ``<repo>/build``."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "profile.yml").write_text("name: demo\n", encoding="utf-8")
    shutil.copytree(rendered, repo / "build")
    (repo / "build" / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    return repo / "build" / "config.yml"


def _container(rendered: Path, tmp_path: Path) -> Path:
    """A container layout: the project is its own render, config.yml at its root."""
    project = tmp_path / "project"
    shutil.copytree(rendered, project)
    (project / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    return project / "config.yml"


@pytest.fixture(params=["host", "container"])
def config_path(request: pytest.FixtureRequest, rendered: Path, tmp_path: Path) -> Path:
    return _host(rendered, tmp_path) if request.param == "host" else _container(rendered, tmp_path)


def _resolve(config_path: Path, kind: object = "orm", **given: Any) -> MeasureArgs:
    arguments: dict[str, Any] = {
        "model": "SR",
        "correctors": None,
        "modulation": None,
        "by": None,
    } | given
    return resolve_measure_args(
        kind,
        arguments["model"],
        arguments["correctors"],
        arguments["modulation"],
        arguments["by"],
        config_path=config_path,
        timeout=given.get("timeout", 600),
    )


def _refusal(config_path: Path, kind: object = "orm", **given: Any) -> str:
    with pytest.raises(ToolError) as refused:
        _resolve(config_path, kind, **given)
    envelope = assert_error(str(refused.value), error_type="validation_error")
    return str(envelope["error_message"])


# --- the render root -------------------------------------------------------------


def test_host_and_container_resolve_the_same_relative_view(rendered: Path, tmp_path: Path):
    host = _resolve(_host(rendered, tmp_path / "h"))
    container = _resolve(_container(rendered, tmp_path / "c"))

    assert host.view_path == container.view_path == "data/pyaml/SR/configuration.yaml"
    assert host.config_path == (tmp_path / "h/repo/build/data/pyaml/SR/configuration.yaml")
    assert container.config_path == (tmp_path / "c/project/data/pyaml/SR/configuration.yaml")
    assert host.render_root == tmp_path / "h/repo/build"
    assert container.render_root == tmp_path / "c/project"
    facts = json.loads((rendered / "data/facility_facts.json").read_text(encoding="utf-8"))
    assert host.view_sha256 == container.view_sha256 == facts["measurement_models"]["SR"]["sha256"]


def test_a_host_render_never_reads_the_repo_root(rendered: Path, tmp_path: Path):
    """The repo's own ``data/`` is the facility source, never the render's views."""
    config_path = _host(rendered, tmp_path)
    (config_path.parent.parent / "data" / "pyaml").mkdir(parents=True)

    assert _resolve(config_path).config_path.is_relative_to(config_path.parent)


def test_a_view_recorded_outside_the_render_s_pyaml_folder_is_refused(config_path: Path):
    facts_path = config_path.parent / "data" / "facility_facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    (config_path.parent / "data" / "elsewhere.yaml").write_text("{}", encoding="utf-8")
    facts["measurement_models"]["SR"]["path"] = "data/elsewhere.yaml"
    facts_path.write_text(json.dumps(facts), encoding="utf-8")

    assert "is not under" in _refusal(config_path)


def test_a_view_linked_out_of_the_render_is_refused(config_path: Path, tmp_path: Path):
    view = config_path.parent / "data" / "pyaml" / "SR" / "configuration.yaml"
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(view.read_bytes())
    view.unlink()
    view.symlink_to(outside)

    assert "is not under" in _refusal(config_path)


def test_a_render_with_no_view_has_nothing_to_measure(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yml").write_text("project_name: demo\n", encoding="utf-8")

    assert "carries no pyAML view" in _refusal(project / "config.yml")


# --- the model --------------------------------------------------------------------


def test_the_model_is_required_when_more_than_one_has_a_view(config_path: Path):
    message = _refusal(config_path, model=None)

    assert message == "model is omitted and more than one model has a pyAML view: LINE, SR"


def test_the_model_may_be_omitted_when_one_alone_has_a_view(config_path: Path):
    facts_path = config_path.parent / "data" / "facility_facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    del facts["measurement_models"]["SR"]
    facts_path.write_text(json.dumps(facts), encoding="utf-8")

    assert _resolve(config_path, model=None).model == "LINE"


def test_a_model_without_a_view_is_refused_listing_those_with_one(config_path: Path):
    assert _refusal(config_path, model="BR") == (
        "model 'BR' has no pyAML view; the models that have one: LINE, SR"
    )


# --- the kind ---------------------------------------------------------------------


def test_an_unknown_kind_is_refused_naming_the_kinds(config_path: Path):
    assert _refusal(config_path, kind="bba") == f"kind 'bba' is not one of {', '.join(KINDS)}"


@pytest.mark.parametrize(
    ("kind", "model", "lacking"),
    [
        ("crm", "SR", "sext"),
        ("chromaticity_monitor", "LINE", "tune and rf"),
        ("dispersion", "LINE", "rf"),
        ("trm", "LINE", "tune"),
    ],
)
def test_a_kind_the_model_lacks_is_refused_naming_what_it_needs(
    config_path: Path, kind: str, model: str, lacking: str
):
    assert _refusal(config_path, kind=kind, model=model) == (
        f"{kind} needs {lacking}, which {model} does not have"
    )


def test_a_kind_the_file_does_not_allow_is_refused_naming_the_allowed(config_path: Path):
    """SR has the tune and the RF, but its measurement file leaves the monitor out."""
    assert _refusal(config_path, kind="chromaticity_monitor") == (
        "chromaticity_monitor is not a kind measurement/SR.yaml allows; it allows: "
        "dispersion, orm, trm"
    )


# --- modulation and by ------------------------------------------------------------


def test_omitted_modulation_and_by_are_bipolar_and_magnet(config_path: Path):
    trm = _resolve(config_path, kind="trm")
    orm = _resolve(config_path, kind="orm")

    assert (trm.modulation, trm.by) == ("bipolar", "magnet")
    assert (orm.modulation, orm.by) == ("bipolar", None)


def test_by_group_and_unipolar_are_taken(config_path: Path):
    trm = _resolve(config_path, kind="trm", modulation="unipolar", by="group")

    assert (trm.modulation, trm.by) == ("unipolar", "group")


def test_by_group_with_modulation_omitted_is_unipolar(config_path: Path):
    resolved = _resolve(config_path, kind="trm", by="group")

    assert (resolved.modulation, resolved.by) == ("unipolar", "group")


def test_by_group_with_bipolar_named_is_refused(config_path: Path):
    assert _refusal(config_path, kind="trm", modulation="bipolar", by="group") == (
        "by group steps each group once from its present setting, so it is unipolar; "
        "modulation is bipolar"
    )


@pytest.mark.parametrize("kind", ["orm", "dispersion"])
def test_by_outside_trm_and_crm_is_refused(config_path: Path, kind: str):
    assert _refusal(config_path, kind=kind, by="group").startswith(
        "by is taken by trm and crm alone"
    )


@pytest.mark.parametrize(("name", "value"), [("modulation", "sinus"), ("by", "family")])
def test_an_unknown_modulation_or_by_is_refused(config_path: Path, name: str, value: str):
    assert _refusal(config_path, kind="trm", **{name: value}).startswith(f"{name} {value!r}")


# --- correctors -------------------------------------------------------------------


def test_correctors_are_setpoint_addresses_in_the_order_given(config_path: Path):
    args = _resolve(config_path, correctors=["SCOR:V:SP", "SCOR:H:SP"])

    assert args.correctors == ("SCOR:V:SP", "SCOR:H:SP")


def test_omitted_correctors_step_every_corrector(config_path: Path):
    assert _resolve(config_path).correctors is None


@pytest.mark.parametrize(
    ("correctors", "message"),
    [
        (["QF:SP"], "QF:SP is no corrector setpoint of SR's hcor or vcor group"),
        (["SCOR:H:SP", "SCOR:H:SP"], "corrector setpoint SCOR:H:SP appears more than once"),
        ([], "correctors is empty"),
        ("SCOR:H:SP", "correctors must be a list of setpoint addresses"),
    ],
)
def test_a_corrector_list_that_is_not_the_model_s_setpoints_is_refused(
    config_path: Path, correctors: Any, message: str
):
    assert _refusal(config_path, correctors=correctors).startswith(message)


@pytest.mark.parametrize(
    ("kind", "why"),
    [
        ("dispersion", "dispersion steps the RF, not the correctors"),
        ("trm", "trm steps no named correctors"),
    ],
)
def test_correctors_outside_orm_are_refused(config_path: Path, kind: str, why: str):
    assert _refusal(config_path, kind=kind, correctors=["SCOR:H:SP"]).startswith(
        f"correctors are taken by orm alone; {why}"
    )


# --- the timeout ------------------------------------------------------------------


@pytest.mark.parametrize("timeout", [None, 0, -1, float("inf"), True, "600"])
def test_a_measurement_needs_a_finite_positive_timeout(config_path: Path, timeout: Any):
    assert _refusal(config_path, timeout=timeout).startswith(
        "python_executor.execution_timeout_seconds is"
    )
