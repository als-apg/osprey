"""Check B, served: a ``pyaml_measure`` measurement on the virtual accelerator meets pyAT.

A measurement taken on a served machine meets pyAT's exact physics on the same
lattice, at the same operating point, taken at the tool's own steps
(``tests/facility/_model_reference.py``). Each lane here takes one measurement
through the ``pyaml_measure`` tool against a container, then takes the same
measurement, with the same arguments, on a replica of the served machine in
this process: the composite over the view and the active set the container
serves, which runs the same pyAT model on the same deck from the same start
values. The tool
renders the same script for both, so both take the same steps through the
same view; the comparison and its bands are ``tests/facility/test_check_b.py``.

The lanes:

* **The served harvested trees.** Every export the supported-tree registry
  says takes check B (``tests/fixtures/mml/_trees.py``) is installed from its
  export, built and served, and measured for every kind -- the orbit and
  dispersion responses unipolar, the tune and chromaticity responses stepped
  by group, and the chromaticity monitor. The measurement file the import
  seeded names the kinds its groups and instruments allow from a first look;
  the lane lists all five in the deployment, as an operator does.
* **The example's transport line.** ``LINE`` takes the orbit response alone;
  a dispersion measurement on it is refused naming the instrument it lacks.

Each deployment starts in the built-in ``still`` scenario, and before
measuring, each lane asserts that every reading of the model carries no
motion (:func:`tests.va._still.assert_readings_still`). The deployments state
``approval.enabled: false``: what is proven here is the measurement, and the
approved path is ``test_pyaml_measure_guarded.py``'s subject.

The whole directory is opt-in behind ``OSPREY_VA_E2E_ENABLE=1``; the skip is
applied by ``conftest.pytest_collection_modifyitems``.
"""

from __future__ import annotations

import io
import json
import shutil
import uuid
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.mml._trees import exports_taking
from tests.va._still import assert_readings_still
from tests.va.e2e import _guarded_measure as gm

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    # The harvests and their containers are module-scoped; one worker runs
    # every lane, so each tree is installed and booted once.
    pytest.mark.xdist_group("check-b-served"),
]

#: The periodic exports that take check B: ``(tree, stem)``. Read at
#: collection, from the registry's literals.
SERVED_EXPORTS = exports_taking("B", lines=False)

#: Each kind, with the arguments check B measures it under.
MEASUREMENTS: dict[str, dict[str, Any]] = {
    "orm": {"kind": "orm", "modulation": "unipolar"},
    "dispersion": {"kind": "dispersion", "modulation": "unipolar"},
    "trm": {"kind": "trm", "by": "group"},
    "crm": {"kind": "crm", "by": "group"},
    "chromaticity_monitor": {"kind": "chromaticity_monitor"},
}

#: The example facility's transport line, and the channel its deployment probes.
LINE = "LINE"
EXAMPLE_PROBE = "SR:VAC:GAUGE:SR01:PRESSURE:RB"

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
# Five kinds over the one served export the registry names, the line's orbit
# response and its refused dispersion; the guard test itself is the eighth
# item, so a floor of 7 reds on the loss of a single lane.
MIN_COLLECTED_TESTS = 7


def _check_b() -> Any:
    """Check B's comparison module, imported on first use rather than at collection."""
    from tests.facility import test_check_b

    return test_check_b


@dataclass(frozen=True)
class Served:
    """A served deployment and the model check B measures on it."""

    deployment: gm.Deployment
    model: str


def _measured_model(repo: Path) -> str:
    """The one model the deployment's measurement files name."""
    files = sorted((repo / "data" / "facility" / "measurement").glob("*.yaml"))
    assert len(files) == 1, f"expected one measured model, found {[f.name for f in files]}"
    return files[0].stem


@pytest.fixture(scope="module", params=SERVED_EXPORTS, ids=[tree for tree, _ in SERVED_EXPORTS])
def harvested(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Served]:
    """One harvested tree, every kind listed, built and served with approval off."""
    from tests.cli import test_mml_build_recipes as recipes
    from tests.va.e2e import test_mml_trees_boot as boot

    tree, _stem = request.param
    built = boot.harvest_and_build(tree, tmp_path_factory.mktemp(f"check-b-{tree}"))
    model = _measured_model(built.repo)
    gm.set_measurement(built.repo, model, kinds=list(MEASUREMENTS))
    probe = recipes.imported_probe(built.repo)
    with gm.deployed(
        built.repo,
        lambda port: gm.isolation_settings(port, probe, approval=False),
        prefix=f"osprey-va-e2e-check-b-{tree}",
        probe=probe,
    ) as served:
        yield Served(deployment=served, model=model)


@pytest.fixture(scope="module")
def example(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    """The example facility's deployment, built and served with approval off."""
    from tests._builds import init_project

    repo = init_project(tmp_path_factory.mktemp("check-b-example"), "control-assistant", "example")
    with gm.deployed(
        repo,
        lambda port: gm.isolation_settings(port, EXAMPLE_PROBE, approval=False),
        prefix="osprey-va-e2e-check-b-example",
        probe=EXAMPLE_PROBE,
    ) as served:
        yield Served(deployment=served, model=LINE)


# ---------------------------------------------------------------------------
# The replica
# ---------------------------------------------------------------------------


def _readings(served: Served) -> list[str]:
    """Every reading the measured model wires: what a measurement of it compares."""
    bindings = served.deployment.view().bindings(model=served.model, served_only=False)
    return sorted({b.address for b in bindings if b.direction == "read"})


def _limits(deployment: gm.Deployment) -> dict[str, Any]:
    """The limits the served run's runtime checks, by address, from the build's limits view."""
    from osprey_connectors.control_system.limits_validator import LimitsValidator

    limits, _ = LimitsValidator._load_limits_database(
        str(deployment.data_root / "channel_limits.json")
    )
    return dict(limits)


async def _replicate(
    served: Served,
    arguments: dict[str, Any],
    scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Take the same measurement on a replica of the served machine; its file and answer.

    The tool resolves the same arguments against the same render and renders
    the same script; the script runs here, in one guarded run, against a
    composite over the view the container serves, from its start values.
    """
    from osprey.mcp_server.python_executor.executor import ExecutionResult
    from osprey.mcp_server.python_executor.tools import pyaml_measure as tool
    from osprey_connectors.simulation.composite import Composite
    from osprey_connectors.simulation.view import VIEW_RELPATH

    composite = Composite(
        served.deployment.data_root / VIEW_RELPATH.name,
        state_dir=served.deployment.state_dir,
        model_log=False,
    )
    _check_b().ReplicaRuntime(composite, _limits(served.deployment)).install(monkeypatch)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")

    async def in_process(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        folder = scratch / f"replica-{uuid.uuid4().hex[:8]}"
        folder.mkdir(parents=True)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            exec(compile(kwargs["code"], "pyaml_measure", "exec"), {"_execution_dir": str(folder)})
        result = ExecutionResult(
            success=True,
            stdout=out.getvalue(),
            stderr=err.getvalue(),
            execution_time_seconds=0.0,
            execution_folder=folder,
        )
        return result, {}

    monkeypatch.setattr(tool, "run_gated_execution", in_process)
    monkeypatch.setattr(tool, "_save_measurement", lambda result, args, folder: None)
    answer = await gm.call("pyaml_measure", {"model": served.model, **arguments})
    monkeypatch.undo()
    assert answer.error is None, f"the replica refused: {answer.text}"
    assert answer.result is not None
    return _saved(answer.result), answer.result


def _saved(result: dict[str, Any]) -> dict[str, Any]:
    """The measurement file a completed run saved; empty for a kind that answers inline."""
    if result.get("data_file") is None:
        return {}
    document: dict[str, Any] = json.loads(Path(result["data_file"]).read_text(encoding="utf-8"))
    return document


async def _measure_and_compare(
    served: Served, arguments: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment = served.deployment
    assert_readings_still(deployment.view(), _readings(served), deployment.active_scenarios())

    with gm.acting_in(deployment, tmp_path / "agent_data"):
        answer = await gm.call("pyaml_measure", {"model": served.model, **arguments})
        assert answer.error is None, answer.text
        assert answer.result is not None and answer.result["completed"] is True, answer.text
        measured = (_saved(answer.result), answer.result)
        exact = await _replicate(served, arguments, tmp_path / "replica", monkeypatch)

    _check_b().assert_meets(str(arguments["kind"]), measured, exact)


# ---------------------------------------------------------------------------
# The lanes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(MEASUREMENTS))
async def test_a_served_measurement_meets_pyat_at_the_same_steps(
    harvested: Served, kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _measure_and_compare(harvested, MEASUREMENTS[kind], tmp_path, monkeypatch)


async def test_the_lines_orbit_response_meets_pyat_at_the_same_steps(
    example: Served, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _measure_and_compare(example, MEASUREMENTS["orm"], tmp_path, monkeypatch)


async def test_a_dispersion_on_the_line_is_refused_naming_its_instrument(
    example: Served, tmp_path: Path
) -> None:
    with gm.acting_in(example.deployment, tmp_path / "agent_data"):
        answer = await gm.call("pyaml_measure", {"kind": "dispersion", "model": LINE})

    assert answer.error is not None and answer.error["error_type"] == "validation_error"
    assert answer.error["error_message"] == f"dispersion needs rf, which {LINE} does not have"


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_check_b_served.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
