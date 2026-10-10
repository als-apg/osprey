"""``osprey sim status`` and ``osprey sim list`` on the simulator view.

``sim list`` names the view's scenarios, marking the active set with ``*``.

``sim status`` connects the deployment's in-process simulator to the simulator view
the build wrote, prints ``<model>: <status>`` per served physics model, then
``log: <absolute path>`` per model, the path being the file the composite
appends to. Overlap records from a model's log follow its ``log:`` line,
prefixed ``(log, …)`` so they never read as the model's status; every other
log record stays out. On a target served by any other connector, each model's
status is read from its status channel through that target's connector.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from osprey.cli.sim import sim_group
from osprey_connectors.control_system.va_in_process_connector import (
    JOURNAL_REJECTED_EVENT,
    simulation_state_dir,
)
from osprey_connectors.simulation import OVERLAP_EVENT
from tests.cli._lifecycle_build import stub_build
from tests.cli._simulator_view import (
    IN_PROCESS_CONFIG,
    SR_ERROR,
    register_stub_engine,
    write_simulator_view,
)

OVERLAP_LINE = "SR (log, instance inprocess, pid 123): scenario-overlap on SR:MAG:HCM:01:CURRENT:SP"


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    register_stub_engine(monkeypatch)


@pytest.fixture
def deployment(lifecycle_repo: Path) -> Path:
    """An in-process deployment whose render holds the simulator view."""
    build = stub_build(lifecycle_repo, config=IN_PROCESS_CONFIG)
    write_simulator_view(build)
    return lifecycle_repo


def _view(repo: Path) -> Path:
    return repo / "build" / "data" / "simulator"


def _activate(repo: Path, *names: str) -> None:
    state = simulation_state_dir(_view(repo))
    state.mkdir(parents=True, exist_ok=True)
    (state / "active_scenarios").write_text("\n".join(names) + "\n", encoding="utf-8")


def _status(repo: Path, *args: str):
    return CliRunner().invoke(sim_group, ["status", "--repo", str(repo), *args])


def _log_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("log: ")]


def test_a_serving_model_reports_ok(deployment: Path) -> None:
    result = _status(deployment)

    assert result.exit_code == 0, result.output
    assert "SR: ok" in result.output.splitlines()


def test_a_failing_sr_scenario_prints_the_engines_error(deployment: Path) -> None:
    _activate(deployment, "nominal", "sr-broken")

    result = _status(deployment)

    assert result.exit_code == 0, result.output
    assert f"SR: {SR_ERROR}" in result.output.splitlines()


def test_the_printed_log_is_the_file_the_composite_appended_to(deployment: Path) -> None:
    result = _status(deployment)

    lines = _log_lines(result.output)
    assert result.exit_code == 0, result.output
    assert len(lines) == 1
    printed = Path(lines[0].removeprefix("log: "))
    assert printed.is_absolute()
    assert printed.samefile(deployment / "var" / "simulator" / "SR.log")
    last = json.loads(printed.read_text(encoding="utf-8").splitlines()[-1])
    assert (last["instance"], last["pid"], last["model"]) == ("inprocess", os.getpid(), "SR")


def test_an_overlap_record_prints_on_its_own_line_and_nothing_else_from_the_log(
    deployment: Path,
) -> None:
    log = deployment / "var" / "simulator" / "SR.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"instance": "inprocess", "pid": 120, "model": "SR", "event": "built", "status": "ok"},
        {"instance": "inprocess", "pid": 121, "model": "SR", "event": "failed", "error": "boom"},
        {
            "instance": "inprocess",
            "pid": 122,
            "model": "SR",
            "event": JOURNAL_REJECTED_EVENT,
            "address": "SR:VAC:PUMP:01:CURRENT:SP",
            "reason": "not writable",
        },
        {
            "instance": "inprocess",
            "pid": 123,
            "model": "SR",
            "event": OVERLAP_EVENT,
            "target": "SR:MAG:HCM:01:CURRENT:SP",
        },
    ]
    text = "\n".join(json.dumps(record) for record in records)
    log.write_text(text + "\nnot json\n", encoding="utf-8")

    result = _status(deployment)

    lines = result.output.splitlines()
    assert result.exit_code == 0, result.output
    assert [line for line in lines if line.startswith("SR")] == ["SR: ok", OVERLAP_LINE]
    assert "boom" not in result.output
    assert "not writable" not in result.output


class _StatusChannels:
    """A connected connector serving status channels as char waveforms."""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.read: list[str] = []
        self.disconnected = False

    async def read_channel(self, address: str, timeout: float | None = None):
        del timeout
        self.read.append(address)
        codes = [*self.values[address].encode("utf-8"), 0, 0]
        return SimpleNamespace(value=codes)

    async def disconnect(self) -> None:
        self.disconnected = True


@pytest.mark.parametrize(
    ("config", "args"),
    [
        (
            "control_system:\n  type: virtual_accelerator\n"
            "  connector:\n    virtual_accelerator:\n      host: localhost\n",
            (),
        ),
        (
            "control_system:\n  type: epics\n"
            "  connector:\n    epics: {}\n    virtual_accelerator:\n      host: localhost\n",
            ("--target", "va"),
        ),
    ],
    ids=["baseline", "flag"],
)
def test_a_virtual_accelerator_target_reports_each_models_status_channel(
    lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch, config: str, args: tuple[str, ...]
) -> None:
    from osprey_connectors.factory import ConnectorFactory

    write_simulator_view(stub_build(lifecycle_repo, config=config))
    connector = _StatusChannels({"ca:SIM:SR:STATUS": SR_ERROR})
    asked: list[str | None] = []

    async def create(section: dict, control_target: str | None = None) -> _StatusChannels:
        del section
        asked.append(control_target)
        return connector

    monkeypatch.setattr(ConnectorFactory, "create_control_system_connector", create)

    result = _status(lifecycle_repo, *args)

    assert result.exit_code == 0, result.output
    assert f"SR: {SR_ERROR}" in result.output.splitlines()
    assert connector.read == ["ca:SIM:SR:STATUS"]
    assert connector.disconnected
    assert asked == [args[1] if args else None]


def test_the_in_process_va_target_answers_from_the_composite(deployment: Path) -> None:
    """``va`` on the simulator in process is the composite, not a status channel."""
    result = _status(deployment, "--target", "va")

    assert result.exit_code == 0, result.output
    assert "SR: ok" in result.output.splitlines()


def test_a_render_without_a_view_is_refused(lifecycle_repo: Path) -> None:
    stub_build(lifecycle_repo, config=IN_PROCESS_CONFIG)

    result = _status(lifecycle_repo)

    assert result.exit_code == 1
    assert "No simulator view" in result.output


def test_a_view_from_another_schema_is_refused_naming_rebuild(deployment: Path) -> None:
    variables = _view(deployment) / "variables.json"
    document = json.loads(variables.read_text(encoding="utf-8"))
    variables.write_text(
        json.dumps({**document, "schema": "osprey.facility.simulator/1"}), encoding="utf-8"
    )

    result = _status(deployment)

    assert result.exit_code == 1
    assert "rebuild" in result.output


def test_list_names_the_views_scenarios_and_marks_the_active_set(deployment: Path) -> None:
    _activate(deployment, "nominal", "sr-broken")

    result = CliRunner().invoke(sim_group, ["list", "--repo", str(deployment)])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "* nominal  (logbook: no)",
        "    Baseline machine.",
        "* sr-broken  (logbook: no)",
        "    SR cannot find a closed orbit.",
    ]


def test_list_loads_no_scenario_apply_code(deployment: Path) -> None:
    code = (
        "import sys\n"
        "from click.testing import CliRunner\n"
        "from osprey.cli.sim import sim_group\n"
        f"result = CliRunner().invoke(sim_group, ['list', '--repo', {str(deployment)!r}])\n"
        "assert result.exit_code == 0, result.output\n"
        "print('osprey.simulation.apply' in sys.modules)\n"
    )

    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "False"
