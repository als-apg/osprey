"""``model_response_matrix``: a served model's orbit response, solved in a child process.

The SR cases read the built demo's render; the line cases read a three-cell
synthetic render. The tool's own cases run the real executor: the generated
script, the readonly sandbox and its child process. The child's solve is the
engine plug-in's ``response_matrix``, so the tool's matrix must equal the
plug-in called directly with the same steps.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402

from osprey.facility.views.simulator import simulator_wiring  # noqa: E402
from osprey.mcp_server.python_executor import executor  # noqa: E402
from osprey.mcp_server.python_executor.tools import (  # noqa: E402
    model_response_matrix as tool,
)
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.response import RESULT_TAG, solve_and_print  # noqa: E402

CORRECTOR_X = "SR:MAG:HCM:01:CURRENT:SP"
CORRECTOR_Y = "SR:MAG:VCM:01:CURRENT:SP"

LINE = "LINE"
LINE_KICK = "COR_1:X:SP"
LINE_TWISS = {
    "beta": [5.0, 3.0],
    "alpha": [0.1, -0.2],
    "closed_orbit": [1.0e-4, 2.0e-5, -5.0e-5, 1.0e-5],
}

#: Wide enough for a cold child to import pyAT and solve two SR columns.
TIMEOUT_S = 120

#: The executor timeout the spinning plug-in is stopped at.
SPIN_TIMEOUT_S = 15


def _envelope(raised: pytest.ExceptionInfo[ToolError]) -> dict[str, Any]:
    envelope: dict[str, Any] = json.loads(str(raised.value))
    return envelope


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sr_render(built_control_assistant: Any) -> Path:
    """The built demo's render, read and never written."""
    return Path(built_control_assistant.build_dir)


def _copy_render(source: Path, target: Path) -> Path:
    """The parts of a render the tool reads, copied so a test may edit them."""
    target.mkdir(parents=True)
    shutil.copy2(source / "facility.json", target / "facility.json")
    shutil.copytree(source / "data" / "simulator", target / "data" / "simulator")
    return target


def _set_step(render: Path, model: str, value: Any) -> None:
    """Set ``corrector_delta`` in a copied render's facility file; ``None`` drops it."""
    path = render / "facility.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for record in document["models"]:
        if record["name"] == model:
            measurement = record.setdefault("measurement", {})
            if value is None:
                measurement.pop(tool.STEP_KEY, None)
            else:
                measurement[tool.STEP_KEY] = value
    path.write_text(json.dumps(document), encoding="utf-8")


def _line_render(root: Path, corrector_delta: float | None) -> Path:
    """A render serving one three-cell line, a corrector and a monitor in each cell."""
    view = root / "data" / "simulator"
    (view / "decks").mkdir(parents=True)
    elements: list[Any] = []
    for cell in (1, 2, 3):
        elements += [
            at.Drift(f"DA_{cell}", 0.5),
            at.Quadrupole(f"QF_{cell}", 0.3, 1.2),
            at.Drift(f"DB_{cell}", 0.5),
            at.Corrector(f"COR_{cell}", 0.0, [0.0, 0.0]),
            at.Monitor(f"BPM_{cell}"),
            at.Drift(f"DC_{cell}", 0.5),
            at.Quadrupole(f"QD_{cell}", 0.3, -1.1),
        ]
    lattice = at.Lattice(elements, energy=3e9, particle="electron", periodicity=1)
    at.save_lattice(lattice, str(view / "decks" / f"{LINE}.json"))
    wiring: list[dict[str, Any]] = [
        {
            "id": f"{LINE}/BPM_{cell}:{axis.upper()}",
            "address": f"BPM_{cell}:{axis.upper()}",
            "direction": "read",
            "element": f"BPM_{cell}",
            "engine": {"axis": axis},
            "role": "monitor",
            "plane": axis,
            "refresh": "pass",
        }
        for cell in (1, 2, 3)
        for axis in ("x", "y")
    ]
    wiring.append(
        {
            "id": f"{LINE}/{LINE_KICK}",
            "address": LINE_KICK,
            "direction": "write",
            "element": "COR_1",
            "engine": {"attribute": "KickAngle", "index": 0},
            "default": 0.0,
            "role": "setpoint",
            "plane": "x",
            "refresh": "pass",
        }
    )
    channels = [
        {
            "address": record["address"],
            "role": "setpoint" if record["direction"] == "write" else "readback",
            "pair": record["address"] if record["direction"] == "write" else None,
            "value_type": "float",
            "unit": None,
            "description": None,
            "writable": record["direction"] == "write",
            "value_range": None,
            "owner": LINE,
            "on": {"device": f"{LINE}/{record['element']}"},
        }
        for record in wiring
    ]
    files = {
        "addresses.json": {
            "schema": "osprey.facility.addresses/1",
            "channels": sorted(channel["address"] for channel in channels),
            "status": [],
        },
        "served_models.json": {"schema": "osprey.facility.served_models/1", "models": [LINE]},
        "variables.json": {
            "schema": "osprey.facility.simulator/2",
            "code": "TEST",
            "models": [
                {
                    "name": LINE,
                    "engine": "pyat",
                    "served": True,
                    "settings": {"pyat": {"solve": "single_pass", "twiss_in": LINE_TWISS}},
                    "deck": f"decks/{LINE}.json",
                    "wiring": wiring,
                }
            ],
            "channels": channels,
        },
    }
    for name, document in files.items():
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    measurement = {} if corrector_delta is None else {tool.STEP_KEY: corrector_delta}
    (root / "facility.json").write_text(
        json.dumps({"models": [{"name": LINE, "measurement": measurement}]}), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# A real executor, reading the render the test names
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A deployment the executor runs readonly children in, rendering from ``render``.

    Set ``state["render"]`` to the render the tool reads; ``state["folders"]``
    collects each execution folder handed out, and ``state["seed"]`` maps file
    names to text each new folder receives first.
    """
    from osprey_connectors import posture_store
    from osprey_connectors.workspace import reset_config_cache

    root = tmp_path / "deployment"
    (root / "var" / "agent_data").mkdir(parents=True)
    config = {
        "project_root": str(root),
        "agent_data": {"base_dir": "var/agent_data"},
        "approval": {"enabled": False},
        "control_system": {"type": "epics", "writes_enabled": False},
        "python_executor": {"execution_timeout_seconds": TIMEOUT_S},
    }
    config_path = root / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root / "var" / "agent_data"))
    reset_config_cache()

    state: dict[str, Any] = {"render": None, "folders": [], "seed": {}, "timeout": TIMEOUT_S}

    def stamp(sandbox_env: dict[str, str]) -> str:
        sandbox_env["OSPREY_CONTROL_TARGET"] = "live"
        return "live"

    def folder() -> Path:
        made = root / "executions" / f"run{len(state['folders'])}"
        (made / "figures").mkdir(parents=True)
        for name, text in state["seed"].items():
            (made / name).write_text(text, encoding="utf-8")
        state["folders"].append(made)
        return made

    monkeypatch.setattr(executor, "_resolve_project_root", lambda: root)
    monkeypatch.setattr(executor, "resolve_agent_interpreter", lambda root=None: sys.executable)
    monkeypatch.setattr(executor, "_apply_target_stamp", stamp)
    monkeypatch.setattr(executor, "_create_execution_folder", folder)
    monkeypatch.setattr(
        executor,
        "_read_config",
        lambda: {"execution_method": "subprocess", "timeout": state["timeout"]},
    )
    monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: state["render"])
    return state


def _call(**arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(asyncio.run(tool.model_response_matrix(**arguments)))
    return result


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


class TestSolve:
    def test_the_matrix_equals_the_plug_in_called_directly(self, sandbox, sr_render):
        sandbox["render"] = sr_render

        result = _call(model="SR", correctors=[CORRECTOR_X, CORRECTOR_Y])

        facility = json.loads((sr_render / "facility.json").read_text(encoding="utf-8"))
        (record,) = [model for model in facility["models"] if model["name"] == "SR"]
        steps = {column["address"]: column["step"] for column in result["columns"]}
        rows, expected = engine.response_matrix(
            sr_render / "data" / "simulator" / "decks" / "SR.json",
            simulator_wiring(facility, "SR"),
            record.get("settings"),
            [CORRECTOR_X, CORRECTOR_Y],
            steps,
        )
        assert [row["address"] for row in result["rows"]] == rows
        np.testing.assert_allclose(np.asarray(result["matrix"]), expected, rtol=1e-12, atol=0)

    def test_rows_and_columns_name_their_address_and_device(self, sandbox, sr_render):
        sandbox["render"] = sr_render

        result = _call(model="SR", correctors=[CORRECTOR_X, CORRECTOR_Y])

        assert result["model"] == "SR"
        assert [column["address"] for column in result["columns"]] == [CORRECTOR_X, CORRECTOR_Y]
        assert [column["device"] for column in result["columns"]] == ["SR/HCM01", "SR/VCM01"]
        first = result["rows"][0]
        assert first == {"address": "SR:DIAG:BPM:01:POSITION:X", "device": "SR/BPM01"}
        assert all(row["device"] for row in result["rows"])

    def test_each_step_spans_corrector_delta_in_physics(self, sandbox, sr_render):
        from osprey.simulation.engines.calibration import curve_from_record, to_physics

        sandbox["render"] = sr_render
        request = tool.resolve_model_request("SR", [CORRECTOR_X], render_root=sr_render)
        facility = json.loads((sr_render / "facility.json").read_text(encoding="utf-8"))
        (record,) = [
            entry for entry in simulator_wiring(facility, "SR") if entry["address"] == CORRECTOR_X
        ]
        curve = curve_from_record(record["calibration"]["curve"])
        width = request.steps[CORRECTOR_X]

        span = to_physics(curve, 0.5 * width) - to_physics(curve, -0.5 * width)

        assert span == pytest.approx(request.corrector_delta, rel=1e-12)
        (model,) = [entry for entry in facility["models"] if entry["name"] == "SR"]
        assert request.corrector_delta == model["measurement"][tool.STEP_KEY]

    def test_omitted_correctors_are_every_wired_corrector_x_then_y(self, sr_render):
        request = tool.resolve_model_request("SR", None, render_root=sr_render)

        planes = [address.split(":")[2] for address in request.correctors]
        assert planes == sorted(planes, key=lambda name: name != "HCM")
        assert len(request.correctors) == 144


class TestSolveErrors:
    def test_an_oversized_step_on_a_line_names_the_corrector(self, sandbox, tmp_path):
        sandbox["render"] = _line_render(tmp_path / "line", corrector_delta=1.0)

        with pytest.raises(ToolError) as raised:
            _call(model=LINE)

        envelope = _envelope(raised)
        assert envelope["error_type"] == "execution_error"
        assert envelope["error_message"].startswith(f"the response of {LINE} has no solve: lost")
        assert envelope["error_message"].endswith(f"(while stepping {LINE_KICK})")

    def test_a_destabilising_step_on_sr_names_the_corrector(self, sandbox, sr_render, tmp_path):
        render = _copy_render(sr_render, tmp_path / "sr")
        _set_step(render, "SR", 0.5)
        sandbox["render"] = render

        with pytest.raises(ToolError) as raised:
            _call(model="SR", correctors=[CORRECTOR_X])

        message = _envelope(raised)["error_message"]
        assert message.startswith("the response of SR has no solve:")
        assert message.endswith(f"(while stepping {CORRECTOR_X})")

    def test_a_result_line_without_a_matrix_ends_without_a_result(self, sr_render, monkeypatch):
        async def solved(**_: Any) -> executor.ExecutionResult:
            line = json.dumps({"rows": ["SR:BPM:01:X:RB"]})
            return executor.ExecutionResult(
                success=True, stdout=f"{RESULT_TAG} {line}\n", stderr=""
            )

        monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: sr_render)
        monkeypatch.setattr(executor, "execute_code", solved)

        with pytest.raises(ToolError) as raised:
            _call(model="SR", correctors=[CORRECTOR_X])

        envelope = _envelope(raised)
        assert envelope["error_type"] == "execution_error"
        assert envelope["error_message"] == "the solve of SR ended without a result"

    def test_an_orbit_solver_that_raises_names_the_corrector(self, sr_render, monkeypatch):
        original = at.find_orbit6
        calls = {"count": 0}

        def raiser(*args: Any, **kwargs: Any) -> Any:
            calls["count"] += 1
            if calls["count"] == 3:
                raise at.AtError("no synchronous phase")
            return original(*args, **kwargs)

        monkeypatch.setattr(at, "find_orbit6", raiser)
        request = tool.resolve_model_request(
            "SR", [CORRECTOR_X, CORRECTOR_Y], render_root=sr_render
        )
        printed = io.StringIO()

        with contextlib.redirect_stdout(printed):
            solve_and_print(tool._child_payload(request))

        tag, payload = printed.getvalue().strip().split(" ", 1)
        assert tag == RESULT_TAG
        error = json.loads(payload)["error"]
        assert error.startswith("closed orbit solve raised AtError:")
        assert error.endswith(f"(while stepping {CORRECTOR_Y})")


class TestRefusals:
    def test_a_model_without_a_corrector_delta_is_refused_naming_its_file(self, tmp_path):
        render = _line_render(tmp_path / "line", corrector_delta=None)

        with pytest.raises(ToolError) as raised:
            tool.resolve_model_request(LINE, None, render_root=render)

        message = _envelope(raised)["error_message"]
        assert message.startswith(f"measurement/{LINE}.yaml states no corrector_delta")

    def test_an_unserved_or_unknown_model_is_refused_listing_the_served(self, sr_render):
        with pytest.raises(ToolError) as raised:
            tool.resolve_model_request("ELSEWHERE", None, render_root=sr_render)

        assert _envelope(raised)["error_message"] == (
            "model 'ELSEWHERE' is not a served deck model of this render; the served ones: LINE, SR"
        )

    @pytest.mark.parametrize(
        ("correctors", "fragment"),
        [
            (["SR:DIAG:BPM:01:POSITION:X"], "is no corrector setpoint SR wires"),
            ([CORRECTOR_X, CORRECTOR_X], "appears more than once"),
            ([], "correctors is empty"),
            (CORRECTOR_X, "must be a list"),
        ],
    )
    def test_a_corrector_list_the_model_cannot_step_is_refused(
        self, sr_render, correctors, fragment
    ):
        with pytest.raises(ToolError) as raised:
            tool.resolve_model_request("SR", correctors, render_root=sr_render)

        assert fragment in _envelope(raised)["error_message"]


# ---------------------------------------------------------------------------
# A solve that never returns
# ---------------------------------------------------------------------------

#: A plug-in whose solve holds the interpreter in pure Python until it is killed.
SPIN_ENGINE = textwrap.dedent(
    """
    import os
    from pathlib import Path


    def response_matrix(deck, wiring, settings, correctors, step):
        (Path(__file__).parent / "spin.pid").write_text(str(os.getpid()))
        while True:
            pass
    """
)


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_solve_that_never_returns_is_refused_at_the_timeout(
    sandbox, tmp_path, monkeypatch
) -> None:
    sandbox["render"] = _line_render(tmp_path / "line", corrector_delta=1.0e-5)
    sandbox["seed"] = {"spin_engine.py": SPIN_ENGINE}
    sandbox["timeout"] = SPIN_TIMEOUT_S
    monkeypatch.setattr(tool, "_engine_module", lambda engine_name: "spin_engine")
    finished: dict[str, float] = {}

    async def solve() -> None:
        try:
            await tool.model_response_matrix(model=LINE)
        finally:
            finished["solve"] = time.monotonic()

    async def execute() -> Any:
        result = await executor.execute_code("print('alongside')", "readonly", "alongside")
        finished["execute"] = time.monotonic()
        return result

    async def both() -> list[Any]:
        return await asyncio.gather(solve(), execute(), return_exceptions=True)

    refusal, alongside = asyncio.run(both())

    assert isinstance(refusal, ToolError)
    envelope = json.loads(str(refusal))
    assert envelope["error_message"].startswith(
        f"the response of {LINE} was not solved within the executor's timeout"
    )
    assert envelope["details"] == {"model": LINE, "kind": executor.FAILURE_KIND_TIMEOUT}
    pid = int((sandbox["folders"][0] / "spin.pid").read_text())
    assert _gone(pid)
    assert alongside.success, alongside.stderr
    assert alongside.stdout.strip() == "alongside"
    assert finished["execute"] < finished["solve"]
