"""``pyaml_measure`` on a served machine: every write under the run lock, under approval.

A real build of the control-assistant deployment, whose one control target is
the virtual accelerator on loopback, is served by a container over the build's
own data root, and ``pyaml_measure`` is driven the way a session drives it:
the build's rendered approval hook answers the PreToolUse call, and an
approved ``ask`` runs the tool with the input the hook rewrote, under the
tool-use id the hook stamped (``_guarded_measure``).

What the lanes hold:

* **The prompt shows the measurement.** Model, kind, modulation, every
  setpoint the call steps and the pyAML view's digest as the build recorded
  it.
* **A killed run leaves the machine where it stopped.** A measurement whose
  sandbox is SIGKILLed once a stepped setpoint reads displaced leaves that
  setpoint displaced; its failure names the channels it attempted, and the
  next approved measurement lists nothing extra and writes no old value back.
  For the bipolar and unipolar orbit response and the tune response stepped by
  group.
* **A stopped run lists what it moved.** A measurement whose sandbox gets
  ``SIGINT`` once a stepped setpoint reads displaced stops through pyAML's own
  abort path and answers a stopped result whose ``moved`` lists each setpoint
  with its value before the call. ``SIGINT`` is the stop: the executor's own
  cancel sends it, and it lands deterministically on the served machine.
* **What is refused.** A measurement on a held target, naming the holder; a
  call no hook stamped; a pyAML write from ``execute``; a view whose bytes
  differ from the facts record; a measurement while writes are off.

Every deployment here starts in the built-in ``still`` scenario, and each lane
asserts that the setpoints it reads back carry no motion before it compares
them (:func:`tests.va._still.assert_readings_still`). The render configures no
target but the simulator, so no write here can reach hardware.

The whole directory is opt-in behind ``OSPREY_VA_E2E_ENABLE=1``; the skip is
applied by ``conftest.pytest_collection_modifyitems``.
"""

from __future__ import annotations

import json
import shutil
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests._guarded_run import held_lock
from tests.va._still import assert_readings_still
from tests.va.e2e import _guarded_measure as gm

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    # The deployments and their containers are module-scoped; one worker runs
    # every lane, so each is built and booted once.
    pytest.mark.xdist_group("pyaml-measure-guarded"),
]

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
# Three killed runs, the stopped run, the held target, the unstamped call, the
# pyAML write from execute, the edited view, writes off and the render's
# targets; the guard test itself is the eleventh item, so a floor of 10 reds on
# the loss of a single lane.
MIN_COLLECTED_TESTS = 10

#: The model measured: the example facility's periodic lattice.
MODEL = "SR"

#: A channel every container here serves, read to know it is up.
PROBE = "SR:VAC:GAUGE:SR01:PRESSURE:RB"


def _repo(parent: Path, name: str) -> Path:
    """The control-assistant deployment, initialised, with every measurement step settling."""
    from tests._builds import init_project

    repo = init_project(parent, "control-assistant", name)
    gm.set_measurement(repo, MODEL, sleep_between_step=gm.SETTLE_S)
    return repo


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[gm.Deployment]:
    """The deployment whose guarded tools ask, built and served."""
    repo = _repo(tmp_path_factory.mktemp("guarded"), "guarded")
    with gm.deployed(
        repo,
        lambda port, pva_port: gm.isolation_settings(port, pva_port, PROBE, approval=True),
        prefix="osprey-va-e2e-guarded",
        probe=PROBE,
    ) as served:
        yield served


@pytest.fixture
def session(deployment: gm.Deployment, tmp_path: Path) -> Iterator[Path]:
    """This test acting in one container of the deployment; its agent-data root."""
    root = tmp_path / "agent_data_a"
    with gm.acting_in(deployment, root):
        assert gm.holder(deployment) is None, "an earlier lane left the target held"
        yield root
    assert gm.holder(deployment) is None, "this lane left the target held"


# ---------------------------------------------------------------------------
# What the build recorded
# ---------------------------------------------------------------------------


def _measurement_block(deployment: gm.Deployment) -> dict[str, Any]:
    """The model's measurement block the approval hook reads."""
    config = json.loads(
        (deployment.build_dir / ".claude" / "hooks" / "hook_config.json").read_text("utf-8")
    )
    block: dict[str, Any] = config["measurement"][MODEL]
    return block


def _view_digest(deployment: gm.Deployment) -> str:
    """The pyAML view's digest as the facts record states it."""
    facts = json.loads((deployment.data_root / "facility_facts.json").read_text("utf-8"))
    return str(facts["measurement_models"][MODEL]["sha256"])


def _stepped(deployment: gm.Deployment, kind: str) -> list[str]:
    """Every setpoint a measurement of ``kind`` over the whole model steps."""
    groups = _measurement_block(deployment)["groups"]
    roles = {"orm": ("hcor", "vcor"), "trm": ("quad",)}[kind]
    return [str(address) for role in roles for address in groups[role]]


def _assert_still(deployment: gm.Deployment, addresses: Any) -> None:
    assert_readings_still(deployment.view(), addresses, deployment.active_scenarios())


def _assert_still_displaced(deployment: gm.Deployment, left: dict[str, Any]) -> None:
    """Every setpoint the stopped run left displaced still reads where it was left."""
    served = gm.ca_read(deployment.port, list(left))
    assert served == left, f"a setpoint moved after the run stopped: {served} against {left}"


async def _put_back(deployment: gm.Deployment, home: dict[str, Any]) -> None:
    """Write the setpoints a lane displaced back, as the operator's own approved Python would."""
    code = f"import osprey.runtime\nosprey.runtime.write_channels({home!r})\n"
    asked, answer = await gm.approved(
        deployment,
        "execute",
        {"code": code, "description": "put back", "execution_mode": "readwrite"},
    )
    del asked
    assert answer.error is None, answer.text
    assert gm.ca_read(deployment.port, list(home)) == home


#: The next measurement: a short approved one, one corrector.
NEXT = {"kind": "orm", "model": MODEL, "correctors": None}


def _next_input(deployment: gm.Deployment) -> dict[str, Any]:
    return {**NEXT, "correctors": [_stepped(deployment, "orm")[-1]]}


# ---------------------------------------------------------------------------
# Killed after the first write: the machine is left where it stopped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"kind": "orm", "model": MODEL}, id="orm-bipolar"),
        pytest.param({"kind": "orm", "model": MODEL, "modulation": "unipolar"}, id="orm-unipolar"),
        pytest.param({"kind": "trm", "model": MODEL, "by": "group"}, id="trm-by-group"),
    ],
)
async def test_a_killed_measurement_leaves_the_machine_where_it_stopped(
    deployment: gm.Deployment, session: Path, arguments: dict[str, Any]
) -> None:
    del session
    stepped = _stepped(deployment, str(arguments["kind"]))
    _assert_still(deployment, stepped)
    home = gm.ca_read(deployment.port, stepped)

    asked = gm.ask(deployment, "pyaml_measure", arguments)

    assert asked.decision == "ask", asked.reason
    assert f"Model: {MODEL}\nKind: {arguments['kind']}\n" in asked.reason
    assert f"  sha256 {_view_digest(deployment)}\n" in asked.reason
    listed = "".join(f'\n  "{address}"' for address in stepped)
    assert f"Setpoints stepped: {len(stepped)}{listed}" in asked.reason
    assert asked.updated_input["approved_view_sha256"] == _view_digest(deployment)

    answer, left = await gm.signalled_after_first_write(
        deployment,
        "pyaml_measure",
        asked.updated_input,
        asked.tool_use_id,
        stepped,
        signal.SIGKILL,
    )

    assert answer.error is not None, answer.text
    attempted = answer.error["details"]["written_channels"]
    displaced = {address: value for address, value in left.items() if value != home[address]}
    assert displaced and set(displaced) <= set(attempted) <= set(stepped), answer.text
    assert "left where the run stopped" in answer.error["error_message"], answer.text
    try:
        _assert_still_displaced(deployment, left)
        following = gm.ask(deployment, "pyaml_measure", _next_input(deployment))
        assert following.decision == "ask", following.reason
        assert "Setpoints stepped: 1\n" in following.reason, following.reason
        assert "interrupted run" not in following.reason
        next_answer = await gm.call("pyaml_measure", following.updated_input, following.tool_use_id)
        assert next_answer.error is None, next_answer.text
        assert next_answer.result is not None and next_answer.result["completed"] is True
        untouched = {
            a: v for a, v in left.items() if a not in _next_input(deployment)["correctors"]
        }
        _assert_still_displaced(deployment, untouched)
    finally:
        await _put_back(deployment, home)


async def test_a_stopped_measurement_lists_each_setpoint_it_moved_with_its_value_before(
    deployment: gm.Deployment, session: Path
) -> None:
    del session
    stepped = _stepped(deployment, "orm")
    _assert_still(deployment, stepped)
    home = gm.ca_read(deployment.port, stepped)
    asked = gm.ask(deployment, "pyaml_measure", {"kind": "orm", "model": MODEL})
    assert asked.decision == "ask", asked.reason

    answer, left = await gm.signalled_after_first_write(
        deployment,
        "pyaml_measure",
        asked.updated_input,
        asked.tool_use_id,
        stepped,
        signal.SIGINT,
    )

    try:
        assert answer.error is None, answer.text
        assert answer.result is not None and answer.result["completed"] is False, answer.text
        moved = {entry["address"]: entry["before"] for entry in answer.result["moved"]}
        assert moved, answer.text
        assert set(moved) <= set(stepped)
        for address, before in moved.items():
            assert before == pytest.approx(home[address], abs=1e-12), address
        assert "GUARDED_RUN_RESTORE" not in answer.text
        _assert_still_displaced(deployment, left)
    finally:
        await _put_back(deployment, home)


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


async def test_a_measurement_on_a_held_target_is_denied_naming_its_holder(
    deployment: gm.Deployment, session: Path
) -> None:
    del session
    with held_lock(gm.VA_TARGET, root=deployment.repo) as holder:
        asked = gm.ask(deployment, "pyaml_measure", {"kind": "orm", "model": MODEL})

    assert asked.decision == "deny", asked.reason
    assert asked.reason.startswith(
        f"a guarded run is in progress on {gm.VA_TARGET} (pid {holder.pid}, started "
    ), asked.reason


async def test_a_call_no_approval_prompt_stamped_is_refused_before_anything_runs(
    deployment: gm.Deployment, session: Path
) -> None:
    del session
    address = _stepped(deployment, "orm")[0]
    home = gm.ca_read(deployment.port, [address])

    answer = await gm.call(
        "pyaml_measure", {"kind": "orm", "model": MODEL, "correctors": [address]}
    )

    assert answer.error is not None
    assert answer.error["details"]["kind"] == "approval_missing", answer.text
    assert gm.ca_read(deployment.port, [address]) == home


async def test_a_pyaml_write_from_execute_is_refused_with_no_put(
    deployment: gm.Deployment, session: Path
) -> None:
    del session
    address = _stepped(deployment, "orm")[0]
    home = gm.ca_read(deployment.port, [address])
    code = (
        "from pyaml.accelerator import Accelerator\n"
        f"sr = Accelerator.load({str(deployment.pyaml_view(MODEL))!r})\n"
        f"magnet = sr.live.magnet.get(name={address!r})\n"
        "magnet.strength.set(magnet.strength.get() + 1.0e-5)\n"
    )
    asked = gm.ask(
        deployment,
        "execute",
        {"code": code, "description": "pyAML write", "execution_mode": "readwrite"},
    )

    answer = await gm.call(
        "execute", {**asked.updated_input, "save_output": False}, asked.tool_use_id
    )

    assert "OspreyWriteRefused" in answer.text, answer.text
    assert gm.ca_read(deployment.port, [address]) == home


async def test_a_view_whose_bytes_differ_from_the_facts_record_is_denied(
    deployment: gm.Deployment, session: Path
) -> None:
    del session
    view = deployment.pyaml_view(MODEL)
    original = view.read_bytes()
    view.write_bytes(original + b"# edited\n")
    try:
        asked = gm.ask(deployment, "pyaml_measure", {"kind": "orm", "model": MODEL})
    finally:
        view.write_bytes(original)

    assert asked.decision == "deny", asked.reason
    assert f"the pyAML view of {MODEL} differs from the one the build recorded" in asked.reason


async def test_with_writes_off_a_measurement_is_refused(
    deployment: gm.Deployment, session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del session
    address = _stepped(deployment, "orm")[0]
    home = gm.ca_read(deployment.port, [address])
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    answer = await gm.call(
        "pyaml_measure", {"kind": "orm", "model": MODEL, "correctors": [address]}
    )

    assert answer.error is not None and answer.error["error_type"] == "safety_error", answer.text
    assert gm.ca_read(deployment.port, [address]) == home


def test_the_render_configures_no_target_but_the_simulator(deployment: gm.Deployment) -> None:
    assert gm.only_simulated_targets(deployment.config) == []
    assert gm.STILL in deployment.active_scenarios()


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_pyaml_measure_guarded.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
