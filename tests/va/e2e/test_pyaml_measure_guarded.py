"""``pyaml_measure`` on a served machine: every write inside one guarded run, under approval.

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
* **A killed run is restored by the next approved run.** A measurement whose
  sandbox is killed after its first write leaves the journal of the
  setpoints it displaced; the next call's prompt lists it, and approving that
  call writes every one back before the measurement starts. For the bipolar
  and unipolar orbit response and the tune response stepped by group.
* **Whose journal does not matter.** A journal left by another identity's
  killed run, and one left by a killed run of another container of the
  deployment (another agent-data root), are listed and restored alike.
* **An approval binds the journal it listed.** A journal planted after the
  prompt refuses the run with ``OspreyJournalChanged`` and no write, and
  leaves the journal byte for byte; an approved ``execute`` whose prompt lists
  that journal restores it without a write, every value being already back.
* **Without approval** a killed run's journal is restored by the next run.
* **What is refused.** A headless dispatch run meeting a pending journal; a
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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

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
# Three killed runs, two foreign journals, approval off, the planted journal,
# dispatch, the unstamped call, the pyAML write from execute, the edited view,
# writes off and the render's targets; the guard test itself is the
# fourteenth item, so a floor of 13 reds on the loss of a single lane.
MIN_COLLECTED_TESTS = 13

#: The model measured: the example facility's periodic lattice.
MODEL = "SR"

#: A channel every container here serves, read to know it is up.
PROBE = "SR:VAC:GAUGE:SR01:PRESSURE:RB"

#: The identity a planted journal names: not the one this module's runs act as.
OTHER_OPERATOR = "operator-b"


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
        lambda port: gm.isolation_settings(port, PROBE, approval=True),
        prefix="osprey-va-e2e-guarded",
        probe=PROBE,
    ) as served:
        yield served


@pytest.fixture(scope="module")
def unasked(tmp_path_factory: pytest.TempPathFactory) -> Iterator[gm.Deployment]:
    """The same deployment with approval off, built and served."""
    repo = _repo(tmp_path_factory.mktemp("unasked"), "unasked")
    with gm.deployed(
        repo,
        lambda port: gm.isolation_settings(port, PROBE, approval=False),
        prefix="osprey-va-e2e-unasked",
        probe=PROBE,
    ) as served:
        yield served


@pytest.fixture
def session(deployment: gm.Deployment, tmp_path: Path) -> Iterator[Path]:
    """This test acting in one container of the deployment; its agent-data root."""
    root = tmp_path / "agent_data_a"
    with gm.acting_in(deployment, root):
        assert gm.pending(deployment) is None, "an earlier lane left a journal pending"
        yield root
    assert gm.pending(deployment) is None, "this lane left a journal pending"


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


def _assert_journal_listed(reason: str, journal: Any) -> None:
    """The prompt names whose run left the journal and lists every value in it."""
    assert f'"{journal.identity}"\'s interrupted run (pid {journal.pid}' in reason, reason
    assert f"left {len(journal.values)} setpoints displaced" in reason, reason
    for address, value in journal.values.items():
        assert f'  "{address}" = {json.dumps(value)}' in reason, reason


def _assert_back(deployment: gm.Deployment, journal: Any) -> None:
    """Every journaled setpoint reads its journaled value and the journal is clear."""
    served = gm.ca_read(deployment.port, list(journal.values))
    off = {
        address: (served[address], value)
        for address, value in journal.values.items()
        if served[address] != pytest.approx(value, abs=1e-12)
    }
    assert not off, f"setpoints not restored to their journaled value: {off}"
    assert gm.pending(deployment) is None


def _assert_displaced(deployment: gm.Deployment, journal: Any) -> None:
    """The killed run left at least one journaled setpoint away from its journaled value."""
    served = gm.ca_read(deployment.port, list(journal.values))
    assert any(
        served[address] != pytest.approx(value, abs=1e-12)
        for address, value in journal.values.items()
    ), f"the kill landed before any write: {served} against {journal.values}"


#: The replay: a short approved measurement, one corrector, which restores
#: the journal before it steps anything.
REPLAY = {"kind": "orm", "model": MODEL, "correctors": None}


def _replay_input(deployment: gm.Deployment) -> dict[str, Any]:
    return {**REPLAY, "correctors": [_stepped(deployment, "orm")[-1]]}


async def _replay(deployment: gm.Deployment, journal: Any) -> None:
    """Ask for the replay, hold its prompt to the journal, approve it and check the restore."""
    asked = gm.ask(deployment, "pyaml_measure", _replay_input(deployment))
    assert asked.decision == "ask", asked.reason
    _assert_journal_listed(asked.reason, journal)
    answer = await gm.call("pyaml_measure", asked.updated_input, asked.tool_use_id)
    assert answer.error is None, answer.text
    assert answer.result is not None and answer.result["completed"] is True, answer.text
    _assert_back(deployment, journal)


# ---------------------------------------------------------------------------
# Killed after the first write, restored by the next approved run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"kind": "orm", "model": MODEL}, id="orm-bipolar"),
        pytest.param({"kind": "orm", "model": MODEL, "modulation": "unipolar"}, id="orm-unipolar"),
        pytest.param({"kind": "trm", "model": MODEL, "by": "group"}, id="trm-by-group"),
    ],
)
async def test_a_killed_measurement_is_restored_by_the_next_approved_run(
    deployment: gm.Deployment, session: Path, arguments: dict[str, Any]
) -> None:
    del session
    stepped = _stepped(deployment, str(arguments["kind"]))
    _assert_still(deployment, stepped)

    asked = gm.ask(deployment, "pyaml_measure", arguments)

    assert asked.decision == "ask", asked.reason
    assert f"Model: {MODEL}\nKind: {arguments['kind']}\n" in asked.reason
    assert f"  sha256 {_view_digest(deployment)}\n" in asked.reason
    listed = "".join(f'\n  "{address}"' for address in stepped)
    assert f"Setpoints stepped: {len(stepped)}{listed}" in asked.reason
    assert asked.updated_input["approved_view_sha256"] == _view_digest(deployment)

    journal, answer = await gm.killed_after_first_write(
        deployment, "pyaml_measure", asked.updated_input, asked.tool_use_id
    )

    assert answer.error is not None, answer.text
    assert set(journal.values) <= set(stepped)
    _assert_displaced(deployment, journal)
    await _replay(deployment, journal)


async def test_another_identitys_journal_is_listed_and_restored(
    deployment: gm.Deployment, session: Path
) -> None:
    del session
    asked = gm.ask(deployment, "pyaml_measure", {"kind": "orm", "model": MODEL})
    killed, _ = await gm.killed_after_first_write(
        deployment, "pyaml_measure", asked.updated_input, asked.tool_use_id
    )
    assert killed.identity != OTHER_OPERATOR
    _plant(deployment, killed.values, identity=OTHER_OPERATOR)
    journal = gm.pending(deployment)
    assert (journal.identity, journal.values) == (OTHER_OPERATOR, killed.values)
    _assert_displaced(deployment, journal)

    await _replay(deployment, journal)


async def test_another_containers_killed_run_is_listed_and_restored(
    deployment: gm.Deployment, tmp_path: Path
) -> None:
    with gm.acting_in(deployment, tmp_path / "container_b"):
        asked = gm.ask(deployment, "pyaml_measure", {"kind": "orm", "model": MODEL})
        journal, _ = await gm.killed_after_first_write(
            deployment, "pyaml_measure", asked.updated_input, asked.tool_use_id
        )
    _assert_displaced(deployment, journal)

    with gm.acting_in(deployment, tmp_path / "container_a"):
        await _replay(deployment, journal)


async def test_without_approval_a_killed_runs_journal_is_restored_by_the_next_run(
    unasked: gm.Deployment, tmp_path: Path
) -> None:
    with gm.acting_in(unasked, tmp_path / "agent_data"):
        assert gm.ask(unasked, "pyaml_measure", {"kind": "orm"}).decision != "ask"
        journal, _ = await gm.killed_after_first_write(
            unasked, "pyaml_measure", {"kind": "orm", "model": MODEL}, None
        )
        _assert_displaced(unasked, journal)

        answer = await gm.call("pyaml_measure", _replay_input(unasked))

        assert answer.error is None, answer.text
        assert answer.result is not None and answer.result["completed"] is True
        _assert_back(unasked, journal)


# ---------------------------------------------------------------------------
# A journal the prompt did not list
# ---------------------------------------------------------------------------


def _plant(
    deployment: gm.Deployment, values: dict[str, Any], *, identity: str = OTHER_OPERATOR
) -> bytes:
    """Write a dead run's journal of ``values`` over the target's journal; its bytes.

    The run named is ``identity``'s, in a process that is not this lock's
    holder: the free lock is what proves a run dead, never its pid.
    """
    from osprey.runtime.journal import DurableJournal

    path = deployment.journal()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    journal = DurableJournal(path)
    try:
        journal.start(
            target=gm.VA_TARGET,
            generation=None,
            identity=identity,
            pid=1,
            started="2026-01-01T00:00:00+00:00",
        )
        for address, value in values.items():
            journal.record(address, value)
    finally:
        journal.close()
    return path.read_bytes()


async def test_a_journal_planted_after_the_prompt_refuses_the_run_and_execute_restores_it(
    deployment: gm.Deployment, session: Path
) -> None:
    del session
    addresses = _stepped(deployment, "orm")[:2]
    _assert_still(deployment, addresses)
    asked = gm.ask(deployment, "pyaml_measure", {"kind": "orm", "model": MODEL})
    assert asked.updated_input["approved_journal_sha256"] == "none"
    home = gm.ca_read(deployment.port, addresses)
    raw = _plant(deployment, {address: float(home[address]) for address in addresses})

    answer = await gm.call("pyaml_measure", asked.updated_input, asked.tool_use_id)

    assert answer.error is not None and "OspreyJournalChanged" in answer.text, answer.text
    assert deployment.journal().read_bytes() == raw
    assert gm.ca_read(deployment.port, addresses) == home

    planted = gm.pending(deployment)
    asked = gm.ask(
        deployment,
        "execute",
        {"code": "restored = True\n", "description": "restore", "execution_mode": "readwrite"},
    )
    assert asked.decision == "ask", asked.reason
    _assert_journal_listed(asked.reason, planted)

    answer = await gm.call(
        "execute", {**asked.updated_input, "save_output": False}, asked.tool_use_id
    )

    assert answer.error is None, answer.text
    assert gm.pending(deployment) is None
    assert gm.ca_read(deployment.port, addresses) == home


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


async def test_a_headless_dispatch_run_meeting_a_pending_journal_is_denied(
    deployment: gm.Deployment, session: Path
) -> None:
    del session
    address = _stepped(deployment, "orm")[0]
    home = gm.ca_read(deployment.port, [address])
    _plant(deployment, {address: float(home[address])})
    try:
        asked = gm.ask(
            deployment,
            "pyaml_measure",
            {"kind": "orm", "model": MODEL},
            environment={"OSPREY_DISPATCH_RUN": "1"},
        )

        assert asked.decision == "deny", asked.reason
        assert "needs a human-approved call" in asked.reason
    finally:
        deployment.journal().write_bytes(b"")
    assert gm.pending(deployment) is None


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
    assert deployment.active_scenarios() == [gm.STILL]


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_pyaml_measure_guarded.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
