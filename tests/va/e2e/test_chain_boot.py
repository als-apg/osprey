"""The chain on the two-model tree, ended in a served machine.

The boot half of ``tests/facility/test_chain.py``. That module installs the
committed NSLS-II tree into a scratch deployment -- both exports in one
``osprey facility import mml`` call, the build, the render, twice over -- and
holds every view to the facility file. This one runs the same chain under the
config a served render carries, hands the render's data root to the virtual
accelerator image, and reads the machine over Channel Access:

* **Every channel of each model is served.** The simulator view names the
  channels each model drives, and every one of them answers a read from the
  host. Each model is read on its own, so a container that resolved one deck
  and not the other fails the model it lost.
* **A setpoint serves what the model owes.** No write reaches the machine
  here, so every setpoint answers with the value an in-process composite over
  the same view holds for it. A server answering with values it never
  computed is not a served machine.
* **A measurement on the chain stays on the simulator.** ``pyaml_measure``
  takes the orbit response of ``LTB`` through the approval hook, in one
  guarded run that clears its journal at the end; the render configures no
  target but the simulator, so every write reaches the virtual accelerator
  and none reaches hardware.

One container serves both models, from the built-in ``still`` scenario. The chain and the boot are module-scoped, and
the module's ``xdist_group`` mark keeps every lane on one worker, so a parallel
run boots the tree once.

The container, its readiness wait and its out-of-process Channel Access client
are ``test_mml_trees_boot``'s; that module's process-boundary note covers the
reads made here.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from typing import Any

import pytest

from osprey_connectors.simulation.view import Model, SimulatorView
from tests.va._still import assert_readings_still
from tests.va.e2e import _guarded_measure as gm
from tests.va.e2e import conftest as e2e_conftest

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    # One worker runs every lane here, so the chain is installed and its
    # container booted once.
    pytest.mark.xdist_group("chain-boot"),
]

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
# Two lanes over two models, one over the view and the measurement; the guard
# test itself is the seventh item, so a floor of 6 reds on the loss of a
# single lane.
MIN_COLLECTED_TESTS = 6

#: The models the chain's tree serves. A literal tuple, because parametrisation
#: is read at COLLECTION; the first lane holds it to the chain's own.
MODELS = ("LTB", "StorageRing")

#: The model the chain's measurement takes the orbit response of.
MEASURED = "LTB"

#: The config the chain renders under: a served render's, every model served,
#: and the active set the container starts in.
CHAIN_CONFIG: dict[str, Any] = {
    **e2e_conftest.VIEW_CONFIG,
    "agent_data": {"base_dir": "var/agent_data"},
    "simulation": {"default_scenarios": [gm.STILL]},
}


def _chain() -> Any:
    """The chain module, imported on first use rather than at collection.

    Behind a call, a chain module that cannot import fails the lanes that need
    it rather than the collection of every module in this directory.
    """
    from tests.facility import test_chain

    return test_chain


def _boot() -> Any:
    """The harvested-tree boot module, whose container and client this one uses."""
    from tests.va.e2e import test_mml_trees_boot

    return test_mml_trees_boot


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """The chain, run under a served render's config, with a container over it."""
    chain = _chain()
    boot = _boot()
    built = chain.run_chain(tmp_path_factory.mktemp("chain-boot") / chain.PROJECT, CHAIN_CONFIG)
    tree = boot.BuiltTree(
        name=chain.TREE,
        repo=built.root,
        view=SimulatorView.of_render(built.build),
        declared_kinds=frozenset(),
        stopped="",
        remedied=(),
    )
    import yaml

    from osprey.simulation.apply import activate_default_scenarios
    from osprey_connectors.workspace import resolve_simulation_state_dir

    # The render's config names the default set; activating it is what
    # `osprey up` does on a first start.
    (built.root / "build" / "config.yml").write_text(yaml.safe_dump(CHAIN_CONFIG), "utf-8")
    assert activate_default_scenarios(CHAIN_CONFIG, built.root), "no scenario was activated"
    state = resolve_simulation_state_dir(CHAIN_CONFIG, built.root)
    gm.readable_by_anyone(state)
    with boot._serving(tree, state_dir=state) as running:
        yield running


def _model(served: Any, name: str) -> Model:
    """The served view's record of one model."""
    model: Model = served.tree.view.model(name)
    return model


def _channels(served: Any, name: str) -> list[str]:
    """Every address one model drives, sorted."""
    bindings = served.tree.view.bindings(model=name, served_only=False)
    return sorted({binding.address for binding in bindings})


def test_the_view_serves_both_models_of_the_chain(served: Any) -> None:
    """The render the container mounts names each model of the tree as served."""
    names = served.tree.view.served()

    assert MODELS == _chain().MODELS
    assert set(MODELS) <= set(names)
    for name in MODELS:
        model = _model(served, name)
        assert model.served, name
        assert model.deck is not None and model.deck.is_file(), name


@pytest.mark.parametrize("name", MODELS)
def test_every_channel_of_the_model_is_served(served: Any, name: str) -> None:
    """Every address the model drives answers a read from the host."""
    channels = _channels(served, name)
    assert channels, f"{name} wires no channel at all"

    answer = served.call({"op": "read", "addresses": channels})

    assert not answer["failed"], (
        f"{name}: {len(answer['failed'])} of {len(channels)} channels are not served: "
        f"{answer['failed']}"
    )
    missing = [address for address, value in answer["values"].items() if value is None]
    assert not missing, f"{name}: served with no value: {missing}"


@pytest.mark.parametrize("name", MODELS)
def test_every_setpoint_of_the_model_serves_what_the_model_owes(served: Any, name: str) -> None:
    """Each setpoint answers with the value the composite over the same view holds.

    Setpoints carry no declared motion and no lane of this module writes, so
    the served value and the owed one are the same number.
    """
    setpoints = [
        address
        for address in _channels(served, name)
        if served.tree.channel(address).role == "setpoint"
    ]
    assert setpoints, f"{name} wires no setpoint"

    values = served.read(*setpoints)

    owed = served.owed(*setpoints)
    for address in setpoints:
        assert values[address] == pytest.approx(owed[address], rel=_boot().READBACK_RTOL), (
            f"{name}: {address} serves {values[address]}, and the model owes {owed[address]}"
        )


def _measuring(served: Any) -> gm.Deployment:
    """The chain's render, made a deployment the measurement tool and its hook can run in.

    The chain renders the facility outputs alone; this adds what a build
    renders beside them for an agent: a ``config.yml`` naming the container
    the chain booted as its one control target, armed for writes, every
    guarded tool asking, and the approval hook with the ``measurement`` block
    the build writes for it.
    """
    import json
    import shutil as files

    import yaml

    from osprey.facility.views.facts import FACTS_FILE, hook_measurement

    repo = served.tree.repo
    render = repo / "build"
    gateway = {"address": "127.0.0.1", "port": served.port, "use_name_server": True}
    config = {
        "project_name": _chain().PROJECT,
        "project_root": str(repo),
        "agent_data": CHAIN_CONFIG["agent_data"],
        "simulation": CHAIN_CONFIG["simulation"],
        "control_system": {
            "type": "virtual_accelerator",
            "writes_enabled": True,
            "limits_checking": {
                "enabled": True,
                "mode": "optional",
                "database_path": "data/channel_limits.json",
            },
            "connector": {
                "virtual_accelerator": {
                    "gateways": {"read_only": gateway, "write_access": gateway},
                    "writes_enabled": True,
                    "timeout_s": 30.0,
                }
            },
        },
        "approval": {"enabled": True, "default_policy": "always"},
        "python_executor": {"execution_timeout_seconds": gm.EXECUTION_TIMEOUT_S},
    }
    (render / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    hooks = render / gm.APPROVAL_HOOK.parent
    hooks.mkdir(parents=True, exist_ok=True)
    source = gm.REPO_ROOT / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
    for hook in source.glob("osprey_*.py"):
        files.copyfile(hook, hooks / hook.name)
    facts = json.loads((render / "data" / FACTS_FILE).read_text(encoding="utf-8"))
    (hooks / "hook_config.json").write_text(
        json.dumps(
            {
                "server_prefixes": ["mcp__python__"],
                "approval_prefixes": ["mcp__python__"],
                "measurement": hook_measurement(facts, render),
            }
        ),
        encoding="utf-8",
    )
    return gm.Deployment(repo=repo, port=served.port)


async def test_a_pyaml_measure_on_the_chain_writes_the_simulator_alone(
    served: Any, tmp_path: Any
) -> None:
    """The orbit response of LTB, approved, under the run lock, which it releases."""
    deployment = _measuring(served)
    readings = sorted(
        {b.address for b in served.tree.view.bindings(model=MEASURED, served_only=False)}
    )
    assert_readings_still(served.tree.view, readings, deployment.active_scenarios())
    assert gm.only_simulated_targets(deployment.config) == []

    with gm.acting_in(deployment, tmp_path / "agent_data"):
        asked, answer = await gm.approved(
            deployment, "pyaml_measure", {"kind": "orm", "model": MEASURED}
        )

    assert f"Model: {MEASURED}\nKind: orm\n" in asked.reason
    assert answer.error is None, answer.text
    assert answer.result is not None and answer.result["completed"] is True, answer.text
    assert gm.holder(deployment) is None
    lock = deployment.lock()
    assert sorted(path.name for path in lock.parent.parent.iterdir()) == [gm.VA_TARGET]
    assert sorted(path.name for path in lock.parent.iterdir()) == [lock.name]


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_chain_boot.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
