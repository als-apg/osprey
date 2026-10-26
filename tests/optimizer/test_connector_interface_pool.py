"""ConnectorInterface through a real connector-host child serving the in-process simulator.

Proves the safety claim the optimizer rests on: its writes are refused by the
channel limits (bounds and ``max_step``) and by an unarmed write posture, in
the child that owns the channel, not by anything in the optimizer.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from osprey_optimizer.environments import interface as iface_module
from osprey_optimizer.environments.connector_env import ConnectorEnvironment
from osprey_optimizer.environments.interface import ConnectorInterface
from osprey_optimizer.environments.spec import EnvironmentFile
from osprey_optimizer.models.exceptions import TimeoutError as OptimizerTimeoutError
from osprey_optimizer.models.exceptions import WriteRefusedError
from tests.facility.served_tree import in_process_config, served_tree

REPO_ROOT = Path(__file__).resolve().parents[3]
IN_PROCESS = "osprey_connectors.control_system._va_in_process_connector.VAInProcessConnector"

pytestmark = pytest.mark.timeout(120)


def _project(tmp_path: Path, *, writes_enabled: bool) -> Path:
    limits = {
        "KNOB:POS:SP": {
            "writable": True,
            "confirm": True,
            "min_value": -5.0,
            "max_value": 5.0,
            "max_step": 1.0,
        },
        "BPM:POS:Y": {"writable": False, "confirm": True},
    }
    (tmp_path / "channel_limits.json").write_text(json.dumps(limits))
    section = {
        "type": IN_PROCESS,
        "writes_enabled": writes_enabled,
        "limits_checking": {
            "enabled": True,
            "database_path": str(tmp_path / "channel_limits.json"),
        },
        # BPM:POS:Y is served as a setpoint so the limits database, not the
        # view, is what refuses a write to it.
        "connector": {
            IN_PROCESS: in_process_config(
                served_tree(tmp_path / "tree", ("KNOB:POS:SP", "BPM:POS:Y")),
                response_delay_ms=1,
            )
        },
    }
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"control_system": section}))
    return config


@pytest.fixture
def project(tmp_path, monkeypatch):
    def make(*, writes_enabled: bool = True) -> Path:
        config = _project(tmp_path, writes_enabled=writes_enabled)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CONFIG_FILE", str(config))
        monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
        monkeypatch.setenv(
            "PYTHONPATH",
            os.pathsep.join(
                [str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src")]
            ),
        )
        return config

    yield make
    iface_module.reset_bridge()


def test_a_write_inside_limits_lands_and_reads_back(project):
    project()
    interface = ConnectorInterface("live")
    interface.set_values({"KNOB:POS:SP": 0.5})
    assert interface.read_many(["KNOB:POS:SP"])["KNOB:POS:SP"].value == pytest.approx(0.5)


def test_a_write_outside_the_channel_limits_is_refused(project):
    project()
    with pytest.raises(WriteRefusedError):
        ConnectorInterface("live").set_values({"KNOB:POS:SP": 6.0})


def test_a_step_larger_than_max_step_is_refused(project):
    project()
    interface = ConnectorInterface("live")
    interface.set_values({"KNOB:POS:SP": 0.5})
    with pytest.raises(WriteRefusedError):
        interface.set_values({"KNOB:POS:SP": 2.0})  # 1.5 from 0.5, max_step 1.0
    assert interface.read_many(["KNOB:POS:SP"])["KNOB:POS:SP"].value == pytest.approx(0.5)


def test_a_read_only_channel_is_refused(project):
    project()
    with pytest.raises(WriteRefusedError):
        ConnectorInterface("live").set_values({"BPM:POS:Y": 1.0})


def test_writes_are_refused_while_the_target_is_not_armed(project):
    project(writes_enabled=False)
    with pytest.raises(WriteRefusedError):
        ConnectorInterface("live").set_values({"KNOB:POS:SP": 0.5})


def test_a_readonly_execution_mode_refuses_writes(project):
    project()
    with pytest.raises(WriteRefusedError):
        ConnectorInterface("live", "readonly").set_values({"KNOB:POS:SP": 0.5})


def test_an_environment_evaluates_through_the_pool(project):
    project()
    spec = EnvironmentFile.model_validate(
        {
            "name": "knob",
            "variables": {"KNOB:POS:SP": {"min": -5, "max": 5, "step_size": 0.001}},
            "observables": {
                "knob_abs": {"reduce": "max", "channels": ["KNOB:POS:SP"], "direction": "MINIMIZE"}
            },
        }
    )
    env = ConnectorEnvironment({"spec": spec, "control_target": "live"})
    out = env.evaluate({"KNOB:POS:SP": 0.25}, ["knob_abs"])
    assert out["knob_abs"].value == pytest.approx(0.25)


def test_an_environment_wider_than_the_limits_database_is_narrowed_to_it(project):
    project()
    spec = EnvironmentFile.model_validate(
        {
            "name": "wide",
            "variables": {"KNOB:POS:SP": {"min": -50, "max": 50, "step_size": 0.01}},
            "observables": {"knob": {"channel": "KNOB:POS:SP"}},
        }
    )
    env = ConnectorEnvironment({"spec": spec, "control_target": "live"})
    meta = env.variables["KNOB:POS:SP"]
    assert (meta.min_value, meta.max_value) == (-5.0, 5.0)


def test_the_pool_kills_a_child_whose_write_timed_out(project):
    project()
    assert iface_module.get_bridge()._pool._kill_on_write_timeout is True


def test_the_bridge_backstop_outlasts_the_pools_own_deadline_and_kill():
    from osprey_connectors.ipc import DEFAULT_CALL_DEADLINE_S, DEFAULT_PING_TIMEOUT_S

    # Deadline, then a ping, then SIGTERM grace and proxy settle on the kill.
    assert iface_module.OP_TIMEOUT > DEFAULT_CALL_DEADLINE_S + DEFAULT_PING_TIMEOUT_S + 4.0


def test_a_write_timeout_kill_is_a_timeout_not_a_refusal():
    from osprey_connectors.ipc import ConnectorHostWriteTimeoutError

    class _Connector:
        async def write_channel_checked(self, *args, **kwargs):
            raise ConnectorHostWriteTimeoutError(
                "write missed its deadline",
                target="live",
                execution_mode=None,
                pid=1,
                cause="write_timeout",
                returncode=None,
            )

    class _Bridge:
        def connector(self, *args):
            return _Connector()

        def run(self, coro):
            return asyncio.run(coro)

    interface = ConnectorInterface("live", bridge=_Bridge())
    with pytest.raises(OptimizerTimeoutError):
        interface.set_values({"KNOB:POS:SP": 0.5})


def test_a_write_leaves_confirmation_to_the_limits_database():
    calls = []

    class _Connector:
        async def write_channel_checked(self, *args, **kwargs):
            calls.append((args, kwargs))

    class _Bridge:
        def connector(self, *args):
            return _Connector()

        def run(self, coro):
            return asyncio.run(coro)

    ConnectorInterface("live", bridge=_Bridge()).set_values({"KNOB:POS:SP": 0.5})
    assert calls == [(("KNOB:POS:SP", 0.5), {})]
