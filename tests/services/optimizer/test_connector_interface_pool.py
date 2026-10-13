"""ConnectorInterface through a real connector-host child serving the mock connector.

Proves the safety claim the optimizer rests on: its writes are refused by the
channel limits (bounds and ``max_step``) and by an unarmed write posture, in
the child that owns the channel, not by anything in the optimizer.
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from osprey.services.optimizer.environments import interface as iface_module
from osprey.services.optimizer.environments.connector_env import ConnectorEnvironment
from osprey.services.optimizer.environments.interface import ConnectorInterface
from osprey.services.optimizer.environments.spec import EnvironmentFile
from osprey.services.optimizer.models.exceptions import WriteRefusedError

REPO_ROOT = Path(__file__).resolve().parents[3]
# A "POS" channel starts at 0.0 in the mock (a plain name starts at 100).
MOCK = "osprey_connectors.control_system.mock_connector.MockConnector"

pytestmark = pytest.mark.timeout(120)


def _project(tmp_path: Path, *, writes_enabled: bool) -> Path:
    limits = {
        "defaults": {"writable": True, "confirm": True},
        "KNOB:POS:SP": {"min_value": -5.0, "max_value": 5.0, "max_step": 1.0},
        "BPM:POS:Y": {"writable": False},
    }
    (tmp_path / "channel_limits.json").write_text(json.dumps(limits))
    section = {
        "type": MOCK,
        "writes_enabled": writes_enabled,
        "limits_checking": {
            "enabled": True,
            "database_path": str(tmp_path / "channel_limits.json"),
            "allow_unlisted_channels": False,
        },
        "connector": {MOCK: {"response_delay_ms": 1, "noise_level": 0.0}},
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
            "variables": {"KNOB:POS:SP": {"min": -50, "max": 50}},
            "observables": {"knob": {"channel": "KNOB:POS:SP"}},
        }
    )
    env = ConnectorEnvironment({"spec": spec, "control_target": "live"})
    meta = env.variables["KNOB:POS:SP"]
    assert (meta.min_value, meta.max_value) == (-5.0, 5.0)
