"""ConnectorInterface through a real connector-host child serving the in-process simulator.

Proves the safety claim the optimizer rests on: its writes are refused by the
channel limits (bounds and ``max_step``), by a write tier that allows none and by
a switch that is off, in the child that owns the channel, not by anything in the
optimizer.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from osprey_connectors_tests._write_fakes import arm_switch
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


def _project(tmp_path: Path, *, writes: str) -> Path:
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
        "writes": writes,
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
    def make(*, writes: str = "allowed", switch: str | None = "live") -> Path:
        """*writes* is the section's write tier; *switch* the target the login's switch is on for."""
        config = _project(tmp_path, writes=writes)
        monkeypatch.chdir(tmp_path)
        arm_switch(monkeypatch, tmp_path / "agent-data", switch, stamp=False)
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


def test_writes_are_refused_while_the_tier_allows_none(project):
    project(writes="never")
    with pytest.raises(WriteRefusedError) as caught:
        ConnectorInterface("live").set_values({"KNOB:POS:SP": 0.5})
    assert caught.value.reason == "WRITES_DISABLED"


def test_writes_are_refused_while_the_switch_is_off(project):
    project(switch=None)
    with pytest.raises(WriteRefusedError) as caught:
        ConnectorInterface("live").set_values({"KNOB:POS:SP": 0.5})
    assert caught.value.reason == "WRITES_DISABLED"
    assert "switch" in str(caught.value).lower()


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


def test_the_pool_kills_a_child_whose_write_timed_out(project, monkeypatch):
    import osprey_connectors.ipc as ipc

    built: list[dict] = []

    class _Recording(ipc.ConnectorHostPool):
        def __init__(self, *args, **kwargs):
            built.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ipc, "ConnectorHostPool", _Recording)
    project()
    iface_module.get_bridge()
    assert [kwargs.get("kill_on_write_timeout") for kwargs in built] == [True]


def test_the_bridge_backstop_outlasts_the_pools_own_deadline_and_kill():
    from osprey_connectors.ipc import DEFAULT_CALL_DEADLINE_S, DEFAULT_PING_TIMEOUT_S

    # Deadline, then a ping, then SIGTERM grace and proxy settle on the kill.
    assert iface_module.OP_TIMEOUT > DEFAULT_CALL_DEADLINE_S + DEFAULT_PING_TIMEOUT_S + 4.0


def test_a_write_timeout_kill_is_a_timeout_not_a_refusal():
    from osprey_connectors.ipc import ConnectorHostWriteTimeoutError

    class _Connector:
        async def write_multiple_channels(self, *args, **kwargs):
            raise ConnectorHostWriteTimeoutError(
                "write missed its deadline",
                target="live",
                execution_mode=None,
                pid=1,
                cause="write_timeout",
                returncode=None,
            )

    class _Bridge:
        def connector(self, *args, **kwargs):
            return _Connector()

        def run(self, coro):
            return asyncio.run(coro)

    interface = ConnectorInterface("live", bridge=_Bridge())
    with pytest.raises(OptimizerTimeoutError):
        interface.set_values({"KNOB:POS:SP": 0.5})


def test_a_write_leaves_confirmation_to_the_limits_database():
    calls = []

    class _Connector:
        async def write_multiple_channels(self, *args, **kwargs):
            calls.append((args, kwargs))
            return []

    class _Bridge:
        def connector(self, *args, **kwargs):
            return _Connector()

        def run(self, coro):
            return asyncio.run(coro)

    ConnectorInterface("live", bridge=_Bridge()).set_values({"KNOB:POS:SP": 0.5})
    assert calls == [(([("KNOB:POS:SP", 0.5)],), {})]


def test_one_set_of_values_is_one_write_call():
    """Every channel of one evaluation rides one batch: one write call, one authorization step."""
    calls = []

    class _Connector:
        async def write_multiple_channels(self, operations, **kwargs):
            calls.append(list(operations))
            return []

    class _Bridge:
        def connector(self, *args, **kwargs):
            return _Connector()

        def run(self, coro):
            return asyncio.run(coro)

    ConnectorInterface("live", bridge=_Bridge()).set_values({"A:SP": 1.0, "B:SP": 2.0, "C:SP": 3.0})
    assert calls == [[("A:SP", 1.0), ("B:SP", 2.0), ("C:SP", 3.0)]]


class _ChannelBridge:
    """A bridge whose connector raises *exc* from every read and write."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def connector(self, *args, **kwargs):
        exc = self._exc

        class _Connector:
            async def read_channel(self, *args, **kwargs):
                raise exc

            async def write_multiple_channels(self, *args, **kwargs):
                raise exc

        return _Connector()

    def run(self, coro):
        return asyncio.run(coro)


def test_an_unreachable_channel_read_is_the_interfaces_connection_error():
    from osprey_connectors.errors import ChannelUnreachableError
    from osprey_optimizer.models.exceptions import ConnectionError as OptimizerConnectionError

    bridge = _ChannelBridge(ChannelUnreachableError("KNOB:POS:SP"))
    with pytest.raises(OptimizerConnectionError, match="KNOB:POS:SP"):
        ConnectorInterface("live", bridge=bridge).read_one("KNOB:POS:SP")


def test_an_unreachable_channel_write_is_the_interfaces_connection_error():
    from osprey_connectors.errors import ChannelUnreachableError
    from osprey_optimizer.models.exceptions import ConnectionError as OptimizerConnectionError

    bridge = _ChannelBridge(ChannelUnreachableError("KNOB:POS:SP"))
    with pytest.raises(OptimizerConnectionError, match="KNOB:POS:SP"):
        ConnectorInterface("live", bridge=bridge).set_values({"KNOB:POS:SP": 0.5})


# -- the write authority -------------------------------------------------------


class _KeyedBridge:
    """A bridge that records the key of every connector asked for, and serves *results*."""

    def __init__(self, results=None, exc=None):
        self.keys = []
        self.calls = []
        self._results = results
        self._exc = exc

    def connector(self, target, execution_mode, **kwargs):
        self.keys.append((target, execution_mode, kwargs.get("owner"), kwargs.get("authorization")))
        bridge = self

        class _Connector:
            async def read_multiple_channels(self, _names, **_kwargs):
                return {}

            async def write_multiple_channels(self, operations, **_kwargs):
                bridge.calls.append(list(operations))
                if bridge._exc is not None:
                    raise bridge._exc
                return bridge._results(operations)

        return _Connector()

    def run(self, coro):
        return asyncio.run(coro)


def _refused(reason):
    from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome
    from osprey_connectors.errors import RefusalReason

    def results(operations):
        return [
            ChannelWriteResult(
                channel_address=address,
                value_written=value,
                outcome=WriteOutcome.REFUSED,
                refusal_reason=RefusalReason.WRITES_DISABLED,
                error_message=f"Write to '{address}' blocked",
                verdict_reason=reason,
            )
            for address, value in operations
        ]

    return results


def test_a_required_authority_holding_none_refuses_before_any_connector_is_asked():
    from osprey_optimizer.environments.write_authority import (
        NO_AUTHORIZATION,
        WriteAuthority,
        use_write_authority,
    )

    bridge = _KeyedBridge(results=lambda ops: [])
    interface = ConnectorInterface("live", bridge=bridge)
    with (
        use_write_authority(WriteAuthority(owner="alice")),
        pytest.raises(WriteRefusedError) as caught,
    ):
        interface.set_values({"KNOB:POS:SP": 0.5})
    assert caught.value.reason == NO_AUTHORIZATION
    assert bridge.calls == []


def test_an_ended_authority_refuses_every_write_before_any_connector_is_asked():
    from osprey_optimizer.environments.write_authority import WriteAuthority, use_write_authority
    from osprey_optimizer.models.exceptions import AuthorizationEndedError

    bridge = _KeyedBridge(results=lambda ops: [])
    authority = WriteAuthority(authorization="a" * 32)
    authority.end("auth_revoked")
    with use_write_authority(authority), pytest.raises(AuthorizationEndedError) as caught:
        ConnectorInterface("live", bridge=bridge).set_values({"KNOB:POS:SP": 0.5})
    assert caught.value.reason == "auth_revoked"
    assert bridge.calls == []


def test_a_write_goes_to_the_child_keyed_by_the_owner_and_the_authorization():
    from osprey_optimizer.environments.write_authority import WriteAuthority, use_write_authority

    bridge = _KeyedBridge(results=lambda ops: [])
    with use_write_authority(WriteAuthority(owner="alice", authorization="a" * 32)):
        interface = ConnectorInterface("live", bridge=bridge)
        interface.set_values({"KNOB:POS:SP": 0.5})
        interface.read_many(["KNOB:POS:SP"])
    # Reads go to the owner's child that names no authorization; the write to its own.
    assert ("live", None, "alice", "a" * 32) in bridge.keys
    assert set(bridge.keys) == {("live", None, "alice", None), ("live", None, "alice", "a" * 32)}


@pytest.mark.parametrize(
    "reason", ["auth_exhausted", "auth_range", "auth_revoked", "auth_lease_lapsed", "auth_channel"]
)
def test_an_auth_refusal_ends_the_authority_and_the_run(reason):
    from osprey_optimizer.environments.write_authority import WriteAuthority, use_write_authority
    from osprey_optimizer.models.exceptions import AuthorizationEndedError

    ended = []
    authority = WriteAuthority(authorization="a" * 32, on_end=lambda why, detail: ended.append(why))
    bridge = _KeyedBridge(results=_refused(reason))
    with use_write_authority(authority):
        interface = ConnectorInterface("live", bridge=bridge)
        with pytest.raises(AuthorizationEndedError) as caught:
            interface.set_values({"A:SP": 1.0, "B:SP": 2.0})
        assert caught.value.reason == reason
        assert authority.ended == reason
        assert ended == [reason]
        # Nothing asks again: the next write is refused here.
        with pytest.raises(AuthorizationEndedError):
            interface.set_values({"A:SP": 1.0})
    assert bridge.calls == [[("A:SP", 1.0), ("B:SP", 2.0)]]


def test_any_other_refusal_keeps_its_verdict_reason_and_leaves_the_authority_standing():
    from osprey_optimizer.environments.write_authority import WriteAuthority, use_write_authority
    from osprey_optimizer.models.exceptions import AuthorizationEndedError

    authority = WriteAuthority(authorization="a" * 32)
    with use_write_authority(authority), pytest.raises(WriteRefusedError) as caught:
        ConnectorInterface(
            "live", bridge=_KeyedBridge(results=_refused("session_held"))
        ).set_values({"A:SP": 1.0})
    assert not isinstance(caught.value, AuthorizationEndedError)
    assert caught.value.verdict_reason == "session_held"
    assert authority.ended is None


def test_a_limits_refusal_out_of_the_batch_is_a_refusal():
    from osprey_connectors.errors import ChannelLimitsViolationError

    exc = ChannelLimitsViolationError(
        channel_address="A:SP", value=9.0, violation_type="MAX_EXCEEDED", violation_reason="max"
    )
    with pytest.raises(WriteRefusedError) as caught:
        ConnectorInterface("live", bridge=_KeyedBridge(exc=exc)).set_values({"A:SP": 9.0})
    assert caught.value.reason == "LIMITS"
