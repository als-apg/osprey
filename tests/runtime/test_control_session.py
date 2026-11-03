"""The control session: one binding per process, opened once and never rebuilt.

Every test opens a real :class:`~osprey.runtime.session.ControlSession` against
a real control-context record in ``tmp_path``. No test touches EPICS: the fake
connector is registered under the virtual accelerator's type inside
``isolated_connector_registries``, so a ``va`` record builds a fake. That
registration exists only here — no production path gives a session a fake
target.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from osprey.runtime import session as session_module
from osprey.runtime import session_env
from osprey.runtime.budget import WriteBudget
from osprey.runtime.session import (
    EXIT_BINDING_MOVED,
    EXIT_START_REFUSED,
    ControlSession,
    SessionEnvironmentError,
    SessionUnservableError,
)
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.control_system import _base as connector_base
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._limits_validator import STEP_READ_TIMEOUT_KEY
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests._harness._control_context_fixtures import _set_switch_to

#: A deployment whose baseline is the virtual accelerator, its write tier allowed.
VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "writes": "allowed",
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}

#: A deployment whose baseline is the stand-in, with a virtual accelerator beside it.
STANDIN_WITH_VA_SECTION = {
    "type": types.LIVE_STANDIN,
    "connector": {
        types.LIVE_STANDIN: {},
        types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0},
    },
}

#: A stand-in baseline with one real machine and no virtual accelerator.
STANDIN_WITH_LIVE_SECTION = {
    "type": types.LIVE_STANDIN,
    "connector": {types.LIVE_STANDIN: {}, types.EPICS: {"response_delay_ms": 0}},
}

#: A stand-in with no simulator beside it and no real machine named: ``live`` has no answer.
STANDIN_ONLY_SECTION = {"type": types.LIVE_STANDIN, "connector": {types.LIVE_STANDIN: {}}}


class _CountingFake(FakeConnector):
    """A fake that counts how many times any instance was connected."""

    connects = 0
    configs: list = []

    async def connect(self, config):
        type(self).connects += 1
        type(self).configs.append(dict(config))
        await super().connect(config)


class _RefusingConnector(FakeConnector):
    """A connector whose control system is not there."""

    async def connect(self, config):  # noqa: ARG002 - the connector connect signature
        raise ConnectionError("no gateway answers")


def _serve(monkeypatch, section):
    def get_config_value(path, default=None, _config_path=None):
        return section if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)


def write_record(state_dir: Path, *, target="va", generation=3, switch=None) -> Path:
    """Write the record and make its switch exactly *switch*: a mapping of target to
    position, or the targets to hold at ``allowed`` (default: *target* allowed)."""
    path = state_dir / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation), path=path
    )
    control_context.invalidate_cache()
    if switch is None:
        wanted = dict.fromkeys({target} & set(types.CONTROL_TARGETS), "allowed")
    elif isinstance(switch, Mapping):
        wanted = dict(switch)
    else:
        wanted = dict.fromkeys(switch, "allowed")
    _set_switch_to(path, wanted)
    control_context.invalidate_cache()
    return path


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """A session process's environment: the bound record directory and nothing stale."""
    directory = tmp_path / "control_target"
    directory.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    for name in list(os.environ):
        if name.startswith("OSPREY_CONTROL_TARGET") or name.startswith("EPICS_"):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)
    # Every record a session files lands in the test's spool zone (the root
    # conftest), never in a real project, and the long-lived mark open() sets
    # is undone after the test.
    monkeypatch.setattr(connector_base, "_long_lived_session", False)
    control_context.invalidate_cache()
    yield directory
    control_context.invalidate_cache()


@pytest.fixture
def mock_va(monkeypatch, state_dir):
    """Serve :data:`VA_SECTION` and build a fake for the virtual accelerator."""
    _serve(monkeypatch, VA_SECTION)
    _CountingFake.connects = 0
    _CountingFake.configs = []
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _CountingFake)
        yield state_dir


@pytest.fixture
def epics_gateways(monkeypatch, state_dir):
    """The virtual accelerator behind a real EPICS connector with a read and a write gateway."""
    gateway = {"address": "127.0.0.1", "port": 5064}
    block = {"gateways": {"read_only": gateway, "write_access": {**gateway, "port": 5065}}}
    _serve(monkeypatch, {**VA_SECTION, "connector": {types.VIRTUAL_ACCELERATOR: block}})
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, EPICSConnector)
        yield state_dir


async def _opened(**kwargs) -> ControlSession:
    return await ControlSession.open(surface="panel", **kwargs)


def test_exit_statuses_are_the_contract_numbers():
    assert EXIT_BINDING_MOVED == 75
    assert EXIT_START_REFUSED == 78


class TestEnvironmentRefusal:
    """``open`` refuses an environment that carries a second source of truth."""

    @pytest.mark.parametrize(
        "name",
        [posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, AUDIT_IDENTITY_ENV, "OSPREY_CONFIG"],
    )
    async def test_a_missing_required_variable_refuses(self, mock_va, monkeypatch, name):
        write_record(mock_va)
        monkeypatch.delenv(name)

        with pytest.raises(SessionEnvironmentError) as refusal:
            await _opened()

        assert any(name in problem for problem in refusal.value.problems)
        assert _CountingFake.connects == 0

    async def test_a_config_file_naming_another_file_refuses(self, mock_va, monkeypatch, tmp_path):
        write_record(mock_va)
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "other.yml"))

        with pytest.raises(SessionEnvironmentError, match="CONFIG_FILE"):
            await _opened()

    async def test_an_absent_config_file_refuses(self, mock_va, monkeypatch):
        write_record(mock_va)
        monkeypatch.delenv("CONFIG_FILE")

        with pytest.raises(SessionEnvironmentError, match="CONFIG_FILE"):
            await _opened()

    @pytest.mark.parametrize(
        "name",
        [
            posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
            posture_store.CONTROL_OWNER_ENV_VAR,
            posture_store.LAUNCH_POSTURE_ENV_VAR,
            "OSPREY_CONTROL_TARGET",
            "OSPREY_CONTROL_TARGET_GENERATION",
            "OSPREY_CONTROL_TARGET_REFUSAL",
            "OSPREY_CONTROL_TARGET_ANYTHING_ELSE",
        ],
    )
    async def test_a_forbidden_variable_refuses(self, mock_va, monkeypatch, name):
        write_record(mock_va)
        monkeypatch.setenv(name, "va")

        with pytest.raises(SessionEnvironmentError) as refusal:
            await _opened()

        assert f"{name} must not be set" in refusal.value.problems
        assert _CountingFake.connects == 0

    @pytest.mark.usefixtures("mock_va")
    async def test_the_refusal_names_every_problem_at_once(self, monkeypatch):
        monkeypatch.delenv(AUDIT_IDENTITY_ENV)
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "sandbox")

        with pytest.raises(SessionEnvironmentError) as refusal:
            await _opened()

        assert len(refusal.value.problems) == 2

    async def test_the_names_refused_on_are_the_runtimes_own_stamps(self):
        """The prefix has to cover every stamp ``osprey.runtime`` reads."""
        import osprey.runtime as runtime

        for name in (
            runtime.ENV_CONTROL_TARGET,
            runtime.ENV_CONTROL_TARGET_GENERATION,
            runtime.ENV_CONTROL_TARGET_REFUSAL,
        ):
            assert name.startswith(session_env.FORBIDDEN_ENV_PREFIX)
        assert set(session_env.FORBIDDEN_ENV) == {
            posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
            posture_store.CONTROL_OWNER_ENV_VAR,
            posture_store.LAUNCH_POSTURE_ENV_VAR,
        }


class TestOpen:
    """A readable record and a buildable connector make a bound session."""

    async def test_the_binding_is_the_records_target_and_generation(self, mock_va):
        write_record(mock_va, target="va", generation=7)

        session = await _opened()

        binding = session.binding
        assert (binding.target, binding.generation) == ("va", 7)
        assert binding.state == "bound"
        assert binding.connector_type == types.VIRTUAL_ACCELERATOR
        assert binding.display_name
        assert binding.id == "va@7"
        await session.close()

    async def test_the_connector_is_built_for_the_target_and_stamped_with_it(self, mock_va):
        write_record(mock_va)

        session = await _opened()

        assert isinstance(session._connector, _CountingFake)
        assert session._connector._connector_type == types.VIRTUAL_ACCELERATOR
        assert session._connector._control_target == "va"
        await session.close()

    async def test_inherited_epics_settings_are_scrubbed(self, mock_va, monkeypatch):
        write_record(mock_va)
        monkeypatch.setenv("EPICS_CA_ADDR_LIST", "10.0.0.1")
        monkeypatch.setenv("EPICS_PVA_NAME_SERVERS", "10.0.0.2")

        session = await _opened()

        assert not [name for name in os.environ if name.startswith("EPICS_")]
        await session.close()

    async def test_builtin_connectors_are_registered(self, state_dir, monkeypatch):
        _serve(monkeypatch, VA_SECTION)
        write_record(state_dir, target="live")
        with isolated_connector_registries(clear=True):
            session = await _opened()
            assert types.VIRTUAL_ACCELERATOR in ConnectorFactory._control_system_connectors
            await session.close()

    async def test_a_warm_step_that_raises_is_not_fatal(self, mock_va, monkeypatch):
        write_record(mock_va)

        def warm(_self):
            raise RuntimeError("libca would not load")

        monkeypatch.setattr(_CountingFake, "warm", warm, raising=False)

        session = await _opened()

        assert session.binding.state == "bound"
        await session.close()

    async def test_a_warm_step_that_hangs_is_bounded(self, mock_va, monkeypatch):
        write_record(mock_va)

        async def warm(_self):
            await asyncio.sleep(60)

        monkeypatch.setattr(_CountingFake, "warm", warm, raising=False)
        monkeypatch.setattr(session_module, "WARM_BOUND_S", 0.05)

        session = await asyncio.wait_for(_opened(), timeout=5)

        assert session.binding.state == "bound"
        await session.close()

    async def test_the_warm_step_reads_no_channel(self, mock_va, monkeypatch):
        write_record(mock_va)
        reads = []

        async def read_channel(_self, channel_address, **_kwargs):
            reads.append(channel_address)
            raise AssertionError("open must not read a channel")

        monkeypatch.setattr(_CountingFake, "read_channel", read_channel)

        session = await _opened()

        assert reads == []
        await session.close()


class TestWriteBudgetAndProcessMark:
    """open() bounds the connector through its block and marks the process long-lived."""

    async def test_the_connector_block_carries_the_bound_budget(self, mock_va):
        write_record(mock_va)
        budget = WriteBudget(T=0.5, S=0.25, D=1.25, grace_s=3.25, relay_timeout_s=4.25)

        session = await _opened(budget=budget)

        (config,) = _CountingFake.configs
        assert config["timeout_s"] == 0.5
        assert config[STEP_READ_TIMEOUT_KEY] == 0.25
        assert session._call_bound_s == 0.5
        assert session._write_bound_s == 1.25
        assert "timeout_s" not in VA_SECTION["connector"][types.VIRTUAL_ACCELERATOR]
        await session.close()

    async def test_a_block_below_the_budget_keeps_its_own_bounds(self, state_dir, monkeypatch):
        section = {
            "type": types.VIRTUAL_ACCELERATOR,
            "connector": {types.VIRTUAL_ACCELERATOR: {"timeout_s": 2, STEP_READ_TIMEOUT_KEY: 1}},
        }
        _serve(monkeypatch, section)
        _CountingFake.configs = []
        with isolated_connector_registries():
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _CountingFake)
            write_record(state_dir)
            session = await _opened()

            (config,) = _CountingFake.configs
            assert (config["timeout_s"], config[STEP_READ_TIMEOUT_KEY]) == (2.0, 1.0)
            assert session._write_bound_s == 5.0
            await session.close()

    async def test_open_marks_the_process_long_lived(self, mock_va, monkeypatch):
        write_record(mock_va)
        monkeypatch.setattr(sys, "argv", ["/tmp/some_server.py"])
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        refused = posture_store.WriteVerdict(
            allowed=False, reason="readonly_run", target="va", kind="simulator"
        )
        before = connector_base._verdict_refusal("TEST:CHANNEL:SP", 1.0, refused)
        assert "readonly execution mode" in before.error_message
        assert "resubmit it with execution_mode='readwrite'" in before.error_message

        session = await _opened()
        after = connector_base._verdict_refusal("TEST:CHANNEL:SP", 1.0, refused)

        # A long-lived session is the deployment's: no script to resubmit.
        assert "readonly execution mode" in after.error_message
        assert "resubmit" not in after.error_message
        await session.close()


class TestFallback:
    """No readable record: the deployment's baseline, served read-only."""

    @pytest.mark.usefixtures("mock_va")
    async def test_no_record_binds_the_baseline_as_fallback(self):
        session = await _opened()

        binding = session.binding
        assert binding.state == "fallback"
        assert binding.id == "fallback"
        assert (binding.target, binding.generation) == ("va", None)
        assert (await session.read("TEST:CHANNEL")).value is not None
        await session.close()

    async def test_a_corrupt_record_is_no_record(self, mock_va):
        (mock_va / control_context.RECORD_FILENAME).write_text("{not json", encoding="utf-8")
        control_context.invalidate_cache()

        session = await _opened()

        assert session.binding.state == "fallback"
        await session.close()

    @pytest.mark.usefixtures("state_dir")
    async def test_a_standin_baseline_falls_back_to_the_va_when_one_exists(self, monkeypatch):
        _serve(monkeypatch, STANDIN_WITH_VA_SECTION)
        with isolated_connector_registries():
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _CountingFake)
            session = await _opened()

        assert (session.binding.target, session.binding.state) == ("va", "fallback")
        await session.close()

    @pytest.mark.usefixtures("state_dir")
    async def test_a_standin_baseline_falls_back_to_live_without_a_va(self, monkeypatch):
        _serve(monkeypatch, STANDIN_WITH_LIVE_SECTION)
        with isolated_connector_registries():
            ConnectorFactory.register_control_system(types.EPICS, _CountingFake)
            session = await _opened()

        binding = session.binding
        assert (binding.target, binding.state) == ("live", "fallback")
        assert binding.connector_type == types.EPICS
        await session.close()

    @pytest.mark.usefixtures("state_dir")
    async def test_a_fallback_that_cannot_be_served_is_unservable(self, monkeypatch):
        """A lone stand-in falls back to ``live``, and nothing here says which machine that is."""
        _serve(monkeypatch, STANDIN_ONLY_SECTION)

        session = await _opened()

        binding = session.binding
        assert (binding.target, binding.state, binding.id) == ("live", "unservable", "unservable")
        assert binding.connector_type is None
        await session.close()


class TestUnservable:
    """A target that cannot be resolved, built or connected raises nothing."""

    async def test_a_target_the_config_cannot_resolve(self, state_dir, monkeypatch):
        _serve(monkeypatch, STANDIN_ONLY_SECTION)
        write_record(state_dir, target="live", generation=2)

        session = await _opened()

        binding = session.binding
        assert (binding.target, binding.generation) == ("live", 2)
        assert (binding.state, binding.id) == ("unservable", "unservable")
        await session.close()

    async def test_a_record_naming_no_known_target_is_no_record(self, mock_va):
        """The record reader drops a target it cannot use, so this is the fallback."""
        write_record(mock_va, target="somewhere", generation=1)

        session = await _opened()

        assert (session.binding.target, session.binding.state) == ("va", "fallback")
        await session.close()

    async def test_a_connector_type_nothing_is_registered_for(self, state_dir, monkeypatch):
        _serve(monkeypatch, VA_SECTION)
        write_record(state_dir, target="va")
        with isolated_connector_registries(clear=True):
            monkeypatch.setattr(
                "osprey_connectors.factory.register_builtin_connectors", lambda: None
            )
            session = await _opened()

        assert session.binding.state == "unservable"
        assert session.binding.connector_type == types.VIRTUAL_ACCELERATOR
        await session.close()

    async def test_a_connect_that_raises(self, state_dir, monkeypatch):
        _serve(monkeypatch, VA_SECTION)
        write_record(state_dir, target="va")
        with isolated_connector_registries():
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _RefusingConnector)
            session = await _opened()

        assert session.binding.state == "unservable"
        await session.close()

    async def test_reads_and_metadata_raise_and_subscriptions_stay_disconnected(
        self, state_dir, monkeypatch
    ):
        _serve(monkeypatch, STANDIN_ONLY_SECTION)
        write_record(state_dir, target="live")
        session = await _opened()
        values, connection = [], []

        with pytest.raises(SessionUnservableError):
            await session.read("TEST:CHANNEL")
        with pytest.raises(SessionUnservableError):
            await session.metadata("TEST:CHANNEL")
        assert issubclass(SessionUnservableError, ConnectionError)

        handle = await session.subscribe(
            "TEST:CHANNEL", values.append, on_connection=connection.append
        )
        await asyncio.sleep(0.02)

        assert (values, connection) == ([], [False])
        await session.unsubscribe(handle)
        await session.close()

    async def test_an_unservable_session_still_sees_the_record_move(self, state_dir, monkeypatch):
        _serve(monkeypatch, STANDIN_ONLY_SECTION)
        write_record(state_dir, target="live", generation=2)
        session = await _opened()
        assert session.moved() is False

        write_record(state_dir, target="va", generation=3)

        assert session.moved() is True
        await session.close()


class TestReadSide:
    """Reads, subscriptions and metadata go to the one connector."""

    async def test_read_and_metadata(self, mock_va):
        write_record(mock_va)
        session = await _opened()
        session._connector.set_value("TEST:CHANNEL", 5.0)

        assert (await session.read("TEST:CHANNEL")).value == pytest.approx(5.0)
        assert (await session.metadata("TEST:CHANNEL")).description
        await session.close()

    async def test_connection_is_seeded_false_and_true_with_the_first_value(self, mock_va):
        write_record(mock_va)
        session = await _opened()
        session._connector.set_value("TEST:CHANNEL", 5.0)
        events = []

        await session.subscribe(
            "TEST:CHANNEL",
            lambda update: events.append(update.value),
            on_connection=events.append,
        )
        session._connector.set_value("TEST:CHANNEL", 6.0)

        assert events == [False, True, 5.0, 6.0]
        await session.close()

    async def test_a_drop_reports_false_and_true_comes_back_with_the_values(self, mock_va):
        write_record(mock_va)
        session = await _opened()
        connector = session._connector
        connector.set_value("TEST:CHANNEL", 5.0)
        events = []
        await session.subscribe(
            "TEST:CHANNEL",
            lambda update: events.append(update.value),
            on_connection=events.append,
        )

        connector.drop("TEST:CHANNEL")
        connector.set_value("TEST:CHANNEL", 7.0)
        connector.restore("TEST:CHANNEL")
        connector.set_value("TEST:CHANNEL", 8.0)

        assert events == [False, True, 5.0, False, True, 7.0, 8.0]
        await session.close()

    async def test_a_channel_down_at_subscribe_reports_false_once_until_it_comes_up(self, mock_va):
        write_record(mock_va)
        session = await _opened()
        connector = session._connector
        connector.set_value("TEST:CHANNEL", 5.0)
        connector.drop("TEST:CHANNEL")
        events = []
        await session.subscribe(
            "TEST:CHANNEL",
            lambda update: events.append(update.value),
            on_connection=events.append,
        )
        await asyncio.sleep(0.02)
        assert events == [False]

        connector.restore("TEST:CHANNEL")

        assert events == [False, True, 5.0]
        await session.close()

    async def test_two_subscriptions_report_their_connection_independently(self, mock_va):
        write_record(mock_va)
        session = await _opened()
        connector = session._connector
        connector.set_value("TEST:CHANNEL", 5.0)
        first, second = [], []
        handle = await session.subscribe(
            "TEST:CHANNEL", lambda update: None, on_connection=first.append
        )
        connector.drop("TEST:CHANNEL")
        await session.subscribe("TEST:CHANNEL", lambda update: None, on_connection=second.append)
        await session.unsubscribe(handle)

        connector.restore("TEST:CHANNEL")

        assert (first, second) == ([False, True, False], [False, True])
        await session.close()

    async def test_subscribe_without_a_connection_callback(self, mock_va):
        write_record(mock_va)
        session = await _opened()
        values = []

        await session.subscribe("TEST:CHANNEL", values.append)

        assert len(values) == 1
        await session.close()

    async def test_unsubscribe_stops_delivery_and_is_idempotent(self, mock_va):
        write_record(mock_va)
        session = await _opened()
        connector = session._connector
        values = []
        handle = await session.subscribe("TEST:CHANNEL", values.append)

        await session.unsubscribe(handle)
        await session.unsubscribe(handle)
        connector.set_value("TEST:CHANNEL", 9.0)

        assert len(values) == 1
        assert connector._subscriptions == {}
        await session.close()

    async def test_two_subscriptions_to_one_channel_are_independent(self, mock_va):
        write_record(mock_va)
        session = await _opened()
        first, second = [], []
        handle = await session.subscribe("TEST:CHANNEL", first.append)
        other = await session.subscribe("TEST:CHANNEL", second.append)
        assert handle != other

        await session.unsubscribe(handle)
        session._connector.set_value("TEST:CHANNEL", 2.0)

        assert (len(first), len(second)) == (1, 2)
        await session.close()


class TestWritesAreRefused:
    """A session that holds no write to make refuses without reaching a connector.

    The full WC6 table is ``test_session_write.py``.
    """

    async def _spy(self, monkeypatch):
        calls = []

        async def write_channel(_self, *args, **_kwargs):
            calls.append(args)
            raise AssertionError("a session write must not reach the connector")

        monkeypatch.setattr(_CountingFake, "write_channel", write_channel)
        return calls

    @pytest.mark.usefixtures("mock_va")
    async def test_a_fallback_session_names_the_missing_record(self, monkeypatch):
        calls = await self._spy(monkeypatch)
        session = await _opened()

        result = await session.write("TEST:CHANNEL:SP", 1.0)

        assert (result.outcome, result.reason) == ("refused", "control_context_unavailable")
        assert result.message == "no control target record; nothing written"
        assert calls == []
        await session.close()

    async def test_an_unservable_session_says_so(self, state_dir, monkeypatch):
        _serve(monkeypatch, STANDIN_ONLY_SECTION)
        write_record(state_dir, target="live")
        session = await _opened()

        result = await session.write("TEST:CHANNEL:SP", 1.0)

        assert (result.outcome, result.reason) == ("refused", "control_context_unavailable")
        assert result.message == "this control target cannot be served here; nothing written"
        await session.close()

    async def test_refuse_answers_a_stale_binding_as_a_moved_target(self, mock_va, monkeypatch):
        calls = await self._spy(monkeypatch)
        write_record(mock_va)
        session = await _opened()

        result = session.refuse("TEST:CHANNEL:SP", 1.0, "switching", origin=None)

        assert (result.outcome, result.reason) == ("target_changed", "switching")
        assert result.message == refusal_phrase("switching", "simulator")
        assert calls == []
        await session.close()

    async def test_refuse_takes_no_reason_of_the_callers_own(self, mock_va, monkeypatch):
        calls = await self._spy(monkeypatch)
        write_record(mock_va)
        session = await _opened()

        with pytest.raises(ValueError, match="not a refusal reason"):
            session.refuse("TEST:CHANNEL:SP", 1.0, "control_target_changed", origin=None)
        assert calls == []
        await session.close()


class TestMoved:
    """A move is another ``(target, generation)`` or another switch position of this target."""

    async def test_an_untouched_record_is_not_a_move(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()

        assert session.moved() is False
        await session.close()

    async def test_a_rewrite_of_the_same_record_is_not_a_move(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()

        write_record(mock_va, target="va", generation=3)

        assert session.moved() is False
        await session.close()

    @pytest.mark.parametrize(("target", "generation"), [("va", 4), ("live", 3), ("live", 4)])
    async def test_another_target_or_generation_is_a_move(self, mock_va, target, generation):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()

        write_record(mock_va, target=target, generation=generation)

        assert session.moved() is True
        assert session.binding.id == "va@3"
        await session.close()

    async def test_a_record_that_appears_under_a_fallback_is_a_move(self, mock_va):
        session = await _opened()
        assert session.moved() is False

        write_record(mock_va, target="va", generation=0)

        assert session.moved() is True
        await session.close()

    async def test_a_record_that_goes_away_is_a_move(self, mock_va):
        path = write_record(mock_va)
        session = await _opened()

        path.unlink()
        control_context.invalidate_cache()

        assert session.moved() is True
        await session.close()

    async def test_switching_the_bound_target_off_is_no_move_and_shows_in_writes(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()

        write_record(mock_va, target="va", generation=3, switch=())

        assert session.moved() is False
        assert session.binding.id == "va@3"
        assert session.writes["reason"] == "switch_off"
        await session.close()

    async def test_switching_the_bound_target_on_is_no_move_and_shows_in_writes(self, mock_va):
        write_record(mock_va, target="va", generation=3, switch=())
        session = await _opened()
        assert session.moved() is False

        write_record(mock_va, target="va", generation=3)

        assert session.moved() is False
        assert session.writes["allowed"] is True
        await session.close()

    @pytest.mark.parametrize(
        ("before", "after", "lags"),
        [
            ({}, {"va": "asked"}, True),
            ({}, {"va": "allowed"}, True),
            ({"va": "asked"}, {}, True),
            ({"va": "allowed"}, {}, True),
            ({"va": "asked"}, {"va": "allowed"}, False),
            ({}, {"live": "allowed"}, False),
        ],
    )
    async def test_a_switch_asking_for_the_other_gateway_is_a_move(
        self, epics_gateways, before, after, lags
    ):
        """DATA at off sits on the read gateway and would refuse every write ``switching``."""
        write_record(epics_gateways, target="va", generation=3, switch=before)
        session = await _opened()
        assert session.moved() is False

        write_record(epics_gateways, target="va", generation=3, switch=after)

        assert session.moved() is lags
        await session.close()

    async def test_a_write_gateway_alone_never_lags(self, monkeypatch, state_dir):
        """Such a block stays on its one gateway whatever the switch says."""
        block = {"gateways": {"write_access": {"address": "127.0.0.1", "port": 5065}}}
        _serve(monkeypatch, {**VA_SECTION, "connector": {types.VIRTUAL_ACCELERATOR: block}})
        with isolated_connector_registries():
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, EPICSConnector)
            write_record(state_dir, target="va", generation=3, switch={})
            session = await _opened()
            write_record(state_dir, target="va", generation=3, switch={"va": "asked"})
            assert session.moved() is False
            await session.close()

    async def test_the_next_process_after_off_to_asked_writes_through_the_write_gateway(
        self, epics_gateways
    ):
        write_record(epics_gateways, target="va", generation=3, switch={})
        stuck = await _opened()
        refusal = stuck.connector._role_refusal("SR:SP", 1.0)
        assert refusal is not None
        assert "switch" in str(refusal.notes or refusal).lower()
        await stuck.close()

        write_record(epics_gateways, target="va", generation=3, switch={"va": "asked"})
        session = await _opened()

        assert session.connector._selected_role == "write_access"
        assert session.connector._role_refusal("SR:SP", 1.0) is None
        assert session.moved() is False
        await session.close()

    async def test_wait_moved_returns_and_marks_the_target_moving_on_off_to_asked(
        self, epics_gateways
    ):
        write_record(epics_gateways, target="va", generation=3, switch={})
        session = await _opened()
        watcher = asyncio.create_task(session.wait_moved())
        await asyncio.sleep(0.05)
        assert not watcher.done()

        write_record(epics_gateways, target="va", generation=3, switch={"va": "asked"})

        await asyncio.wait_for(watcher, timeout=5)
        assert session.target_moving is True
        await session.close()

    async def test_another_targets_write_state_is_not_a_move(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()

        write_record(mock_va, target="va", generation=3, switch=("va", "live"))

        assert session.moved() is False
        await session.close()

    async def test_wait_moved_returns_when_the_record_moves(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()
        waiter = asyncio.ensure_future(session.wait_moved())

        await asyncio.sleep(0.05)
        assert not waiter.done()
        write_record(mock_va, target="live", generation=4)

        await asyncio.wait_for(waiter, timeout=5)
        await session.close()

    async def test_wait_moved_reads_the_record_only_when_its_signature_changes(
        self, mock_va, monkeypatch
    ):
        """Unchanged ticks read nothing; a changed signature reads the record once."""
        write_record(mock_va, target="va", generation=3)
        session = await _opened()
        reads = []
        read_record = session_module._read_record
        monkeypatch.setattr(
            session_module, "_read_record", lambda: reads.append(1) or read_record()
        )

        waiter = asyncio.ensure_future(session.wait_moved())
        await asyncio.sleep(0.1)
        assert reads == []

        write_record(mock_va, target="va", generation=3, switch=("va", "live"))
        await asyncio.sleep(0.1)
        assert len(reads) == 1
        assert not waiter.done()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await session.close()

    async def test_wait_moved_returns_at_once_when_already_moved(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()
        write_record(mock_va, target="va", generation=4)

        await asyncio.wait_for(session.wait_moved(), timeout=5)
        await session.close()

    async def test_a_move_never_rebuilds_the_connector(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()
        connector = session._connector

        write_record(mock_va, target="va", generation=4)
        assert session.moved() is True
        await session.read("TEST:CHANNEL")
        handle = await session.subscribe("TEST:CHANNEL", lambda update: None)
        await session.unsubscribe(handle)

        assert session._connector is connector
        assert _CountingFake.connects == 1
        assert session.binding.id == "va@3"
        assert not hasattr(session, "rebuild")
        await session.close()


class TestClose:
    """Close stops writes, drains, drops subscriptions and disconnects, in that order."""

    async def test_drain_with_nothing_in_flight_returns_at_once(self, mock_va):
        write_record(mock_va)
        session = await _opened()

        session.stop_writes()

        assert await asyncio.wait_for(session.drain(30.0), timeout=1) is True
        await session.close()

    async def test_close_runs_the_steps_in_order(self, mock_va, monkeypatch):
        write_record(mock_va)
        session = await _opened()
        connector = session._connector
        await session.subscribe("TEST:CHANNEL", lambda update: None)
        steps = []

        stop_writes, drain = session.stop_writes, session.drain
        unsubscribe, disconnect = connector.unsubscribe, connector.disconnect

        def spy_stop():
            steps.append("stop_writes")
            stop_writes()

        async def spy_drain(timeout):
            steps.append("drain")
            return await drain(timeout)

        async def spy_unsubscribe(subscription_id):
            steps.append("unsubscribe")
            await unsubscribe(subscription_id)

        async def spy_disconnect():
            steps.append("disconnect")
            await disconnect()

        monkeypatch.setattr(session, "stop_writes", spy_stop)
        monkeypatch.setattr(session, "drain", spy_drain)
        monkeypatch.setattr(connector, "unsubscribe", spy_unsubscribe)
        monkeypatch.setattr(connector, "disconnect", spy_disconnect)

        await session.close()
        await session.close()

        assert steps == ["stop_writes", "drain", "unsubscribe", "disconnect"]

    async def test_a_closed_session_serves_nothing(self, mock_va):
        write_record(mock_va)
        session = await _opened()
        await session.close()
        connection = []

        with pytest.raises(SessionUnservableError):
            await session.read("TEST:CHANNEL")
        await session.subscribe(
            "TEST:CHANNEL", lambda update: None, on_connection=connection.append
        )

        assert connection == [False]
        assert _CountingFake.connects == 1


# ---------------------------------------------------------------------------
# ControlSession.moved() over pin_verdict: the recordless (fallback) case
# ---------------------------------------------------------------------------


@pytest.fixture
def pin_env(tmp_path, monkeypatch):
    """A session process bound to a record directory under ``tmp_path``, fake VA served."""
    import os

    from osprey.runtime import session as session_module
    from osprey_connectors import control_context, posture_store, types
    from osprey_connectors.factory import ConnectorFactory
    from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
    from osprey_connectors_tests._fake_connector import FakeConnector
    from osprey_connectors_tests._harness import isolated_connector_registries

    directory = tmp_path / "control_target"
    directory.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    for name in list(os.environ):
        if name.startswith("OSPREY_CONTROL_TARGET") or name.startswith("EPICS_"):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)

    section = {
        "type": types.VIRTUAL_ACCELERATOR,
        "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
    }

    def get_config_value(path, default=None, _config_path=None):
        return section if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)
    control_context.invalidate_cache()
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, FakeConnector)
        yield directory
    control_context.invalidate_cache()


def _write_record(directory, *, target="va", generation=3):
    from osprey_connectors import control_context

    path = directory / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation, posture={}),
        path=path,
    )
    control_context.invalidate_cache()
    return path


def _corrupt_record(directory):
    from osprey_connectors import control_context

    path = directory / control_context.RECORD_FILENAME
    path.write_text("{not json", encoding="utf-8")
    control_context.invalidate_cache()
    return path


async def _open():
    from osprey.runtime.session import ControlSession

    return await ControlSession.open(surface="panel")


class TestSessionMovedOverPinVerdict:
    @pytest.mark.usefixtures("pin_env")
    async def test_a_recordless_session_with_still_no_record_has_not_moved(self):
        session = await _open()

        assert session.binding.state == "fallback"
        assert session.moved() is False
        await session.close()

    async def test_a_recordless_session_has_not_moved_on_an_unreadable_record(self, pin_env):
        session = await _open()

        _corrupt_record(pin_env)

        assert session.moved() is False
        await session.close()

    async def test_a_recordless_session_moves_when_a_record_appears(self, pin_env):
        session = await _open()

        _write_record(pin_env, target="va", generation=0)

        assert session.moved() is True
        await session.close()

    async def test_a_bound_session_moves_when_its_record_becomes_unreadable(self, pin_env):
        _write_record(pin_env)
        session = await _open()
        assert session.moved() is False

        _corrupt_record(pin_env)

        assert session.moved() is True
        await session.close()
