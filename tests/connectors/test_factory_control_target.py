"""Every connector carries the session target it was built for.

``ConnectorFactory.build_control_system_connector`` — and
``create_control_system_connector``, which is that build plus ``connect()`` —
stamps two things on the instance it returns: the connector *type*, which selects the deployment's
connector block and the per-type write posture, and the control *target*, which
indexes the per-(session, target) posture store. They answer different questions
and neither is derivable from the other — a two-lane deployment's VA lane and
its live baseline resolve to different targets with the same connector type, and
a degraded bridge lane has a target with no type at all.

Every site that builds a connector therefore has to name the target it means,
and there are five of them. Each gets a test here that drives the real site and
asserts the value that site is entitled to name is the value that reaches the
instance, because the failure mode is silent: an unstamped connector reads the
wrong session's posture rather than raising.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from osprey_connectors import factory as factory_module
from osprey_connectors.control_system.mock_connector import MockConnector
from osprey_connectors.factory import ConnectorFactory, isolated_connector_registries

#: The mock connector by dotted path, which ``resolve_target`` returns verbatim
#: for ``live`` — the whole real factory path with no Channel Access anywhere.
MOCK_TYPE = "osprey_connectors.control_system.mock_connector.MockConnector"


@pytest.fixture
def registered_mock():
    """Only the mock, so nothing in this file can dial a real control system."""
    with isolated_connector_registries(clear=True):
        ConnectorFactory.register_control_system("mock", MockConnector)
        yield


class _RecordingFactory:
    """Stands in for the factory, recording the target each site names.

    Stamps the instance the way the real factory does, so a site that clears or
    rewrites a stamp afterwards (the degraded bridge lane) is observed doing it.
    Both entry points are recorded: the one-call ``create`` and the two-step
    ``build`` a site uses when it has to reach the instance before ``connect()``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str | None]] = []

    def _stamped(self, config: Any, control_target: str | None) -> Any:
        return SimpleNamespace(
            _connector_type=(config or {}).get("type"),
            _control_target=control_target,
            connect=_noop_connect,
            disconnect=_noop_disconnect,
        )

    async def __call__(self, config: Any = None, *, control_target: str | None = None) -> Any:
        self.calls.append((config, control_target))
        return self._stamped(config, control_target)

    def build(
        self, config: Any = None, *, control_target: str | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.calls.append((config, control_target))
        connector = self._stamped(config, control_target)
        type_config = (config or {}).get("connector", {}).get((config or {}).get("type"), {})
        return connector, type_config

    @property
    def target(self) -> str | None:
        """The target named by the one call this factory was asked to make."""
        assert len(self.calls) == 1, f"expected exactly one construction, got {self.calls}"
        return self.calls[0][1]


async def _noop_connect(type_config: Any = None) -> None:
    return None


async def _noop_disconnect() -> None:
    return None


@pytest.fixture
def recording_factory(monkeypatch: pytest.MonkeyPatch) -> _RecordingFactory:
    """Replace the factory classmethod; every site imports it at call time."""
    spy = _RecordingFactory()
    monkeypatch.setattr(
        factory_module.ConnectorFactory, "create_control_system_connector", spy, raising=True
    )
    monkeypatch.setattr(
        factory_module.ConnectorFactory, "build_control_system_connector", spy.build, raising=True
    )
    monkeypatch.setattr(factory_module, "register_builtin_connectors", lambda: None)
    return spy


# ---------------------------------------------------------------------------
# The factory itself
# ---------------------------------------------------------------------------


class TestFactoryStamp:
    @pytest.mark.asyncio
    async def test_named_target_reaches_the_instance(self, registered_mock):
        connector = await ConnectorFactory.create_control_system_connector(
            {"type": "mock", "connector": {"mock": {}}}, control_target="standin"
        )
        try:
            assert connector._control_target == "standin"
            # The type stamp is untouched: the two are independent.
            assert connector._connector_type == "mock"
        finally:
            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_target_defaults_to_none(self, registered_mock):
        """Every pre-existing caller keeps working, naming no target."""
        connector = await ConnectorFactory.create_control_system_connector(
            {"type": "mock", "connector": {"mock": {}}}
        )
        try:
            assert connector._control_target is None
        finally:
            await connector.disconnect()

    def test_unbuilt_connector_names_no_target(self):
        """An instance nobody built through the factory has the same default."""
        assert MockConnector()._control_target is None


# ---------------------------------------------------------------------------
# Site 1 — the connector-host child names the target it was pointed at
# ---------------------------------------------------------------------------


class TestConnectorHostChild:
    @pytest.mark.asyncio
    async def test_child_stamps_the_payload_target(self, registered_mock, tmp_path, monkeypatch):
        """The init payload's target, through the real ``_build_connector``."""
        from osprey_connectors.ipc import host

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        connector, report = await host._build_connector(
            {
                "control_system": {
                    "type": MOCK_TYPE,
                    "connector": {MOCK_TYPE: {"response_delay_ms": 0, "noise_level": 0.0}},
                },
                "target": "live",
            }
        )
        try:
            assert connector._control_target == "live"
            assert report["target"] == "live"
        finally:
            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_child_stamps_a_target_that_is_not_the_baseline(
        self, registered_mock, tmp_path, monkeypatch
    ):
        """A child on ``va`` says ``va`` even though the section's own type is live."""
        from osprey_connectors.ipc import host

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        connector, _ = await host._build_connector(
            {
                "control_system": {
                    "type": MOCK_TYPE,
                    "connector": {"virtual_accelerator": {"channel_prefix": "VA:"}},
                },
                "target": "va",
            }
        )
        try:
            assert connector._control_target == "va"
            assert connector._connector_type == "virtual_accelerator"
        finally:
            await connector.disconnect()


# ---------------------------------------------------------------------------
# Site 2 — the executor sandbox names the target it was stamped with
# ---------------------------------------------------------------------------


class TestRuntimeSandbox:
    @pytest.fixture(autouse=True)
    def clear_runtime_connector(self):
        import osprey.runtime as runtime

        runtime._runtime_connector = None
        yield
        runtime._runtime_connector = None

    @pytest.mark.asyncio
    async def test_stamped_sandbox_names_its_target(self, recording_factory, monkeypatch, tmp_path):
        import osprey.runtime as runtime

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(runtime.ENV_CONTROL_TARGET, "va")
        # The config half is not what is under test here; keep it out of reach.
        monkeypatch.setattr(runtime, "_target_connector_config", lambda: {"type": "mock"})

        await runtime._get_connector()

        assert recording_factory.target == "va"

    @pytest.mark.asyncio
    async def test_unstamped_sandbox_names_no_target(
        self, recording_factory, monkeypatch, tmp_path
    ):
        import osprey.runtime as runtime

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(runtime.ENV_CONTROL_TARGET, raising=False)
        monkeypatch.setattr(runtime, "_target_connector_config", lambda: None)

        await runtime._get_connector()

        assert recording_factory.target is None


# ---------------------------------------------------------------------------
# Site 3 — the in-process controls server names its deployment baseline
# ---------------------------------------------------------------------------


class TestServerContextInProcess:
    @staticmethod
    def _context(raw: dict[str, Any]):
        from osprey.mcp_server.control_system.server_context import (
            ConnectorEntry,
            ControlSystemContext,
            MCPServerConfig,
        )

        context = ControlSystemContext()
        context._config = MCPServerConfig(raw=raw)
        # The switch-capable path never reaches the factory: it serves a child.
        context._switch_capable = False
        context._connectors["control_system"] = ConnectorEntry(
            config=context._config.control_system, connector_type="control_system"
        )
        return context

    @pytest.mark.asyncio
    async def test_baseline_target_is_named(self, recording_factory):
        """A virtual-accelerator deployment's in-process connector says ``va``."""
        context = self._context(
            {"control_system": {"type": "virtual_accelerator", "connector": {}}}
        )

        await context._get_connector("control_system")

        assert recording_factory.target == "va"
        # Derived from config, not by asking the supervisor: a deployment that
        # cannot switch must never construct one (test_switch_lifecycle.py's
        # TestNonCapableDeploymentIsUntouched pins the same invariant).
        assert context._connector_hosts is None

    @pytest.mark.asyncio
    async def test_rebuild_after_invalidation_names_the_same_target(self, recording_factory):
        """Dropping the instance and rebuilding carries the stamp, not a blank."""
        context = self._context({"control_system": {"type": "epics", "connector": {}}})

        first = await context._get_connector("control_system")
        await context.invalidate_connector("control_system")
        second = await context._get_connector("control_system")

        assert first is not second
        assert [target for _, target in recording_factory.calls] == ["live", "live"]


# ---------------------------------------------------------------------------
# Site 4 — a health run names the target its own section describes
# ---------------------------------------------------------------------------


class TestHealthRuntime:
    @pytest.mark.asyncio
    async def test_health_names_the_configured_baseline(self, recording_factory):
        from osprey.health.runtime import HealthRuntime

        async with HealthRuntime({"type": "live_standin", "connector": {}}) as runtime:
            await runtime.get_connector()

        assert recording_factory.target == "standin"

    @pytest.mark.asyncio
    async def test_health_on_a_live_deployment_names_live(self, recording_factory):
        from osprey.health.runtime import HealthRuntime

        async with HealthRuntime({"type": "epics", "connector": {}}) as runtime:
            await runtime.get_connector()

        assert recording_factory.target == "live"


# ---------------------------------------------------------------------------
# Site 5 — a bridge worker names its own lane's target
# ---------------------------------------------------------------------------


class TestBlueskyWorker:
    @staticmethod
    def _patch_lane(
        monkeypatch: pytest.MonkeyPatch,
        *,
        connector_type: str,
        lane_target: str,
        degraded: str | None = None,
    ) -> None:
        from osprey.services.bluesky_bridge import queue_backend

        monkeypatch.setattr(
            queue_backend, "resolve_lane_connector_type", lambda: (connector_type, degraded)
        )
        monkeypatch.setattr(
            queue_backend, "resolve_lane_identity", lambda: ("bluesky", lane_target)
        )

    @pytest.mark.asyncio
    async def test_worker_names_its_lane_target_not_the_baseline(
        self, recording_factory, monkeypatch
    ):
        """The shipped control-assistant render: a ``standin`` lane."""
        from osprey.services.bluesky_bridge import qserver_startup

        self._patch_lane(monkeypatch, connector_type="live_standin", lane_target="standin")

        connector = await qserver_startup.create_connector()

        assert recording_factory.target == "standin"
        assert connector._control_target == "standin"
        assert connector._connector_type == "live_standin"

    @pytest.mark.asyncio
    async def test_va_lane_diverges_from_a_live_baseline(self, recording_factory, monkeypatch):
        from osprey.services.bluesky_bridge import qserver_startup

        self._patch_lane(monkeypatch, connector_type="virtual_accelerator", lane_target="va")

        await qserver_startup.create_connector()

        assert recording_factory.target == "va"

    @pytest.mark.asyncio
    async def test_degraded_lane_keeps_its_cleared_type_and_its_target(
        self, recording_factory, monkeypatch, caplog
    ):
        """Rung 3 clears the TYPE stamp only.

        A degraded lane was built as the deployment baseline while addressing a
        machine that baseline's block does not describe, so the type stamp is
        cleared and the deployment-wide key becomes its whole write posture. The
        target stamp survives: the lane's declared target is still the honest
        answer to which machine this worker addresses, and it indexes the
        session store rather than any config block.
        """
        from osprey.services.bluesky_bridge import qserver_startup

        self._patch_lane(
            monkeypatch,
            connector_type="mock",
            lane_target="live",
            degraded="Lane 'bluesky' declares the 'live' target, which this deployment cannot "
            "resolve to a control system",
        )

        connector = await qserver_startup.create_connector()

        assert connector._connector_type is None
        assert connector._control_target == "live"
        assert recording_factory.target == "live"
