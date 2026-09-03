"""Failure envelopes name the control-system target they were pointed at (#697).

A dead-IOC timeout on the live machine is a materially different situation
than one on the simulator, and the operator should not have to reconstruct
which one it was from session memory. ``connector_error_handler`` therefore
stamps ``details["active_target"]`` (name/label, endpoint where known) and a
human ``(active target: ...)`` clause into every control-system envelope.

Two layers are pinned separately:

* the resolver (``describe_active_target``), which reads the supervisor's
  display cache rather than re-deriving identity from config, including its
  fail-soft collapse to ``None``;
* the wiring — every branch of the handler carries the identity, the archiver
  (which has no live/VA axis) carries none, and an unresolvable identity
  leaves each envelope byte-identical to what it was before #697.

The composed proof — a real supervisor, a killed child, the real tool —
lives in ``tests/integration/test_target_switch_mock_pair.py`` and the bench
outage suite; this file is the fast, hardware-free layer.
"""

from __future__ import annotations

import pytest

from osprey.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from osprey.mcp_server.control_system import error_handling, server_context
from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
from osprey.mcp_server.control_system.error_handling import (
    connector_error_handler,
    describe_active_target,
)
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from tests.mcp_server.conftest import assert_raises_error

IDENTITY = {"name": "live", "label": "LIVE MACHINE", "endpoint": "localhost:5064"}
CLAUSE = "(active target: LIVE MACHINE at localhost:5064)"


# ------------------------------------------------------------- the resolver


def _slot(endpoint: str = "localhost:5064", role: str = "write_access") -> dict:
    """One target slot in the shape the supervisor's display cache holds."""
    return {
        "label": "LIVE MACHINE",
        "display_name": "the live machine",
        "endpoint": endpoint,
        "selected_role": role,
        "real_machine": True,
        "probe_channel": "",
    }


class _FakeManager:
    """The two supervisor reads: the target of record and the display cache."""

    def __init__(self, slot: dict | None = None) -> None:
        self._display = {"live": slot if slot is not None else _slot()}

    def active_target(self) -> str:
        return "live"

    def display_metadata(self) -> dict[str, dict]:
        return self._display


class _FakeContext:
    """Just the one attribute the resolver reads, nothing else."""

    def __init__(self, slot: dict | None = None) -> None:
        self.connector_hosts = _FakeManager(slot)


class _RealManagerContext:
    """A real supervisor that has never spawned a child, and nothing else."""

    def __init__(self, raw: dict) -> None:
        self.connector_hosts = ConnectorHostManager(MCPServerConfig(raw=raw))


def _epics_raw(port: int = 5064) -> dict:
    gateway = {"address": "localhost", "port": port, "use_name_server": True}
    return {
        "control_system": {
            "type": "epics",
            "connector": {
                "epics": {
                    "timeout": 1.0,
                    "gateways": {
                        "read_only": dict(gateway),
                        "write_access": dict(gateway),
                    },
                }
            },
        }
    }


def test_the_resolver_names_target_label_and_endpoint(monkeypatch):
    monkeypatch.setattr(server_context, "get_server_context", lambda: _FakeContext())
    assert describe_active_target() == IDENTITY


def test_the_resolver_prints_the_read_gateway_the_writer_rendered(monkeypatch):
    """The whole point of reading the cache instead of re-deriving.

    A session that has given up writes is served through the read gateway,
    and the writer rendered that. Re-deriving from config here would name
    the write gateway and tell the operator the wrong port.
    """
    slot = _slot(endpoint="localhost:5065", role="read_only")
    monkeypatch.setattr(server_context, "get_server_context", lambda: _FakeContext(slot))
    assert describe_active_target() == {
        "name": "live",
        "label": "LIVE MACHINE",
        "endpoint": "localhost:5065",
    }


def test_the_resolver_names_the_target_before_any_child_exists(monkeypatch):
    """A real supervisor with no child still answers with label and endpoint.

    The first child is spawned lazily, so the earliest failures happen while
    the manager has nothing running. The cache renders on demand, which is
    what keeps those envelopes from carrying an empty identity line.
    """
    context = _RealManagerContext(_epics_raw())
    monkeypatch.setattr(server_context, "get_server_context", lambda: context)
    assert describe_active_target() == IDENTITY


def test_the_resolver_fails_soft_to_none(monkeypatch):
    """No server context — the state every unit test in this repo runs in.

    ``None`` here is what keeps every pre-#697 envelope byte-identical, so
    this is the assertion the rest of the suite silently leans on.
    """

    def boom() -> None:
        raise RuntimeError("not initialized")

    monkeypatch.setattr(server_context, "get_server_context", boom)
    assert describe_active_target() is None


# ------------------------------------------------------- the handler wiring


@pytest.fixture
def resolved_identity(monkeypatch):
    """Pin the resolver so each branch's wiring is tested in isolation."""
    monkeypatch.setattr(error_handling, "describe_active_target", lambda: dict(IDENTITY))

    async def no_invalidate(connector_name: str) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(error_handling, "invalidate_active_connector", no_invalidate)


async def _envelope_for(exc: BaseException, connector_name: str = "control_system") -> dict:
    with assert_raises_error() as captured:
        async with connector_error_handler("channel_write", connector_name=connector_name):
            raise exc
    return captured["envelope"]


async def test_a_connection_failure_names_the_machine(resolved_identity):
    envelope = await _envelope_for(ConnectionError("connect timeout after 1.0s"))
    assert envelope["error_type"] == "connection_error"
    assert CLAUSE in envelope["error_message"]
    assert envelope["details"]["active_target"] == IDENTITY
    # The first suggestion points at the actual machine, not a generic service.
    assert "LIVE MACHINE at localhost:5064" in envelope["suggestions"][0]


async def test_a_timeout_names_the_machine(resolved_identity):
    envelope = await _envelope_for(TimeoutError("no response"))
    assert envelope["error_type"] == "timeout_error"
    assert CLAUSE in envelope["error_message"]
    assert envelope["details"]["active_target"] == IDENTITY


async def test_a_limits_violation_carries_the_target_in_its_details(resolved_identity):
    envelope = await _envelope_for(
        ChannelLimitsViolationError(
            channel_address="X:Y:SP",
            value=99.0,
            violation_type="range",
            violation_reason="above maximum",
        )
    )
    assert envelope["error_type"] == "limits_violation"
    assert envelope["details"]["active_target"] == IDENTITY


async def test_a_control_system_refusal_names_the_machine_that_refused(resolved_identity):
    envelope = await _envelope_for(ChannelWriteBlockedError("X:Y:SP", "CONTROL_SYSTEM_REFUSED"))
    assert envelope["error_type"] == "write_refused"
    assert CLAUSE in envelope["error_message"]
    assert envelope["details"]["active_target"] == IDENTITY
    assert envelope["details"]["reason"] == "CONTROL_SYSTEM_REFUSED"


async def test_a_reference_monitor_refusal_carries_the_target(resolved_identity):
    envelope = await _envelope_for(ChannelWriteBlockedError("X:Y:SP", "WRITES_DISABLED"))
    assert envelope["details"]["active_target"] == IDENTITY


async def test_an_internal_error_names_the_machine(resolved_identity):
    envelope = await _envelope_for(ValueError("surprise"))
    assert envelope["error_type"] == "internal_error"
    assert CLAUSE in envelope["error_message"]
    assert envelope["details"]["active_target"] == IDENTITY


async def test_the_archiver_envelope_carries_no_target(resolved_identity, monkeypatch):
    """The archiver has no live/VA axis — stamping the control-system target
    onto its failures would attribute an archiver outage to a machine."""

    def must_not_resolve() -> dict:
        raise AssertionError("describe_active_target() must not run for the archiver")

    monkeypatch.setattr(error_handling, "describe_active_target", must_not_resolve)
    envelope = await _envelope_for(TimeoutError("no response"), connector_name="archiver")
    assert "details" not in envelope
    assert "active target" not in envelope["error_message"]


async def test_an_unresolvable_identity_leaves_the_envelope_unchanged():
    """No context installed: exactly the pre-#697 envelope, key for key."""
    envelope = await _envelope_for(ChannelWriteBlockedError("X:Y:SP", "WRITES_DISABLED"))
    assert envelope["details"] == {"channel": "X:Y:SP", "reason": "WRITES_DISABLED"}

    envelope = await _envelope_for(TimeoutError("no response"))
    assert "details" not in envelope
    assert "active target" not in envelope["error_message"]
