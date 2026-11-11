"""A ``channel_unreachable`` envelope states whether the active target answers (#2483).

One silent channel does not say whether the machine is up, and an agent left to
work that out from further reads hedges. Reachability is a measured fact, so the
envelope measures the active target's gateway when the read fails and says what
it found: the target is not answering, the target answers and only this channel
is silent, or reachability is unmeasured.

The two measured cases run against real sockets, as the prober's own tests do:
a listener that accepts, and a reserved port nothing is bound to.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from osprey.mcp_server.control_system import server_context
from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from osprey_connectors.errors import ChannelUnreachableError
from tests.mcp_server.conftest import assert_raises_error

LABEL = "Live — writes move hardware"
CHANNEL = "SR:DEAD"


def _raw(port: int, *, use_name_server: bool = True) -> dict[str, Any]:
    gateway = {"address": "127.0.0.1", "port": port, "use_name_server": use_name_server}
    return {
        "control_system": {
            "type": "epics",
            "connector": {
                "epics": {
                    "gateways": {"read_only": dict(gateway), "write_access": dict(gateway)},
                }
            },
        }
    }


class _Manager:
    """The two supervisor reads the envelope's identity comes from."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def active_target(self) -> str:
        return "live"

    def display_metadata(self) -> dict[str, dict]:
        return {"live": {"label": LABEL, "endpoint": self._endpoint}}


class _Context:
    def __init__(self, raw: dict[str, Any], endpoint: str) -> None:
        self.config = MCPServerConfig(raw=raw)
        self.connector_hosts = _Manager(endpoint)


def _serve(monkeypatch, raw: dict[str, Any], endpoint: str) -> None:
    context = _Context(raw, endpoint)
    monkeypatch.setattr(server_context, "get_server_context", lambda: context)


@pytest.fixture
async def listener():
    """A real TCP listener on an ephemeral port; yields its port."""

    async def _handle(_reader: Any, writer: Any) -> None:
        writer.close()

    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    try:
        yield int(server.sockets[0].getsockname()[1])
    finally:
        server.close()
        await server.wait_closed()


async def _envelope(connector_name: str = "control_system") -> dict:
    with assert_raises_error(error_type="channel_unreachable") as captured:
        async with connector_error_handler("channel_read", connector_name=connector_name):
            raise ChannelUnreachableError(CHANNEL, f"{CHANNEL} timed out")
    return captured["envelope"]


async def test_a_dead_gateway_is_reported_as_the_target_not_answering(monkeypatch, reserve_port):
    port = reserve_port()
    _serve(monkeypatch, _raw(port), f"127.0.0.1:{port}")

    envelope = await _envelope()

    stated = f"the {LABEL} at 127.0.0.1:{port} is not answering"
    assert stated in envelope["error_message"]
    assert envelope["suggestions"] == [
        f"Report that the {LABEL} is not answering, so {CHANNEL} could not be read."
    ]
    reachability = envelope["details"]["target_reachability"]
    assert reachability["state"] == "down"
    assert reachability["gateway"] == f"127.0.0.1:{port}"
    assert reachability["detail"]


async def test_a_live_gateway_is_reported_as_only_the_channel_silent(monkeypatch, listener):
    _serve(monkeypatch, _raw(listener), f"127.0.0.1:{listener}")

    envelope = await _envelope()

    stated = f"the {LABEL} at 127.0.0.1:{listener} answers; only this channel is silent"
    assert stated in envelope["error_message"]
    assert envelope["suggestions"] == [
        f"Report that {CHANNEL} did not answer while the {LABEL} itself answers."
    ]
    assert envelope["details"]["target_reachability"] == {
        "state": "reached",
        "gateway": f"127.0.0.1:{listener}",
    }


async def test_an_address_list_gateway_is_unmeasured_and_never_dialled(monkeypatch, listener):
    """CA search on an address list is UDP: a TCP connect would prove nothing."""

    async def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an address-list gateway was dialled")

    _serve(monkeypatch, _raw(listener, use_name_server=False), f"127.0.0.1:{listener}")
    monkeypatch.setattr(asyncio, "open_connection", _forbidden)

    envelope = await _envelope()

    assert f"whether the {LABEL} is answering was not measured" in envelope["error_message"]
    assert "is not answering" not in envelope["error_message"]
    assert "answers;" not in envelope["error_message"]
    reachability = envelope["details"]["target_reachability"]
    assert reachability["state"] == "unmeasured"
    assert "UDP" in reachability["detail"]


async def test_a_target_with_no_gateway_is_unmeasured(monkeypatch):
    """The mock connector, or the simulator served in-process: nothing to dial."""
    _serve(monkeypatch, {"control_system": {"type": "mock"}}, "")

    envelope = await _envelope()

    assert f"whether the {LABEL} is answering was not measured" in envelope["error_message"]
    assert envelope["suggestions"] == [
        f"Report that {CHANNEL} did not answer and that whether the {LABEL} is answering "
        "is unmeasured."
    ]
    assert envelope["details"]["target_reachability"]["state"] == "unmeasured"


async def test_a_measurement_that_fails_is_unmeasured(monkeypatch, listener):
    """A broken instrument claims neither state, and never costs the envelope."""
    from osprey.mcp_server.control_system import endpoint_prober

    async def _broken(*_args: Any) -> Any:
        raise RuntimeError("instrument broke")

    _serve(monkeypatch, _raw(listener), f"127.0.0.1:{listener}")
    monkeypatch.setattr(endpoint_prober.EndpointProber, "measure", _broken)

    envelope = await _envelope()

    assert envelope["details"]["channel"] == CHANNEL
    assert envelope["details"]["target_reachability"]["state"] == "unmeasured"


async def test_no_server_context_is_unmeasured():
    """The state most unit tests run in: no identity, so nothing is claimed."""
    envelope = await _envelope()

    assert "whether the control system is answering was not measured" in envelope["error_message"]
    assert envelope["details"]["target_reachability"]["state"] == "unmeasured"


async def test_the_archiver_has_no_target_to_measure():
    envelope = await _envelope("archiver")

    assert "whether the archiver is answering was not measured" in envelope["error_message"]
    assert "active_target" not in envelope["details"]
