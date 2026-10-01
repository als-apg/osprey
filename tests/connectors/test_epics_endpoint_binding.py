"""One process, one endpoint per pvapy provider — and pvapy only on the workers.

pvapy reads ``EPICS_CA_*`` when the first Channel Access channel of a process
is created, and ``EPICS_PVA_*`` at the first PVAccess one, and never again. A
second ``connect()`` in the same process (a notebook kernel whose control
target moved, say) that sets different values would leave the rebuilt
connector talking to the OLD gateway under the new target's name. It is
refused instead; the same endpoint reconnects freely.

The second half pins the ``max_step`` reader to the connector's own worker
threads: the runtime calls it on whatever thread it is on, and on macOS a
thread that touched pvapy hangs forever when it is joined.
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import threading
import time

import pytest

from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey_connectors.errors import ClientEndpointConflictError
from tests.connectors._epics_fakes import (
    PVA_GLOB,
    VALUE_REQUEST,
    FakePvaccess,
    ca_connector,
    clean_epics_env,  # noqa: F401 - fixture, used by name
    fake_pvaccess,  # noqa: F401 - fixture, used by name
    record,
)
from tests.connectors._epics_fakes import patch_writes_enabled as _patch_writes_enabled

PVA_ADDRESS = "SR:CAM1:IMAGE"


def _gateway(address: str, port: int = 5064) -> dict:
    return {"gateways": {"read_only": {"address": address, "port": port}}}


def _pva(pva_address: str, ca_address: str = "ca-gw") -> dict:
    return {
        **_gateway(ca_address),
        "pva_channels": [PVA_GLOB],
        "pva_gateway": {"address": pva_address},
    }


async def _connected(config: dict) -> EPICSConnector:
    connector = EPICSConnector()
    await connector.connect(config)
    return connector


@pytest.fixture
def served(fake_pvaccess: FakePvaccess, monkeypatch) -> FakePvaccess:  # noqa: F811
    """The fake client, serving one CA and one PVA channel, writes off."""
    _patch_writes_enabled(monkeypatch, False)
    fake_pvaccess.serve("SR:CH", record(1.0))
    fake_pvaccess.serve(PVA_ADDRESS, record(2.0))
    return fake_pvaccess


@pytest.mark.usefixtures("clean_epics_env", "served")
class TestTheBoundEndpoint:
    @pytest.mark.asyncio
    async def test_a_different_ca_endpoint_after_the_first_channel_is_refused(self):
        first = await _connected(_gateway("old-gw"))
        await first.read_channel("SR:CH")
        await first.disconnect()

        second = EPICSConnector()
        with pytest.raises(ClientEndpointConflictError) as caught:
            await second.connect(_gateway("new-gw", 5065))

        message = str(caught.value)
        assert "EPICS_CA_ADDR_LIST=new-gw, EPICS_CA_SERVER_PORT=5065" in message
        assert "EPICS_CA_ADDR_LIST=old-gw, EPICS_CA_SERVER_PORT=5064" in message
        assert "fresh process" in message
        assert caught.value.provider == "ca"
        # Refused before anything changed: the environment still names the
        # endpoint pvapy is bound to, and the connector never connected.
        assert os.environ["EPICS_CA_ADDR_LIST"] == "old-gw"
        assert os.environ["EPICS_CA_SERVER_PORT"] == "5064"
        assert second._connected is False
        assert second._pvaccess is None

    @pytest.mark.asyncio
    async def test_the_same_endpoint_reconnects(self):
        first = await _connected(_gateway("gw"))
        await first.read_channel("SR:CH")
        await first.disconnect()

        second = await _connected(_gateway("gw"))

        assert (await second.read_channel("SR:CH")).value == 1.0

    @pytest.mark.asyncio
    async def test_nothing_is_bound_before_the_first_channel(self):
        """connect() opens no channel, so a reconnect before any read is free."""
        await _connected(_gateway("old-gw"))

        second = await _connected(_gateway("new-gw"))

        assert os.environ["EPICS_CA_ADDR_LIST"] == "new-gw"
        assert (await second.read_channel("SR:CH")).value == 1.0

    @pytest.mark.asyncio
    async def test_a_connector_overtaken_before_its_first_channel_is_refused_there(self, served):
        """Its own endpoint was replaced in the environment before pvapy read it."""
        overtaken = await _connected(_gateway("old-gw"))
        newer = await _connected(_gateway("new-gw"))
        await newer.read_channel("SR:CH")
        gets = len(served.calls("get"))

        with pytest.raises(ClientEndpointConflictError, match="old-gw"):
            await overtaken.read_channel("SR:CH")
        assert len(served.calls("get")) == gets  # nothing reached the wire

    @pytest.mark.asyncio
    async def test_a_different_pva_endpoint_after_the_first_pva_channel_is_refused(self):
        first = await _connected(_pva("old-pva"))
        await first.read_channel(PVA_ADDRESS)

        with pytest.raises(ClientEndpointConflictError, match="PVAccess") as caught:
            await EPICSConnector().connect(_pva("new-pva"))

        assert caught.value.provider == "pva"
        assert "EPICS_PVA_ADDR_LIST=old-pva" in str(caught.value)
        assert "EPICS_PVA_ADDR_LIST=new-pva" in str(caught.value)

    @pytest.mark.asyncio
    async def test_a_connector_routing_nothing_over_pva_ignores_the_pva_binding(self):
        """Only a provider the new connector will use can conflict."""
        first = await _connected(_pva("old-pva"))
        await first.read_channel(PVA_ADDRESS)

        ca_only = await _connected(_gateway("ca-gw"))

        assert (await ca_only.read_channel("SR:CH")).value == 1.0

    @pytest.mark.asyncio
    async def test_each_client_module_carries_its_own_binding(self, monkeypatch):
        """Keyed by the client, so a fresh client (a fresh process's) starts unbound."""
        first = await _connected(_gateway("old-gw"))
        await first.read_channel("SR:CH")

        monkeypatch.setitem(sys.modules, "pvaccess", FakePvaccess())

        await _connected(_gateway("new-gw"))  # does not raise


class TestTheStepReaderRunsOnTheWorkers:
    """No caller thread ever touches pvapy, whichever thread calls the reader."""

    @staticmethod
    def _recording_connector(delay: float = 0.0) -> tuple[EPICSConnector, list[str]]:
        threads: list[str] = []
        pvaccess = FakePvaccess()
        pvaccess.serve("SR:CH", record(4.0))

        def hook(request: str):
            if request == VALUE_REQUEST:
                threads.append(threading.current_thread().name)
                time.sleep(delay)
            return None

        pvaccess.get_hooks["SR:CH"] = hook
        return ca_connector(pvaccess), threads

    def test_a_call_from_the_caller_thread_reads_on_a_worker(self):
        connector, threads = self._recording_connector()

        assert connector._current_value_reader()("SR:CH") == 4.0

        assert len(threads) == 1
        assert threads[0].startswith("epics-worker-")

    def test_a_call_from_a_joined_pool_thread_reads_on_a_worker(self):
        """The notebook path: the runtime's coroutine runs on a ThreadPoolExecutor."""
        connector, threads = self._recording_connector()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            value = pool.submit(connector._current_value_reader(), "SR:CH").result()

        assert value == 4.0
        assert threads[0].startswith("epics-worker-")

    def test_a_read_that_does_not_answer_in_its_backstop_fails_closed(self):
        connector, _threads = self._recording_connector(delay=1.5)
        connector._step_read_timeout = 0.05  # backstop: 3 * 0.05 + 1 = 1.15 s

        start = time.monotonic()
        value = connector._current_value_reader()("SR:CH")

        assert value is None
        assert time.monotonic() - start < 1.45
