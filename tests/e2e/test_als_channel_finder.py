"""E2E tests for ALSChannelFinderBackend against the real ALS ChannelFinder API.

Requires an SSH tunnel to be open:
    ./scripts/als_channel_finder_tunnel.sh

Run with:
    pytest tests/e2e/test_als_channel_finder.py -v
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from osprey.services.channel_finder.backends.als_channel_finder import (
    ALSChannelFinderBackend,
)


def _is_tunnel_available() -> bool:
    try:
        s = socket.create_connection(("localhost", 8443), timeout=2)
        s.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _is_tunnel_available(),
        reason="SSH tunnel to ALS ChannelFinder not available on localhost:8443",
    ),
]


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def backend():
    b = ALSChannelFinderBackend("https://localhost:8443/ChannelFinder")
    yield b
    _run(b.close())


class TestALSChannelFinderE2E:
    def test_search_bpm_pvs(self, backend):
        result = _run(backend.search("SR*BPM*"))
        assert result.total_count > 0
        assert len(result.records) > 0
        for rec in result.records:
            assert "BPM" in rec.name
            assert rec.name  # non-empty

    def test_pagination(self, backend):
        page1 = _run(backend.search("SR*", page=1, page_size=10))
        page2 = _run(backend.search("SR*", page=2, page_size=10))

        assert len(page1.records) == 10
        assert len(page2.records) > 0

        names1 = {r.name for r in page1.records}
        names2 = {r.name for r in page2.records}
        assert names1.isdisjoint(names2), "Pages should have different records"

    def test_record_type_filter(self, backend):
        result = _run(backend.search("SR*", record_type="ai", page_size=20))
        assert result.total_count > 0
        for rec in result.records:
            assert rec.record_type == "ai"

    def test_get_metadata(self, backend):
        # First find a PV name via search
        search = _run(backend.search("SR*BPM*", page_size=1))
        assert len(search.records) > 0
        pv_name = search.records[0].name

        records = _run(backend.get_metadata([pv_name]))
        assert len(records) == 1
        assert records[0].name == pv_name
        assert records[0].host  # should have a hostname

    def test_large_result_count(self, backend):
        """Verify the API can report large counts (265k+ PVs at ALS)."""
        result = _run(backend.search("*", page_size=10))
        assert result.total_count > 1000
        assert result.has_more is True
