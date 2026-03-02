"""Tests for MockPVInfoBackend."""

import asyncio

import pytest

from osprey.services.channel_finder.backends.base import PVRecord, SearchResult
from osprey.services.channel_finder.backends.mock import MockPVInfoBackend


@pytest.fixture
def backend():
    return MockPVInfoBackend()


def _run(coro):
    return asyncio.run(coro)


class TestMockPVInfoBackend:
    def test_total_pv_count(self, backend):
        assert backend.total_pv_count > 400

    def test_search_all(self, backend):
        result = _run(backend.search("*"))
        assert isinstance(result, SearchResult)
        assert result.total_count == backend.total_pv_count
        assert result.page == 1

    def test_search_bpm_pattern(self, backend):
        result = _run(backend.search("SR:*:BPM:*"))
        assert result.total_count > 0
        for rec in result.records:
            assert "BPM" in rec.name

    def test_search_sector_pattern(self, backend):
        result = _run(backend.search("SR:01:*"))
        assert result.total_count > 0
        for rec in result.records:
            assert rec.name.startswith("SR:01:")

    def test_search_global_devices(self, backend):
        result = _run(backend.search("SR:DCCT:*"))
        assert result.total_count > 0
        for rec in result.records:
            assert "DCCT" in rec.name

    def test_filter_by_record_type(self, backend):
        result = _run(backend.search("*", record_type="ao"))
        assert result.total_count > 0
        for rec in result.records:
            assert rec.record_type == "ao"

    def test_filter_by_ioc(self, backend):
        result = _run(backend.search("*", ioc="IOC:BPM:01"))
        assert result.total_count > 0
        for rec in result.records:
            assert rec.ioc == "IOC:BPM:01"

    def test_filter_by_description(self, backend):
        result = _run(backend.search("*", description_contains="beam current"))
        assert result.total_count > 0
        for rec in result.records:
            assert "beam current" in rec.description.lower() or "Beam Current" in rec.description

    def test_pagination(self, backend):
        page1 = _run(backend.search("*", page=1, page_size=10))
        assert len(page1.records) == 10
        assert page1.has_more is True
        assert page1.page == 1

        page2 = _run(backend.search("*", page=2, page_size=10))
        assert len(page2.records) == 10
        assert page2.page == 2

        # Pages should have different records
        names1 = {r.name for r in page1.records}
        names2 = {r.name for r in page2.records}
        assert names1.isdisjoint(names2)

    def test_page_size_capped_at_200(self, backend):
        result = _run(backend.search("*", page_size=999))
        assert result.page_size == 200

    def test_empty_search(self, backend):
        result = _run(backend.search("NONEXISTENT:*"))
        assert result.total_count == 0
        assert result.records == []
        assert result.has_more is False

    def test_get_metadata(self, backend):
        # First find a PV name
        search = _run(backend.search("SR:01:BPM:*", page_size=1))
        pv_name = search.records[0].name

        records = _run(backend.get_metadata([pv_name]))
        assert len(records) == 1
        assert isinstance(records[0], PVRecord)
        assert records[0].name == pv_name
        assert records[0].units != ""

    def test_get_metadata_missing(self, backend):
        records = _run(backend.get_metadata(["DOES:NOT:EXIST"]))
        assert records == []

    def test_get_metadata_mixed(self, backend):
        search = _run(backend.search("SR:01:BPM:*", page_size=1))
        pv_name = search.records[0].name

        records = _run(backend.get_metadata([pv_name, "DOES:NOT:EXIST"]))
        assert len(records) == 1
        assert records[0].name == pv_name

    def test_pv_record_fields(self, backend):
        search = _run(backend.search("SR:01:BPM:*", page_size=1))
        rec = search.records[0]
        assert rec.name
        assert rec.record_type in ("ai", "ao")
        assert rec.description
        assert rec.host
        assert rec.ioc
        assert rec.units
        assert isinstance(rec.tags, dict)

    def test_deterministic_generation(self):
        b1 = MockPVInfoBackend()
        b2 = MockPVInfoBackend()
        assert b1.total_pv_count == b2.total_pv_count
        r1 = _run(b1.search("*", page_size=5))
        r2 = _run(b2.search("*", page_size=5))
        assert [r.name for r in r1.records] == [r.name for r in r2.records]
