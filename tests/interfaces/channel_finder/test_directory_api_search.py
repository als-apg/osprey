"""The PV directory search routes: ``search``, ``count`` and ``export.csv``.

A fake service stands in for ``DirectoryService``: the routes pass its answers
through and build no contract type, so the tests need no connectors release.
"""

from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.channel_finder import directory_api

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "directory_routes.json").read_text(encoding="utf-8")
)
_COLUMNS = [
    "name",
    "origin",
    "status",
    "description",
    "record_type",
    "ioc",
    "host",
    "last_announced",
    "alarm",
    "archived",
]
#: The fixture page's row count: a page of this size comes back full.
_PAGE_ROWS = len(_FIXTURE["search"]["rows"])
_UNAVAILABLE = {
    "outcome": "unavailable",
    "total": None,
    "source": _FIXTURE["count"]["source"],
    "reason": "no answer within 6 s",
}


class FakeService:
    """Answers every search with the fixture's page; records each call's keywords."""

    def __init__(
        self, *, total: int = 2, count: dict[str, Any] | None = None, filter_cap: int = 5000
    ) -> None:
        self.filter_cap = filter_cap
        self.searches: list[dict[str, Any]] = []
        self.counts: list[dict[str, Any]] = []
        self._total = total
        self._count = count

    async def search(self, **fields: Any) -> dict[str, Any]:
        self.searches.append(fields)
        page = copy.deepcopy(_FIXTURE["search"])
        page["total"] = self._total
        return page

    async def count(self, **fields: Any) -> dict[str, Any]:
        self.counts.append(fields)
        if self._count is not None:
            return self._count
        return {**_FIXTURE["count"], "total": self._total}


def _client(service: FakeService) -> TestClient:
    app = FastAPI()
    app.include_router(directory_api.router, prefix="/api/directory")
    app.state.directory_service = service
    return TestClient(app)


def _cell(row: dict[str, Any], column: str) -> str:
    value = row.get(column)
    if column == "alarm":
        severity = (value or {}).get("severity")
        return "" if severity in (None, "OK") else severity
    if column == "archived":
        return {True: "yes", False: "no"}.get(value, "")
    return "" if value is None else str(value)


def _expected_rows() -> list[list[str]]:
    return [[_cell(row, column) for column in _COLUMNS] for row in _FIXTURE["search"]["rows"]]


class TestSearch:
    def test_the_payload_matches_the_shared_fixture(self):
        resp = _client(FakeService()).get("/api/directory/search", params={"q": "BPM01"})

        assert resp.status_code == 200
        assert resp.json() == _FIXTURE["search"]

    def test_repeated_filters_reach_the_service_as_keywords(self):
        service = FakeService()
        params = [
            ("q", "BPM01"),
            ("name", "DEMO:*"),
            ("origin", "curated"),
            ("origin", "directory"),
            ("ioc", "demo-ioc-bpm"),
            ("ioc", "demo-ioc-rf"),
            ("record_type", "ai"),
            ("record_type", "calc"),
            ("alarm", "MAJOR"),
            ("alarm", "none"),
            ("archived", "no"),
            ("page", "2"),
            ("page_size", "25"),
        ]

        _client(service).get("/api/directory/search", params=params)

        assert service.searches == [
            {
                "words": "BPM01",
                "name": "DEMO:*",
                "origin": ("curated", "directory"),
                "ioc": ("demo-ioc-bpm", "demo-ioc-rf"),
                "record_type": ("ai", "calc"),
                "alarm": ("MAJOR", "none"),
                "archived": ("no",),
                "page": 2,
                "page_size": 25,
            }
        ]

    def test_no_parameters_search_every_pv_from_the_first_page(self):
        service = FakeService()

        _client(service).get("/api/directory/search")

        assert service.searches == [
            {
                "words": "",
                "name": None,
                "origin": (),
                "ioc": (),
                "record_type": (),
                "alarm": (),
                "archived": (),
                "page": 0,
                "page_size": 50,
            }
        ]

    @pytest.mark.parametrize("params", [{"page_size": "51"}, {"page_size": "0"}, {"page": "-1"}])
    def test_an_out_of_range_page_is_refused(self, params):
        service = FakeService()

        resp = _client(service).get("/api/directory/search", params=params)

        assert resp.status_code == 422
        assert service.searches == []

    def test_the_routes_are_read_only(self):
        client = _client(FakeService())
        for path in ("search", "count", "export.csv"):
            assert client.post(f"/api/directory/{path}").status_code == 405


class TestCount:
    def test_without_q_it_counts_the_whole_directory(self):
        service = FakeService()

        resp = _client(service).get("/api/directory/count")

        assert resp.json() == _FIXTURE["count"]
        assert service.counts == [{"words": "", "alarm": (), "archived": ()}]

    def test_with_q_it_counts_the_words(self):
        service = FakeService()

        resp = _client(service).get("/api/directory/count", params={"q": "BPM01"})

        assert resp.json() == _FIXTURE["count"]
        assert service.counts == [{"words": "BPM01", "alarm": (), "archived": ()}]

    def test_an_alarm_filter_reaches_the_count(self):
        service = FakeService()

        params = [("q", "BPM01"), ("alarm", "MAJOR"), ("alarm", "MINOR"), ("archived", "no")]
        _client(service).get("/api/directory/count", params=params)

        assert service.counts == [
            {"words": "BPM01", "alarm": ("MAJOR", "MINOR"), "archived": ("no",)}
        ]

    def test_an_unavailable_count_carries_its_reason(self):
        resp = _client(FakeService(count=_UNAVAILABLE)).get("/api/directory/count")

        assert resp.status_code == 200
        assert resp.json() == _UNAVAILABLE


class TestExport:
    def test_the_csv_names_the_search_then_the_columns_then_one_line_per_row(self):
        resp = _client(FakeService()).get("/api/directory/export.csv", params={"q": "BPM01"})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert resp.headers["x-content-type-options"] == "nosniff"
        lines = list(csv.reader(io.StringIO(resp.text)))
        assert lines[0] == [_FIXTURE["search"]["searched"]]
        assert lines[1] == _COLUMNS
        assert lines[2:] == _expected_rows()
        assert [line[-2:] for line in lines[2:]] == [["MAJOR", "yes"], ["", "no"]]

    def test_an_alarm_filter_reaches_the_export(self):
        service = FakeService()

        _client(service).get(
            "/api/directory/export.csv", params={"alarm": "none", "archived": "no"}
        )

        assert service.counts[0]["alarm"] == ("none",)
        assert service.searches[0]["alarm"] == ("none",)
        assert service.counts[0]["archived"] == ("no",)
        assert service.searches[0]["archived"] == ("no",)

    def test_within_the_filter_cap_one_search_fetches_the_whole_set(self):
        service = FakeService(total=120)
        params = [("q", "BPM01"), ("ioc", "demo-ioc-bpm"), ("page", "1"), ("page_size", "25")]

        _client(service).get("/api/directory/export.csv", params=params)

        filters = {"words": "BPM01", "name": None, "origin": (), "ioc": ("demo-ioc-bpm",)}
        filters.update(record_type=(), alarm=(), archived=())
        assert service.counts == [filters]
        assert service.searches == [{**filters, "page": 0, "page_size": 120}]

    def test_above_the_filter_cap_it_carries_the_fetched_pages(self):
        service = FakeService(total=11, filter_cap=10)
        params = {"q": "BPM01", "page": "2", "page_size": str(_PAGE_ROWS)}

        resp = _client(service).get("/api/directory/export.csv", params=params)

        assert [(call["page"], call["page_size"]) for call in service.searches] == [
            (0, 2),
            (1, 2),
            (2, 2),
        ]
        lines = list(csv.reader(io.StringIO(resp.text)))
        assert lines[2:] == _expected_rows() * 3

    def test_an_unknown_count_carries_the_fetched_pages(self):
        service = FakeService(count=_UNAVAILABLE)

        params = {"page": "1", "page_size": str(_PAGE_ROWS)}
        _client(service).get("/api/directory/export.csv", params=params)

        assert [(call["page"], call["page_size"]) for call in service.searches] == [
            (0, 2),
            (1, 2),
        ]

    def test_a_short_page_ends_the_fetched_pages(self):
        service = FakeService(count=_UNAVAILABLE)

        params = {"page": "2", "page_size": "25"}
        _client(service).get("/api/directory/export.csv", params=params)

        assert [(call["page"], call["page_size"]) for call in service.searches] == [(0, 25)]

    def test_an_over_cap_page_is_clamped_to_the_filter_cap(self):
        service = FakeService(total=11, filter_cap=3)

        params = {"page": "1000000", "page_size": str(_PAGE_ROWS)}
        resp = _client(service).get("/api/directory/export.csv", params=params)

        assert [(call["page"], call["page_size"]) for call in service.searches] == [
            (0, 2),
            (1, 2),
        ]
        lines = list(csv.reader(io.StringIO(resp.text)))
        assert lines[2:] == (_expected_rows() * 2)[:3]

    @pytest.mark.parametrize("trigger", ["=", "+", "-", "@", "\t", "\r"])
    def test_a_cell_starting_a_formula_is_exported_quoted(self, trigger):
        text = f"{trigger}HYPERLINK(1)"

        class Formula(FakeService):
            async def search(self, **fields: Any) -> dict[str, Any]:
                page = await super().search(**fields)
                page["searched"] = text
                page["rows"][0]["description"] = text
                return page

        resp = _client(Formula()).get("/api/directory/export.csv")

        lines = list(csv.reader(io.StringIO(resp.text)))
        assert lines[0] == [f"'{text}"]
        assert lines[2][_COLUMNS.index("description")] == f"'{text}"

    def test_an_unavailable_search_answers_its_reason(self):
        class Unavailable(FakeService):
            async def search(self, **fields: Any) -> dict[str, Any]:
                return copy.deepcopy(_FIXTURE["search no source"])

        resp = _client(Unavailable(count=_UNAVAILABLE)).get("/api/directory/export.csv")

        assert resp.status_code == 503
        assert resp.json() == {"detail": "not available for this target"}
