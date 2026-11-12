"""The caPutLog backend, against the response shapes a caputlog server answers."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from osprey_connectors.put_log import PutLog, PutRow
from osprey_connectors.put_log._caputlog import CaPutLog
from osprey_connectors.service_answers import BackendSpec, Outcome, Unavailable

PV = "DEMO:RF1:CAV:TempLimit-SP"

#: One result as the service boxes it, field for field.
RECORDED_HIT = {
    "id": {"raw": "Xk2q9"},
    "_meta": {
        "id": "Xk2q9",
        "rawHit": {"_index": "putlog-2026-10-28", "_id": "Xk2q9", "_score": 1},
    },
    "pv": {"raw": f"{PV}.HIHI"},
    "old": {"raw": "45"},
    "new": {"raw": "50"},
    "old_float": {"raw": 45.0},
    "new_float": {"raw": 50.0},
    "user": {"raw": "opsuser"},
    "client": {"raw": "opi-02.demo.lab"},
    "ioctimestamp": {"raw": "2026-10-28T21:03:11.000Z"},
    "@timestamp": {"raw": "2026-10-28T21:03:12.884366304Z"},
    "timestamp": {"raw": "28-Oct-26 14:03:11"},
    "message": {
        "raw": f"28-Oct-26 14:03:11 opi-02.demo.lab opsuser {PV}.HIHI new=50 old=45",
    },
    "event": {"raw": {"original": f"28-Oct-26 14:03:11 opi-02.demo.lab opsuser {PV}.HIHI"}},
    "@version": {"raw": "1"},
}


def _hit(name: str = PV, *, minute: int = 0, **fields: Any) -> dict[str, Any]:
    recorded = {
        "pv": name,
        "old": "1",
        "new": "2",
        "user": "opsuser",
        "client": "opi-02.demo.lab",
        "ioctimestamp": f"2026-10-28T21:{minute:02d}:00.000Z",
        **fields,
    }
    return {key: {"raw": value} for key, value in recorded.items() if value is not None}


def _envelope(results: list[dict[str, Any]], total: Any = None) -> dict[str, Any]:
    return {
        "facets": {},
        "pagingEnd": len(results),
        "pagingStart": 1,
        "rawResponse": None,
        "requestId": "",
        "resultSearchTerm": "",
        "results": results,
        "totalPages": 1,
        "totalResults": len(results) if total is None else total,
        # False on a success, as measured: never read as a signal.
        "wasSearched": False,
    }


class _Transport:
    """Answers what it is given and records what it was asked."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        self.calls.append({"method": method, "path": path, **kwargs})
        return self.answer


def _spec(**settings: Any) -> BackendSpec:
    return BackendSpec(
        kind="put_log", type="caputlog", target="live", settings=settings, rehearsal=False
    )


def _backend(answer: Any, **settings: Any) -> tuple[CaPutLog, _Transport]:
    transport = _Transport(answer)
    return CaPutLog(_spec(**settings), transport=transport), transport


async def test_the_search_is_the_measured_body_and_never_a_get() -> None:
    backend, transport = _backend(_envelope([]))

    await backend.history(PV, None, 25, 5.0)

    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert (call["method"], call["path"]) == ("POST", "/search")
    assert call["deadline"] == 5.0
    assert call["json_body"] == {
        "state": {"searchTerm": f"{PV}.*", "current": 1, "resultsPerPage": 26},
        "queryConfig": {"search_fields": {"pv": {}}},
    }
    assert CaPutLog.ALLOWED_REQUESTS == (("POST", "/search"),)


async def test_a_recorded_result_unboxes_to_a_row_as_recorded() -> None:
    backend, _ = _backend(_envelope([RECORDED_HIT]))

    page = await backend.history(PV, None, 25, 5.0)

    assert page.outcome is Outcome.FOUND
    assert page.rows == [
        PutRow(
            at="2026-10-28T21:03:11.000Z",
            pv=f"{PV}.HIHI",
            field=".HIHI",
            old="45",
            new="50",
            user="opsuser",
            client="opi-02.demo.lab",
        )
    ]
    assert page.retention_known is False
    assert page.truncated is False
    assert page.reason is None
    assert (page.source.target, page.source.service) == ("live", "caputlog")


async def test_a_put_to_the_pv_itself_has_no_field() -> None:
    backend, _ = _backend(_envelope([_hit()]))

    page = await backend.history(PV, None, 25, 5.0)

    assert page.rows[0].pv == PV
    assert page.rows[0].field is None


async def test_a_missing_ioc_stamp_falls_back_to_the_ingest_stamp() -> None:
    hit = _hit(ioctimestamp=None, **{"@timestamp": "2026-10-28T21:03:12.884Z"})
    backend, _ = _backend(_envelope([hit]))

    page = await backend.history(PV, None, 25, 5.0)

    assert page.rows[0].at == "2026-10-28T21:03:12.884Z"


async def test_a_field_absent_from_the_record_reads_none() -> None:
    backend, _ = _backend(_envelope([_hit(user=None, old=None)]))

    row = (await backend.history(PV, None, 25, 5.0)).rows[0]

    assert (row.user, row.old, row.new) == (None, None, "2")


async def test_rows_of_other_pvs_the_text_match_returned_are_not_this_pvs_history() -> None:
    hits = [
        _hit(f"{PV}.HIHI", minute=3),
        _hit(f"{PV}2", minute=2),
        _hit("DEMO:RF2:CAV:TempLimit-SP", minute=1),
        _hit(PV.lower(), minute=4),
    ]
    backend, _ = _backend(_envelope(hits))

    page = await backend.history(PV, None, 25, 5.0)

    assert [row.pv for row in page.rows] == [f"{PV}.HIHI"]


async def test_only_rows_of_other_pvs_is_no_logged_puts() -> None:
    backend, _ = _backend(_envelope([_hit(f"{PV}2")]))

    page = await backend.history(PV, None, 25, 5.0)

    assert page.outcome is Outcome.NO_MATCH
    assert page.rows == []


async def test_rows_come_newest_first_whatever_order_the_service_ranks_them() -> None:
    backend, _ = _backend(_envelope([_hit(minute=1), _hit(minute=3), _hit(minute=2)]))

    page = await backend.history(PV, None, 25, 5.0)

    assert [row.at[14:16] for row in page.rows] == ["03", "02", "01"]


async def test_one_row_past_the_limit_truncates_and_keeps_the_newest() -> None:
    backend, _ = _backend(_envelope([_hit(minute=1), _hit(minute=3), _hit(minute=2)]))

    page = await backend.history(PV, None, 2, 5.0)

    assert [row.at[14:16] for row in page.rows] == ["03", "02"]
    assert page.truncated is True


async def test_a_page_that_exactly_fills_the_limit_is_not_truncated() -> None:
    backend, _ = _backend(_envelope([_hit(minute=1), _hit(minute=2)]))

    page = await backend.history(PV, None, 2, 5.0)

    assert len(page.rows) == 2
    assert page.truncated is False


async def test_matches_the_service_holds_beyond_the_page_truncate() -> None:
    backend, _ = _backend(_envelope([_hit(f"{PV}2"), _hit(minute=1), _hit(f"{PV}3")], total=132))

    page = await backend.history(PV, None, 2, 5.0)

    assert len(page.rows) == 1
    assert page.truncated is True


async def test_other_pvs_filling_a_fully_read_page_do_not_truncate() -> None:
    backend, _ = _backend(_envelope([_hit(f"{PV}2"), _hit(minute=1), _hit(f"{PV}3")]))

    page = await backend.history(PV, None, 2, 5.0)

    assert len(page.rows) == 1
    assert page.truncated is False


async def test_without_a_total_a_full_page_still_says_truncated() -> None:
    answer = _envelope([_hit(f"{PV}2"), _hit(minute=1), _hit(f"{PV}3")])
    del answer["totalResults"]
    backend, _ = _backend(answer)

    page = await backend.history(PV, None, 2, 5.0)

    assert page.truncated is True


async def test_since_narrows_the_page_on_this_side_and_is_not_sent() -> None:
    backend, transport = _backend(_envelope([_hit(minute=1), _hit(minute=3)]))

    page = await backend.history(PV, datetime(2026, 10, 28, 21, 2, tzinfo=UTC), 25, 5.0)

    assert [row.at[14:16] for row in page.rows] == ["03"]
    assert "@timestamp" not in json.dumps(transport.calls[0]["json_body"])
    assert "filters" not in transport.calls[0]["json_body"]["state"]


async def test_since_compares_instants_across_zones_and_reads_a_naive_one_as_utc() -> None:
    backend, _ = _backend(_envelope([_hit(minute=1), _hit(minute=3)]))
    pacific = timezone(timedelta(hours=-7))

    zoned = await backend.history(PV, datetime(2026, 10, 28, 14, 2, tzinfo=pacific), 25, 5.0)
    naive = await backend.history(PV, datetime(2026, 10, 28, 21, 2), 25, 5.0)

    assert [row.at[14:16] for row in zoned.rows] == ["03"]
    assert [row.at[14:16] for row in naive.rows] == ["03"]


async def test_a_row_whose_time_cannot_be_read_stays_and_sorts_last() -> None:
    hits = [_hit(ioctimestamp="28-Oct-26 14:03:11"), _hit(minute=3), _hit(minute=1)]
    backend, _ = _backend(_envelope(hits))

    page = await backend.history(PV, datetime(2026, 10, 28, 21, 2, tzinfo=UTC), 25, 5.0)

    assert [row.at for row in page.rows] == ["2026-10-28T21:03:00.000Z", "28-Oct-26 14:03:11"]


async def test_since_past_every_row_is_no_logged_puts() -> None:
    backend, _ = _backend(_envelope([_hit(minute=1)]))

    page = await backend.history(PV, datetime(2027, 1, 1, tzinfo=UTC), 25, 5.0)

    assert page.outcome is Outcome.NO_MATCH
    assert page.reason == "no logged puts"


async def test_an_empty_result_says_no_logged_puts_and_never_that_it_never_changed() -> None:
    backend, _ = _backend(_envelope([]))

    page = await backend.history(PV, None, 25, 5.0)

    assert page.outcome is Outcome.NO_MATCH
    assert page.rows == []
    assert page.reason == "no logged puts"
    assert "never" not in page.reason
    assert page.retention_known is False
    assert page.truncated is False


async def test_a_service_with_no_answer_is_unavailable_with_its_reason() -> None:
    backend, _ = _backend(Unavailable("no answer within 5 s"))

    page = await backend.history(PV, None, 25, 5.0)

    assert page.outcome is Outcome.UNAVAILABLE
    assert page.reason == "no answer within 5 s"
    assert page.rows == []
    assert page.retention_known is False


@pytest.mark.parametrize(
    "answer",
    [
        [],
        {"error": "boom"},
        {"results": {"raw": "x"}},
        {"results": ["not a hit"]},
    ],
)
async def test_an_answer_that_is_not_the_envelope_is_malformed(answer: Any) -> None:
    backend, _ = _backend(answer)

    page = await backend.history(PV, None, 25, 5.0)

    assert page.outcome is Outcome.UNAVAILABLE
    assert page.reason == "malformed response"


@pytest.mark.parametrize(("limit", "asked"), [(0, 2), (1, 2), (200, 201), (5000, 201)])
async def test_the_limit_is_bounded_before_it_reaches_the_service(limit: int, asked: int) -> None:
    backend, transport = _backend(_envelope([]))

    await backend.history(PV, None, limit, 5.0)

    assert transport.calls[0]["json_body"]["state"]["resultsPerPage"] == asked


async def test_probe_says_the_search_answered_through_the_only_allowed_request() -> None:
    backend, transport = _backend(_envelope([_hit()], total=132))

    assert await backend.probe(3.0) == "search answered"
    call = transport.calls[0]
    assert (call["method"], call["path"], call["deadline"]) == ("POST", "/search", 3.0)
    assert call["json_body"]["state"]["resultsPerPage"] == 1


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (Unavailable("answered HTTP 503"), "answered HTTP 503"),
        ({"status": "ok"}, "malformed response"),
    ],
)
async def test_probe_reports_why_not(answer: Any, reason: str) -> None:
    backend, _ = _backend(answer)

    assert await backend.probe(3.0) == Unavailable(reason)


def test_it_is_a_put_log_and_building_one_touches_nothing() -> None:
    backend = CaPutLog(_spec(url="https://nowhere.invalid/caputlog-server"), project_dir=None)

    assert isinstance(backend, PutLog)


@pytest.fixture
def caputlog_server() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    received: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            received.append({"path": self.path, "body": json.loads(self.rfile.read(length))})
            body = json.dumps(_envelope([RECORDED_HIT])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/caputlog-server", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_without_a_transport_it_builds_the_read_only_one_from_its_settings(
    caputlog_server: tuple[str, list[dict[str, Any]]],
) -> None:
    url, received = caputlog_server
    backend = CaPutLog(_spec(url=url))

    page = await backend.history(PV, None, 25, 5.0)

    assert page.outcome is Outcome.FOUND
    assert page.rows[0].field == ".HIHI"
    assert received[0]["path"] == "/caputlog-server/search"
    assert received[0]["body"]["state"]["searchTerm"] == f"{PV}.*"
