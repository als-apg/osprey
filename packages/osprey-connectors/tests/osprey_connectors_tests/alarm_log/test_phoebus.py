"""The alarm logger answers from recorded row shapes: three state rows, two config rows."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from osprey_connectors._service_http import LOGIN_PAGE_REASON, ReadOnlyTransport
from osprey_connectors.alarm_log import AlarmHistoryRow
from osprey_connectors.alarm_log._phoebus import PhoebusAlarmLogger
from osprey_connectors.alarm_state import AlarmConfigRecord
from osprey_connectors.service_answers import BackendSpec, Outcome, Unavailable

DEADLINE = 6.0
PV = "DEMO:RF1:CAV:Temp-I"
PATH = f"/Demo/RF/Cavity1/{PV}"
LABEL = "last logged configuration"


def _state(when: str, *, severity: str = "MAJOR", status: str = "NO_ALARM", **extra: Any) -> dict:
    return {
        "severity": severity,
        "message": "HIHI_ALARM",
        "value": "48.7",
        "time": when,
        "current_severity": "OK",
        "current_message": status,
        "message_time": when.replace(".120Z", ".402Z"),
        "config": f"state:{PATH}",
        "pv": PV,
        "enabled": True,
        **extra,
    }


def _config(when: str, *, user: str = "opsuser", path: str = PATH, **body: Any) -> dict:
    message = body or {
        "user": user,
        "host": "alarm-srv.demo.lab",
        "description": "Cavity 1 over-temperature",
        "enabled": True,
        "latching": True,
        "annunciating": False,
        "delay": 5,
        "count": 0,
        "guidance": [{"title": "Action", "details": "Check cooling water flow", "delay": 0}],
        "displays": [{"title": "Cavity", "details": "/opi/cavity.bob"}],
        "actions": [],
        "config": f"config:{path}",
        "message_time": 1792512131.009,
    }
    return {
        "message_time": when,
        "config": f"config:{path}",
        "config_msg": json.dumps(message),
        "user": user,
        "host": "alarm-srv.demo.lab",
        "enabled": True,
    }


#: Newest first, interleaved, as the service answers: three state rows and two config rows.
RECORDED = [
    _state("2026-10-29T03:14:07.120Z"),
    _config("2026-10-29T03:13:00.000Z"),
    _state("2026-10-20T16:40:11.120Z", severity="OK", status="NO_ALARM", mode="maintenance"),
    _config("2026-10-20T16:02:11.009Z", user="previous", guidance=[], latching=False),
    _state("2026-05-26T16:25:55.120Z", severity="MINOR"),
]


class Recording:
    """Answers each path from a table and keeps every request it was asked for."""

    def __init__(self, **answers: Any) -> None:
        self.answers = answers
        self.requests: list[dict[str, Any]] = []

    async def request(self, method: str, path: str, *, params: Any = None, deadline: float) -> Any:
        self.requests.append(
            {"method": method, "path": path, "params": params, "deadline": deadline}
        )
        key = {"/": "root", "/search/alarm": "search", "/search/alarm/config": "config"}[path]
        return self.answers[key]


def _logger(transport: Any = None, **settings: Any) -> PhoebusAlarmLogger:
    spec = BackendSpec(
        kind="alarm_log",
        type="phoebus_alarm_logger",
        target="live",
        settings={"url": "https://alarm-logger.demo.lab/", **settings},
        rehearsal=False,
    )
    return PhoebusAlarmLogger(spec, transport=transport)


@pytest.fixture
def loopback() -> Iterator[Any]:
    """A logger's front end: JSON as ``text/plain``, and 503 when the slash is missing."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            body, status, kind = Handler.routes.get(
                self.path, (b"<html>down</html>", 503, "text/html")
            )
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    root = json.dumps(
        {"name": "Alarm logging Service", "version": "4.7.3", "elastic": {"status": "GREEN"}}
    )
    Handler.routes = {"/alarm-logger/": (root.encode(), 200, "text/plain")}
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.routes = Handler.routes  # type: ignore[attr-defined]
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


# --- history ------------------------------------------------------------------------


async def test_history_keeps_both_kinds_as_recorded_and_newest_first():
    transport = Recording(search=RECORDED)
    page = await _logger(transport).history(PV, None, 50, DEADLINE)

    assert page.outcome is Outcome.FOUND
    assert [row.kind for row in page.rows] == ["state", "config", "state", "config", "state"]
    assert page.rows[0] == AlarmHistoryRow(
        time="2026-10-29T03:14:07.120Z",
        severity="MAJOR",
        status="NO_ALARM",
        message="HIHI_ALARM",
        user=None,
        kind="state",
    )
    assert page.rows[1] == AlarmHistoryRow(
        time="2026-10-29T03:13:00.000Z",
        message="Cavity 1 over-temperature",
        user="opsuser",
        kind="config",
    )
    assert page.truncated is False
    assert page.retention_known is False
    assert page.source.target == "live"
    assert transport.requests == [
        {
            "method": "GET",
            "path": "/search/alarm",
            "params": {"pv": PV, "size": 51},
            "deadline": DEADLINE,
        }
    ]


async def test_history_window_is_sent_in_days_and_applied_exactly():
    # The service ignored the window and answered everything; a fresh row keeps it in range.
    since = datetime.now(UTC) - timedelta(days=2, hours=1)
    fresh = _state((since + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.120Z"))
    stale = _state((since - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.120Z"))
    transport = Recording(search=[fresh, stale])

    page = await _logger(transport).history(PV, since, 50, DEADLINE)

    assert [row.time for row in page.rows] == [fresh["time"]]
    params = transport.requests[0]["params"]
    assert params["start"] == "3days"
    assert params["end"] == "now"


async def test_history_window_reads_a_config_row_by_its_message_time():
    since = datetime(2026, 10, 25, tzinfo=UTC)
    page = await _logger(Recording(search=RECORDED)).history(PV, since, 50, DEADLINE)
    assert [row.kind for row in page.rows] == ["state", "config"]


async def test_history_naive_since_reads_as_utc():
    since = datetime(2026, 10, 25)
    page = await _logger(Recording(search=RECORDED)).history(PV, since, 50, DEADLINE)
    assert len(page.rows) == 2


async def test_history_keeps_a_row_whose_time_cannot_be_read():
    row = {"config": f"state:{PATH}", "pv": PV, "time": "yesterday", "severity": "OK"}
    since = datetime(2026, 10, 25, tzinfo=UTC)
    page = await _logger(Recording(search=[row])).history(PV, since, 50, DEADLINE)
    assert len(page.rows) == 1


async def test_history_a_full_page_is_truncated_at_the_limit():
    page = await _logger(Recording(search=RECORDED[:3])).history(PV, None, 2, DEADLINE)
    assert len(page.rows) == 2
    assert page.truncated is True


async def test_history_a_page_that_fills_exactly_is_not_truncated():
    page = await _logger(Recording(search=RECORDED[:2])).history(PV, None, 2, DEADLINE)
    assert len(page.rows) == 2
    assert page.truncated is False


async def test_history_reaching_the_window_end_is_not_truncated():
    since = datetime(2026, 10, 25, tzinfo=UTC)
    page = await _logger(Recording(search=RECORDED[:3])).history(PV, since, 2, DEADLINE)
    assert len(page.rows) == 2
    assert page.truncated is False


async def test_history_filtered_rows_on_a_full_page_still_report_truncation():
    other = _state("2026-10-29T03:14:08.120Z", config=f"state:/Demo/{PV}-X", pv=f"{PV}-X")
    page = await _logger(Recording(search=[other, *RECORDED[:2]])).history(PV, None, 2, DEADLINE)
    assert len(page.rows) == 2
    assert page.truncated is True


async def test_history_no_rows_is_no_logged_events_not_no_alarm():
    page = await _logger(Recording(search=[])).history(PV, None, 50, DEADLINE)
    assert page.outcome is Outcome.NO_MATCH
    assert page.rows == []
    assert page.reason == "no logged events"


async def test_history_drops_rows_of_a_name_the_search_matched_loosely():
    near = _state("2026-10-29T03:14:08.120Z", config=f"state:/Demo/{PV}-X", pv=f"{PV}-X")
    page = await _logger(Recording(search=[near, "not a row"])).history(PV, None, 50, DEADLINE)
    assert page.outcome is Outcome.NO_MATCH


async def test_history_service_failure_carries_its_reason():
    page = await _logger(Recording(search=Unavailable(LOGIN_PAGE_REASON))).history(
        PV, None, 50, DEADLINE
    )
    assert page.outcome is Outcome.UNAVAILABLE
    assert page.reason == LOGIN_PAGE_REASON


async def test_history_a_bare_object_is_malformed():
    page = await _logger(Recording(search={"hits": []})).history(PV, None, 50, DEADLINE)
    assert page.outcome is Outcome.UNAVAILABLE
    assert page.reason == "malformed response"


async def test_history_a_deleted_config_row_says_so():
    row = _config("2026-10-29T03:13:00.000Z", delete="Deleting", user="opsuser")
    page = await _logger(Recording(search=[row])).history(PV, None, 50, DEADLINE)
    assert page.rows[0].message == "Deleting"
    assert page.rows[0].kind == "config"


# --- config_for ---------------------------------------------------------------------


async def test_config_found_reads_the_newest_config_row():
    transport = Recording(search=RECORDED)
    record = await _logger(transport).config_for(PV, DEADLINE)

    assert record == AlarmConfigRecord(
        in_tree="yes",
        path=PATH,
        guidance=[{"title": "Action", "details": "Check cooling water flow", "delay": 0}],
        displays=[{"title": "Cavity", "details": "/opi/cavity.bob"}],
        enabled=True,
        latching=True,
        label=LABEL,
    )
    assert [request["path"] for request in transport.requests] == ["/search/alarm"]
    assert transport.requests[0]["params"]["pv"] == PV


async def test_config_path_check_is_exact_on_the_last_segment():
    # The near name is newest, and is another PV's configuration.
    near = _config("2026-10-30T00:00:00.000Z", path=f"/Demo/{PV}-X")
    record = await _logger(Recording(search=[near, *RECORDED])).config_for(PV, DEADLINE)
    assert record.path == PATH

    alone = await _logger(Recording(search=[near])).config_for(PV, DEADLINE)
    assert alone.in_tree == "no"
    assert alone.path is None


async def test_config_path_check_reads_an_escaped_slash_and_a_scheme():
    pv = "loc://x/y"
    row = _config("2026-10-29T03:13:00.000Z", path="/Demo/loc:\\/\\/x\\/y")
    record = await _logger(Recording(search=[row])).config_for(pv, DEADLINE)
    assert record.in_tree == "yes"

    row = _config("2026-10-29T03:13:00.000Z", path=f"/Demo/ca://{PV}")
    row["config"] = f"config:/Demo/ca:\\/\\/{PV}"
    record = await _logger(Recording(search=[row])).config_for(PV, DEADLINE)
    assert record.in_tree == "yes"


async def test_config_not_in_tree_when_the_logger_holds_nothing():
    transport = Recording(search=[])
    record = await _logger(transport).config_for(PV, DEADLINE)
    assert record == AlarmConfigRecord(in_tree="no", label=LABEL)
    # Nothing names a path to ask the second route about.
    assert len(transport.requests) == 1


async def test_config_search_failure_keeps_its_reason():
    # A lookup with no answer at all carries why; the record has no field for a reason.
    record = await _logger(Recording(search=Unavailable("no answer within 6 s"))).config_for(
        PV, DEADLINE
    )
    assert record == Unavailable("no answer within 6 s")


async def test_config_fallback_runs_only_after_state_rows_with_no_config_row():
    transport = Recording(search=[RECORDED[0]], config=[RECORDED[1]])
    record = await _logger(transport).config_for(PV, DEADLINE)
    assert record.in_tree == "yes"
    assert record.label == LABEL
    assert transport.requests[1]["path"] == "/search/alarm/config"
    assert transport.requests[1]["params"] == {"config": f"config:{PATH}"}
    assert 0 < transport.requests[1]["deadline"] <= DEADLINE


@pytest.mark.parametrize(
    "fallback",
    [Unavailable("answered HTTP 404"), [], [_config("2026-10-29", path="/Demo/other")], {"a": 1}],
    ids=["route-fails", "no-rows", "another-pv", "not-a-list"],
)
async def test_config_lookup_failure_is_unknown(fallback: Any):
    # The state row proves the path; the unmeasured route's failure proves nothing.
    transport = Recording(search=[RECORDED[0]], config=fallback)
    record = await _logger(transport).config_for(PV, DEADLINE)
    assert record == AlarmConfigRecord(in_tree="unknown", path=PATH, label=LABEL)


async def test_config_state_row_without_a_path_is_unknown_with_no_second_request():
    row = {"pv": PV, "severity": "OK", "time": "2026-10-29T03:14:07.120Z"}
    transport = Recording(search=[row])
    record = await _logger(transport).config_for(PV, DEADLINE)
    assert record == AlarmConfigRecord(in_tree="unknown", path=None, label=LABEL)
    assert len(transport.requests) == 1


async def test_config_deleted_means_removed():
    deleted = _config("2026-10-29T03:13:00.000Z", delete="Deleting", user="opsuser")
    record = await _logger(Recording(search=[deleted, *RECORDED])).config_for(PV, DEADLINE)
    assert record == AlarmConfigRecord(in_tree="no", path=PATH, label=LABEL)


@pytest.mark.parametrize("config_msg", ["{not json", None, "[1]"], ids=["bad", "absent", "list"])
async def test_config_row_that_cannot_be_read_is_unknown(config_msg: Any):
    row = {**_config("2026-10-29T03:13:00.000Z"), "config_msg": config_msg}
    record = await _logger(Recording(search=[row])).config_for(PV, DEADLINE)
    assert record == AlarmConfigRecord(in_tree="unknown", path=PATH, label=LABEL)


async def test_config_record_never_carries_the_notification_actions():
    # The record has no field for them, so a conversion to JSON cannot leak one.
    row = _config("2026-10-29T03:13:00.000Z")
    message = json.loads(row["config_msg"])
    message["actions"] = [{"title": "Mail", "details": "mailto:someone@demo.lab"}]
    row["config_msg"] = json.dumps(message)
    record = await _logger(Recording(search=[row])).config_for(PV, DEADLINE)
    assert "someone@demo.lab" not in json.dumps(asdict(record))


# --- probe and the real transport ---------------------------------------------------


async def test_probe_on_a_text_plain_root(loopback: Any):
    url = f"http://127.0.0.1:{loopback.server_address[1]}/alarm-logger"
    logger = _logger(None, url=url)
    answer = await logger.probe(DEADLINE)
    assert answer == "root answered (Alarm logging Service 4.7.3); this does not prove ingestion"


async def test_probe_sends_only_the_slashed_root():
    transport = Recording(root={"name": "Alarm logging Service"})
    answer = await _logger(transport).probe(DEADLINE)
    assert "does not prove ingestion" in answer
    assert [(r["method"], r["path"]) for r in transport.requests] == [("GET", "/")]


async def test_probe_failure_and_a_root_that_is_not_an_object():
    assert await _logger(Recording(root=Unavailable("x"))).probe(DEADLINE) == Unavailable("x")
    answer = await _logger(Recording(root=[1])).probe(DEADLINE)
    assert answer == Unavailable("malformed response")


async def test_probe_names_nothing_it_was_not_told():
    answer = await _logger(Recording(root={})).probe(DEADLINE)
    assert answer == "root answered (unnamed service); this does not prove ingestion"


async def test_search_over_the_real_transport_reaches_the_service(loopback: Any):
    port = loopback.server_address[1]
    rows = json.dumps(RECORDED).encode()
    path = "/alarm-logger/search/alarm?pv=" + PV.replace(":", "%3A") + "&size=51"
    loopback.routes[path] = (rows, 200, "text/plain")
    logger = _logger(None, url=f"http://127.0.0.1:{port}/alarm-logger/")

    page = await logger.history(PV, None, 50, DEADLINE)

    assert page.outcome is Outcome.FOUND
    assert len(page.rows) == 5


async def test_the_allowed_requests_admit_the_routes_and_refuse_a_walk():
    transport = ReadOnlyTransport(
        {"url": "https://alarm-logger.demo.lab/"},
        PhoebusAlarmLogger.ALLOWED_REQUESTS,
        where="pv_directory.alarm_log.settings",
    )
    assert transport._allows("GET", "/")
    assert transport._allows("GET", "/search/alarm")
    assert transport._allows("GET", "/search/alarm/config")
    assert not transport._allows("POST", "/search/alarm")
    assert not transport._allows("GET", "/search/../import")
    assert not transport._allows("GET", "/import")


def test_building_one_does_no_io():
    # An unusable block is a value on the first read, not an exception here.
    logger = _logger(None, url=None, tls="not a block")
    assert logger.spec.type == "phoebus_alarm_logger"
