"""The one HTTP path of every directory backend, over a real loopback server."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from osprey_connectors import _service_http as transport_module
from osprey_connectors._service_http import LOGIN_PAGE_REASON, ReadOnlyTransport
from osprey_connectors.service_answers import Unavailable

CHANNELFINDER = (
    ("GET", "/resources/channels"),
    ("GET", "/resources/channels/count"),
    ("GET", "/resources/properties"),
    ("GET", "/resources/tags"),
)
ALARM_LOGGER = (("GET", "/"), ("GET", "/search/*"))
PUT_LOG = (("POST", "/search"),)
JSON = "application/json"


class _Service:
    """A loopback HTTP service: answers by path and records every request it receives."""

    def __init__(self) -> None:
        self.answers: dict[str, tuple[int, dict[str, str], bytes]] = {}
        self.received: list[dict] = []
        self.hang_s = 0.0
        self.url = ""

    def answer(self, path: str, status: int, content_type: str | None, body: bytes = b"") -> None:
        headers = {"Content-Type": content_type} if content_type else {}
        self.answers[path] = (status, headers, body)

    def redirect(self, path: str, status: int, location: str) -> None:
        self.answers[path] = (status, {"Location": location}, b"")


@pytest.fixture
def service() -> Iterator[_Service]:
    state = _Service()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            state.received.append(
                {
                    "method": self.command,
                    "target": self.path,
                    "headers": dict(self.headers),
                    "body": self.rfile.read(length),
                }
            )
            if state.hang_s:
                time.sleep(state.hang_s)
            status, headers, body = state.answers.get(
                self.path.split("?")[0], (404, {"Content-Type": JSON}, b'{"error": "absent"}')
            )
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_PUT = do_DELETE = _reply

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def _transport(service: _Service, allowed=CHANNELFINDER, **settings) -> ReadOnlyTransport:
    return ReadOnlyTransport(
        {"url": service.url, **settings}, allowed, where="pv_directory.settings"
    )


@contextmanager
def _serving(do_get: Callable[[BaseHTTPRequestHandler], None]) -> Iterator[ReadOnlyTransport]:
    """A transport onto a service whose handler writes its reply by hand."""

    class Handler(BaseHTTPRequestHandler):
        do_GET = do_get

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    try:
        yield ReadOnlyTransport(
            {"url": f"http://127.0.0.1:{server.server_address[1]}"},
            CHANNELFINDER,
            where="pv_directory.settings",
        )
    finally:
        server.shutdown()
        server.server_close()


# --- the allowlist ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("allowed", "method", "path"),
    [
        (CHANNELFINDER, "POST", "/resources/channels"),
        (CHANNELFINDER, "PUT", "/resources/channels"),
        (CHANNELFINDER, "DELETE", "/resources/channels"),
        (CHANNELFINDER, "GET", "/resources/channels/SR:PV"),
        (CHANNELFINDER, "GET", "/resources/scroll"),
        (CHANNELFINDER, "GET", "/"),
        (CHANNELFINDER, "get", "/resources/channels"),
        (CHANNELFINDER, "GET", "/resources/channels?~size=1"),
        (ALARM_LOGGER, "POST", "/"),
        (ALARM_LOGGER, "GET", ""),
        (ALARM_LOGGER, "GET", "/search"),
        (ALARM_LOGGER, "GET", "/search/"),
        (ALARM_LOGGER, "POST", "/search/alarm"),
        (ALARM_LOGGER, "GET", "/search/../admin"),
        (ALARM_LOGGER, "GET", "/search/%2e%2e/admin"),
        (ALARM_LOGGER, "GET", "/search/alarm/./x"),
        (ALARM_LOGGER, "GET", "/search//alarm"),
        (ALARM_LOGGER, "GET", "/search/alarm?pv=X"),
        (ALARM_LOGGER, "GET", "/search/alarm#frag"),
        (ALARM_LOGGER, "GET", "/searching/alarm"),
        (ALARM_LOGGER, "GET", "/search/*"),
        (PUT_LOG, "GET", "/search"),
        (PUT_LOG, "POST", "/autocomplete"),
        (PUT_LOG, "POST", "/search/x"),
    ],
)
async def test_a_refused_request_never_reaches_the_network(service, allowed, method, path):
    answer = await _transport(service, allowed).request(method, path, deadline=5)

    assert isinstance(answer, Unavailable)
    assert method in answer.reason and "refused" in answer.reason
    assert service.received == []


async def test_a_refusal_comes_before_a_broken_connection_block():
    transport = ReadOnlyTransport({"headers": {}}, CHANNELFINDER, where="pv_directory.settings")

    answer = await transport.request("DELETE", "/resources/channels", deadline=5)

    assert isinstance(answer, Unavailable) and "refused" in answer.reason


@pytest.mark.parametrize(
    ("allowed", "method", "path", "sent"),
    [
        (CHANNELFINDER, "GET", "/resources/channels", "/resources/channels"),
        (CHANNELFINDER, "GET", "/resources/channels/count", "/resources/channels/count"),
        (CHANNELFINDER, "GET", "/resources/properties", "/resources/properties"),
        (CHANNELFINDER, "GET", "/resources/tags", "/resources/tags"),
        (ALARM_LOGGER, "GET", "/", "/"),
        (ALARM_LOGGER, "GET", "/search/alarm", "/search/alarm"),
        (ALARM_LOGGER, "GET", "/search/alarm/config", "/search/alarm/config"),
        (PUT_LOG, "POST", "/search", "/search"),
    ],
)
async def test_an_allowed_request_is_sent(service, allowed, method, path, sent):
    service.answer(sent, 200, JSON, b"[]")

    answer = await _transport(service, allowed).request(method, path, deadline=5)

    assert answer == []
    assert [(r["method"], r["target"]) for r in service.received] == [(method, sent)]


# --- what is sent -------------------------------------------------------------------


async def test_the_base_path_is_kept_and_the_trailing_slash_of_root_is_sent(service):
    service.answer("/alarm-logger/", 200, JSON, b"{}")
    transport = ReadOnlyTransport(
        {"url": f"{service.url}/alarm-logger/"}, ALARM_LOGGER, where="alarm_log.settings"
    )

    assert await transport.request("GET", "/", deadline=5) == {}
    assert service.received[0]["target"] == "/alarm-logger/"


async def test_params_are_percent_encoded_in_the_query_and_may_repeat(service):
    service.answer("/resources/channels", 200, JSON, b"[]")

    await _transport(service).request(
        "GET",
        "/resources/channels",
        params=[("~name", "SR *"), ("~size", 50), ("~tag", "a"), ("~tag", "b")],
        deadline=5,
    )

    assert service.received[0]["target"] == (
        "/resources/channels?~name=SR%20%2A&~size=50&~tag=a&~tag=b"
    )


async def test_a_search_posts_its_json_body(service):
    service.answer("/search", 200, JSON, b'{"results": []}')

    answer = await _transport(service, PUT_LOG).request(
        "POST", "/search", json_body={"query": "SR:PV"}, deadline=5
    )

    assert answer == {"results": []}
    assert json.loads(service.received[0]["body"]) == {"query": "SR:PV"}
    assert service.received[0]["headers"]["Content-Type"] == JSON


async def test_a_token_login_is_sent_from_the_named_variable(service, monkeypatch):
    monkeypatch.setenv("DIRECTORY_TOKEN", "s3cret")
    service.answer("/resources/tags", 200, JSON, b"[]")

    await _transport(service, auth={"token_env": "DIRECTORY_TOKEN"}).request(
        "GET", "/resources/tags", deadline=5
    )

    assert service.received[0]["headers"]["Authorization"] == "Bearer s3cret"


# --- how an answer is read ----------------------------------------------------------


@pytest.mark.parametrize(
    ("content_type", "body", "parsed"),
    [
        (JSON, b'[{"name": "SR:PV"}]', [{"name": "SR:PV"}]),
        (JSON, b"1234", 1234),
        ("text/plain", b'{"name": "Alarm logging Service"}', {"name": "Alarm logging Service"}),
        ("text/plain;charset=UTF-8", b"[]", []),
        (None, b"[1]", [1]),
    ],
)
async def test_any_content_type_but_html_is_parsed_as_json(service, content_type, body, parsed):
    service.answer("/resources/channels", 200, content_type, body)

    assert await _transport(service).request("GET", "/resources/channels", deadline=5) == parsed


@pytest.mark.parametrize(
    ("status", "content_type", "body"),
    [
        (200, "text/html", b'{"looks": "like json"}'),
        (200, "TEXT/HTML; charset=utf-8", b"[]"),
        (200, "application/json", b"<!DOCTYPE html><html><body>Sign in</body></html>"),
        (200, "text/plain", b"  \n<html><body>Sign in</body></html>"),
        (200, None, b"<!-- proxy --><html></html>"),
        (401, "text/html", b"<html></html>"),
        (403, "text/html", b"<html></html>"),
        (403, "text/plain", b"<!doctype html><html></html>"),
    ],
)
async def test_html_reads_as_a_login_page(service, status, content_type, body):
    service.answer("/resources/channels", status, content_type, body)

    answer = await _transport(service).request("GET", "/resources/channels", deadline=5)

    assert answer == Unavailable(LOGIN_PAGE_REASON)
    assert answer.reason == (
        "answered with a login page (an access proxy is in front of the service)"
    )


@pytest.mark.parametrize(
    ("status", "content_type", "body"),
    [
        (201, "text/html", b"<html></html>"),
        (400, "text/html", b"<html></html>"),
        (404, "text/html", b"<html>Tomcat</html>"),
        (500, "text/html", b"<html></html>"),
        (503, "text/html; charset=iso-8859-1", b"<html>Apache</html>"),
        (503, "text/plain", b"<!doctype html><html>Apache</html>"),
    ],
)
async def test_html_with_any_other_status_names_the_status(service, status, content_type, body):
    service.answer("/resources/channels", status, content_type, body)

    answer = await _transport(service).request("GET", "/resources/channels", deadline=5)

    assert answer == Unavailable(f"answered HTTP {status}")


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_a_redirect_reads_as_a_login_page_and_is_not_followed(service, status):
    service.redirect("/resources/channels", status, f"{service.url}/sso/login")
    service.answer("/sso/login", 200, JSON, b"[]")

    answer = await _transport(service).request("GET", "/resources/channels", deadline=5)

    assert answer == Unavailable(LOGIN_PAGE_REASON)
    assert [r["target"] for r in service.received] == ["/resources/channels"]


async def test_a_redirect_of_a_post_is_not_followed(service):
    service.redirect("/search", 307, f"{service.url}/elsewhere")

    answer = await _transport(service, PUT_LOG).request("POST", "/search", json_body={}, deadline=5)

    assert answer == Unavailable(LOGIN_PAGE_REASON)
    assert [r["target"] for r in service.received] == ["/search"]


@pytest.mark.parametrize("body", [b"{not json", b"", b"\xff\xfe", b"<rss></rss>", b"[" * 200_000])
async def test_a_body_that_is_not_json_is_malformed(service, body):
    service.answer("/resources/channels", 200, "application/json", body)

    answer = await _transport(service).request("GET", "/resources/channels", deadline=5)

    assert answer == Unavailable("malformed response")


@pytest.mark.parametrize("status", [404, 500, 503])
async def test_an_error_status_is_named(service, status):
    service.answer("/resources/channels", status, JSON, b"{}")

    answer = await _transport(service).request("GET", "/resources/channels", deadline=5)

    assert answer == Unavailable(f"answered HTTP {status}")


# --- the deadline -------------------------------------------------------------------


async def test_a_silent_service_answers_no_answer_within_the_deadline(service):
    service.hang_s = 3.0
    service.answer("/resources/channels", 200, JSON, b"[]")

    started = time.monotonic()
    answer = await _transport(service).request("GET", "/resources/channels", deadline=0.2)

    assert answer == Unavailable("no answer within 0.2 s")
    assert time.monotonic() - started < 2.0


async def test_a_shorter_configured_timeout_bounds_a_longer_deadline(service):
    service.hang_s = 3.0

    answer = await _transport(service, timeout_s=0.2).request(
        "GET", "/resources/channels", deadline=60
    )

    assert answer == Unavailable("no answer within 0.2 s")


async def test_a_spent_deadline_sends_nothing(service):
    answer = await _transport(service).request("GET", "/resources/channels", deadline=0)

    assert answer == Unavailable("no answer within 0 s")
    assert service.received == []


async def test_a_body_dripped_past_the_deadline_times_out():
    # A per-socket timeout alone never fires while bytes keep arriving.
    class Drip(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", JSON)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Drip)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    try:
        transport = ReadOnlyTransport(
            {"url": f"http://127.0.0.1:{server.server_address[1]}"},
            CHANNELFINDER,
            where="pv_directory.settings",
        )
        started = time.monotonic()
        answer = await transport.request("GET", "/resources/channels", deadline=0.3)
    finally:
        server.shutdown()
        server.server_close()

    assert answer == Unavailable("no answer within 0.3 s")
    assert time.monotonic() - started < 2.0


# --- a reply too large or cut short ------------------------------------------------


async def test_a_reply_at_the_size_cap_is_read_and_one_byte_over_is_refused(service, monkeypatch):
    monkeypatch.setattr(transport_module, "_MAX_BODY_BYTES", 2**20)
    service.answer("/resources/channels", 200, JSON, b" " * (2**20 - 1) + b"1")
    service.answer("/resources/tags", 200, JSON, b" " * 2**20 + b"1")
    transport = _transport(service)

    assert await transport.request("GET", "/resources/channels", deadline=5) == 1
    answer = await transport.request("GET", "/resources/tags", deadline=5)
    assert answer == Unavailable("answered more than 1 MiB")


async def test_a_declared_length_over_the_cap_is_refused_before_reading(monkeypatch):
    monkeypatch.setattr(transport_module, "_MAX_BODY_BYTES", 2**20)

    def do_get(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", str(2**20 + 1))
        handler.end_headers()
        handler.wfile.flush()
        time.sleep(5.0)

    with _serving(do_get) as transport:
        started = time.monotonic()
        answer = await transport.request("GET", "/resources/channels", deadline=30)

    assert answer == Unavailable("answered more than 1 MiB")
    assert time.monotonic() - started < 4.0


async def test_an_undeclared_reply_over_the_cap_stops_the_read(monkeypatch):
    monkeypatch.setattr(transport_module, "_MAX_BODY_BYTES", 2**20)
    total = 512 * 2**20
    chunk = b" " * 2**16
    sent = 0
    finished = threading.Event()

    def do_get(handler: BaseHTTPRequestHandler) -> None:
        nonlocal sent
        handler.send_response(200)
        handler.end_headers()
        try:
            while sent < total:
                handler.wfile.write(chunk)
                sent += len(chunk)
        except OSError:
            pass
        finished.set()

    with _serving(do_get) as transport:
        answer = await transport.request("GET", "/resources/channels", deadline=30)
        assert finished.wait(5)

    assert answer == Unavailable("answered more than 1 MiB")
    assert sent < total / 2


def _declared_five_sends_four(handler: BaseHTTPRequestHandler, status: int) -> None:
    handler.send_response(status)
    handler.send_header("Content-Length", "5")
    handler.end_headers()
    handler.wfile.write(b"1234")


def _chunked(handler: BaseHTTPRequestHandler, written: bytes) -> None:
    handler.send_response(200)
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()
    handler.wfile.write(written)


@pytest.mark.parametrize(
    "do_get",
    [
        pytest.param(lambda h: _declared_five_sends_four(h, 200), id="count-cut-short"),
        pytest.param(lambda h: _declared_five_sends_four(h, 503), id="error-status-cut-short"),
        pytest.param(lambda h: _chunked(h, b"5\r\n1234"), id="inside-a-chunk"),
        pytest.param(lambda h: _chunked(h, b"4\r\n1234\r\n"), id="between-chunks"),
    ],
)
async def test_a_body_cut_short_is_never_read_as_data(do_get):
    with _serving(do_get) as transport:
        answer = await transport.request("GET", "/resources/channels/count", deadline=5)

    assert isinstance(answer, Unavailable)
    assert answer.reason.startswith("could not reach the service")
    assert "before the answer ended" in answer.reason


async def test_a_complete_chunked_body_is_read():
    def do_get(handler: BaseHTTPRequestHandler) -> None:
        _chunked(handler, b"4\r\n1234\r\n0\r\n\r\n")

    with _serving(do_get) as transport:
        assert await transport.request("GET", "/resources/channels/count", deadline=5) == 1234


# --- a service or a block that cannot be used ---------------------------------------


async def test_a_closed_port_is_unavailable():
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_address[1]
    server.server_close()
    transport = ReadOnlyTransport(
        {"url": f"http://127.0.0.1:{port}"}, CHANNELFINDER, where="pv_directory.settings"
    )

    answer = await transport.request("GET", "/resources/channels", deadline=5)

    assert isinstance(answer, Unavailable)
    assert answer.reason.startswith("could not reach the service")


@pytest.mark.parametrize(
    ("settings", "named"),
    [
        ({}, "pv_directory.settings.url"),
        ({"url": "ftp://host"}, "pv_directory.settings.url"),
        ({"url": "http://host", "verify": False, "headers": {}}, "pv_directory.settings.headers"),
        ({"url": "http://host", "auth": {"token_env": "UNSET_VAR_X"}}, "UNSET_VAR_X"),
        ({"url": "https://host", "tls": {"ca_bundle": "/no/such/ca.pem"}}, "tls.ca_bundle"),
    ],
)
async def test_a_block_that_cannot_be_used_is_unavailable_naming_the_key(settings, named):
    transport = ReadOnlyTransport(settings, CHANNELFINDER, where="pv_directory.settings")

    answer = await transport.request("GET", "/resources/channels", deadline=5)

    assert isinstance(answer, Unavailable)
    assert named in answer.reason


@pytest.mark.parametrize(
    ("auth", "secret"),
    [
        ({"token_env": "DIRECTORY_SECRET"}, "s3cret\n"),
        ({"token_env": "DIRECTORY_SECRET"}, "s3cret\r\nX-Injected: 1"),
        ({"token_env": "DIRECTORY_SECRET"}, "s3cret\u20ac"),
        ({"username": "reader", "password_env": "DIRECTORY_SECRET"}, "s3cret\udcff"),
    ],
    ids=["newline", "header-injection", "not-latin-1", "password-not-encodable"],
)
async def test_a_login_that_cannot_be_sent_is_unavailable_without_the_secret(
    service, monkeypatch, caplog, auth, secret
):
    monkeypatch.setenv("DIRECTORY_SECRET", secret)
    caplog.set_level(logging.DEBUG)

    answer = await _transport(service, auth=auth).request("GET", "/resources/channels", deadline=5)

    assert answer == Unavailable("`pv_directory.settings`: the url or login cannot be sent")
    assert service.received == []
    assert "s3cret" not in caplog.text


@pytest.mark.parametrize(
    "url",
    [
        "http://a..b/",
        "http://127.0.0.1:1/gr\u00f6\u00dfe",
        "http://127.0.0.1:notaport/",
        "http://127.0.0.1:1/with space",
        "http://127.0.0.1:1/line\nbreak",
    ],
)
async def test_a_url_that_cannot_be_sent_is_unavailable_naming_the_block(url):
    transport = ReadOnlyTransport({"url": url}, CHANNELFINDER, where="pv_directory.settings")

    answer = await transport.request("GET", "/resources/channels", deadline=5)

    assert answer == Unavailable("`pv_directory.settings`: the url or login cannot be sent")


async def test_a_refused_login_names_the_status_not_the_login(service, monkeypatch):
    monkeypatch.setenv("DIRECTORY_SECRET", "s3cret")
    service.answer("/resources/channels", 401, JSON, b'{"error": "no"}')

    answer = await _transport(service, auth={"token_env": "DIRECTORY_SECRET"}).request(
        "GET", "/resources/channels", deadline=5
    )

    assert answer == Unavailable("answered HTTP 401")


@pytest.mark.parametrize(
    "params",
    [[("~name", "\ud800")], {"~name": "a\udcff"}],
    ids=["pairs", "mapping"],
)
async def test_a_query_that_cannot_be_encoded_sends_nothing(service, monkeypatch, params):
    monkeypatch.setenv("DIRECTORY_SECRET", "tok-secret")
    transport = _transport(service, auth={"token_env": "DIRECTORY_SECRET"})

    answer = await transport.request("GET", "/resources/channels", params=params, deadline=5)

    assert answer == Unavailable("the query cannot be sent")
    assert service.received == []


@pytest.mark.parametrize("suffix", ["/?a=", "/?", "/#frag", "/#", "?a=1"])
async def test_a_url_with_a_query_or_fragment_sends_nothing(service, suffix):
    transport = ReadOnlyTransport(
        {"url": service.url + suffix}, CHANNELFINDER, where="pv_directory.settings"
    )

    answer = await transport.request("GET", "/resources/channels", deadline=5)

    assert answer == Unavailable("`pv_directory.settings.url` must not carry a query or fragment")
    assert service.received == []


async def test_building_a_transport_opens_nothing(service):
    _transport(service)

    assert service.received == []
