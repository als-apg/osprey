"""Tests for the attachment fetch helpers: origins, file sources, redaction and fetching."""

from __future__ import annotations

import asyncio
import errno
import os
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from osprey.services.ariel_search.attachments import fetch as fetch_mod
from osprey.services.ariel_search.attachments.fetch import (
    COPY_ENTRY_DEADLINE,
    FetchOutcome,
    _redirect_allowed,
    fetch_attachment_bytes,
    is_file_source,
    make_fetch_session,
    origin_of,
    origins_for,
    redact_url,
)
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter
from osprey.services.ariel_search.ingestion.adapters.generic import GenericJSONAdapter
from osprey.services.ariel_search.ingestion.base import FacilityAdapter

# These tests exercise the real fetcher, so the conftest fake stays out of the way.
pytestmark = pytest.mark.real_fetch


def _config(source_url: str | None, *, allowed_origins: list[str] | None = None) -> ARIELConfig:
    data: dict = {"database": {"uri": "postgresql://test"}}
    if source_url is not None:
        data["ingestion"] = {"adapter": "generic_json", "source_url": source_url}
    if allowed_origins is not None:
        data["attachments"] = {"allowed_origins": allowed_origins}
    return ARIELConfig.from_dict(data)


class _StubAdapter(FacilityAdapter):
    """A registered-style adapter OSPREY knows nothing about."""

    @property
    def source_system_name(self) -> str:
        return "Stub"

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in ():  # never yields
            yield entry


class _StubWithSourceUrl(_StubAdapter):
    def __init__(self, config: ARIELConfig, source_url: str) -> None:
        super().__init__(config)
        self.source_url = source_url


# -- origin_of ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://h.example", ("https", "h.example", 443)),
        ("https://h.example:443/a/b.png", ("https", "h.example", 443)),
        ("http://h.example/x", ("http", "h.example", 80)),
        ("http://h.example:80", ("http", "h.example", 80)),
        ("HTTPS://H.Example/", ("https", "h.example", 443)),
    ],
)
def test_origin_default_ports_normalise(url, expected):
    assert origin_of(url) == expected


def test_origin_different_port_is_different_origin():
    assert origin_of("https://h.example:8443/") == ("https", "h.example", 8443)
    assert origin_of("https://h.example:8443/") != origin_of("https://h.example/")


@pytest.mark.parametrize(
    "url",
    [
        "/data/logbook.json",
        "file:///data/x.png",
        "ftp://h/x",
        "images/a.png",
        "",
        "https://",
        "https://h:notaport/",
    ],
)
def test_origin_none_for_non_http_or_malformed(url):
    assert origin_of(url) is None


# -- adapter origins -----------------------------------------------------------


def test_origin_stub_adapter_with_http_source_url_gets_its_own_origin():
    adapter = _StubWithSourceUrl(_config(None), "https://logbook.site.example:8443/api/entries")
    assert adapter.attachment_origins() == frozenset({("https", "logbook.site.example", 8443)})
    assert adapter.attachment_file_base() is None
    assert is_file_source(adapter) is False


def test_origin_stub_adapter_without_source_url_uses_configured_ingestion_source_url():
    adapter = _StubAdapter(_config("http://elog.site.example/feed"))
    assert not hasattr(adapter, "source_url")
    assert adapter.attachment_origins() == frozenset({("http", "elog.site.example", 80)})


def test_origin_stub_adapter_with_no_source_at_all_has_no_origin():
    adapter = _StubAdapter(_config(None))
    assert adapter.attachment_origins() == frozenset()
    assert is_file_source(adapter) is False


def test_origin_stub_adapter_with_file_source_url_has_no_origin():
    adapter = _StubWithSourceUrl(_config(None), "/data/logbook.jsonl")
    assert adapter.attachment_origins() == frozenset()


def test_origin_als_adapter_answers_prefix_origin():
    adapter = ALSLogbookAdapter(_config("/fake/path.jsonl"))
    prefix = urlsplit(adapter.attachment_url_prefix)
    assert prefix.scheme == "https"
    assert adapter.attachment_origins() == frozenset({("https", prefix.hostname, 443)})


def test_origin_als_adapter_http_source_still_answers_prefix_origin():
    adapter = ALSLogbookAdapter(_config("http://localhost:9999/elog"))
    prefix = urlsplit(adapter.attachment_url_prefix)
    assert prefix.scheme == "https"
    assert adapter.attachment_origins() == frozenset({("https", prefix.hostname, 443)})


def test_origin_generic_file_source_has_no_origin_and_a_file_base(tmp_path: Path):
    source = tmp_path / "logbook.json"
    adapter = GenericJSONAdapter(_config(str(source)))
    assert adapter.attachment_origins() == frozenset()
    assert adapter.attachment_file_base() == tmp_path
    assert is_file_source(adapter) is True


def test_origin_generic_http_source_has_origin_and_no_file_base():
    adapter = GenericJSONAdapter(_config("https://json.site.example/entries.json"))
    assert adapter.attachment_origins() == frozenset({("https", "json.site.example", 443)})
    assert adapter.attachment_file_base() is None
    assert is_file_source(adapter) is False


# -- origins_for ----------------------------------------------------------------


def test_origins_for_unions_adapter_origin_and_allowed_origins():
    config = _config(
        "https://json.site.example/entries.json",
        allowed_origins=["https://cdn.site.example", "http://files.site.example:8080"],
    )
    adapter = GenericJSONAdapter(config)
    assert origins_for(adapter, config) == frozenset(
        {
            ("https", "json.site.example", 443),
            ("https", "cdn.site.example", 443),
            ("http", "files.site.example", 8080),
        }
    )


def test_origins_for_file_source_is_allowed_origins_only(tmp_path: Path):
    config = _config(str(tmp_path / "x.json"), allowed_origins=["https://cdn.site.example"])
    adapter = GenericJSONAdapter(config)
    assert origins_for(adapter, config) == frozenset({("https", "cdn.site.example", 443)})


def test_origins_for_returns_frozenset_with_no_allowed_origins():
    config = _config("https://json.site.example/e.json")
    result = origins_for(GenericJSONAdapter(config), config)
    assert isinstance(result, frozenset)
    assert result == frozenset({("https", "json.site.example", 443)})


# -- redact_url -------------------------------------------------------------------


def test_redact_socks_proxy_credentials():
    assert redact_url("socks5://u:secret@h:1080") == "socks5://***@h:1080"


def test_redact_username_only():
    out = redact_url("socks5://alice@proxy.example:1080")
    assert out == "socks5://***@proxy.example:1080"
    assert "alice" not in out


def test_redact_http_without_explicit_port_uses_effective_port():
    out = redact_url("https://u:pw@h.example/path?q=1")
    assert out == "https://***@h.example:443"
    assert "pw" not in out


def test_redact_leaves_url_without_userinfo_unchanged():
    assert redact_url("socks5://h:1080") == "socks5://h:1080"
    assert redact_url("https://h.example/a.png") == "https://h.example/a.png"


def test_redact_ipv6_host_keeps_brackets():
    assert redact_url("socks5://u:p@[::1]:1080") == "socks5://***@[::1]:1080"


def test_redact_never_leaks_secret_on_odd_input():
    out = redact_url("socks5://u:secret@h:notaport")
    assert "secret" not in out


async def test_proxy_connector_log_redacts_the_proxy_credentials(caplog):
    adapter = _StubAdapter(_config(None))
    adapter.proxy_url = "socks5://user:secret@proxy.example:1080"

    with caplog.at_level("INFO"):
        connector = adapter._create_connector()
    await connector.close()

    assert "socks5://***@proxy.example:1080" in caplog.text
    assert "secret" not in caplog.text
    assert "user" not in caplog.text


# -- fetch_attachment_bytes over HTTP ---------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 100


class _CountingAdapter(_StubAdapter):
    """Stub adapter that counts the connectors it builds."""

    def __init__(self, config: ARIELConfig) -> None:
        super().__init__(config)
        self.connectors = 0

    def _create_connector(self) -> aiohttp.BaseConnector:
        self.connectors += 1
        return super()._create_connector()


async def _pic(_request: web.Request) -> web.Response:
    return web.Response(body=PNG)


async def _big_length(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(headers={"Content-Length": "3000000000"})
    await resp.prepare(request)
    if request.method == "HEAD":
        return resp
    await resp.write(b"x" * 10)
    await asyncio.sleep(5)
    return resp


async def _chunked_big(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse()
    resp.enable_chunked_encoding()
    await resp.prepare(request)
    for _ in range(10):
        await resp.write(b"y" * 100)
    await resp.write_eof()
    return resp


async def _slow_body(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse()
    resp.enable_chunked_encoding()
    await resp.prepare(request)
    await resp.write(b"z")
    await asyncio.sleep(5)
    return resp


async def _slow_headers(_request: web.Request) -> web.Response:
    await asyncio.sleep(5)
    return web.Response(body=PNG)


def _status(code: int):
    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=code, body=b"nope")

    return handler


def _redirect(location: str, code: int = 302):
    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=code, headers={"Location": location})

    return handler


async def _chain_handler(request: web.Request) -> web.Response:
    n = int(request.match_info["n"])
    if n == 0:
        return web.Response(body=PNG)
    return web.Response(status=302, headers={"Location": f"/chain/{n - 1}"})


@pytest.fixture
async def server() -> AsyncIterator[TestServer]:
    app = web.Application()
    app.router.add_get("/pic.png", _pic)
    app.router.add_route("*", "/big", _big_length)
    app.router.add_get("/chunked", _chunked_big)
    app.router.add_get("/slow-body", _slow_body)
    app.router.add_get("/slow-headers", _slow_headers)
    for code in (400, 401, 403, 404, 408, 410, 429, 500, 503):
        app.router.add_get(f"/status/{code}", _status(code))
    app.router.add_get("/redir/same", _redirect("/pic.png"))
    app.router.add_get("/redir/off", _redirect("http://elsewhere.invalid/pic.png"))
    app.router.add_get("/redir/none", _status(302))
    app.router.add_get("/chain/{n}", _chain_handler)
    srv = TestServer(app)
    await srv.start_server()
    try:
        yield srv
    finally:
        await srv.close()


def _origins(srv: TestServer) -> frozenset:
    return frozenset({("http", srv.host, srv.port)})


def _url(srv: TestServer, path: str) -> str:
    return str(srv.make_url(path))


@pytest.fixture
def adapter() -> _CountingAdapter:
    return _CountingAdapter(_config(None))


async def test_fetch_success_returns_bytes(server, adapter):
    out = await fetch_attachment_bytes(_url(server, "/pic.png"), 1000, _origins(server), adapter)
    assert out == FetchOutcome(data=PNG)
    assert out.code is None and out.transient is False and out.host_up is False
    assert out.observed_size is None


async def test_fetch_off_origin_url_is_origin_not_allowed_without_request(server, adapter):
    sent: list[int] = []
    out = await fetch_attachment_bytes(
        _url(server, "/pic.png"),
        1000,
        frozenset({("http", server.host, server.port + 1)}),
        adapter,
        on_sent=lambda: sent.append(1),
    )
    assert out == FetchOutcome(code="origin_not_allowed")
    assert adapter.connectors == 0 and sent == []


async def test_fetch_empty_origin_set_allows_nothing(server, adapter):
    out = await fetch_attachment_bytes(_url(server, "/pic.png"), 1000, frozenset(), adapter)
    assert out.code == "origin_not_allowed"
    assert adapter.connectors == 0


async def test_fetch_head_outside_origin_set_is_origin_not_allowed(server, adapter):
    out = await fetch_attachment_bytes(
        _url(server, "/pic.png"), 1000, frozenset(), adapter, method="HEAD"
    )
    assert out.code == "origin_not_allowed"
    assert adapter.connectors == 0


@pytest.mark.parametrize("url", ["ftp://h/x.png", "file:///etc/passwd", "images/a.png", ""])
async def test_fetch_non_http_on_http_source_is_source_refused_never_transient(adapter, url):
    out = await fetch_attachment_bytes(url, 1000, frozenset({("ftp", "h", 21)}), adapter)
    assert out == FetchOutcome(code="source_refused")
    assert adapter.connectors == 0


async def test_fetch_relative_path_on_file_source_skips_the_origin_rule(tmp_path, monkeypatch):
    adapter = GenericJSONAdapter(_config(str(tmp_path / "logbook.json")))
    seen: list[str] = []

    async def fake_file_source(url, _cap, _adapter, _method):
        seen.append(url)
        return FetchOutcome(data=b"file")

    monkeypatch.setattr(fetch_mod, "_fetch_file_source", fake_file_source)
    out = await fetch_attachment_bytes("images/a.png", 1000, frozenset(), adapter)
    assert out == FetchOutcome(data=b"file")
    assert seen == ["images/a.png"]


async def test_fetch_content_length_3gb_is_size_cap(server, adapter):
    out = await asyncio.wait_for(
        fetch_attachment_bytes(_url(server, "/big"), 1000, _origins(server), adapter), 3
    )
    assert out == FetchOutcome(code="size_cap", observed_size=1001)


async def test_fetch_content_length_just_over_cap_records_content_length(server, adapter):
    out = await fetch_attachment_bytes(
        _url(server, "/pic.png"), len(PNG) - 1, _origins(server), adapter
    )
    assert out == FetchOutcome(code="size_cap", observed_size=len(PNG))


async def test_fetch_streamed_body_over_cap_is_size_cap(server, adapter):
    out = await fetch_attachment_bytes(_url(server, "/chunked"), 250, _origins(server), adapter)
    assert out == FetchOutcome(code="size_cap", observed_size=251)


async def test_fetch_body_exactly_at_cap_is_accepted(server, adapter):
    out = await fetch_attachment_bytes(
        _url(server, "/pic.png"), len(PNG), _origins(server), adapter
    )
    assert out.data == PNG


async def test_fetch_head_returns_no_body(server, adapter):
    out = await fetch_attachment_bytes(
        _url(server, "/pic.png"), 1000, _origins(server), adapter, method="HEAD"
    )
    assert out == FetchOutcome(data=b"")


async def test_fetch_head_still_applies_content_length_cap(server, adapter):
    out = await asyncio.wait_for(
        fetch_attachment_bytes(
            _url(server, "/big"), 1000, _origins(server), adapter, method="HEAD"
        ),
        3,
    )
    assert out == FetchOutcome(code="size_cap", observed_size=1001)


@pytest.mark.parametrize("code", [404, 410])
async def test_fetch_gone_statuses_are_source_gone(server, adapter, code):
    out = await fetch_attachment_bytes(
        _url(server, f"/status/{code}"), 1000, _origins(server), adapter
    )
    assert out == FetchOutcome(code="source_gone")


@pytest.mark.parametrize("code", [400, 401, 403])
async def test_fetch_other_4xx_are_source_refused(server, adapter, code):
    out = await fetch_attachment_bytes(
        _url(server, f"/status/{code}"), 1000, _origins(server), adapter
    )
    assert out == FetchOutcome(code="source_refused")


@pytest.mark.parametrize("code", [408, 429, 500, 503])
async def test_fetch_host_up_statuses_are_transient_with_host_up(server, adapter, code):
    out = await fetch_attachment_bytes(
        _url(server, f"/status/{code}"), 1000, _origins(server), adapter
    )
    assert out == FetchOutcome(transient=True, host_up=True)


async def test_fetch_connection_refused_is_transient_host_down(adapter):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    sent: list[int] = []
    out = await fetch_attachment_bytes(
        f"http://127.0.0.1:{port}/pic.png",
        1000,
        frozenset({("http", "127.0.0.1", port)}),
        adapter,
        on_sent=lambda: sent.append(1),
    )
    assert out == FetchOutcome(transient=True, host_up=False)
    assert sent == []


async def test_fetch_total_timeout_after_send_is_host_up(server, adapter):
    sent: list[int] = []
    out = await fetch_attachment_bytes(
        _url(server, "/slow-headers"),
        1000,
        _origins(server),
        adapter,
        total=0.5,
        on_sent=lambda: sent.append(1),
    )
    assert out == FetchOutcome(transient=True, host_up=True)
    assert sent == [1]


async def test_fetch_sock_read_timeout_is_host_up(server, adapter, monkeypatch):
    monkeypatch.setattr(fetch_mod, "_SOCK_READ_TIMEOUT", 0.3)
    out = await fetch_attachment_bytes(_url(server, "/slow-body"), 1000, _origins(server), adapter)
    assert out == FetchOutcome(transient=True, host_up=True)


async def test_fetch_connect_timeout_is_host_down(adapter, monkeypatch):
    class _Timeout:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise aiohttp.ConnectionTimeoutError("connect timed out")

        async def __aexit__(self, *a):
            return False

    async with make_fetch_session(adapter) as session:
        monkeypatch.setattr(session, "request", _Timeout)
        out = await fetch_attachment_bytes(
            "http://h.example/a.png",
            1000,
            frozenset({("http", "h.example", 80)}),
            adapter,
            session=session,
        )
    assert out == FetchOutcome(transient=True, host_up=False)


async def test_fetch_same_origin_redirect_is_followed(server, adapter):
    out = await fetch_attachment_bytes(_url(server, "/redir/same"), 1000, _origins(server), adapter)
    assert out.data == PNG


async def test_fetch_three_hops_are_followed(server, adapter):
    out = await fetch_attachment_bytes(_url(server, "/chain/3"), 1000, _origins(server), adapter)
    assert out.data == PNG


async def test_fetch_fourth_hop_is_source_refused(server, adapter):
    out = await fetch_attachment_bytes(_url(server, "/chain/4"), 1000, _origins(server), adapter)
    assert out == FetchOutcome(code="source_refused")


async def test_fetch_off_origin_location_is_source_refused(server, adapter):
    out = await fetch_attachment_bytes(_url(server, "/redir/off"), 1000, _origins(server), adapter)
    assert out == FetchOutcome(code="source_refused")


async def test_fetch_redirect_without_location_is_source_refused(server, adapter):
    out = await fetch_attachment_bytes(_url(server, "/redir/none"), 1000, _origins(server), adapter)
    assert out == FetchOutcome(code="source_refused")


@pytest.mark.parametrize(
    ("current", "target", "allowed"),
    [
        (("http", "h", 80), ("http", "h", 80), True),
        (("http", "h", 80), ("https", "h", 443), True),
        (("http", "h", 8080), ("https", "h", 443), False),
        (("http", "h", 80), ("https", "h", 8443), False),
        (("https", "h", 443), ("http", "h", 80), False),
        (("http", "h", 80), ("https", "other", 443), False),
        (("https", "h", 443), ("https", "h", 8443), False),
        (("https", "h", 443), None, False),
    ],
)
def test_fetch_redirect_origin_rule(current, target, allowed):
    assert _redirect_allowed(current, target) is allowed


async def test_fetch_twenty_fetches_through_one_session_open_one_connector(server, adapter):
    async with make_fetch_session(adapter) as session:
        connector = session.connector
        for _ in range(20):
            out = await fetch_attachment_bytes(
                _url(server, "/pic.png"), 1000, _origins(server), adapter, session=session
            )
            assert out.data == PNG
        assert session.connector is connector
    assert adapter.connectors == 1


async def test_fetch_without_session_builds_a_one_shot_session(server, adapter):
    out = await fetch_attachment_bytes(_url(server, "/pic.png"), 1000, _origins(server), adapter)
    assert out.data == PNG
    assert adapter.connectors == 1


async def test_fetch_uses_the_adapter_ssl_context(server, adapter, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(adapter, "_ssl_context", lambda: calls.append(1) or False)
    await fetch_attachment_bytes(_url(server, "/pic.png"), 1000, _origins(server), adapter)
    assert calls == [1]


async def test_fetch_on_sent_called_once_on_success(server, adapter):
    sent: list[int] = []
    await fetch_attachment_bytes(
        _url(server, "/chain/2"), 1000, _origins(server), adapter, on_sent=lambda: sent.append(1)
    )
    assert sent == [1]


async def test_fetch_on_sent_called_with_foreign_session(server, adapter):
    sent: list[int] = []
    async with aiohttp.ClientSession() as session:
        await fetch_attachment_bytes(
            _url(server, "/pic.png"),
            1000,
            _origins(server),
            adapter,
            session=session,
            on_sent=lambda: sent.append(1),
        )
    assert sent == [1]


async def test_fetch_on_sent_not_called_while_waiting_on_semaphore(server, adapter):
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    sent: list[int] = []
    task = asyncio.create_task(
        fetch_attachment_bytes(
            _url(server, "/pic.png"),
            1000,
            _origins(server),
            adapter,
            semaphore=sem,
            on_sent=lambda: sent.append(1),
        )
    )
    await asyncio.sleep(0.2)
    assert not task.done()
    assert sent == []
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sent == []
    assert adapter.connectors == 0


async def test_fetch_semaphore_limits_concurrency(server, adapter):
    sem = asyncio.Semaphore(1)
    out = await asyncio.gather(
        *(
            fetch_attachment_bytes(
                _url(server, "/pic.png"), 1000, _origins(server), adapter, semaphore=sem
            )
            for _ in range(3)
        )
    )
    assert all(o.data == PNG for o in out)
    assert sem._value == 1


async def test_fetch_cancellation_propagates(server, adapter):
    task = asyncio.create_task(
        fetch_attachment_bytes(_url(server, "/slow-headers"), 1000, _origins(server), adapter)
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_fetch_rejects_unknown_method(adapter):
    with pytest.raises(ValueError):
        await fetch_attachment_bytes("http://h/x", 1, frozenset(), adapter, method="POST")


def test_fetch_default_deadline_is_sixty_seconds():
    assert COPY_ENTRY_DEADLINE == 60.0


# -- file sources --------------------------------------------------------------


class _StubFileSource(_StubAdapter):
    """A registered adapter with a file base and no ``source_url`` attribute."""

    def __init__(self, config: ARIELConfig, base: Path) -> None:
        super().__init__(config)
        self._base = base

    def attachment_file_base(self) -> Path | None:
        return self._base


def _file_source(tmp_path: Path) -> GenericJSONAdapter:
    return GenericJSONAdapter(_config(str(tmp_path / "logbook.json")))


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


async def test_fetch_file_source_reads_relative_path(tmp_path):
    _write(tmp_path / "images" / "a.png", b"png-bytes")
    out = await fetch_attachment_bytes("images/a.png", 1000, frozenset(), _file_source(tmp_path))
    assert out == FetchOutcome(data=b"png-bytes")


async def test_fetch_file_source_stub_without_source_url_reads_from_its_base(tmp_path):
    base = tmp_path / "attachments"
    _write(base / "x.png", b"stub")
    adapter = _StubFileSource(_config(None), base)
    assert not hasattr(adapter, "source_url")
    out = await fetch_attachment_bytes("x.png", 1000, frozenset(), adapter)
    assert out == FetchOutcome(data=b"stub")


async def test_fetch_file_source_body_at_cap_is_accepted_and_over_cap_is_size_cap(tmp_path):
    _write(tmp_path / "ten.bin", b"0123456789")
    adapter = _file_source(tmp_path)
    assert await fetch_attachment_bytes("ten.bin", 10, frozenset(), adapter) == FetchOutcome(
        data=b"0123456789"
    )
    assert await fetch_attachment_bytes("ten.bin", 9, frozenset(), adapter) == FetchOutcome(
        code="size_cap", observed_size=10
    )


async def test_fetch_file_source_head_reads_no_body(tmp_path):
    _write(tmp_path / "a.png", b"abc")
    out = await fetch_attachment_bytes("a.png", 1000, frozenset(), _file_source(tmp_path), "HEAD")
    assert out == FetchOutcome(data=b"")


async def test_fetch_file_source_head_still_applies_the_cap(tmp_path):
    _write(tmp_path / "a.png", b"abcdef")
    out = await fetch_attachment_bytes("a.png", 2, frozenset(), _file_source(tmp_path), "HEAD")
    assert out == FetchOutcome(code="size_cap", observed_size=3)


@pytest.mark.parametrize(
    "url",
    ["../x", "a/../../x", "a/../x", "./x", "a//x", "a/", "/etc/passwd", "a/./x", "..", "x\x00y"],
)
async def test_fetch_file_source_refuses_dot_dot_absolute_and_empty_parts(tmp_path, url):
    (tmp_path / "base").mkdir()
    _write(tmp_path / "x", b"outside")
    _write(tmp_path / "base" / "x", b"inside")
    _write(tmp_path / "base" / "a" / "x", b"inside")
    adapter = _StubFileSource(_config(None), tmp_path / "base")
    out = await fetch_attachment_bytes(url, 1000, frozenset(), adapter)
    assert out == FetchOutcome(code="source_refused")


async def test_fetch_file_source_without_a_base_is_refused():
    out = await fetch_mod._fetch_file_source("x.png", 1000, _StubAdapter(_config(None)), "GET")
    assert out == FetchOutcome(code="source_refused")


async def test_fetch_file_source_symlinked_leaf_is_not_a_regular_file(tmp_path, monkeypatch):
    target = _write(tmp_path / "secret.txt", b"secret")
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.png").symlink_to(target)
    reads: list[int] = []
    real_read = fetch_mod.os.read
    monkeypatch.setattr(fetch_mod.os, "read", lambda fd, n: reads.append(fd) or real_read(fd, n))
    out = await fetch_attachment_bytes(
        "a.png", 1000, frozenset(), _StubFileSource(_config(None), base)
    )
    assert out == FetchOutcome(code="not_a_regular_file")
    assert reads == []


async def test_fetch_file_source_symlinked_intermediate_directory_is_not_a_regular_file(
    tmp_path, monkeypatch
):
    _write(tmp_path / "outside" / "a.png", b"secret")
    base = tmp_path / "base"
    base.mkdir()
    (base / "images").symlink_to(tmp_path / "outside", target_is_directory=True)
    reads: list[int] = []
    real_read = fetch_mod.os.read
    monkeypatch.setattr(fetch_mod.os, "read", lambda fd, n: reads.append(fd) or real_read(fd, n))
    out = await fetch_attachment_bytes(
        "images/a.png", 1000, frozenset(), _StubFileSource(_config(None), base)
    )
    assert out == FetchOutcome(code="not_a_regular_file")
    assert reads == []


@pytest.mark.timeout(10)
async def test_fetch_file_source_fifo_leaf_does_not_block(tmp_path):
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)
    out = await asyncio.wait_for(
        fetch_attachment_bytes("pipe.png", 1000, frozenset(), _file_source(tmp_path)), 5
    )
    assert out == FetchOutcome(code="not_a_regular_file")


async def test_fetch_file_source_directory_leaf_is_not_a_regular_file(tmp_path):
    (tmp_path / "dir.png").mkdir()
    out = await fetch_attachment_bytes("dir.png", 1000, frozenset(), _file_source(tmp_path))
    assert out == FetchOutcome(code="not_a_regular_file")


async def test_fetch_file_source_missing_leaf_is_source_gone(tmp_path):
    out = await fetch_attachment_bytes("nope.png", 1000, frozenset(), _file_source(tmp_path))
    assert out == FetchOutcome(code="source_gone")


async def test_fetch_file_source_missing_directory_is_source_gone(tmp_path):
    out = await fetch_attachment_bytes("nodir/a.png", 1000, frozenset(), _file_source(tmp_path))
    assert out == FetchOutcome(code="source_gone")


async def test_fetch_file_source_file_as_intermediate_is_source_gone(tmp_path):
    _write(tmp_path / "plain", b"x")
    out = await fetch_attachment_bytes("plain/a.png", 1000, frozenset(), _file_source(tmp_path))
    assert out == FetchOutcome(code="source_gone")


async def test_fetch_file_source_missing_base_is_source_gone(tmp_path):
    adapter = _StubFileSource(_config(None), tmp_path / "gone")
    out = await fetch_attachment_bytes("a.png", 1000, frozenset(), adapter)
    assert out == FetchOutcome(code="source_gone")


_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


@pytest.mark.skipif(_ROOT, reason="root ignores file modes")
async def test_fetch_file_source_mode_000_file_is_source_refused(tmp_path):
    path = _write(tmp_path / "locked.png", b"x")
    path.chmod(0)
    try:
        out = await fetch_attachment_bytes("locked.png", 1000, frozenset(), _file_source(tmp_path))
    finally:
        path.chmod(0o600)
    assert out == FetchOutcome(code="source_refused")


@pytest.mark.skipif(_ROOT, reason="root ignores file modes")
async def test_fetch_file_source_mode_000_directory_is_source_refused(tmp_path):
    _write(tmp_path / "locked" / "a.png", b"x")
    (tmp_path / "locked").chmod(0)
    try:
        out = await fetch_attachment_bytes(
            "locked/a.png", 1000, frozenset(), _file_source(tmp_path)
        )
    finally:
        (tmp_path / "locked").chmod(0o700)
    assert out == FetchOutcome(code="source_refused")


@pytest.mark.parametrize(
    ("err", "code"),
    [
        (errno.ENOENT, "source_gone"),
        (errno.ENOTDIR, "source_gone"),
        (errno.EACCES, "source_refused"),
        (errno.EPERM, "source_refused"),
        (errno.ELOOP, "not_a_regular_file"),
    ],
)
async def test_fetch_file_source_errno_maps_to_outcome(tmp_path, monkeypatch, err, code):
    _write(tmp_path / "a" / "b.png", b"x")
    real_open = fetch_mod.os.open

    def fake_open(path, flags, *args, **kwargs):
        if path == "b.png":
            raise OSError(err, os.strerror(err))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(fetch_mod.os, "open", fake_open)
    out = await fetch_attachment_bytes("a/b.png", 1000, frozenset(), _file_source(tmp_path))
    assert out == FetchOutcome(code=code)


async def test_fetch_file_source_closes_every_descriptor(tmp_path, monkeypatch):
    _write(tmp_path / "a" / "b" / "c.png", b"x")
    opened: list[int] = []
    closed: list[int] = []
    real_open, real_close = fetch_mod.os.open, fetch_mod.os.close

    def track_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def track_close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(fetch_mod.os, "open", track_open)
    monkeypatch.setattr(fetch_mod.os, "close", track_close)
    adapter = _file_source(tmp_path)
    assert (await fetch_attachment_bytes("a/b/c.png", 1000, frozenset(), adapter)).ok
    assert (await fetch_attachment_bytes("a/b/none.png", 1000, frozenset(), adapter)).code
    assert sorted(opened) == sorted(closed)
    assert len(opened) == 7
