"""Tests for the Slack Web API client's file download guard and three-step upload.

Every test drives the real :class:`SlackWebClient` over an
:class:`httpx.MockTransport` that records what reached it, so "no request was
made" and "no credential was sent" are observed on the wire rather than assumed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from osprey.bridges.core import CoreConfig
from osprey.bridges.slack.client import (
    API_BASE,
    FILES_HOST,
    FileTooLargeError,
    ForeignHostError,
    NotAFileError,
    SlackApiError,
    SlackWebClient,
)
from osprey.bridges.slack.config import SlackBridgeConfig

BOT_TOKEN = "xoxb-1111-2222-bot-secret"
APP_TOKEN = "xapp-1-A111-3333-app-secret"  # gitleaks:allow - a fake test token
FILE_URL = "https://files.slack.com/files-pri/T1-F1/download/report.csv"
UPLOAD_URL = "https://files.slack.com/upload/v1/ticketF1"
LOGIN_PAGE = b"<!DOCTYPE html><html><head><title>Slack</title></head><body>Sign in</body></html>"

Reply = httpx.Response | Callable[[httpx.Request], httpx.Response]


class Endpoint:
    """Records every request and answers from a FIFO script (``ok`` when dry)."""

    def __init__(self, *replies: Reply) -> None:
        self.replies: list[Reply] = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.replies:
            return httpx.Response(200, json={"ok": True})
        reply = self.replies.pop(0)
        return reply(request) if callable(reply) else reply

    def form(self, index: int) -> dict[str, str]:
        body = self.requests[index].content.decode()
        return {k: v[0] for k, v in parse_qs(body, keep_blank_values=True).items()}


def ok(**body: Any) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, **body})


def make(*replies: Reply) -> tuple[SlackWebClient, Endpoint]:
    endpoint = Endpoint(*replies)
    http = httpx.Client(transport=httpx.MockTransport(endpoint))
    cfg = SlackBridgeConfig(
        bot_token=BOT_TOKEN, app_token=APP_TOKEN, core=CoreConfig(trust_env=False)
    )
    return SlackWebClient(cfg, http=http, sleep=lambda _s: None), endpoint


def file_reply(content: bytes, content_type: str = "text/csv", **headers: str) -> httpx.Response:
    return httpx.Response(200, content=content, headers={"Content-Type": content_type, **headers})


def streamed(chunks: list[bytes], content_type: str = "application/octet-stream") -> Reply:
    """A reply whose body is a stream with no declared length."""

    def _iter() -> Iterator[bytes]:
        yield from chunks

    return httpx.Response(200, content=_iter(), headers={"Content-Type": content_type})


# --- download: host guard -----------------------------------------------------------


def test_files_host_is_slacks_file_host() -> None:
    assert FILES_HOST == "files.slack.com"


def test_download_sends_the_bot_bearer_to_the_files_host_and_returns_bytes() -> None:
    client, endpoint = make(file_reply(b"a,b\n1,2\n"))

    data = client.download(FILE_URL, max_bytes=1024)

    assert data == b"a,b\n1,2\n"
    (request,) = endpoint.requests
    assert request.method == "GET"
    assert str(request.url) == FILE_URL
    assert request.headers["Authorization"] == f"Bearer {BOT_TOKEN}"


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com/files-pri/T1-F1/report.csv",
        "https://files.slack.com.evil.example/x",
        "https://evilfiles.slack.com/x",
        "https://slack.com/files-pri/T1-F1/report.csv",
        "https://cdn.files.slack.com/x",
        "http://files.slack.com/files-pri/T1-F1/report.csv",
        "https://files.slack.com:8443/files-pri/T1-F1/report.csv",
        "https://user:pw@files.slack.com/files-pri/T1-F1/report.csv",
        "ftp://files.slack.com/x",
        "files.slack.com/x",
        "",
        "not a url at all",
    ],
)
def test_a_foreign_or_malformed_url_raises_and_no_request_is_made(url: str) -> None:
    client, endpoint = make(file_reply(b"secret-free"))

    with pytest.raises(ForeignHostError):
        client.download(url, max_bytes=1024)

    assert endpoint.requests == []


def test_a_foreign_host_never_receives_an_authorization_header() -> None:
    client, endpoint = make()

    for url in ("https://drive.example.org/f", "https://files.slack.com.evil.example/f"):
        with pytest.raises(ForeignHostError):
            client.download(url, max_bytes=10)

    assert not any("Authorization" in r.headers for r in endpoint.requests)
    assert endpoint.requests == []


def test_the_host_match_is_case_insensitive() -> None:
    client, endpoint = make(file_reply(b"x"))

    assert client.download("https://FILES.Slack.COM/files-pri/T1-F1/x", max_bytes=10) == b"x"
    assert endpoint.requests[0].headers["Authorization"] == f"Bearer {BOT_TOKEN}"


def test_foreign_host_error_never_quotes_a_token() -> None:
    client, _ = make()

    with pytest.raises(ForeignHostError) as info:
        client.download(f"https://evil.example/{BOT_TOKEN}", max_bytes=10)

    assert BOT_TOKEN not in str(info.value)


def test_a_redirect_is_not_followed() -> None:
    client, endpoint = make(
        httpx.Response(302, headers={"Location": "https://evil.example/steal"}),
        file_reply(b"never"),
    )

    with pytest.raises(SlackApiError):
        client.download(FILE_URL, max_bytes=1024)

    assert len(endpoint.requests) == 1


# --- download: not-a-file guard -----------------------------------------------------


@pytest.mark.parametrize("content_type", ["text/html", "text/html; charset=utf-8", "TEXT/HTML"])
def test_an_html_login_page_is_rejected_as_not_a_file(content_type: str) -> None:
    client, _ = make(file_reply(LOGIN_PAGE, content_type))

    with pytest.raises(NotAFileError):
        client.download(FILE_URL, max_bytes=1024)


def test_a_declared_length_over_the_cap_is_rejected() -> None:
    client, _ = make(file_reply(b"x" * 100))

    with pytest.raises(NotAFileError):
        client.download(FILE_URL, max_bytes=99)


def test_a_stream_without_declared_length_is_abandoned_past_the_cap() -> None:
    client, _ = make(streamed([b"x" * 60, b"x" * 60, b"x" * 60]))

    with pytest.raises(NotAFileError):
        client.download(FILE_URL, max_bytes=100)


def test_a_size_overrun_is_typed_too_large_and_is_a_not_a_file_error() -> None:
    client, _ = make(file_reply(b"x" * 11))

    with pytest.raises(FileTooLargeError) as info:
        client.download(FILE_URL, max_bytes=10)

    assert isinstance(info.value, NotAFileError)


def test_a_body_exactly_at_the_cap_is_accepted() -> None:
    client, _ = make(streamed([b"x" * 5, b"x" * 5]))

    assert client.download(FILE_URL, max_bytes=10) == b"x" * 10


def test_a_non_positive_cap_is_a_value_error_and_no_request_is_made() -> None:
    client, endpoint = make()

    with pytest.raises(ValueError):
        client.download(FILE_URL, max_bytes=0)

    assert endpoint.requests == []


def test_an_error_status_is_a_slack_api_error_without_the_token() -> None:
    client, _ = make(httpx.Response(404, text=f"gone {BOT_TOKEN}"))

    with pytest.raises(SlackApiError) as info:
        client.download(FILE_URL, max_bytes=1024)

    assert BOT_TOKEN not in str(info.value)


def test_a_transport_failure_is_a_slack_api_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused {BOT_TOKEN}", request=request)

    client, _ = make(boom)

    with pytest.raises(SlackApiError) as info:
        client.download(FILE_URL, max_bytes=1024)

    assert BOT_TOKEN not in str(info.value)


# --- upload: the three steps --------------------------------------------------------


def _upload_script() -> tuple[Reply, ...]:
    return (
        ok(file_id="F1", upload_url=UPLOAD_URL),
        httpx.Response(200, text="OK - 5"),
        ok(files=[{"id": "F1", "title": "plot.png"}]),
    )


def test_upload_runs_get_url_then_post_bytes_then_complete() -> None:
    client, endpoint = make(*_upload_script())

    client.upload("C1", "1700000000.000100", "plot.png", b"\x89PNG!", "image/png")

    first, second, third = endpoint.requests
    assert str(first.url) == f"{API_BASE}/files.getUploadURLExternal"
    assert endpoint.form(0) == {"filename": "plot.png", "length": "5"}
    assert first.headers["Authorization"] == f"Bearer {BOT_TOKEN}"

    assert second.method == "POST"
    assert str(second.url) == UPLOAD_URL
    assert second.content == b"\x89PNG!"
    assert second.headers["Content-Type"] == "image/png"

    assert str(third.url) == f"{API_BASE}/files.completeUploadExternal"
    assert third.headers["Authorization"] == f"Bearer {BOT_TOKEN}"
    form = endpoint.form(2)
    assert json.loads(form["files"]) == [{"id": "F1", "title": "plot.png"}]
    assert form["channel_id"] == "C1"


def test_the_upload_url_receives_no_bearer() -> None:
    client, endpoint = make(*_upload_script())

    client.upload("C1", None, "plot.png", b"\x89PNG!", "image/png")

    assert "Authorization" not in endpoint.requests[1].headers
    assert BOT_TOKEN not in json.dumps(dict(endpoint.requests[1].headers))


def test_upload_into_a_thread_carries_thread_ts() -> None:
    client, endpoint = make(*_upload_script())

    client.upload("C1", "1700000000.000100", "plot.png", b"\x89PNG!", "image/png")

    assert endpoint.form(2)["thread_ts"] == "1700000000.000100"


@pytest.mark.parametrize("thread_ts", [None, ""])
def test_upload_at_top_level_omits_thread_ts(thread_ts: str | None) -> None:
    client, endpoint = make(*_upload_script())

    client.upload("D1", thread_ts, "plot.png", b"\x89PNG!", "image/png")

    assert "thread_ts" not in endpoint.form(2)
    assert endpoint.form(2)["channel_id"] == "D1"


def test_upload_returns_the_completion_body() -> None:
    client, _ = make(*_upload_script())

    body = client.upload("C1", None, "plot.png", b"\x89PNG!", "image/png")

    assert body["files"] == [{"id": "F1", "title": "plot.png"}]


def test_a_get_url_answer_without_url_or_id_raises_and_posts_nothing() -> None:
    client, endpoint = make(ok(file_id="F1"))

    with pytest.raises(SlackApiError):
        client.upload("C1", None, "a.txt", b"hello", "text/plain")

    assert len(endpoint.requests) == 1


def test_a_failed_byte_post_raises_and_never_completes() -> None:
    client, endpoint = make(
        ok(file_id="F1", upload_url=UPLOAD_URL),
        httpx.Response(500, text=f"boom {BOT_TOKEN}"),
    )

    with pytest.raises(SlackApiError) as info:
        client.upload("C1", None, "a.txt", b"hello", "text/plain")

    assert len(endpoint.requests) == 2
    assert BOT_TOKEN not in str(info.value)


def test_a_transport_failure_on_the_byte_post_raises_slack_api_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client, endpoint = make(ok(file_id="F1", upload_url=UPLOAD_URL), boom)

    with pytest.raises(SlackApiError):
        client.upload("C1", None, "a.txt", b"hello", "text/plain")

    assert len(endpoint.requests) == 2


def test_a_get_url_refusal_propagates_from_call() -> None:
    client, endpoint = make(httpx.Response(200, json={"ok": False, "error": "missing_scope"}))

    with pytest.raises(SlackApiError) as info:
        client.upload("C1", None, "a.txt", b"hello", "text/plain")

    assert info.value.error == "missing_scope"
    assert len(endpoint.requests) == 1
