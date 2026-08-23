"""Contract tests for :class:`osprey.interfaces.common_middleware.WebAuthMiddleware`.

Most of these drive raw ASGI scopes rather than a test client. Two reasons: the
middleware's whole job is to answer *before* an application exists, so a scope
is the honest input; and a raw scope is immune to whatever default credentials a
shared test-client fixture may inject later — a refusal test that a fixture can
silently turn into an admission is a test that stops testing.

The handful of client-driven tests below clear cookies explicitly for the same
reason, so they keep asserting what they say they assert.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.interfaces import common_middleware
from osprey.interfaces.common_middleware import (
    EXTERNAL_ORIGIN_ENV,
    MAX_BODY_PEEK_BYTES,
    MAX_SESSION_COOKIE_CANDIDATES,
    OPERATOR_SECRET_HEADER,
    SESSION_COOKIE_BASE,
    STATIC_MOUNT_PREFIXES,
    TERMINAL_USER_ENV,
    WEB_PORT_ENV,
    WEBSOCKET_REFUSAL_CODE,
    WebAuthMiddleware,
    is_exempt_path,
    session_cookie_name,
)
from osprey.interfaces.web_auth import WebCredentials, reset_web_credentials

#: Read off the module rather than retyped, so a rename cannot leave the
#: caplog assertions below silently watching a logger nothing writes to.
MIDDLEWARE_LOGGER = common_middleware.logger.name

OPERATOR_SECRET = "operator-secret-value"
PANEL_TOKEN = "panel-token-value"
COOKIE_NAME = "osprey_terminal_session_8080"


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class RecordingApp:
    """A downstream ASGI app that records what reached it and answers 200."""

    def __init__(self) -> None:
        self.scopes: list[dict[str, Any]] = []
        self.bodies: list[bytes] = []

    async def __call__(self, scope, receive, send) -> None:
        self.scopes.append(scope)
        if scope["type"] == "http":
            body = b""
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    break
                body += message.get("body") or b""
                if not message.get("more_body", False):
                    break
            self.bodies.append(body)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"downstream"})
        elif scope["type"] == "websocket":
            await send({"type": "websocket.accept"})

    @property
    def called(self) -> bool:
        return bool(self.scopes)


def encode_headers(headers: dict[str, str] | None) -> list[tuple[bytes, bytes]]:
    return [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()]


def http_scope(
    path: str = "/api/config",
    method: str = "GET",
    headers: dict[str, str] | None = None,
    *,
    app: Any = None,
    scheme: str = "http",
    raw_headers: list[tuple[bytes, bytes]] | None = None,
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "scheme": scheme,
        "headers": raw_headers if raw_headers is not None else encode_headers(headers),
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8080),
        "app": app,
    }


def ws_scope(
    path: str = "/ws",
    headers: dict[str, str] | None = None,
    *,
    app: Any = None,
    extensions: dict[str, Any] | None = None,
    scheme: str = "ws",
) -> dict[str, Any]:
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "scheme": scheme,
        "headers": encode_headers(headers),
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8080),
        "app": app,
    }
    if extensions is not None:
        scope["extensions"] = extensions
    return scope


def drive(middleware, scope, incoming: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Run one connection through ``middleware`` and return everything it sent."""
    pending = list(incoming if incoming is not None else [{"type": "http.request", "body": b""}])
    sent: list[dict[str, Any]] = []

    async def receive():
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def status_of(sent: list[dict[str, Any]]) -> int | None:
    for message in sent:
        if message["type"] in ("http.response.start", "websocket.http.response.start"):
            return message["status"]
    return None


def _content_type_of(sent: list[dict[str, Any]]) -> str:
    for message in sent:
        if message["type"] in ("http.response.start", "websocket.http.response.start"):
            for name, value in message["headers"]:
                if name == b"content-type":
                    return value.decode("latin-1")
    raise AssertionError(f"no content-type in {sent}")


def _body_of(sent: list[dict[str, Any]]) -> bytes:
    for message in sent:
        if message["type"] in ("http.response.body", "websocket.http.response.body"):
            return message["body"]
    raise AssertionError(f"no body in {sent}")


def detail_of(sent: list[dict[str, Any]]) -> str:
    for message in sent:
        if message["type"] in ("http.response.body", "websocket.http.response.body"):
            return json.loads(message["body"])["detail"]
    raise AssertionError(f"no JSON body in {sent}")


@pytest.fixture
def credentials() -> WebCredentials:
    """A known credential holder, with the process holder emptied around it."""
    reset_web_credentials()
    yield WebCredentials(operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN)
    reset_web_credentials()


@pytest.fixture
def app_stub(credentials: WebCredentials) -> Any:
    """The minimum an ASGI ``scope["app"]`` needs to carry the credentials."""
    return SimpleNamespace(state=SimpleNamespace(web_credentials=credentials))


@pytest.fixture
def downstream() -> RecordingApp:
    return RecordingApp()


@pytest.fixture
def middleware(downstream: RecordingApp) -> WebAuthMiddleware:
    return WebAuthMiddleware(downstream, cookie_name=COOKIE_NAME)


def session_cookie(credentials: WebCredentials, name: str = COOKIE_NAME) -> dict[str, str]:
    return {"cookie": f"{name}={credentials.create_session()}"}


# --------------------------------------------------------------------------- #
# Pass-through: protocols and exempt paths
# --------------------------------------------------------------------------- #


def test_lifespan_scope_passes_through_untouched(middleware, downstream):
    """A lifespan scope reaches the app: an unauthenticated app must still boot."""
    asyncio.run(middleware({"type": "lifespan"}, _never_receive, _drop_send))
    assert downstream.called


async def _never_receive():  # pragma: no cover - lifespan test never reads
    raise AssertionError("lifespan receive should be handed straight to the app")


async def _drop_send(message):  # pragma: no cover - the stub app sends nothing
    return None


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/static/session.html",
        "/static",
        "/static/js/app.js",
        "/static/fonts/inter.woff2",
        "/design-system",
        "/design-system/tokens.css",
    ],
)
def test_exempt_paths_need_no_credential(middleware, downstream, app_stub, path):
    """The healthcheck, the token-exchange page and the static mounts stay open."""
    sent = drive(middleware, http_scope(path, app=app_stub))
    assert status_of(sent) == 200
    assert downstream.called


@pytest.mark.parametrize(
    "path",
    ["/health/", "/healthz", "/static-secrets/keys.json", "/design-systems", "/", "/api/panels"],
)
def test_lookalike_paths_are_not_exempt(middleware, downstream, app_stub, path):
    """Exemption is exact or mount-scoped — never a bare prefix match."""
    sent = drive(middleware, http_scope(path, app=app_stub))
    assert status_of(sent) == 401
    assert not downstream.called


def test_is_exempt_path_matches_mount_paths_and_children():
    """The importable predicate is what a route-walking test can check against."""
    for prefix in STATIC_MOUNT_PREFIXES:
        assert is_exempt_path(prefix)
        assert is_exempt_path(f"{prefix}/asset.css")
    # A sibling of a mount is not the mount. ``/static/fonts-elsewhere`` is
    # deliberately absent from this list: it lives *under* the ``/static`` mount
    # and is served by it, so it is exempt for that reason rather than by
    # accident of string matching.
    assert not is_exempt_path("/static-secrets/keys.json")
    assert not is_exempt_path("/design-systems/tokens.css")
    assert is_exempt_path("/health")
    assert not is_exempt_path("/api/config")


# --------------------------------------------------------------------------- #
# Refusal shape
# --------------------------------------------------------------------------- #


def test_missing_credential_is_401_json(middleware, downstream, app_stub):
    sent = drive(middleware, http_scope(app=app_stub))
    assert status_of(sent) == 401
    assert detail_of(sent) == "authentication required"
    headers = dict(sent[0]["headers"])
    assert headers[b"content-type"] == b"application/json"
    assert not downstream.called


# --------------------------------------------------------------------------- #
# Operator secret header
# --------------------------------------------------------------------------- #


def test_operator_header_grants_full_access(middleware, downstream, app_stub):
    sent = drive(
        middleware,
        http_scope("/api/config", "POST", {OPERATOR_SECRET_HEADER: OPERATOR_SECRET}, app=app_stub),
    )
    assert status_of(sent) == 200
    assert downstream.called


def test_wrong_operator_header_is_refused(middleware, downstream, app_stub):
    sent = drive(middleware, http_scope(headers={OPERATOR_SECRET_HEADER: "wrong"}, app=app_stub))
    assert status_of(sent) == 401
    assert detail_of(sent) == "invalid credential"
    assert not downstream.called


def test_wrong_operator_header_does_not_fall_back_to_a_valid_cookie(
    middleware, downstream, app_stub, credentials
):
    """One comparison per request: a presented-but-wrong header ends the request.

    Falling through would make the work — and so the timing — depend on how many
    of the caller's credentials were wrong.
    """
    headers = {OPERATOR_SECRET_HEADER: "wrong", **session_cookie(credentials)}
    sent = drive(middleware, http_scope(headers=headers, app=app_stub))
    assert status_of(sent) == 401
    assert not downstream.called


def test_blank_operator_header_is_treated_as_absent(middleware, downstream, app_stub, credentials):
    """An empty header is no credential, so the cookie beside it still counts."""
    headers = {OPERATOR_SECRET_HEADER: "   ", **session_cookie(credentials)}
    sent = drive(middleware, http_scope(headers=headers, app=app_stub))
    assert status_of(sent) == 200
    assert downstream.called


def test_operator_header_with_a_foreign_origin_is_403(middleware, downstream, app_stub):
    """The Origin check is not scoped to the cookie — the header path runs it too.

    It was, on the reasoning that no browser sends this header. In the
    multi-user shape nginx *injects* it on every authenticated browser request,
    so scoping the check to cookies switched app-level CSRF off for exactly the
    deployment that has real users, leaving it resting on the sidecar cookie's
    SameSite and the absence of CORS.
    """
    headers = {
        OPERATOR_SECRET_HEADER: OPERATOR_SECRET,
        "origin": "https://evil.test",
        "host": "localhost:8080",
    }
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 403
    assert detail_of(sent) == "cross-origin request refused"
    assert not downstream.called


def test_operator_header_with_no_origin_is_allowed(middleware, downstream, app_stub):
    """A non-browser caller sends no ``Origin`` and must not be refused for it.

    This is the difference between the header path and the cookie path: a
    script, a probe or an in-process companion carrying the operator secret has
    no ``Origin`` to offer, and demanding one would refuse every one of them.
    """
    headers = {OPERATOR_SECRET_HEADER: OPERATOR_SECRET, "host": "localhost:8080"}
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 200
    assert downstream.called


def test_operator_header_with_a_matching_origin_is_allowed(middleware, downstream, app_stub):
    """A genuine browser request behind nginx carries its own Origin and passes."""
    headers = {
        OPERATOR_SECRET_HEADER: OPERATOR_SECRET,
        "origin": "http://localhost:8080",
        "host": "localhost:8080",
    }
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 200
    assert downstream.called


def test_operator_header_websocket_with_a_foreign_origin_is_403(middleware, downstream, app_stub):
    """A websocket handshake is checked whatever credential opens it."""
    headers = {
        OPERATOR_SECRET_HEADER: OPERATOR_SECRET,
        "origin": "http://evil.test",
        "host": "localhost:8080",
    }
    sent = drive(middleware, ws_scope(headers=headers, app=app_stub, extensions=WS_EXTENSIONS))
    assert status_of(sent) == 403
    assert not downstream.called


def test_external_origin_env_is_consulted_on_the_header_path(
    middleware, downstream, app_stub, monkeypatch
):
    """The proxy shape's declared origin governs the header path too.

    Behind nginx the app's own ``Host`` is an internal service name no browser
    has seen, so ``OSPREY_TERMINAL_EXTERNAL_ORIGIN`` is the only thing that can
    say what a legitimate ``Origin`` looks like. Now that the check applies to
    the injected operator header, that variable is load-bearing for the whole
    browser surface rather than for the cookie alone.
    """
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, "https://osprey.example.org")
    base = {OPERATOR_SECRET_HEADER: OPERATOR_SECRET, "host": "web-terminal-alice:8087"}

    allowed = drive(
        middleware,
        http_scope(
            "/api/config", "POST", {**base, "origin": "https://osprey.example.org"}, app=app_stub
        ),
    )
    refused = drive(
        middleware,
        http_scope("/api/config", "POST", {**base, "origin": "https://evil.test"}, app=app_stub),
    )

    assert status_of(allowed) == 200
    assert status_of(refused) == 403


# --------------------------------------------------------------------------- #
# Session cookie
# --------------------------------------------------------------------------- #


def test_valid_cookie_grants_a_safe_request(middleware, downstream, app_stub, credentials):
    sent = drive(middleware, http_scope(headers=session_cookie(credentials), app=app_stub))
    assert status_of(sent) == 200
    assert downstream.called


def test_expired_or_forged_cookie_is_refused(middleware, downstream, app_stub):
    headers = {"cookie": f"{COOKIE_NAME}=never-issued"}
    sent = drive(middleware, http_scope(headers=headers, app=app_stub))
    assert status_of(sent) == 401
    assert detail_of(sent) == "invalid credential"
    assert not downstream.called


def test_revoked_session_is_refused(middleware, downstream, app_stub, credentials):
    session_id = credentials.create_session()
    credentials.revoke_session(session_id)
    headers = {"cookie": f"{COOKIE_NAME}={session_id}"}
    assert status_of(drive(middleware, http_scope(headers=headers, app=app_stub))) == 401


@pytest.mark.parametrize(
    "cookie_header",
    [
        "",
        "garbage-with-no-equals",
        ";;;",
        "other=value",
        f"{COOKIE_NAME}",
        "=value",
        f"{COOKIE_NAME}=",
    ],
)
def test_malformed_cookie_headers_read_as_no_credential(
    middleware, downstream, app_stub, cookie_header
):
    """Junk in the cookie header is an absent credential, never an exception."""
    sent = drive(middleware, http_scope(headers={"cookie": cookie_header}, app=app_stub))
    assert status_of(sent) == 401
    assert detail_of(sent) == "authentication required"
    assert not downstream.called


def test_adversarial_cookie_bytes_are_refused_not_crashed(middleware, downstream, app_stub):
    """NUL bytes and non-ASCII in a forged cookie refuse rather than 500.

    ``secrets.compare_digest`` raises ``TypeError`` on a non-ASCII ``str``, so
    the decoded value has to reach ``verify_session`` in a shape it can compare.
    """
    raw = [(b"cookie", COOKIE_NAME.encode() + b"=\x00\x00\xff\xfe" + b"caf\xc3\xa9")]
    sent = drive(middleware, http_scope(raw_headers=raw, app=app_stub))
    assert status_of(sent) == 401
    assert not downstream.called


def test_quoted_cookie_value_is_unquoted(middleware, downstream, app_stub, credentials):
    session_id = credentials.create_session()
    headers = {"cookie": f'other=x; {COOKIE_NAME}="{session_id}"'}
    assert status_of(drive(middleware, http_scope(headers=headers, app=app_stub))) == 200


def test_split_cookie_headers_are_joined(middleware, downstream, app_stub, credentials):
    """HTTP/2 clients may split the cookie header; the session must survive it."""
    session_id = credentials.create_session()
    raw = [
        (b"cookie", b"theme=dark"),
        (b"cookie", f"{COOKIE_NAME}={session_id}".encode()),
    ]
    assert status_of(drive(middleware, http_scope(raw_headers=raw, app=app_stub))) == 200


def test_cookie_under_another_name_is_ignored(middleware, downstream, app_stub, credentials):
    """The name carries the port, so a neighbouring server's cookie is not ours."""
    headers = session_cookie(credentials, name="osprey_terminal_session_9999")
    assert status_of(drive(middleware, http_scope(headers=headers, app=app_stub))) == 401


def test_cookie_name_is_derived_from_the_port_environment(
    downstream, app_stub, credentials, monkeypatch
):
    """An un-pinned middleware reads the settled port per request, as 2.1 writes it."""
    monkeypatch.setenv(WEB_PORT_ENV, "8123")
    guard = WebAuthMiddleware(downstream)
    headers = session_cookie(credentials, name="osprey_terminal_session_8123")
    assert status_of(drive(guard, http_scope(headers=headers, app=app_stub))) == 200


@pytest.mark.parametrize(
    ("port", "expected"),
    [
        (8080, "osprey_terminal_session_8080"),
        ("8080", "osprey_terminal_session_8080"),
        ("", SESSION_COOKIE_BASE),
        ("not-a-port", SESSION_COOKIE_BASE),
        ("80 80", SESSION_COOKIE_BASE),
    ],
)
def test_session_cookie_name_shapes(port, expected):
    assert session_cookie_name(port) == expected


def test_session_cookie_name_falls_back_when_port_unset(monkeypatch):
    monkeypatch.delenv(WEB_PORT_ENV, raising=False)
    assert session_cookie_name() == SESSION_COOKIE_BASE


# --------------------------------------------------------------------------- #
# Origin / CSRF
# --------------------------------------------------------------------------- #


def test_mutating_cookie_request_with_matching_origin_is_allowed(
    middleware, downstream, app_stub, credentials
):
    headers = {
        "host": "localhost:8080",
        "origin": "http://localhost:8080",
        **session_cookie(credentials),
    }
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 200
    assert downstream.called


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_mutating_cookie_request_from_a_foreign_origin_is_403(
    middleware, downstream, app_stub, credentials, method
):
    headers = {
        "host": "localhost:8080",
        "origin": "http://evil.test",
        **session_cookie(credentials),
    }
    sent = drive(middleware, http_scope("/api/config", method, headers, app=app_stub))
    assert status_of(sent) == 403
    assert detail_of(sent) == "cross-origin request refused"
    assert not downstream.called


def test_origin_match_is_whole_string_not_a_prefix(middleware, downstream, app_stub, credentials):
    """``http://localhost:8080.evil.test`` starts with the real origin."""
    headers = {
        "host": "localhost:8080",
        "origin": "http://localhost:8080.evil.test",
        **session_cookie(credentials),
    }
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 403


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_safe_methods_skip_the_origin_check(middleware, downstream, app_stub, credentials, method):
    """A cross-site GET can be made but not read, so it needs no Origin."""
    headers = {"host": "localhost:8080", "origin": "http://evil.test"}
    headers.update(session_cookie(credentials))
    sent = drive(middleware, http_scope("/api/panels", method, headers, app=app_stub))
    assert status_of(sent) == 200


def test_sec_fetch_site_same_origin_stands_in_for_a_missing_origin(
    middleware, downstream, app_stub, credentials
):
    headers = {
        "host": "localhost:8080",
        "sec-fetch-site": "Same-Origin",
        **session_cookie(credentials),
    }
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 200


@pytest.mark.parametrize("site", ["cross-site", "same-site", "none"])
def test_other_sec_fetch_site_values_are_refused(
    middleware, downstream, app_stub, credentials, site
):
    headers = {"host": "localhost:8080", "sec-fetch-site": site, **session_cookie(credentials)}
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 403


def test_mutating_cookie_request_with_no_origin_headers_at_all_is_403(
    middleware, downstream, app_stub, credentials
):
    """A browser sends one or the other; their absence means it is not a browser."""
    headers = {"host": "localhost:8080", **session_cookie(credentials)}
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 403
    assert not downstream.called


def test_origin_with_no_host_to_compare_against_is_403(
    middleware, downstream, app_stub, credentials
):
    headers = {"origin": "http://localhost:8080", **session_cookie(credentials)}
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 403


def test_external_origin_env_overrides_the_host_header(
    middleware, downstream, app_stub, credentials, monkeypatch
):
    """The nginx shape: the app's own Host is a service name no browser sends."""
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, "https://osprey.example.test")
    headers = {
        "host": "osprey-user-terminal:8080",
        "origin": "https://osprey.example.test",
        **session_cookie(credentials),
    }
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 200


def test_external_origin_env_is_compared_exactly(
    middleware, downstream, app_stub, credentials, monkeypatch
):
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, "https://osprey.example.test")
    headers = {
        "host": "osprey.example.test",
        "origin": "https://osprey.example.test.evil.test",
        **session_cookie(credentials),
    }
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == 403


def test_constructor_origin_beats_the_environment(downstream, app_stub, credentials, monkeypatch):
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, "https://ignored.test")
    guard = WebAuthMiddleware(
        downstream, cookie_name=COOKIE_NAME, external_origin="https://pinned.test"
    )
    headers = {"host": "anything", "origin": "https://pinned.test", **session_cookie(credentials)}
    assert status_of(drive(guard, http_scope("/api/config", "POST", headers, app=app_stub))) == 200


def test_https_request_derives_an_https_origin(middleware, downstream, app_stub, credentials):
    headers = {
        "host": "console.test",
        "origin": "https://console.test",
        **session_cookie(credentials),
    }
    scope = http_scope("/api/config", "POST", headers, app=app_stub, scheme="https")
    assert status_of(drive(middleware, scope)) == 200


# --------------------------------------------------------------------------- #
# Origin serialization: a scheme's default port
# --------------------------------------------------------------------------- #


#: ``(the origin the app resolves, the origin the browser sends, admitted?)``.
#: A browser elides a scheme's default port when it serializes an ``Origin``
#: (RFC 6454 / WHATWG URL); the producers on the app's side do not — the
#: deployment renderer always writes ``http://<fqdn>:<nginx_port>`` when TLS is
#: off, and the single-user ``Host`` fallback carries whatever port the client
#: typed. The two spellings mean the same origin and must compare equal; a
#: genuinely different port, or a different scheme, must not.
ORIGIN_DEFAULT_PORT_CASES = [
    ("http://h:80", "http://h", True),
    ("https://h:443", "https://h", True),
    ("http://h:9080", "http://h:9080", True),
    ("http://h:9080", "http://h", False),
    ("http://h", "https://h", False),
]


@pytest.mark.parametrize(("expected", "origin", "allowed"), ORIGIN_DEFAULT_PORT_CASES)
def test_declared_origin_is_compared_with_default_ports_elided(
    middleware, downstream, app_stub, credentials, monkeypatch, expected, origin, allowed
):
    """``nginx_port: 80`` must not refuse every write in the proxy shape.

    ``OSPREY_TERMINAL_EXTERNAL_ORIGIN`` is rendered as
    ``http://<fqdn>:<nginx_port>`` with the port always explicit when TLS is
    off, so a plain-HTTP deployment on port 80 declares ``http://host:80``
    while every browser on it sends ``Origin: http://host``. Normalizing in
    the gate rather than at the producer also covers a hand-written value.
    """
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, expected)
    headers = {
        "host": "web-terminal-alice:8087",
        "origin": origin,
        **session_cookie(credentials),
    }
    sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))
    assert status_of(sent) == (200 if allowed else 403)


@pytest.mark.parametrize(("expected", "origin", "allowed"), ORIGIN_DEFAULT_PORT_CASES)
def test_host_derived_origin_is_compared_with_default_ports_elided(
    middleware, downstream, app_stub, credentials, monkeypatch, expected, origin, allowed
):
    """The single-user fallback needs the same normalization.

    There the origin is built from the request's own ``Host``, which carries
    ``:80`` whenever the client sent it explicitly — a redirect, a proxy or a
    typed URL — while the ``Origin`` beside it does not.
    """
    monkeypatch.delenv(EXTERNAL_ORIGIN_ENV, raising=False)
    scheme, _, authority = expected.partition("://")
    headers = {"host": authority, "origin": origin, **session_cookie(credentials)}
    scope = http_scope("/api/config", "POST", headers, app=app_stub, scheme=scheme)
    assert status_of(drive(middleware, scope)) == (200 if allowed else 403)


def test_operator_header_origin_is_normalized_too(middleware, downstream, app_stub, monkeypatch):
    """The header path is the one nginx injects on real browser requests."""
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, "http://osprey.example.org:80")
    headers = {
        OPERATOR_SECRET_HEADER: OPERATOR_SECRET,
        "host": "web-terminal-alice:8087",
        "origin": "http://osprey.example.org",
    }
    sent = drive(middleware, http_scope("/api/config", "PATCH", headers, app=app_stub))
    assert status_of(sent) == 200
    assert downstream.called


# --------------------------------------------------------------------------- #
# Origin refusals are audible
# --------------------------------------------------------------------------- #


def test_an_origin_refusal_names_both_origins_in_the_log(
    middleware, app_stub, credentials, monkeypatch, caplog
):
    """The one refusal whose cause is deployment configuration must not be silent.

    Neither the received nor the resolved origin is a credential, and an
    operator whose writes all 403 has nothing else to go on: the page loads,
    the buttons do nothing, and the JSON detail says only "cross-origin
    request refused".
    """
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, "https://osprey.example.org")
    headers = {
        OPERATOR_SECRET_HEADER: OPERATOR_SECRET,
        "host": "web-terminal-alice:8087",
        "origin": "https://evil.test",
    }
    with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
        sent = drive(middleware, http_scope("/api/config", "PATCH", headers, app=app_stub))

    assert status_of(sent) == 403
    records = [record for record in caplog.records if record.name == MIDDLEWARE_LOGGER]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "https://evil.test" in message
    assert "https://osprey.example.org" in message
    assert "PATCH" in message
    assert "/api/config" in message
    assert OPERATOR_SECRET not in message


def test_an_absent_origin_refusal_names_sec_fetch_site(
    middleware, app_stub, credentials, monkeypatch, caplog
):
    """With no ``Origin`` the header that decided it is what the log has to carry."""
    monkeypatch.delenv(EXTERNAL_ORIGIN_ENV, raising=False)
    headers = {
        "host": "console.test",
        "sec-fetch-site": "cross-site",
        **session_cookie(credentials),
    }
    with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
        sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))

    assert status_of(sent) == 403
    message = "\n".join(
        record.getMessage() for record in caplog.records if record.name == MIDDLEWARE_LOGGER
    )
    assert "sec-fetch-site" in message.lower()
    assert "cross-site" in message
    assert "http://console.test" in message


def test_a_matching_origin_logs_nothing(middleware, app_stub, credentials, caplog):
    """The warning fires only on requests that are being refused anyway."""
    headers = {
        "host": "console.test",
        "origin": "http://console.test",
        **session_cookie(credentials),
    }
    with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
        sent = drive(middleware, http_scope("/api/config", "POST", headers, app=app_stub))

    assert status_of(sent) == 200
    assert [record for record in caplog.records if record.name == MIDDLEWARE_LOGGER] == []


# --------------------------------------------------------------------------- #
# Panel token tier
# --------------------------------------------------------------------------- #


def bearer(token: str = PANEL_TOKEN) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def test_panel_token_reaches_a_panel_tier_route(middleware, downstream, app_stub):
    sent = drive(middleware, http_scope("/api/panels", "GET", bearer(), app=app_stub))
    assert status_of(sent) == 200
    assert downstream.called


def test_panel_token_is_refused_on_an_operator_route(middleware, downstream, app_stub):
    sent = drive(middleware, http_scope("/api/config", "POST", bearer(), app=app_stub))
    assert status_of(sent) == 401
    assert detail_of(sent) == "this route requires operator credentials"
    assert not downstream.called


def test_wrong_panel_token_is_refused(middleware, downstream, app_stub):
    sent = drive(middleware, http_scope("/api/panels", "GET", bearer("wrong"), app=app_stub))
    assert status_of(sent) == 401
    assert detail_of(sent) == "invalid credential"


@pytest.mark.parametrize("header", ["Bearer", "Bearer   ", "Basic abc", "token-with-no-scheme"])
def test_unusable_authorization_headers_read_as_no_credential(
    middleware, downstream, app_stub, header
):
    sent = drive(middleware, http_scope(headers={"authorization": header}, app=app_stub))
    assert status_of(sent) == 401
    assert detail_of(sent) == "authentication required"


def test_bearer_scheme_is_case_insensitive(middleware, downstream, app_stub):
    headers = {"authorization": f"bearer {PANEL_TOKEN}"}
    assert (
        status_of(drive(middleware, http_scope("/api/panels", "GET", headers, app=app_stub))) == 200
    )


def test_panel_token_with_no_origin_is_allowed(middleware, downstream, app_stub):
    """The panel token's normal caller is in-process and sends no ``Origin``."""
    headers = {"host": "localhost:8080", **bearer()}
    sent = drive(middleware, http_scope("/api/panel-focus", "POST", headers, app=app_stub))
    assert status_of(sent) == 200
    assert downstream.called


def test_panel_token_with_a_foreign_origin_is_403(middleware, downstream, app_stub):
    """A present-and-foreign ``Origin`` is refused on the panel path as well.

    Nothing should be able to reach a mutating route from another site, whatever
    credential it presents — the check keys off the request being a browser
    request, not off which credential authenticated it.
    """
    headers = {"host": "localhost:8080", "origin": "http://evil.test", **bearer()}
    sent = drive(middleware, http_scope("/api/panel-focus", "POST", headers, app=app_stub))
    assert status_of(sent) == 403
    assert not downstream.called


def register_scope(app_stub, headers: dict[str, str] | None = None):
    return http_scope("/api/panels/register", "POST", {**bearer(), **(headers or {})}, app=app_stub)


def body_messages(payload: bytes) -> list[dict[str, Any]]:
    return [{"type": "http.request", "body": payload, "more_body": False}]


def test_url_free_registration_is_panel_tier_and_the_body_is_replayed(
    middleware, downstream, app_stub
):
    payload = json.dumps({"name": "scan", "panel_id": "p1"}).encode()
    sent = drive(middleware, register_scope(app_stub), body_messages(payload))
    assert status_of(sent) == 200
    assert downstream.bodies == [payload]


def test_url_backed_registration_is_operator_only(middleware, downstream, app_stub):
    payload = json.dumps({"name": "scan", "url": "http://127.0.0.1:9/"}).encode()
    sent = drive(middleware, register_scope(app_stub), body_messages(payload))
    assert status_of(sent) == 401
    assert not downstream.called


def test_null_url_still_counts_as_a_url_key(middleware, downstream, app_stub):
    """The key's presence decides, not its truthiness — a null is still a repoint."""
    sent = drive(middleware, register_scope(app_stub), body_messages(b'{"url": null}'))
    assert status_of(sent) == 401


@pytest.mark.parametrize(
    "payload", [b"", b"not json at all", b"[1, 2, 3]", b'"a string"', b"\xff\xfe\x00"]
)
def test_unreadable_registration_bodies_fail_closed(middleware, downstream, app_stub, payload):
    """A body the gate cannot parse is treated as the dangerous one."""
    sent = drive(middleware, register_scope(app_stub), body_messages(payload))
    assert status_of(sent) == 401
    assert not downstream.called


def test_streamed_registration_body_is_buffered_and_replayed(middleware, downstream, app_stub):
    payload = json.dumps({"name": "scan"}).encode()
    messages = [
        {"type": "http.request", "body": payload[:5], "more_body": True},
        {"type": "http.request", "body": payload[5:], "more_body": False},
    ]
    sent = drive(middleware, register_scope(app_stub), messages)
    assert status_of(sent) == 200
    assert downstream.bodies == [payload]


def test_oversized_registration_body_fails_closed_without_buffering_it_all(
    middleware, downstream, app_stub
):
    chunk = b"x" * (MAX_BODY_PEEK_BYTES + 1)
    sent = drive(middleware, register_scope(app_stub), body_messages(b'{"name": "' + chunk))
    assert status_of(sent) == 401
    assert not downstream.called


def test_disconnect_mid_body_fails_closed(middleware, downstream, app_stub):
    messages = [
        {"type": "http.request", "body": b'{"na', "more_body": True},
        {"type": "http.disconnect"},
    ]
    sent = drive(middleware, register_scope(app_stub), messages)
    assert status_of(sent) == 401
    assert not downstream.called


def test_body_is_only_peeked_for_the_registration_route(middleware, downstream, app_stub):
    """Every other route decides on method and path alone — no body is read."""
    payload = json.dumps({"url": "http://evil.test"}).encode()
    sent = drive(
        middleware,
        http_scope("/api/panel-arrange", "POST", bearer(), app=app_stub),
        body_messages(payload),
    )
    assert status_of(sent) == 200
    assert downstream.bodies == [payload]


def test_operator_secret_reaches_url_backed_registration(middleware, downstream, app_stub):
    payload = json.dumps({"url": "http://127.0.0.1:9/"}).encode()
    scope = http_scope(
        "/api/panels/register", "POST", {OPERATOR_SECRET_HEADER: OPERATOR_SECRET}, app=app_stub
    )
    sent = drive(middleware, scope, body_messages(payload))
    assert status_of(sent) == 200
    assert downstream.bodies == [payload]


# --------------------------------------------------------------------------- #
# Websockets
# --------------------------------------------------------------------------- #

WS_EXTENSIONS = {"websocket.http.response": {}}


def test_unauthenticated_websocket_is_refused_at_the_handshake(middleware, downstream, app_stub):
    """A 401 the client can read, not an accepted socket that then closes."""
    sent = drive(middleware, ws_scope(app=app_stub, extensions=WS_EXTENSIONS), incoming=[])
    assert [message["type"] for message in sent] == [
        "websocket.http.response.start",
        "websocket.http.response.body",
    ]
    assert status_of(sent) == 401
    assert detail_of(sent) == "authentication required"
    assert not downstream.called


def test_websocket_refusal_falls_back_to_close_without_the_extension(
    middleware, downstream, app_stub
):
    """Only when the server does not advertise it — uvicorn's sansio impl does."""
    sent = drive(middleware, ws_scope(app=app_stub), incoming=[])
    assert sent == [{"type": "websocket.close", "code": WEBSOCKET_REFUSAL_CODE}]
    assert not downstream.called


def test_websocket_is_never_accepted_before_a_refusal(middleware, downstream, app_stub):
    sent = drive(middleware, ws_scope(app=app_stub, extensions=WS_EXTENSIONS), incoming=[])
    assert all(message["type"] != "websocket.accept" for message in sent)


def test_operator_secret_opens_a_websocket(middleware, downstream, app_stub):
    scope = ws_scope(
        headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET}, app=app_stub, extensions=WS_EXTENSIONS
    )
    sent = drive(middleware, scope, incoming=[])
    assert sent == [{"type": "websocket.accept"}]
    assert downstream.called


def test_panel_token_never_opens_a_websocket(middleware, downstream, app_stub):
    """No websocket is panel-tier; the terminal socket carries keystrokes."""
    scope = ws_scope(headers=bearer(), app=app_stub, extensions=WS_EXTENSIONS)
    sent = drive(middleware, scope, incoming=[])
    assert status_of(sent) == 401
    assert detail_of(sent) == "this route requires operator credentials"
    assert not downstream.called


def test_cookie_websocket_needs_a_matching_origin(middleware, downstream, app_stub, credentials):
    """A handshake is a cross-site request a page can make with the cookie attached."""
    headers = {
        "host": "localhost:8080",
        "origin": "http://evil.test",
        **session_cookie(credentials),
    }
    scope = ws_scope(headers=headers, app=app_stub, extensions=WS_EXTENSIONS)
    sent = drive(middleware, scope, incoming=[])
    assert status_of(sent) == 403
    assert detail_of(sent) == "cross-origin request refused"
    assert not downstream.called


def test_cookie_websocket_from_the_page_itself_is_accepted(
    middleware, downstream, app_stub, credentials
):
    """``wss``/``ws`` scopes compare against the page's ``https``/``http`` origin."""
    headers = {
        "host": "console.test",
        "origin": "https://console.test",
        **session_cookie(credentials),
    }
    scope = ws_scope(headers=headers, app=app_stub, extensions=WS_EXTENSIONS, scheme="wss")
    sent = drive(middleware, scope, incoming=[])
    assert sent == [{"type": "websocket.accept"}]


# --------------------------------------------------------------------------- #
# Credential resolution
# --------------------------------------------------------------------------- #


def test_credentials_come_from_the_process_holder_when_the_app_has_none(
    middleware, downstream, monkeypatch
):
    """An app that never cached them still gates on the process's credentials."""
    from osprey.interfaces import web_auth

    reset_web_credentials()
    monkeypatch.setenv(web_auth.OPERATOR_SECRET_ENV, "process-wide-secret")
    monkeypatch.delenv(web_auth.BIND_HOST_ENV, raising=False)
    try:
        headers = {OPERATOR_SECRET_HEADER: "process-wide-secret"}
        assert status_of(drive(middleware, http_scope(headers=headers))) == 200
        assert (
            status_of(drive(middleware, http_scope(headers={OPERATOR_SECRET_HEADER: "no"}))) == 401
        )
    finally:
        reset_web_credentials()


def test_app_state_credentials_win_over_the_process_holder(middleware, downstream, app_stub):
    """A companion app carrying the hub's holder authenticates against that one."""
    sent = drive(
        middleware,
        http_scope(headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET}, app=app_stub),
    )
    assert status_of(sent) == 200


# --------------------------------------------------------------------------- #
# Through a real Starlette stack
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(credentials):
    """A test client over a real app behind the gate, with no ambient cookies.

    Cookies are cleared explicitly rather than trusted to be absent: a shared
    fixture that injects a session into every client would otherwise turn the
    refusal tests below into admissions without failing anything.
    """
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.testclient import TestClient

    async def config(request):
        return PlainTextResponse("config")

    async def socket(websocket):
        await websocket.accept()
        await websocket.send_text("open")
        await websocket.close()

    app = Starlette(
        routes=[
            Route("/api/config", config, methods=["GET", "POST"]),
            Route("/health", lambda request: PlainTextResponse("ok")),
            WebSocketRoute("/ws", socket),
        ]
    )
    app.add_middleware(WebAuthMiddleware, cookie_name=COOKIE_NAME)
    app.state.web_credentials = credentials
    with TestClient(app) as test_client:
        test_client.cookies.clear()
        yield test_client


@pytest.mark.no_auth_seam
def test_client_refusal_is_json_401(client):
    response = client.get("/api/config")
    assert response.status_code == 401
    assert response.json() == {"detail": "authentication required"}


@pytest.mark.no_auth_seam
def test_client_health_stays_open(client):
    # Without the marker the seam force-sets the operator header on this
    # request, so the assertion would hold even if /health had stopped being
    # exempt — the one thing it exists to prove.
    assert client.get("/health").status_code == 200


@pytest.mark.no_auth_seam
def test_client_operator_header_reaches_the_route(client):
    # The marker is what makes the header below the *only* credential in play;
    # under the seam the same header is injected anyway and the test would pass
    # whether or not this one was read.
    response = client.get("/api/config", headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET})
    assert response.status_code == 200
    assert response.text == "config"


@pytest.mark.no_auth_seam
def test_client_cookie_post_needs_the_origin(client, credentials):
    client.cookies.set(COOKIE_NAME, credentials.create_session())
    assert client.post("/api/config").status_code == 403
    allowed = client.post("/api/config", headers={"origin": "http://testserver"})
    assert allowed.status_code == 200


@pytest.mark.no_auth_seam
def test_client_websocket_denial_is_an_http_401(client):
    from starlette.testclient import WebSocketDenialResponse

    with pytest.raises(WebSocketDenialResponse) as refused:
        with client.websocket_connect("/ws"):
            pass  # pragma: no cover - the handshake must never complete
    assert refused.value.status_code == 401
    assert refused.value.json() == {"detail": "authentication required"}


@pytest.mark.no_auth_seam
def test_client_websocket_opens_with_the_operator_secret(client):
    # As above: the seam also carries the operator header onto the handshake,
    # so without the marker this would not be testing the header it passes.
    headers = {OPERATOR_SECRET_HEADER: OPERATOR_SECRET}
    with client.websocket_connect("/ws", headers=headers) as socket:
        assert socket.receive_text() == "open"


# --------------------------------------------------------------------------- #
# Refusal rendering: JSON for machines, HTML for a navigating browser
# --------------------------------------------------------------------------- #


def test_refusal_is_json_for_a_fetch(middleware, downstream, app_stub):
    """``fetch``/``XHR`` default to ``Accept: */*`` and keep the JSON body."""
    sent = drive(middleware, http_scope(headers={"accept": "*/*"}, app=app_stub))
    assert status_of(sent) == 401
    assert detail_of(sent) == "authentication required"
    assert _content_type_of(sent) == "application/json"


def test_refusal_is_html_for_a_navigating_browser(middleware, downstream, app_stub, monkeypatch):
    """A navigation gets a readable page, not raw JSON.

    Single-user has no perimeter in front of the app, so when a session expires
    the gate itself is the only thing that can tell the operator what happened —
    and ``api.js`` reloads the page precisely so this navigation happens. The
    status is unchanged; only the rendering differs.
    """
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    accept = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    sent = drive(middleware, http_scope(headers={"accept": accept}, app=app_stub))

    assert status_of(sent) == 401
    assert _content_type_of(sent).startswith("text/html")
    body = _body_of(sent).decode("utf-8")
    assert "<!DOCTYPE html>" in body
    # It has to say what to do next, not merely that something went wrong.
    assert "?token=" in body
    # And it must never quote a credential back at the browser.
    assert OPERATOR_SECRET not in body


def test_html_refusal_is_self_contained(middleware, downstream, app_stub):
    """No asset references: the page must render for an operator whose session is gone."""
    sent = drive(middleware, http_scope(headers={"accept": "text/html"}, app=app_stub))
    body = _body_of(sent).decode("utf-8")
    for reference in ("<link", "<script", "src=", "href="):
        assert reference not in body


def test_html_refusal_points_single_user_at_the_launcher_line(
    middleware, downstream, app_stub, monkeypatch
):
    """With no per-user mount the way back in is the ``Open:`` line, and it exists."""
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    sent = drive(middleware, http_scope(headers={"accept": "text/html"}, app=app_stub))
    body = _body_of(sent).decode("utf-8")

    assert "Open:" in body
    assert "?token=" in body
    assert "osprey users login-url" not in body


def test_html_refusal_points_multi_user_at_the_login_url_command(
    middleware, downstream, app_stub, monkeypatch
):
    """In the deployment shape the launcher prints no ``Open:`` line to re-open.

    A browser only reaches this page when the app's own gate is the outermost
    one — ``auth.method: none`` or a roster entry with ``login: false`` — and
    there the container's ``osprey web`` suppresses the announcement because
    nginx owns the way in. The verb that helps is ``osprey users login-url``.
    """
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    sent = drive(middleware, http_scope(headers={"accept": "text/html"}, app=app_stub))
    body = _body_of(sent).decode("utf-8")

    assert "osprey users login-url" in body
    assert "Open:" not in body
    # Still self-contained, and still quoting no credential.
    for reference in ("<link", "<script", "src=", "href="):
        assert reference not in body
    assert OPERATOR_SECRET not in body


def test_websocket_refusal_is_json_even_with_an_html_accept(middleware, downstream, app_stub):
    """A handshake refusal stays machine-readable; a websocket client parses it."""
    headers = {"accept": "text/html"}
    sent = drive(
        middleware, ws_scope(headers=headers, app=app_stub, extensions=WS_EXTENSIONS), incoming=[]
    )
    assert status_of(sent) == 401
    assert detail_of(sent) == "authentication required"


# --------------------------------------------------------------------------- #
# Shadowing session cookies
# --------------------------------------------------------------------------- #


def test_a_shadowing_cookie_does_not_hide_the_real_session(
    middleware, downstream, app_stub, credentials
):
    """A same-named cookie sent first must not lock the operator out.

    A page on a sibling host under the same registrable domain can set a
    ``Domain``-scoped cookie of this name, and the browser then sends it
    alongside the app's own — in an order the app does not control. Reading only
    the first value made that a permanent, undiagnosable denial of service.
    """
    live = credentials.create_session()
    headers = {"cookie": f"{COOKIE_NAME}=shadow-value; {COOKIE_NAME}={live}"}
    sent = drive(middleware, http_scope(headers=headers, app=app_stub))

    assert status_of(sent) == 200
    assert downstream.called


def test_the_candidate_list_is_bounded(middleware, downstream, app_stub, credentials):
    """Past the cap the real session is not reached — the work stays constant.

    Trying every copy would make the per-request cost the length of an
    attacker-written header. The cap is the trade: it covers the shadowing a
    browser can actually be made to do, and refuses to be turned into unbounded
    work.
    """
    live = credentials.create_session()
    decoys = "; ".join(
        f"{COOKIE_NAME}=decoy-{index}" for index in range(MAX_SESSION_COOKIE_CANDIDATES)
    )
    sent = drive(
        middleware, http_scope(headers={"cookie": f"{decoys}; {COOKIE_NAME}={live}"}, app=app_stub)
    )

    assert status_of(sent) == 401
    assert not downstream.called


def test_replayed_receive_delegates_past_the_buffered_body(app_stub):
    """A handler that keeps reading gets the real stream, not a truncated one.

    The gate buffers the body to decide the tier; anything it did not buffer —
    a trailing disconnect, a later chunk — has to keep arriving, or a streaming
    handler behind the gate would hang or see the connection as still open.
    """
    seen: list[dict[str, Any]] = []

    class GreedyApp:
        async def __call__(self, scope, receive, send):
            for _ in range(3):
                seen.append(await receive())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

    guard = WebAuthMiddleware(GreedyApp(), cookie_name=COOKIE_NAME)
    payload = json.dumps({"name": "scan"}).encode()
    sent = drive(guard, register_scope(app_stub), body_messages(payload))
    assert status_of(sent) == 200
    assert seen[0]["body"] == payload
    assert seen[-1] == {"type": "http.disconnect"}


# --------------------------------------------------------------------------- #
# Review remediation: structural guarantees
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [
        "/static/../api/config",
        "/design-system/../api/config",
        "/static/fonts/../../api/config",
        "/static/..",
    ],
)
def test_traversal_paths_are_not_exempt(path):
    """The gate's own matching must not be talked past by a ``..`` segment."""
    assert is_exempt_path(path) is False


def test_traversal_request_is_refused_by_the_gate(middleware, downstream, app_stub):
    """A raw scope whose path traverses out of a static mount gets a 401, not a pass."""
    sent = drive(middleware, http_scope("/static/../api/config", app=app_stub))
    assert status_of(sent) == 401
    assert not downstream.called


def test_double_dot_inside_a_segment_is_still_exempt():
    """Only a whole ``..`` segment is traversal; ``..woff2`` is a filename."""
    assert is_exempt_path("/static/fonts/..woff2") is True


def test_panel_token_websocket_is_refused_structurally(middleware, downstream, app_stub):
    """A panel-token WS to a panel-tier HTTP route is refused, not admitted.

    ``('GET', '/api/panels')`` is panel-tier, and the WS branch would synthesise
    ``GET`` — so without the unconditional refusal a keystroke-carrying socket
    could open on the weak credential.
    """
    scope = ws_scope(path="/api/panels", headers=bearer(), app=app_stub, extensions=WS_EXTENSIONS)
    sent = drive(middleware, scope, incoming=[])
    assert status_of(sent) == 401
    assert detail_of(sent) == "this route requires operator credentials"
    assert not downstream.called


def test_unavailable_credentials_yield_a_clean_503(downstream, monkeypatch):
    """A container shape that cannot populate refuses with 503, never a 500."""
    from osprey.interfaces import web_auth

    reset_web_credentials()
    monkeypatch.setenv(web_auth.BIND_HOST_ENV, "0.0.0.0")
    monkeypatch.delenv(web_auth.OPERATOR_SECRET_ENV, raising=False)
    try:
        guard = WebAuthMiddleware(downstream, cookie_name=COOKIE_NAME)
        # No app.state credentials, and the process holder cannot populate.
        sent = drive(guard, http_scope(headers={OPERATOR_SECRET_HEADER: "anything"}))
        assert status_of(sent) == 503
        assert detail_of(sent) == "authentication is not available; see the service log"
        assert not downstream.called
    finally:
        reset_web_credentials()


def test_unavailable_credentials_refuse_a_websocket_with_503(downstream, monkeypatch):
    from osprey.interfaces import web_auth

    reset_web_credentials()
    monkeypatch.setenv(web_auth.BIND_HOST_ENV, "0.0.0.0")
    monkeypatch.delenv(web_auth.OPERATOR_SECRET_ENV, raising=False)
    try:
        guard = WebAuthMiddleware(downstream, cookie_name=COOKIE_NAME)
        sent = drive(guard, ws_scope(app=None, extensions=WS_EXTENSIONS), incoming=[])
        assert status_of(sent) == 503
        assert not downstream.called
    finally:
        reset_web_credentials()
