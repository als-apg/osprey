"""The readers' origin: the second nginx server, and the one origin behind it.

The rendered server is pinned byte for byte in ``golden/reader_origin/``, from
its opening comment to the end of the file, for the shapes its text can take:
an origin derived from the deploy host over plain HTTP, and one stated beside
an external origin with TLS on — each without the history page (token sign-in)
and with it (password sign-in). The terminal servers above it are pinned
by ``test_golden_render.py``; a deployment without the store renders exactly
those and nothing more, which the goldens there already hold.

Regenerate the goldens with
``python tests/deployment/web_terminals/test_reader_origin_render.py`` after a
deliberate change to the server, and read the diff: every line of it is a
change to what a reader's browser is allowed to do.
"""

from __future__ import annotations

import copy
import re
import socket
from pathlib import Path
from typing import Any

import pytest

from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_credentials
from osprey.deployment.web_terminals.render import (
    NGINX_TEMPLATES_OUTPUT_DIR,
    TERMINAL_SECRET_HEADER,
    deployment_external_origin,
    reader_origin,
    render_web_terminals,
    serves_reader_origin,
)
from osprey.services.auth_sidecar.identity_headers import (
    ACCOUNT_HEADER,
    ROLE_HEADER,
    ROLE_SOURCE_HEADER,
    SUBJECT_HEADER,
)
from osprey.services.auth_sidecar.routes import history as verify_history
from osprey.services.auth_sidecar.routes.entry import ENTRY_PATH
from osprey.services.auth_sidecar.routes.logout import LOGOUT_PATH

_GOLDEN_DIR = Path(__file__).parent / "golden" / "reader_origin"
_FACILITY_NAME = "Demo Light Source"

#: The first line of the readers' part of the rendered file.
_MARKER = "# The readers' origin.\n"

_HASH = tc.PLAY_SCRIPT_SHA256["26.8.18.2"]


def _config(web: dict[str, Any] | None = None, store: dict[str, Any] | None = None) -> dict:
    """A two-user web tier in front of a deployed store."""
    return {
        "project_name": "dls_controls",
        "facility": {"prefix": "dls"},
        "system": {"timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "deployed_services": ["telemetry_store", "telemetry_gateway"],
        "services": {"telemetry_store": {"readers": [{"name": "carol"}], **(store or {})}},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": ["alice", "bob"],
                "default_persona": "assistant",
                "personas": {"assistant": {"project": "dls_controls-assistant"}},
                **(web or {}),
            }
        },
    }


def _stated_config() -> dict:
    """A load balancer in front: both origins stated, TLS on a rootless port."""
    return _config(
        web={
            "external_origin": "https://terminals.dls.example.org",
            "reader_origin": "https://terminals.dls.example.org:8443",
            "reader_port": 18443,
            "tls": {
                "enabled": True,
                "port": 8444,
                "cert": "/etc/nginx/certs/dls.crt",
                "key": "/etc/nginx/certs/dls.key",
            },
        },
        store={"port": 19000, "limits": {"seconds": 90}},
    )


def _signed_in(config: dict) -> dict:
    """``config`` with password sign-in, which is what deploys the history page."""
    web = config["modules"]["web_terminals"]
    web["auth"] = {"method": "password"}
    if not web.get("tls"):
        web["auth"]["allow_insecure_http"] = True
    return config


def _history_config() -> dict:
    return _signed_in(_config())


def _history_stated_config() -> dict:
    return _signed_in(_stated_config())


_VARIANTS = {
    "derived.conf": _config,
    "stated.conf": _stated_config,
    "history_derived.conf": _history_config,
    "history_stated.conf": _history_stated_config,
}


def _nginx(config: dict) -> str:
    return render_web_terminals(config, facility_name=_FACILITY_NAME)["nginx/nginx.conf"]


def _reader_part(config: dict) -> str:
    conf = _nginx(config)
    assert conf.count(_MARKER) == 1
    return conf[conf.index(_MARKER) :]


def _location(part: str, opening: str) -> str:
    """The body of the one location that opens with ``opening``."""
    start = part.index(f"    {opening} {{\n")
    return part[start : part.index("\n    }\n", start)]


# --- goldens -----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_the_readers_server_matches_its_golden_byte_for_byte(name: str) -> None:
    golden = _GOLDEN_DIR / name
    assert golden.is_file(), f"missing golden fixture: {golden}"
    assert _reader_part(_VARIANTS[name]()) == golden.read_text(encoding="utf-8")


def test_the_goldens_hold_what_each_shape_is_for() -> None:
    """A stale or truncated golden cannot pass by matching an equally wrong render."""
    derived = (_GOLDEN_DIR / "derived.conf").read_text(encoding="utf-8")
    stated = (_GOLDEN_DIR / "stated.conf").read_text(encoding="utf-8")

    assert "    listen 10002;\n    listen [::]:10002;\n" in derived
    assert "proxy_pass http://127.0.0.1:10804/?$osprey_reader_query;" in derived
    assert (
        "return 302 /telemetry/play?url=http%3A%2F%2Fdls-deploy.dls.example.org"
        "%3A10002%2Ftelemetry%2F&$args;"
    ) in derived
    assert "proxy_read_timeout 40s;" in derived

    assert "    listen 18443 ssl;\n    listen [::]:18443 ssl;\n" in stated
    assert "ssl_certificate /etc/nginx/certs/dls.crt;" in stated
    assert "proxy_pass http://127.0.0.1:19000/?$osprey_reader_query;" in stated
    assert (
        "return 302 /telemetry/play?url=https%3A%2F%2Fterminals.dls.example.org"
        "%3A8443%2Ftelemetry%2F&$args;"
    ) in stated
    assert "proxy_read_timeout 100s;" in stated
    for golden in (derived, stated):
        assert golden.rstrip().endswith("}")
        assert f"script-src 'sha256-{_HASH}' 'wasm-unsafe-eval'" in golden


# --- when the server exists --------------------------------------------------


def test_no_store_means_no_readers_server() -> None:
    config = _config()
    config["deployed_services"] = ["telemetry_gateway"]
    assert not serves_reader_origin(config)
    conf = _nginx(config)
    assert _MARKER not in conf
    assert "osprey_reader" not in conf
    assert "10002" not in conf


def test_the_server_is_gated_exactly_as_the_port_preflight_is() -> None:
    """The preflight probes the readers' port for the configs that listen on it."""
    from osprey.deployment.host_ports import derive_reader_origin_bindings

    with_store = _config()
    without_store = _config()
    without_store["deployed_services"] = []
    web_off = _config(web={"enabled": False})
    for config in (with_store, without_store, web_off):
        assert bool(derive_reader_origin_bindings(config)) is serves_reader_origin(config)
    assert serves_reader_origin(with_store)
    assert not serves_reader_origin(web_off)


def test_the_server_listens_on_the_port_the_preflight_probes() -> None:
    from osprey.deployment.host_ports import derive_reader_origin_bindings

    config = _config(web={"reader_port": 12345})
    (binding,) = derive_reader_origin_bindings(config)
    assert binding.host_port == 12345
    assert "    listen 12345;\n" in _reader_part(config)


# --- the query endpoint ------------------------------------------------------


def test_only_four_request_headers_go_upstream_and_never_the_body_length() -> None:
    body = _location(_reader_part(_config()), "location = /telemetry/")
    assert "proxy_pass_request_headers off;" in body
    sent = re.findall(r"^\s*proxy_set_header (\S+) (\S+);$", body, flags=re.MULTILINE)
    assert sent == [
        ("Authorization", "$http_authorization"),
        ("Content-Type", "$http_content_type"),
        ("Content-Encoding", "$http_content_encoding"),
        ("Accept-Encoding", "$http_accept_encoding"),
    ]
    assert "proxy_set_header Content-Length" not in _reader_part(_config())


def test_the_wait_parameter_is_appended_and_the_query_string_is_never_edited() -> None:
    part = _reader_part(_config())
    assert (
        "map $args $osprey_reader_query {\n"
        '    "" "http_wait_end_of_query=1";\n'
        '    default "$args&http_wait_end_of_query=1";\n'
        "}\n"
    ) in part
    # One mechanism: nothing rewrites or filters what the reader sent.
    directives = "\n".join(line for line in part.splitlines() if not line.lstrip().startswith("#"))
    assert "rewrite" not in directives
    assert "$arg_wait_end_of_query" not in directives
    assert directives.count("http_wait_end_of_query") == 2


def _anonymous(authorization: str, user: str, password: str) -> bool:
    """The 401 map, evaluated as nginx evaluates it: an exact match on the key."""
    part = _reader_part(_config())
    match = re.search(
        r'^map "(?P<key>[^"]+)" \$osprey_reader_anonymous \{\n'
        r'    "" 1;\n'
        r"    default 0;\n"
        r"\}$",
        part,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert match.group("key") == "$http_authorization$arg_user$arg_password"
    return authorization + user + password == ""


@pytest.mark.parametrize(
    ("authorization", "user", "password", "refused"),
    [
        pytest.param("Basic Y2Fyb2w6a2V5", "", "", False, id="basic"),
        pytest.param("never", "carol", "key", False, id="never-with-parameters"),
        pytest.param("", "carol", "key", False, id="parameters-only"),
        pytest.param("", "", "", True, id="nothing"),
    ],
)
def test_only_a_request_naming_no_account_is_answered_401_by_nginx(
    authorization: str, user: str, password: str, refused: bool
) -> None:
    assert _anonymous(authorization, user, password) is refused


def test_the_401_carries_the_basic_challenge_and_nothing_else_does() -> None:
    part = _reader_part(_config())
    endpoint = _location(part, "location = /telemetry/")
    assert "if ($osprey_reader_anonymous) {\n            return 401;\n        }" in endpoint
    assert "error_page 401 @osprey_reader_401;" in endpoint
    challenge = _location(part, "location @osprey_reader_401")
    assert "add_header WWW-Authenticate 'Basic realm=\"telemetry\"' always;" in challenge
    assert challenge.rstrip().endswith("return 401;")
    assert part.count("WWW-Authenticate") == 1


@pytest.mark.parametrize("opening", ["location = /telemetry/", "location @osprey_reader_401"])
def test_no_result_can_run_as_a_document(opening: str) -> None:
    body = _location(_reader_part(_config()), opening)
    assert "add_header X-Content-Type-Options nosniff always;" in body
    assert "add_header Content-Security-Policy \"default-src 'none'; sandbox\" always;" in body


def test_the_stores_cors_grant_is_not_passed_on() -> None:
    part = _reader_part(_config())
    for opening in ("location = /telemetry/", "location = /telemetry/play"):
        assert "proxy_hide_header Access-Control-Allow-Origin;" in _location(part, opening)


#: Response headers a reader could name through `http_response_headers` that
#: act on the browser or the host beyond the answer itself.
_BROWSER_ACTING_HEADERS = (
    "Set-Cookie",
    "Refresh",
    "Location",
    "Link",
    "Clear-Site-Data",
    "Strict-Transport-Security",
    "Alt-Svc",
    "NEL",
    "Report-To",
    "Reporting-Endpoints",
    "Content-Security-Policy",
    "Content-Security-Policy-Report-Only",
    "Access-Control-Allow-Credentials",
    "Access-Control-Allow-Headers",
    "Access-Control-Allow-Methods",
    "Access-Control-Expose-Headers",
    "Access-Control-Allow-Private-Network",
)


def test_no_header_a_query_names_acts_on_the_browser_or_on_nginx() -> None:
    body = _location(_reader_part(_config()), "location = /telemetry/")
    for header in _BROWSER_ACTING_HEADERS:
        assert f"proxy_hide_header {header};" in body, header
    # Download names its file and its type; both still reach the browser.
    assert "proxy_hide_header Content-Disposition;" not in body
    assert "proxy_hide_header Content-Type;" not in body
    assert (
        "proxy_ignore_headers X-Accel-Redirect X-Accel-Expires X-Accel-Limit-Rate "
        "X-Accel-Buffering X-Accel-Charset;" in body
    )


def test_nothing_throttles_and_the_logs_hold_no_query_string() -> None:
    part = _reader_part(_config())
    directives = "\n".join(line for line in part.splitlines() if not line.lstrip().startswith("#"))
    assert "limit_req" not in directives
    assert "limit_conn" not in directives
    assert "access_log /dev/stdout osprey_reader;" in directives
    assert "error_log /dev/stderr crit;" in directives
    log_format = directives[directives.index("log_format osprey_reader") :]
    log_format = log_format[: log_format.index(";")]
    assert "$uri" in log_format
    for leaky in ("$request_uri", "$request ", "$args", "$query_string", "$http_referer"):
        assert leaky not in log_format


# --- every other path --------------------------------------------------------


def test_every_other_path_is_a_404_and_the_root_answers_the_pages_ping() -> None:
    part = _reader_part(_config())
    assert _location(part, "location /telemetry/").rstrip().endswith("return 404;")
    assert _location(part, "location /").rstrip().endswith("return 404;")
    root = _location(part, "location = /")
    assert "if ($request_method = OPTIONS) {\n            return 204;\n        }" in root
    assert root.rstrip().endswith("return 404;")
    assert "\n    root " not in part
    assert "try_files" not in part
    assert re.findall(r"^    location (.+) \{$", part, flags=re.MULTILINE) == [
        "= /",
        "= /telemetry/",
        "@osprey_reader_401",
        "= /telemetry/play",
        "/telemetry/",
        "/",
    ]


def test_redirects_stay_relative() -> None:
    assert "\n    absolute_redirect off;\n" in _reader_part(_config())
    assert "\n    absolute_redirect off;\n" in _reader_part(_stated_config())


# --- the query page ----------------------------------------------------------


def test_the_query_page_runs_one_script_named_by_the_images_hash() -> None:
    page = _location(_reader_part(_config()), "location = /telemetry/play")
    assert (
        "add_header Content-Security-Policy \"default-src 'none'; "
        f"script-src 'sha256-{_HASH}' 'wasm-unsafe-eval'; style-src 'unsafe-inline'; "
        "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'\" always;"
    ) in page
    policy = page[page.index("script-src") : page.index("style-src")]
    assert "'self'" not in policy
    assert "'unsafe-inline'" not in policy


def test_the_query_page_is_fetched_without_the_readers_parameters() -> None:
    page = _location(_reader_part(_config()), "location = /telemetry/play")
    assert "if ($request_method != GET) {\n            return 405;\n        }" in page
    assert "set $osprey_reader_play /play;" in page
    assert "proxy_pass http://127.0.0.1:10804$osprey_reader_play;" in page
    assert "proxy_pass_request_headers off;" in page
    assert "$osprey_reader_anonymous" not in page
    directives = [line for line in page.splitlines() if not line.lstrip().startswith("#")]
    assert not any("$args" in line and "return 302" not in line for line in directives)


@pytest.mark.parametrize(
    ("config", "origin"),
    [
        pytest.param(_config(), "http://dls-deploy.dls.example.org:10002", id="derived"),
        pytest.param(_stated_config(), "https://terminals.dls.example.org:8443", id="stated"),
    ],
)
def test_a_bare_bookmark_is_healed_from_the_resolved_origin(config: dict, origin: str) -> None:
    """The redirect names the origin the printed link names, never the request's own."""
    page = _location(_reader_part(config), "location = /telemetry/play")
    link = tc.reader_play_url(reader_origin(config).origin)
    assert link.startswith(f"{origin}/telemetry/play?url=")
    relative = link[len(origin) :]
    assert f'if ($arg_url = "") {{\n            return 302 {relative}&$args;\n        }}' in page
    for forbidden in ("$scheme", "$http_host", "$host", "$is_args", "$server_port"):
        assert forbidden not in page


def test_an_image_nobody_has_read_serves_no_query_page() -> None:
    config = _config(store={"image": "clickhouse/clickhouse-server:99.1.1.1"})
    part = _reader_part(config)
    page = _location(part, "location = /telemetry/play")
    assert page.rstrip().endswith("return 404;")
    assert "PLAY_SCRIPT_SHA256" in page
    assert "proxy_pass" not in page
    assert "script-src" not in part
    # The query endpoint does not depend on the page.
    assert "proxy_pass http://127.0.0.1:10804/?$osprey_reader_query;" in part


def test_a_render_asks_no_store_and_writes_the_same_policy_every_time(monkeypatch) -> None:
    """A roster change re-renders the web tier with the store down, and must not move the hash."""

    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the render opened a network connection")

    first = _reader_part(_config())
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    grown = _config()
    grown["modules"]["web_terminals"]["users"] = ["alice", "bob", "dana"]
    again = _reader_part(grown)
    assert again == first
    assert f"'sha256-{_HASH}'" in again


# --- the one origin ----------------------------------------------------------


def test_the_derived_origin_is_the_deploy_host_on_the_readers_port() -> None:
    origin = reader_origin(_config())
    assert (origin.origin, origin.scheme, origin.host) == (
        "http://dls-deploy.dls.example.org:10002",
        "http",
        "dls-deploy.dls.example.org",
    )
    moved = reader_origin(_config(web={"reader_port": 12345}))
    assert moved.origin == "http://dls-deploy.dls.example.org:12345"


def test_tls_makes_the_derived_origin_https_on_the_same_port() -> None:
    tls = {"enabled": True, "cert": "/c", "key": "/k"}
    assert reader_origin(_config(web={"tls": tls})).origin == (
        "https://dls-deploy.dls.example.org:10002"
    )


def test_a_stated_origin_is_used_verbatim() -> None:
    config = _config(web={"reader_origin": " https://Readers.example.org "})
    origin = reader_origin(config)
    assert origin.origin == "https://Readers.example.org"
    assert origin.host == "readers.example.org"


@pytest.mark.parametrize(
    "stated", ["", "readers.example.org", "https://readers.example.org/", "ftp://x", 8443]
)
def test_a_stated_origin_that_is_not_one_is_refused_by_name(stated: Any) -> None:
    with pytest.raises(ValueError, match=r"modules\.web_terminals\.reader_origin"):
        reader_origin(_config(web={"reader_origin": stated}))


def test_an_external_origin_alone_cannot_name_the_readers_origin() -> None:
    config = _config(web={"external_origin": "https://terminals.dls.example.org"})
    with pytest.raises(ValueError, match="reader_origin is not"):
        reader_origin(config)
    with pytest.raises(ValueError, match="reader_origin is not"):
        _nginx(config)


def test_no_deploy_host_and_no_stated_origin_is_refused_by_name() -> None:
    config = _config()
    config["deploy"] = {"host": "dls-deploy"}
    with pytest.raises(ValueError, match=r"deploy\.fqdn"):
        reader_origin(config)


def test_a_deployment_without_the_store_is_never_asked_for_a_readers_origin() -> None:
    config = _config(web={"external_origin": "https://terminals.dls.example.org"})
    config["deployed_services"] = []
    assert _MARKER not in _nginx(config)


# --- the history page ----------------------------------------------------------

#: Every header the sidecar's history answer carries, in the sidecar's spelling.
_HISTORY_HEADERS = (
    verify_history.PERSON_HEADER,
    verify_history.SCOPE_HEADER,
    verify_history.VIEW_HEADER,
    verify_history.ROLE_HEADER,
    verify_history.TABLES_HEADER,
    verify_history.ENTRY_HEADER,
    verify_history.CHOICES_HEADER,
)
_IDENTITY_HEADERS = (ACCOUNT_HEADER, SUBJECT_HEADER, ROLE_HEADER, ROLE_SOURCE_HEADER)
_GATE_TEMPLATE = f"{NGINX_TEMPLATES_OUTPUT_DIR}/history-gate.conf.template"
_GATE_INCLUDE = "include /etc/nginx/osprey/history-gate.conf;"


def _var(header: str) -> str:
    """The variable a history header is held in between the subrequest and the proxy."""
    return "$osprey_history_" + header.removeprefix("X-Osprey-History-").lower()


def _upstream(header: str) -> str:
    """nginx's spelling of a subrequest's response header."""
    return "$upstream_http_" + header.lower().replace("-", "_")


def _directive_lines(body: str) -> list[str]:
    return [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _sent(body: str) -> list[tuple[str, str]]:
    return re.findall(r"^\s*proxy_set_header (\S+) (.+);$", body, flags=re.MULTILINE)


def _page(config: dict) -> str:
    return _location(_reader_part(config), "location /history/")


def test_the_history_goldens_hold_what_each_shape_is_for() -> None:
    """The proxy target, the headers, the policy and the sign-in origin, in both shapes."""
    derived = (_GOLDEN_DIR / "history_derived.conf").read_text(encoding="utf-8")
    stated = (_GOLDEN_DIR / "history_stated.conf").read_text(encoding="utf-8")
    for golden in (derived, stated):
        assert "        proxy_pass http://127.0.0.1:10003/;\n" in golden
        assert "        auth_request /verify-history;\n" in golden
        for header in _HISTORY_HEADERS:
            assert f"auth_request_set {_var(header)} {_upstream(header)};" in golden
            assert f"proxy_set_header {header} {_var(header)};" in golden
        assert "proxy_set_header X-Original-URI $request_uri;" in golden
        assert 'proxy_set_header Cookie "";' in golden
        assert _GATE_INCLUDE in golden
        assert (
            "add_header Content-Security-Policy \"default-src 'self'; frame-ancestors 'none'\""
            " always;"
        ) in golden
        assert "add_header Cross-Origin-Resource-Policy same-origin always;" in golden
        assert "error_page 403 /history-403.html;" in golden
        assert "location = /auth/logout {" in golden
    # The terminal origin a signed-out visitor is sent to: derived from the
    # deploy host and the terminals' port, or the stated external origin.
    assert (
        "return 302 http://dls-deploy.dls.example.org:10000/auth/enter?next="
        "http%3A%2F%2Fdls-deploy.dls.example.org%3A10002$osprey_history_next;"
    ) in derived
    assert (
        "return 302 https://terminals.dls.example.org/auth/enter?next="
        "https%3A%2F%2Fterminals.dls.example.org%3A8443$osprey_history_next;"
    ) in stated


@pytest.mark.parametrize(
    ("config", "deployed"),
    [
        pytest.param(_config(), False, id="token"),
        pytest.param(_signed_in(_config(web={"auth": {}})), True, id="password"),
        pytest.param(_config(web={"auth": {"method": "none"}}), False, id="none"),
    ],
)
def test_the_page_exists_exactly_when_the_history_app_is_deployed(
    config: dict, deployed: bool
) -> None:
    assert tc.history_page_deployed(config) is deployed
    artifacts = render_web_terminals(config, facility_name=_FACILITY_NAME)
    part = artifacts["nginx/nginx.conf"][artifacts["nginx/nginx.conf"].index(_MARKER) :]
    for text in ("/history", "verify-history", "osprey_history", "/auth/logout"):
        assert (text in part) is deployed, text
    # The snippet and its include are emitted together or not at all: an
    # include with no file stops nginx at start.
    assert (_GATE_TEMPLATE in artifacts) is deployed
    assert (_GATE_INCLUDE in part) is deployed


def test_the_history_locations_sit_before_the_catch_alls() -> None:
    part = _reader_part(_history_config())
    assert re.findall(r"^    location (.+) \{$", part, flags=re.MULTILINE) == [
        "= /",
        "= /telemetry/",
        "@osprey_reader_401",
        "= /telemetry/play",
        "/history/",
        "= /history",
        "= /verify-history",
        "@osprey_history_401",
        "= /history-403.html",
        "= /auth/logout",
        "/telemetry/",
        "/",
    ]


def test_the_page_is_proxied_with_its_prefix_stripped() -> None:
    page = _page(_history_config())
    port = tc.history_port(_history_config())
    assert f"proxy_pass http://127.0.0.1:{port}/;" in page
    assert re.findall(r"proxy_pass (\S+);", page) == [f"http://127.0.0.1:{port}/"]
    bookmark = _location(_reader_part(_history_config()), "location = /history")
    assert "return 301 /history/;" in bookmark


def test_the_page_is_authorized_by_the_sidecars_history_answer() -> None:
    config = _history_config()
    part = _reader_part(config)
    page = _page(config)
    assert f"auth_request {verify_history.VERIFY_HISTORY_PATH};" in page
    target = _location(part, f"location = {verify_history.VERIFY_HISTORY_PATH}")
    lines = _directive_lines(target)
    assert lines[1] == "internal;"
    assert any(
        re.fullmatch(
            rf"proxy_pass http://127\.0\.0\.1:\d+{re.escape(verify_history.VERIFY_HISTORY_PATH)};",
            line,
        )
        for line in lines
    ), lines
    # The selector travels in the original URI; the subrequest's own carries no
    # query string.
    assert "proxy_set_header X-Original-URI $request_uri;" in target
    assert "proxy_pass_request_body off;" in target
    assert 'proxy_set_header Content-Length "";' in target
    for header in _IDENTITY_HEADERS:
        assert f'proxy_set_header {header} "";' in target
    # The sidecar reads the session cookie, so the subrequest keeps it.
    assert "Cookie" not in "\n".join(lines)


def test_every_history_header_is_lifted_and_forwarded_once() -> None:
    page = _page(_history_config())
    sent = dict(_sent(page))
    for header in _HISTORY_HEADERS:
        assert page.count(f"auth_request_set {_var(header)} {_upstream(header)};") == 1
        assert sent[header] == _var(header)


def test_nothing_the_client_sent_as_a_credential_or_an_identity_reaches_the_app() -> None:
    page = _page(_history_config())
    names = [name for name, _value in _sent(page)]
    assert len(names) == len(set(names)), "a header written twice in one location"
    assert "proxy_pass_request_headers off;" in page
    sent = dict(_sent(page))
    for header in (*_IDENTITY_HEADERS, "Authorization", "Cookie"):
        assert sent[header] == '""', header
    # Only the request headers a page load needs are passed on, by name.
    passed = {name for name, value in sent.items() if value.startswith("$http_")}
    assert all(not name.lower().startswith("x-osprey") for name in passed)
    assert "Cookie" not in passed and "Authorization" not in passed


def test_the_gate_secret_comes_from_the_envsubst_snippet_and_nowhere_else() -> None:
    config = _history_config()
    artifacts = render_web_terminals(config, facility_name=_FACILITY_NAME)
    page = _page(config)
    assert _GATE_INCLUDE in page
    assert TERMINAL_SECRET_HEADER not in "\n".join(_directive_lines(page))
    snippet = artifacts[_GATE_TEMPLATE]
    directives = _directive_lines(snippet)
    assert directives == [
        f'proxy_set_header {TERMINAL_SECRET_HEADER} "${{{telemetry_credentials.HISTORY_GATE_ENV}}}";'
    ]
    comments = [line for line in snippet.splitlines() if line.startswith("#")]
    assert comments and not any("$" in line for line in comments)
    # The variable passes the nginx container's envsubst filter.
    assert telemetry_credentials.HISTORY_GATE_ENV.startswith("OSPREY_TERMINAL_SECRET_")
    # The included file is rendered nowhere else in the readers' server.
    assert _reader_part(config).count("history-gate.conf") == 1


def test_the_page_may_not_be_framed_or_embedded_by_another_origin() -> None:
    part = _reader_part(_history_config())
    page = _page(_history_config())
    assert (
        "add_header Content-Security-Policy \"default-src 'self'; frame-ancestors 'none'\" always;"
    ) in page
    assert "add_header Cross-Origin-Resource-Policy same-origin always;" in page
    assert (
        'map "$http_sec_fetch_site:$http_sec_fetch_mode:$http_sec_fetch_dest" '
        "$osprey_history_embedded {\n"
        "    default 0;\n"
        '    "~^(same-site|cross-site):navigate:document\\z" 0;\n'
        '    "~^(same-site|cross-site):" 1;\n'
        "}\n"
    ) in part
    # Refused before the sign-in is asked about, so an embed spends nothing.
    guard = "if ($osprey_history_embedded) {\n            return 403;\n        }"
    assert guard in page
    assert page.index(guard) < page.index("auth_request ")


@pytest.mark.parametrize(
    ("site", "mode", "dest", "refused"),
    [
        pytest.param("same-site", "navigate", "document", False, id="link-from-the-terminals"),
        pytest.param("cross-site", "navigate", "document", False, id="sign-in-round-trip"),
        pytest.param("same-site", "navigate", "iframe", True, id="iframe"),
        pytest.param("same-site", "no-cors", "image", True, id="img"),
        pytest.param("cross-site", "cors", "empty", True, id="fetch"),
        pytest.param("same-origin", "no-cors", "script", False, id="own-asset"),
        pytest.param("none", "navigate", "document", False, id="typed-url"),
        pytest.param("", "", "", False, id="no-fetch-metadata"),
    ],
)
def test_the_embed_map_refuses_only_cross_origin_non_navigations(
    site: str, mode: str, dest: str, refused: bool
) -> None:
    """The map, evaluated as nginx evaluates it: regexes in order, first match wins."""
    part = _reader_part(_history_config())
    body = part[part.index("$osprey_history_embedded {\n") :]
    body = body[: body.index("\n}\n")]
    value = 0
    for pattern, result in re.findall(r'^    "~(.+)" (\d);$', body, flags=re.MULTILINE):
        if re.search(pattern.replace("\\z", r"\Z"), f"{site}:{mode}:{dest}"):
            value = int(result)
            break
    assert bool(value) is refused


@pytest.mark.parametrize(
    ("config", "terminal_origin"),
    [
        pytest.param(_history_config(), "http://dls-deploy.dls.example.org:10000", id="derived"),
        pytest.param(_history_stated_config(), "https://terminals.dls.example.org", id="stated"),
    ],
)
def test_a_signed_out_navigation_is_sent_to_the_terminals_sign_in(
    config: dict, terminal_origin: str
) -> None:
    part = _reader_part(config)
    assert deployment_external_origin(config) == terminal_origin
    page = _page(config)
    assert "error_page 401 = @osprey_history_401;" in page
    handler = _location(part, "location @osprey_history_401")
    from urllib.parse import quote

    target = (
        f"{terminal_origin}{ENTRY_PATH}?next="
        f"{quote(reader_origin(config).origin, safe='')}$osprey_history_next"
    )
    assert (
        f"if ($osprey_auth_wants_login_page) {{\n            return 302 {target};\n        }}"
    ) in handler
    assert handler.rstrip().endswith("return 401;")


def test_the_return_path_is_an_allowlisted_history_path_or_the_page_itself() -> None:
    part = _reader_part(_history_config())
    assert (
        "map $uri $osprey_history_next {\n"
        '    default "/history/";\n'
        '    "~^/history/[A-Za-z0-9._~/-]*\\z" $uri;\n'
        "}\n"
    ) in part
    pattern = r"^/history/[A-Za-z0-9._~/-]*\Z"
    for good in ("/history/", "/history/session/abc-123"):
        assert re.match(pattern, good)
    for bad in ("/history/\r\nSet-Cookie:x", "/history/a&b", "/history/a?b", "/history/a\n"):
        assert not re.match(pattern, bad)


def test_a_sign_in_naming_no_roster_person_gets_the_403_page() -> None:
    part = _reader_part(_history_config())
    page = _page(_history_config())
    # No `=`: the page is served under the sidecar's own 403.
    assert "error_page 403 /history-403.html;" in page
    body = _location(part, "location = /history-403.html")
    lines = _directive_lines(body)
    assert lines[1] == "internal;"
    assert "default_type text/html;" in lines
    assert "names no roster person" in body
    returned = next(line for line in lines if line.startswith("return "))
    assert returned.startswith("return 403 '")
    assert "$" not in returned
    assert (
        "add_header Content-Security-Policy \"default-src 'none'; frame-ancestors 'none'\"" in body
    )


def test_sign_out_is_proxied_to_the_sidecar_and_names_no_identity() -> None:
    part = _reader_part(_history_config())
    body = _location(part, f"location = {LOGOUT_PATH}")
    assert re.search(rf"proxy_pass http://127\.0\.0\.1:\d+{re.escape(LOGOUT_PATH)};", body)
    for header in _IDENTITY_HEADERS:
        assert f'proxy_set_header {header} "";' in body
    assert "auth_request" not in body


def test_every_history_location_names_each_header_once() -> None:
    part = _reader_part(_history_config())
    for opening in (
        "location /history/",
        f"location = {verify_history.VERIFY_HISTORY_PATH}",
        f"location = {LOGOUT_PATH}",
    ):
        names = [name for name, _value in _sent(_location(part, opening))]
        assert len(names) == len(set(names)), opening
        for header in _IDENTITY_HEADERS:
            assert names.count(header) == 1, (opening, header)


# --- the nginx service carries the gate secret ----------------------------------


def _nginx_service(config: dict) -> dict:
    import yaml

    compose = render_web_terminals(config, facility_name=_FACILITY_NAME)
    return yaml.safe_load(compose["docker-compose.web.yml"])["services"]["nginx"]


def test_nginx_reads_the_gate_file_and_is_recreated_when_it_changes() -> None:
    service = _nginx_service(_history_config())
    assert service["env_file"] == [telemetry_credentials.HISTORY_GATE_FILE]
    assert service["labels"][telemetry_credentials.DIGEST_LABEL] == (
        f"${{{telemetry_credentials.digest_env_name('nginx')}:-}}"
    )
    # Delivered by the file, never interpolated into the service definition.
    assert not any(
        telemetry_credentials.HISTORY_GATE_ENV in entry for entry in service["environment"]
    )


@pytest.mark.parametrize(
    "config",
    [pytest.param(_config(), id="token"), pytest.param(_history_config(), id="no-store")],
)
def test_without_the_page_nginx_names_no_gate_file(config: dict) -> None:
    config = copy.deepcopy(config)
    if config["modules"]["web_terminals"].get("auth"):
        config["deployed_services"] = []
    service = _nginx_service(config)
    assert "env_file" not in service
    assert telemetry_credentials.DIGEST_LABEL not in service["labels"]


def _regenerate() -> None:
    """Overwrite this module's goldens from today's renderer."""
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in _VARIANTS.items():
        path = _GOLDEN_DIR / name
        path.write_text(_reader_part(copy.deepcopy(build())), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
