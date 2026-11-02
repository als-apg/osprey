"""The history page's origin: the second nginx server, and the one origin behind it.

The server exists exactly when the history page does, and serves that page and
nothing else. It is pinned byte for byte in
``golden/history_origin/``, from its opening comment to the end of the file, for
the shapes its text can take: an origin derived from the deploy host over plain
HTTP, and one stated beside an external origin with TLS on. The terminal
servers above it are pinned by ``test_golden_render.py``; a deployment without
the page renders exactly those and nothing more.

Regenerate the goldens with
``python tests/deployment/web_terminals/test_history_origin_render.py`` after a
deliberate change to the server, and read the diff: every line of it is a
change to what a browser on that origin is allowed to do.
"""

from __future__ import annotations

import copy
import re
import socket
from pathlib import Path
from typing import Any

import pytest

from osprey.agent_record import credentials as record_credentials
from osprey.agent_record.config import history_app_port, history_page_deployed
from osprey.deployment.web_terminals.render import (
    NGINX_TEMPLATES_OUTPUT_DIR,
    TERMINAL_SECRET_HEADER,
    deployment_external_origin,
    history_origin,
    render_web_terminals,
    serves_history_origin,
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
from tests._renders import tier_render

_GOLDEN_DIR = Path(__file__).parent / "golden" / "history_origin"
_FACILITY_NAME = "Demo Light Source"

#: The first line of the history page's part of the rendered file.
_MARKER = "# The history page's origin.\n"


def _config(web: dict[str, Any] | None = None) -> dict:
    """A two-user web tier."""
    return {
        "project_name": "dls_controls",
        "facility": {"prefix": "dls"},
        "system": {"timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "deployed_services": [],
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": ["alice", "bob"],
                "default_tier": "assistant",
                "renders": {"assistant": tier_render("assistant", "dls_controls-assistant")},
                **(web or {}),
            }
        },
    }


def _stated_config() -> dict:
    """A load balancer in front: both origins stated, TLS on a rootless port."""
    return _config(
        web={
            "external_origin": "https://terminals.dls.example.org",
            "history_origin": "https://terminals.dls.example.org:8443",
            "history_port": 18443,
            "tls": {
                "enabled": True,
                "port": 8444,
                "cert": "/etc/nginx/certs/dls.crt",
                "key": "/etc/nginx/certs/dls.key",
            },
        },
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
    "history_derived.conf": _history_config,
    "history_stated.conf": _history_stated_config,
}


def _nginx(config: dict) -> str:
    return render_web_terminals(config, facility_name=_FACILITY_NAME)["nginx/nginx.conf"]


def _history_part(config: dict) -> str:
    conf = _nginx(config)
    assert conf.count(_MARKER) == 1
    return conf[conf.index(_MARKER) :]


def _location(part: str, opening: str) -> str:
    """The body of the one location that opens with ``opening``."""
    start = part.index(f"    {opening} {{\n")
    return part[start : part.index("\n    }\n", start)]


# --- goldens -----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_the_history_server_matches_its_golden_byte_for_byte(name: str) -> None:
    golden = _GOLDEN_DIR / name
    assert golden.is_file(), f"missing golden fixture: {golden}"
    assert _history_part(_VARIANTS[name]()) == golden.read_text(encoding="utf-8")


def test_only_the_history_goldens_remain() -> None:
    """A deployment without the page renders no history server, so it has no golden."""
    assert sorted(path.name for path in _GOLDEN_DIR.iterdir()) == sorted(_VARIANTS)


# --- when the server exists --------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_config(), id="token-sign-in"),
        pytest.param(_stated_config(), id="stated-origins"),
        pytest.param(_config(web={"auth": {"method": "none"}}), id="no-sign-in"),
    ],
)
def test_a_deployment_without_the_history_page_renders_no_history_server(config: dict) -> None:
    assert not history_page_deployed(config)
    assert not serves_history_origin(config)
    conf = _nginx(config)
    assert _MARKER not in conf
    assert "osprey_history" not in conf
    assert "\n    listen 10002;" not in conf


def test_the_server_exists_exactly_when_the_history_page_does() -> None:
    config = _history_config()
    web_off = _history_config()
    web_off["modules"]["web_terminals"]["enabled"] = False
    assert serves_history_origin(config)
    assert _MARKER in _nginx(config)
    assert not serves_history_origin(web_off)


def test_the_server_listens_on_the_port_the_preflight_probes() -> None:
    from osprey.deployment.host_ports import derive_history_origin_bindings

    config = _signed_in(_config(web={"history_port": 12345}))
    (binding,) = derive_history_origin_bindings(config)
    assert binding.host_port == 12345
    assert "    listen 12345;\n" in _history_part(config)


# --- throttling and logging ----------------------------------------------------


def test_nothing_throttles_and_the_logs_hold_no_query_string() -> None:
    part = _history_part(_history_config())
    directives = "\n".join(line for line in part.splitlines() if not line.lstrip().startswith("#"))
    assert "limit_req" not in directives
    assert "limit_conn" not in directives
    assert "access_log /dev/stdout osprey_history;" in directives
    assert "error_log /dev/stderr crit;" in directives
    log_format = directives[directives.index("log_format osprey_history") :]
    log_format = log_format[: log_format.index(";")]
    assert "$uri" in log_format
    for leaky in ("$request_uri", "$request ", "$args", "$query_string", "$http_referer"):
        assert leaky not in log_format


# --- every other path --------------------------------------------------------


def test_every_other_path_is_a_404() -> None:
    part = _history_part(_history_config())
    assert _location(part, "location /").rstrip().endswith("return 404;")
    assert "\n    root " not in part
    assert "try_files" not in part


def test_redirects_stay_relative() -> None:
    assert "\n    absolute_redirect off;\n" in _history_part(_history_config())
    assert "\n    absolute_redirect off;\n" in _history_part(_history_stated_config())


def test_a_render_opens_no_connection_and_writes_the_same_server_every_time(monkeypatch) -> None:
    """A roster change re-renders the web tier with every service down."""

    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the render opened a network connection")

    first = _history_part(_history_config())
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    grown = _history_config()
    grown["modules"]["web_terminals"]["users"] = ["alice", "bob", "dana"]
    assert _history_part(grown) == first


# --- the one origin ----------------------------------------------------------


def test_the_derived_origin_is_the_deploy_host_on_the_history_port() -> None:
    origin = history_origin(_config())
    assert (origin.origin, origin.scheme, origin.host) == (
        "http://dls-deploy.dls.example.org:10002",
        "http",
        "dls-deploy.dls.example.org",
    )
    moved = history_origin(_config(web={"history_port": 12345}))
    assert moved.origin == "http://dls-deploy.dls.example.org:12345"


def test_tls_makes_the_derived_origin_https_on_the_same_port() -> None:
    tls = {"enabled": True, "cert": "/c", "key": "/k"}
    assert history_origin(_config(web={"tls": tls})).origin == (
        "https://dls-deploy.dls.example.org:10002"
    )


def test_a_stated_origin_is_used_verbatim() -> None:
    config = _config(web={"history_origin": " https://History.example.org "})
    origin = history_origin(config)
    assert origin.origin == "https://History.example.org"
    assert origin.host == "history.example.org"


@pytest.mark.parametrize(
    "stated", ["", "history.example.org", "https://history.example.org/", "ftp://x", 8443]
)
def test_a_stated_origin_that_is_not_one_is_refused_by_name(stated: Any) -> None:
    with pytest.raises(ValueError, match=r"modules\.web_terminals\.history_origin"):
        history_origin(_config(web={"history_origin": stated}))


def test_an_external_origin_alone_cannot_name_the_history_origin() -> None:
    config = _signed_in(_config(web={"external_origin": "https://terminals.dls.example.org"}))
    config["modules"]["web_terminals"]["tls"] = {"enabled": True, "cert": "/c", "key": "/k"}
    with pytest.raises(ValueError, match="history_origin is not"):
        history_origin(config)
    with pytest.raises(ValueError, match="history_origin is not"):
        _nginx(config)


def test_no_deploy_host_and_no_stated_origin_is_refused_by_name() -> None:
    config = _config()
    config["deploy"] = {"host": "dls-deploy"}
    with pytest.raises(ValueError, match=r"deploy\.fqdn"):
        history_origin(config)


def test_a_deployment_without_the_page_is_never_asked_for_a_history_origin() -> None:
    config = _config(web={"external_origin": "https://terminals.dls.example.org"})
    assert _MARKER not in _nginx(config)


# --- the history page ----------------------------------------------------------

#: Every header the sidecar's history answer carries, in the sidecar's spelling.
_HISTORY_HEADERS = (
    verify_history.PERSON_HEADER,
    verify_history.SCOPE_HEADER,
    verify_history.VIEW_HEADER,
    verify_history.ROLE_HEADER,
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
    return _location(_history_part(config), "location /history/")


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
    assert history_page_deployed(config) is deployed
    artifacts = render_web_terminals(config, facility_name=_FACILITY_NAME)
    conf = artifacts["nginx/nginx.conf"]
    assert (_MARKER in conf) is deployed
    part = conf[conf.index(_MARKER) :] if deployed else ""
    for text in ("/history", "verify-history", "osprey_history", "/auth/logout"):
        assert (text in part) is deployed, text
    # The snippet and its include are emitted together or not at all: an
    # include with no file stops nginx at start.
    assert (_GATE_TEMPLATE in artifacts) is deployed
    assert (_GATE_INCLUDE in part) is deployed


def test_the_history_locations_sit_before_the_catch_alls() -> None:
    part = _history_part(_history_config())
    assert re.findall(r"^    location (.+) \{$", part, flags=re.MULTILINE) == [
        "/history/",
        "= /history",
        "= /verify-history",
        "@osprey_history_401",
        "= /history-403.html",
        "= /auth/logout",
        "/",
    ]


def test_the_page_is_proxied_with_its_prefix_stripped() -> None:
    page = _page(_history_config())
    port = history_app_port(_history_config())
    assert f"proxy_pass http://127.0.0.1:{port}/;" in page
    assert re.findall(r"proxy_pass (\S+);", page) == [f"http://127.0.0.1:{port}/"]
    bookmark = _location(_history_part(_history_config()), "location = /history")
    assert "return 301 /history/;" in bookmark


def test_the_page_is_authorized_by_the_sidecars_history_answer() -> None:
    config = _history_config()
    part = _history_part(config)
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
        f'proxy_set_header {TERMINAL_SECRET_HEADER} "${{{record_credentials.HISTORY_GATE_ENV}}}";'
    ]
    comments = [line for line in snippet.splitlines() if line.startswith("#")]
    assert comments and not any("$" in line for line in comments)
    # The variable passes the nginx container's envsubst filter.
    assert record_credentials.HISTORY_GATE_ENV.startswith("OSPREY_TERMINAL_SECRET_")
    # The included file is rendered nowhere else in the history server.
    assert _history_part(config).count("history-gate.conf") == 1


def test_the_page_may_not_be_framed_or_embedded_by_another_origin() -> None:
    part = _history_part(_history_config())
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
    part = _history_part(_history_config())
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
    part = _history_part(config)
    assert deployment_external_origin(config) == terminal_origin
    page = _page(config)
    assert "error_page 401 = @osprey_history_401;" in page
    handler = _location(part, "location @osprey_history_401")
    from urllib.parse import quote

    target = (
        f"{terminal_origin}{ENTRY_PATH}?next="
        f"{quote(history_origin(config).origin, safe='')}$osprey_history_next"
    )
    assert (
        f"if ($osprey_auth_wants_login_page) {{\n            return 302 {target};\n        }}"
    ) in handler
    assert handler.rstrip().endswith("return 401;")


def test_the_return_path_is_an_allowlisted_history_path_or_the_page_itself() -> None:
    part = _history_part(_history_config())
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
    part = _history_part(_history_config())
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
    part = _history_part(_history_config())
    body = _location(part, f"location = {LOGOUT_PATH}")
    assert re.search(rf"proxy_pass http://127\.0\.0\.1:\d+{re.escape(LOGOUT_PATH)};", body)
    for header in _IDENTITY_HEADERS:
        assert f'proxy_set_header {header} "";' in body
    assert "auth_request" not in body


def test_every_history_location_names_each_header_once() -> None:
    part = _history_part(_history_config())
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
    assert service["env_file"] == [record_credentials.HISTORY_GATE_FILE]
    assert service["labels"][record_credentials.DIGEST_LABEL] == (
        f"${{{record_credentials.digest_env_name('nginx')}:-}}"
    )
    # One label for the file, the record's.
    assert [label for label in service["labels"] if label.endswith(".cred.digest")] == [
        record_credentials.DIGEST_LABEL
    ]
    # Delivered by the file, never interpolated into the service definition.
    assert not any(record_credentials.HISTORY_GATE_ENV in entry for entry in service["environment"])


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_config(), id="token"),
        pytest.param(_config(web={"auth": {"method": "none"}}), id="auth-none"),
    ],
)
def test_without_the_page_nginx_names_no_gate_file(config: dict) -> None:
    service = _nginx_service(config)
    assert "env_file" not in service
    assert record_credentials.DIGEST_LABEL not in service["labels"]


def _regenerate() -> None:
    """Overwrite this module's goldens from today's renderer."""
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in _VARIANTS.items():
        path = _GOLDEN_DIR / name
        path.write_text(_history_part(copy.deepcopy(build())), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
