"""Regression guard for the auth/TLS seams in web-terminal nginx rendering.

Deploy-lifecycle work (up/down/reconcile/status, etc.) around
:func:`osprey.deployment.web_terminals.render.render_web_terminals` must not move
the two security seams this deployment's safety rests on. Both remain entirely
opt-in — a config that sets neither stanza renders the pre-auth plain-http
posture byte for byte — and both fail closed rather than silently degrade:

1. AUTH FAIL-CLOSED: setting ``modules.web_terminals.auth.method`` to a real
   method (``password``/``oidc``) must gate EVERY proxied ``/u/<user>/``
   location behind an ``auth_request`` naming that user's own ``internal``
   ``/_osprey_auth/<user>`` target, which asks the sidecar
   ``GET /verify?user=<user>`` with the username fixed at render time. No
   location may be left ungated, no target may be reachable from outside, and
   nothing anywhere may authorize by returning success. Leaving ``auth.method``
   absent/``"none"`` must emit no auth surface at all, even with TLS on.
2. TLS FAIL-FAST: ``modules.web_terminals.tls.enabled: true`` without both
   ``tls.cert`` and ``tls.key`` set must raise ``ValueError`` at render time
   rather than emit an incoherent nginx config. With both paths set, the plain
   port becomes a redirect-only server and the ``listen 443 ssl`` server becomes
   the sole content server — so with auth also on, there is no cleartext path to
   a login form and therefore no cleartext origin a session cookie can be issued
   on.

This file is the authority to point at if a future lifecycle change is
suspected of moving either seam — it duplicates a handful of assertions
already present in test_render.py by design (a single, explicit,
narrowly-scoped file that stays correct even if test_render.py's broader
coverage shifts), and it deliberately keeps its own helpers rather than
importing test_render.py's.
"""

from __future__ import annotations

import copy
import re

import pytest

from osprey.deployment.web_terminals.render import render_web_terminals

_BASE_PORTS = {"web": 9091, "artifact": 9291, "ariel": 9391, "lattice": 9491}

_TLS_STANZA = {
    "enabled": True,
    "cert": "/etc/nginx/certs/dls.crt",
    "key": "/etc/nginx/certs/dls.key",
}


def _config(users: list[str]) -> dict:
    """Minimal-but-complete facility config that exercises render_web_terminals()."""
    return {
        "facility": {
            "name": "Demo Light Source",
            "prefix": "dls",
            "timezone": "America/Los_Angeles",
        },
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": 9080,
                "web_base_port": _BASE_PORTS["web"],
                "artifact_base_port": _BASE_PORTS["artifact"],
                "ariel_base_port": _BASE_PORTS["ariel"],
                "lattice_base_port": _BASE_PORTS["lattice"],
                "users": users,
            }
        },
    }


_MULTI_USER_CONFIG = _config(["alice", "bob", "carol"])


def _auth_config(users: list[str] | None = None, *, tls: bool = False) -> dict:
    """A rendering config with authentication on, over cleartext unless `tls`.

    Without TLS the render refuses `auth.method` outright unless the facility
    accepts the risk, so `allow_insecure_http` is set in that case — the seam
    under test here is the auth surface itself, not that separate gate.
    """
    config = _config(users if users is not None else ["alice", "bob", "carol"])
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["auth"] = {"method": "password"}
    if tls:
        web_terminals["tls"] = copy.deepcopy(_TLS_STANZA)
    else:
        web_terminals["auth"]["allow_insecure_http"] = True
    return config


def _render_nginx(config: dict) -> str:
    return render_web_terminals(config)["nginx/nginx.conf"]


def _directives(conf: str) -> str:
    """`conf` with its comment lines dropped.

    Counting and absence assertions have to ignore comment prose, which in this
    template frequently names the very construct being counted or ruled out (the
    comments explain `auth_request`, `$request_uri` and `/verify` at length).
    """
    return "\n".join(line for line in conf.splitlines() if not line.lstrip().startswith("#"))


def _server_blocks(conf: str) -> list[str]:
    """The top-level `server { … }` bodies, in render order.

    Under TLS there are TWO: the redirect-only plain-port server first, then the
    443 content server. Split on the closing brace in column 0, which only a
    top-level block has.
    """
    blocks = re.findall(r"^server \{\n(.*?)^\}$", conf, flags=re.DOTALL | re.MULTILINE)
    assert blocks, "no top-level server block in the rendered fragment"
    return blocks


def _location_body(conf: str, header: str) -> str:
    match = re.search(rf"{re.escape(header)} \{{(.*?)\n    \}}", conf, flags=re.DOTALL)
    assert match is not None, f"no {header!r} block in the rendered fragment"
    return match.group(1)


def test_seam_auth_method_set_gates_every_user_behind_its_own_internal_target() -> None:
    """SEAM 1 (auth fail-closed): a configured `auth.method` must gate every
    proxied `/u/<user>/` location behind an `auth_request` naming that user's own
    `internal` target — and must emit no construct that authorizes by returning
    success.

    The pin is a coverage invariant, not a substring: one `auth_request` per
    proxied user location, so a roster entry rendered ungated fails here rather
    than passing because some other user's directive is present.
    """
    # Arrange
    config = _auth_config()
    users = config["modules"]["web_terminals"]["users"]

    # Act
    nginx_conf = _render_nginx(config)
    directives = _directives(nginx_conf)

    # Assert — every proxied user location is gated, and by its OWN target
    gated_locations = directives.count("location /u/")
    assert gated_locations == len(users)
    assert directives.count("auth_request ") == gated_locations
    for user in users:
        assert f"auth_request /_osprey_auth/{user};" in _location_body(
            nginx_conf, f"location /u/{user}/"
        )
        assert f"location = /_osprey_auth/{user} {{" in nginx_conf

    # Assert — the auth targets are unreachable from outside, one per user
    assert directives.count("internal;") == len(users)

    # Assert — nothing authorizes, and the pre-sidecar shared stub is gone
    # rather than merely supplemented: neither a permissive `return 200;` nor
    # the single blanket `return 403;` target it used to deny with survives.
    assert "return 200;" not in nginx_conf
    assert "return 403;" not in directives
    assert "location = /_osprey_auth {" not in nginx_conf


def test_seam_auth_target_asks_the_sidecar_about_a_render_time_username() -> None:
    """SEAM 1 (identity is structural): each internal target's `?user=` is a
    render-time literal from the roster, so no part of a client's request
    selects which user it is authorized as.

    The subrequest must also carry no body (an inherited Content-Length the
    sidecar never reads hangs every authenticated POST) and must fail in seconds
    rather than hold a gated request on nginx's 60s defaults.
    """
    # Arrange
    config = _auth_config(["alice", "bob"])
    users = config["modules"]["web_terminals"]["users"]

    # Act
    nginx_conf = _render_nginx(config)
    directives = _directives(nginx_conf)

    # Assert
    for user in users:
        body = _location_body(nginx_conf, f"location = /_osprey_auth/{user}")
        assert f"proxy_pass http://127.0.0.1:9070/verify?user={user};" in body
        assert "proxy_pass_request_body off;" in body
        assert 'proxy_set_header Content-Length "";' in body
        assert "proxy_connect_timeout 2s;" in body
        assert "proxy_read_timeout 5s;" in body

    # The identity is never reconstructed from client-controlled input — the two
    # places a crafted URL could otherwise smuggle a different username in.
    assert "$request_uri" not in directives
    assert "X-Original-URI" not in directives
    # `/verify` is reachable only through the internal targets: it is not a
    # location of its own and is not exposed under the public `/auth/` prefix.
    assert "location /verify" not in directives
    assert "location = /verify" not in directives
    assert len([line for line in directives.splitlines() if "/verify" in line]) == len(users)


def test_seam_auth_denied_subrequest_redirects_only_an_allowlisted_return_path() -> None:
    """SEAM 1 (a refusal never emits an attacker-chosen header value): the shared
    401 handler's login redirect carries `$osprey_auth_next` — the allowlisted
    form of the path — and neither raw path variable.

    `$uri` is nginx's DECODED path, so a `%0d%0a` in the request path is a real
    CR LF by the time `return 302` copies it into the Location header; emitting
    it verbatim let a crafted link append arbitrary response headers.
    `$request_uri` is injection-safe but carries whatever the client typed,
    including a leading `//` a browser follows off-site. Everything that is not
    a plain terminal path must collapse to the empty default.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config())
    handler = _location_body(nginx_conf, "location @osprey_auth_401")
    handler_directives = _directives(handler)

    # Assert — the sanitized variable is what the redirect emits
    assert "return 302 /auth/login?user=$osprey_auth_user&next=$osprey_auth_next;" in handler
    assert "next=$uri" not in handler_directives
    assert "$request_uri" not in handler_directives
    # Programs get a bare status they can act on, not a page of login HTML.
    assert "return 401;" in handler_directives

    # Assert — the allowlist is an allowlist: a default of "" and an anchored
    # pattern. `\z`, not `$`: PCRE's `$` also matches before a trailing newline,
    # so a path ending in `%0a` would otherwise be emitted, newline and all.
    allowlist = _directives(nginx_conf)
    assert "map $uri $osprey_auth_next {" in allowlist
    assert '\n    default "";' in allowlist
    assert re.search(r'\n    "~\^/u/\[[^"]+\\z" \$uri;', allowlist) is not None
    assert "map $http_upgrade$http_accept $osprey_auth_wants_login_page {" in allowlist


def test_seam_auth_redirects_stay_on_the_clients_own_origin() -> None:
    """SEAM 1 (no redirect walks a browser off TLS): `absolute_redirect off` sits
    at SERVER scope — before the first location, exactly once — so it covers
    every redirect this server can emit, not just the one that first needed it.

    Left at nginx's default, a relative Location becomes an absolute URL built
    from `$host` plus nginx's OWN listening port: a browser on
    `https://facility/u/alice` behind a facility TLS terminator would be walked
    to `http://facility:9080/u/alice/`. It is gated on auth being on, so the
    `method: none` render must not carry it at all.
    """
    # Act
    content = _server_blocks(_render_nginx(_auth_config(["alice"])))[0]

    # Assert — server scope, before any location, and only there
    preamble = _directives(content).split("location", 1)[0]
    assert "absolute_redirect off;" in preamble
    assert content.count("absolute_redirect off;") == 1

    # Assert — gated on auth: the unauthenticated render is unchanged by it
    assert "absolute_redirect" not in _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))


def test_seam_auth_none_or_absent_emits_no_auth_surface_even_with_tls_on() -> None:
    """SEAM 1 (auth gated off by default): with `auth.method` absent/"none", no
    part of the auth surface is emitted — no `auth_request`, no internal target,
    no public `/auth/` prefix, no 401 handler — including when TLS is separately
    enabled. The two seams are independently gated; TLS opting in must not
    accidentally opt in auth too."""
    # Arrange — absent auth stanza, TLS enabled
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "auth" not in config["modules"]["web_terminals"]
    config["modules"]["web_terminals"]["tls"] = copy.deepcopy(_TLS_STANZA)

    # Act
    nginx_conf = _render_nginx(config)

    # Assert
    assert "listen 443 ssl;" in nginx_conf
    for absent in ("auth_request", "_osprey_auth", "location /auth/", "error_page"):
        assert absent not in nginx_conf

    # Arrange — explicit auth.method: "none" (not merely absent)
    config_explicit_none = copy.deepcopy(_MULTI_USER_CONFIG)
    config_explicit_none["modules"]["web_terminals"]["auth"] = {"method": "none"}

    # Act
    nginx_conf_explicit_none = _render_nginx(config_explicit_none)

    # Assert
    for absent in ("auth_request", "_osprey_auth", "location /auth/", "error_page"):
        assert absent not in nginx_conf_explicit_none


def test_seam_tls_enabled_without_both_cert_and_key_raises_value_error() -> None:
    """SEAM 2 (TLS fail-fast): `tls.enabled: true` without both `cert` and `key`
    must raise ValueError at render time rather than emit a nginx config pointing
    at a missing cert/key path."""
    # Arrange — enabled with neither cert nor key
    config_neither = copy.deepcopy(_MULTI_USER_CONFIG)
    config_neither["modules"]["web_terminals"]["tls"] = {"enabled": True}

    # Act / Assert
    with pytest.raises(ValueError, match="tls"):
        render_web_terminals(config_neither)

    # Arrange — enabled with cert but no key
    config_cert_only = copy.deepcopy(_MULTI_USER_CONFIG)
    config_cert_only["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
    }

    # Act / Assert
    with pytest.raises(ValueError, match="tls"):
        render_web_terminals(config_cert_only)

    # Arrange — enabled with key but no cert
    config_key_only = copy.deepcopy(_MULTI_USER_CONFIG)
    config_key_only["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act / Assert
    with pytest.raises(ValueError, match="tls"):
        render_web_terminals(config_key_only)


def test_seam_auth_over_cleartext_is_refused_unless_explicitly_accepted() -> None:
    """SEAM 2 (no session cookie over cleartext by accident): `auth.method` set
    with `tls.enabled` false must raise rather than render a login flow whose
    cookies cross the network in the clear. A facility fronted by its own TLS
    terminator opts in explicitly via `auth.allow_insecure_http`."""
    # Arrange — auth on, no TLS, no explicit acceptance
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = {"method": "password"}

    # Act / Assert
    with pytest.raises(ValueError, match="allow_insecure_http"):
        render_web_terminals(config)

    # Act / Assert — the explicit opt-in renders the auth surface
    assert "auth_request " in _directives(_render_nginx(_auth_config()))


def test_seam_tls_enabled_with_both_cert_and_key_emits_ssl_listen_and_paths() -> None:
    """SEAM 2 (TLS renders when fully configured): with both `tls.cert` and
    `tls.key` set, the fragment splits into a redirect-only server on the plain
    port and a `listen 443 ssl` content server carrying the configured
    `ssl_certificate`/`ssl_certificate_key` paths."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["tls"] = copy.deepcopy(_TLS_STANZA)

    # Act
    blocks = _server_blocks(_render_nginx(config))

    # Assert — two servers: the plain port redirects, 443 serves
    assert len(blocks) == 2
    redirect, content = blocks
    assert "return 301 https://$host$request_uri;" in redirect
    assert "location" not in redirect
    assert "listen 443 ssl;" in content
    assert "ssl_certificate /etc/nginx/certs/dls.crt;" in content
    assert "ssl_certificate_key /etc/nginx/certs/dls.key;" in content
    assert "location = / {" in content


def test_seam_auth_surface_exists_only_on_the_tls_content_server() -> None:
    """SEAMS 1+2 together: with both on, every gated location and every auth
    target lives in the 443 server and nowhere else. A duplicate left on the
    cleartext listener would be a plain-http path to a login form, and any
    session cookie issued there would cross the network in the clear."""
    # Arrange
    config = _auth_config(tls=True)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    redirect, content = _server_blocks(_render_nginx(config))

    # Assert — the cleartext listener carries nothing but the 301
    assert redirect.count("location") == 0
    assert "_osprey_auth" not in redirect
    assert _directives(redirect).count("auth_request ") == 0

    # Assert — the secure listener carries the whole gated surface, in full
    assert _directives(content).count("location /u/") == len(users)
    assert _directives(content).count("auth_request ") == len(users)
    assert content.count("location = /_osprey_auth/") == len(users)
    assert "location /auth/ {" in content
    assert "return 200;" not in content
