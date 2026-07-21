"""Regression guard for the phase-4 auth/TLS seams in web-terminal nginx rendering.

Phase 3 changes the deploy *lifecycle* (up/down/reconcile/status, etc.) around
:func:`osprey.deployment.web_terminals.render.render_web_terminals`, but must not
touch the two security seams that Phase 4 (real auth backend, real TLS
provisioning) is scoped to fill in. Those seams are deliberately gated OFF by
default and, when opted into via config, fail closed/fast rather than silently
degrade:

1. AUTH FAIL-CLOSED: setting ``modules.web_terminals.auth.method`` to anything
   other than ``"none"`` must still emit the ``auth_request`` -> ``return 403``
   stub per proxied ``/u/<user>/`` location (no real auth backend exists yet —
   the placeholder must deny, never silently authorize). Leaving ``auth.method``
   absent/``"none"`` must emit no ``auth_request`` at all, even with TLS on.
2. TLS FAIL-FAST: ``modules.web_terminals.tls.enabled: true`` without both
   ``tls.cert`` and ``tls.key`` set must raise ``ValueError`` at render time
   rather than emit an incoherent nginx config. With both paths set, it must
   render the ``listen 443 ssl`` listener and the configured cert/key paths.

This file is the authority to point at if a future lifecycle change is
suspected of moving either seam — it duplicates a handful of assertions
already present in test_render.py by design (a single, explicit,
narrowly-scoped file that stays correct even if test_render.py's broader
coverage shifts).
"""

from __future__ import annotations

import copy

import pytest

from osprey.deployment.web_terminals.render import render_web_terminals

_BASE_PORTS = {"web": 9091, "artifact": 9291, "ariel": 9391, "lattice": 9491}


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


def test_seam_auth_method_set_emits_fail_closed_auth_request_stub() -> None:
    """SEAM 1 (auth fail-closed): a configured `auth.method` (not "none") must emit
    `auth_request` in every proxied `/u/<user>/` location plus a `return 403;`
    internal stub target — never `return 200;`. This is the fail-closed placeholder
    Phase 4 is scoped to replace with a real backend; a lifecycle change must not
    turn it into a silent allow."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]
    config["modules"]["web_terminals"]["auth"] = {"method": "oauth2_proxy"}

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert nginx_conf.count("auth_request /_osprey_auth;") == len(users)
    assert nginx_conf.count("internal;") == 1
    assert "return 403;" in nginx_conf
    assert "return 200;" not in nginx_conf


def test_seam_auth_none_or_absent_emits_no_auth_request_even_with_tls_on() -> None:
    """SEAM 1 (auth gated off by default): with `auth.method` absent/"none", no
    `auth_request` directive is emitted at all — including when TLS is separately
    enabled. The two seams are independently gated; TLS opting in must not
    accidentally opt in auth too."""
    # Arrange — absent auth stanza, TLS enabled
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "auth" not in config["modules"]["web_terminals"]
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "listen 443 ssl;" in nginx_conf
    assert "auth_request" not in nginx_conf

    # Arrange — explicit auth.method: "none" (not merely absent)
    config_explicit_none = copy.deepcopy(_MULTI_USER_CONFIG)
    config_explicit_none["modules"]["web_terminals"]["auth"] = {"method": "none"}

    # Act
    nginx_conf_explicit_none = render_web_terminals(config_explicit_none)["nginx/nginx.conf"]

    # Assert
    assert "auth_request" not in nginx_conf_explicit_none


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


def test_seam_tls_enabled_with_both_cert_and_key_emits_ssl_listen_and_paths() -> None:
    """SEAM 2 (TLS renders when fully configured): with both `tls.cert` and
    `tls.key` set, render emits `listen 443 ssl` plus the configured
    `ssl_certificate`/`ssl_certificate_key` paths inside the `server{}` block."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    nginx_conf = render_web_terminals(config)["nginx/nginx.conf"]

    # Assert
    assert "listen 443 ssl;" in nginx_conf
    assert "ssl_certificate /etc/nginx/certs/dls.crt;" in nginx_conf
    assert "ssl_certificate_key /etc/nginx/certs/dls.key;" in nginx_conf
    server_index = nginx_conf.index("server {")
    ssl_index = nginx_conf.index("listen 443 ssl;")
    assert server_index < ssl_index, "the ssl listener must sit inside server{}, not above it"
