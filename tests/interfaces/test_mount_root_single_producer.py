"""The per-user mount root is spelled once, and every producer derives from it."""

from __future__ import annotations

from osprey.deployment.web_terminals.render import _user_card, terminal_login_url
from osprey.interfaces import common_middleware
from osprey.interfaces.common_middleware import (
    TERMINAL_USER_ENV,
    URL_MOUNT_ROOT,
    compute_url_prefix,
    url_mount_prefix,
)
from osprey.services.auth_sidecar.return_to import safe_return_to


def test_the_mount_prefix_is_the_root_and_one_user_segment():
    assert URL_MOUNT_ROOT == "/u"
    assert url_mount_prefix("alice") == "/u/alice"


def test_the_container_prefix_derives_from_the_root(monkeypatch):
    monkeypatch.setattr(common_middleware, "URL_MOUNT_ROOT", "/m")
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    assert compute_url_prefix() == "/m/alice"


def test_the_login_url_and_the_landing_card_derive_from_the_mount_root(monkeypatch):
    monkeypatch.setattr(common_middleware, "URL_MOUNT_ROOT", "/m")
    config = {"modules": {"web_terminals": {"external_origin": "https://ops.example.org"}}}
    assert (
        terminal_login_url(config, "alice", "s/t") == "https://ops.example.org/m/alice/?token=s%2Ft"
    )
    assert _user_card({"name": "alice", "persona": None}, frozenset())["url"] == "/m/alice/"


def test_the_sidecars_default_return_to_derives_from_the_root(monkeypatch):
    monkeypatch.setattr(common_middleware, "URL_MOUNT_ROOT", "/m")
    assert safe_return_to("", "alice") == "/m/alice/"
    assert safe_return_to("https://evil.example/", "alice") == "/m/alice/"
