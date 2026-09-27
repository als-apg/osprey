"""The per-user mount root is spelled once, and every producer derives from it."""

from __future__ import annotations

from osprey.interfaces import common_middleware
from osprey.interfaces.common_middleware import (
    TERMINAL_USER_ENV,
    URL_MOUNT_ROOT,
    compute_url_prefix,
    url_mount_prefix,
)


def test_the_mount_prefix_is_the_root_and_one_user_segment():
    assert URL_MOUNT_ROOT == "/u"
    assert url_mount_prefix("alice") == "/u/alice"


def test_the_container_prefix_derives_from_the_root(monkeypatch):
    monkeypatch.setattr(common_middleware, "URL_MOUNT_ROOT", "/m")
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    assert compute_url_prefix() == "/m/alice"
