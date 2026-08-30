"""A config-declared panel's ``path`` may name the per-user URL prefix (#784).

Behind the multi-user front door every terminal is mounted at ``/u/<user>/``
and its panels at ``/u/<user>/panel/<id>/``. The proxy rewrites root-absolute
literals in what a backend serves, but a URL the browser assembles at runtime
from a query parameter never passes through it — noVNC's ``vnc.html?path=``
is resolved against the origin root, so a static ``path=panel/x/websockify``
opens ``ws://host/panel/x/websockify`` and 404s. The correct value is per user,
so the profile spells a placeholder and ``_load_panel_config`` resolves it from
``compute_url_prefix()`` once per container.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osprey.interfaces.common_middleware import TERMINAL_USER_ENV
from osprey.interfaces.web_terminal.app import _load_panel_config, _substitute_url_prefix

_NOVNC = "/vnc.html?path={url_prefix_dir}panel/phoebus/websockify&autoconnect=1"


def _config(path: str) -> dict:
    return {"web": {"panels": {"phoebus": {"url": "http://127.0.0.1:19922", "path": path}}}}


def _custom_path(path: str) -> str:
    with patch("osprey.utils.workspace.load_osprey_config", return_value=_config(path)):
        _enabled, custom, _default = _load_panel_config()
    (panel,) = custom
    return panel["path"]


def test_placeholders_resolve_behind_the_front_door(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    assert _custom_path(_NOVNC) == ("/vnc.html?path=u/alice/panel/phoebus/websockify&autoconnect=1")
    assert _custom_path("{url_prefix}/dashboard") == "/u/alice/dashboard"


def test_placeholders_resolve_to_nothing_in_the_single_origin_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No prefix ⇒ the noVNC path has no leading slash and the root path stays root."""
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    assert _custom_path(_NOVNC) == "/vnc.html?path=panel/phoebus/websockify&autoconnect=1"
    assert _custom_path("{url_prefix}/dashboard") == "/dashboard"


def test_a_path_without_a_placeholder_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    assert _custom_path("/panel/") == "/panel/"
    assert _custom_path("/dashboard?x={not_a_prefix}") == "/dashboard?x={not_a_prefix}"


def test_substitution_tolerates_a_non_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed spec reaches the browser as it always did, not as an exception."""
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    assert _substitute_url_prefix(None) is None  # type: ignore[arg-type]
