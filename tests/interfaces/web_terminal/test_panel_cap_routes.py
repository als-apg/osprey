"""C-CAP: the web-terminal app is the layer that decides whether a cap is good.

nginx exempts a capability path from the login and strips every credential on
the way in (``tests/deployment/web_terminals/test_auth_serving.py`` proves that
against the real perimeter), but it never looks at the cap itself. So these
tests drive the real panel routes — both trees a cap can frame, the discovered
bundle at ``/panel-static/<id>/~c/<cap>/`` and the runtime panel proxied at
``/panel/<id>/~c/<cap>/`` — and pin the decision:

* a valid cap is served;
* a forged cap, a cap minted for another panel, and an expired cap answer
  ``403``, serve none of the panel's bytes, and never reach a runtime backend.

Every request runs under ``no_auth_seam`` and sends no cookie: the suite-wide
seam would otherwise authenticate it, and a pass would prove nothing about the
cap. Marked like its nginx half (``dockerbuild``) so the two run together on the
gate host; this half needs no engine and has no skip.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import PANEL_CAP_SEGMENT
from osprey.interfaces.web_auth import get_web_credentials
from osprey.interfaces.web_terminal.panel_cap import CAP_TTL_SECONDS, OPERATOR_LOGIN, mint_cap

from ._proxy_fakes import panel_app
from .test_panel_cap import _RUNTIME_PANEL, _bundle, _credentials, _discovered_entry

pytestmark = [pytest.mark.dockerbuild, pytest.mark.no_auth_seam]

#: What an opaque panel's own requests carry: no cookie, an opaque origin.
_OPAQUE = {"Origin": "null"}

_RUNTIME_BODY = "<html><body>RUNTIME-DASH</body></html>"


@pytest.fixture
def hub(tmp_path) -> Iterator[tuple[FastAPI, TestClient, list[str]]]:
    """The whole app: two discovered bundles and one runtime panel.

    The runtime panel's backend is a recorder, so a refusal can also be shown
    never to have reached it. Yields ``(app, client, backend_calls)``.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    bundles = tmp_path / "panels"
    alpha = _bundle(bundles, "alpha", "alpha")
    beta = _bundle(bundles, "beta", "beta")
    custom = [_discovered_entry("alpha"), _discovered_entry("beta"), _RUNTIME_PANEL]
    calls: list[str] = []

    def backend(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        calls.append(url)
        return httpx.Response(200, text=_RUNTIME_BODY, headers={"content-type": "text/html"})

    for app, client in panel_app(workspace, custom):
        app.state.discovered_panel_dirs = {"alpha": alpha, "beta": beta}
        app.state.proxy_client.request = AsyncMock(side_effect=backend)
        yield app, client, calls


def _mint(app: FastAPI, panel_id: str, **kwargs) -> str:
    return mint_cap(get_web_credentials(app), OPERATOR_LOGIN, panel_id, **kwargs)


#: (tree, panel id, a file under it, the marker its bytes carry)
_TREES = {
    "discovered-bundle": ("panel-static", "alpha", "index.html", "ALPHA"),
    "runtime-panel": ("panel", "agent-dash", "view", "RUNTIME-DASH"),
}


def _url(tree: str, panel_id: str, cap: str, file: str) -> str:
    return f"/{tree}/{panel_id}/{PANEL_CAP_SEGMENT}/{cap}/{file}"


def _forged(app: FastAPI, panel_id: str) -> str:
    """A cap whose signature was tampered with after minting."""
    expiry, tag, signature = _mint(app, panel_id).split(".")
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    return f"{expiry}.{tag}.{flipped}"


def _foreign_key(app: FastAPI, panel_id: str) -> str:  # noqa: ARG001
    """A well-formed cap signed under a secret this deployment does not hold."""
    return mint_cap(_credentials("t" * 43), OPERATOR_LOGIN, panel_id)


def _other_panel(app: FastAPI, panel_id: str) -> str:
    """A valid cap — for a different panel than the one it is used on."""
    return _mint(app, "beta" if panel_id != "beta" else "alpha")


def _expired(app: FastAPI, panel_id: str) -> str:
    """A cap that was valid, minted a full lifetime and a minute ago."""
    return _mint(app, panel_id, now=time.time() - CAP_TTL_SECONDS - 60)


_INVALID: dict[str, Callable[[FastAPI, str], str]] = {
    "forged-signature": _forged,
    "foreign-key": _foreign_key,
    "another-panel": _other_panel,
    "expired": _expired,
}


@pytest.mark.parametrize("tree_id", list(_TREES))
def test_a_valid_cap_is_served(hub, tree_id: str) -> None:
    """The control for every refusal below: the same request with a good cap."""
    # Arrange
    app, client, calls = hub
    tree, panel_id, file, marker = _TREES[tree_id]

    # Act
    response = client.get(_url(tree, panel_id, _mint(app, panel_id), file), headers=_OPAQUE)

    # Assert
    assert response.status_code == 200, response.text
    assert marker in response.text
    assert "set-cookie" not in response.headers
    assert len(calls) == (1 if tree == "panel" else 0)


@pytest.mark.parametrize("cap_id", list(_INVALID))
@pytest.mark.parametrize("tree_id", list(_TREES))
def test_an_invalid_cap_is_refused_403(hub, tree_id: str, cap_id: str) -> None:
    """Forged, foreign, cross-panel and expired caps are all a 403, not a read."""
    # Arrange
    app, client, calls = hub
    tree, panel_id, file, marker = _TREES[tree_id]
    cap = _INVALID[cap_id](app, panel_id)

    # Act
    response = client.get(_url(tree, panel_id, cap, file), headers=_OPAQUE)

    # Assert
    assert response.status_code == 403, response.text
    assert marker not in response.text
    assert calls == [], "a refused cap reached the runtime panel's backend"


def test_a_cap_minted_for_one_tree_does_not_open_the_other(hub) -> None:
    """A runtime panel's cap names its id, not a bundle; the bundle tree refuses it."""
    # Arrange
    app, client, calls = hub
    runtime_cap = _mint(app, "agent-dash")
    bundle_cap = _mint(app, "alpha")

    # Act
    into_bundle = client.get(_url("panel-static", "alpha", runtime_cap, "index.html"))
    into_runtime = client.get(_url("panel", "agent-dash", bundle_cap, "view"))

    # Assert
    assert into_bundle.status_code == 403
    assert "ALPHA" not in into_bundle.text
    assert into_runtime.status_code == 403
    assert calls == []


def test_an_expired_cap_is_refused_one_second_past_its_expiry(hub) -> None:
    """Expiry is a hard edge: a cap one second past it is refused, not tolerated."""
    # Arrange
    app, client, _calls = hub
    cap = _mint(app, "alpha", ttl=-1)

    # Act
    response = client.get(_url("panel-static", "alpha", cap, "data.json"), headers=_OPAQUE)

    # Assert
    assert response.status_code == 403
    assert "alpha-data" not in response.text
