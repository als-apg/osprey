"""C-ASSETS in the multi-user perimeter: asset paths skip nginx's ``auth_request``.

An agent-authored document runs at an opaque origin, so the browser fetches the
modules, stylesheets and fonts it imports with ``Origin: null`` and without the
SameSite=Lax cookie. Behind the auth sidecar every such fetch would be denied by
the per-user ``auth_request``. The render therefore emits one exemption location
per user covering exactly the C-ASSETS paths, and these tests pin it:

* it matches every C-ASSETS path and none that is agent-writable or relayed to a
  panel backend (artifact files, ``/panel-static``, Jupyter ``files/``, a runtime
  panel's own static tree, ``/api``);
* it carries no ``auth_request`` and no secret, forwards none of the client's
  headers, claims every identity name by clearing it, cuts the cookie jar and
  allows GET only;
* it is rendered only where there is an ``auth_request`` to be exempt from.

The C-CAP exemption (a panel framed at ``…/<id>/~c/<cap>/``) is a second
location per user, pinned the same way and against the app's own predicate: no
``/panel*`` path skips ``auth_request`` unless C-ASSETS or C-CAP covers it.

Assertions run on the comment-stripped render: the template explains these
directives in prose directly above them.
"""

from __future__ import annotations

import copy
import re

import pytest

from osprey.deployment.web_terminals.render import TERMINAL_SECRET_HEADER, render_web_terminals
from osprey.interfaces.common_middleware import (
    FIRST_PARTY_PANEL_STATIC,
    PANEL_ASSET_SUBPATHS,
    PANEL_CAP_SEGMENT,
    URL_MOUNT_ROOT,
    is_panel_asset_path,
    is_panel_cap_path,
)
from osprey.services.auth_sidecar.identity_headers import (
    ACCOUNT_HEADER,
    ROLE_HEADER,
    ROLE_SOURCE_HEADER,
    SUBJECT_HEADER,
)
from tests._renders import tier_render

_USERS = ["alice", "bob"]
_IDENTITY_HEADERS = (ACCOUNT_HEADER, SUBJECT_HEADER, ROLE_HEADER, ROLE_SOURCE_HEADER)


def _config(auth: dict | None) -> dict:
    config = {
        "facility": {"prefix": "dls"},
        "system": {"timezone": "America/Los_Angeles"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": copy.deepcopy(_USERS),
                "default_tier": "assistant",
                "renders": {"assistant": tier_render("assistant", "demo-assistant")},
            }
        },
    }
    if auth is not None:
        config["modules"]["web_terminals"]["auth"] = auth
    return config


_PASSWORD = {"method": "password", "allow_insecure_http": True}
_OPEN = {"method": "none"}


def _directives(conf: str) -> str:
    return "\n".join(line for line in conf.splitlines() if not line.lstrip().startswith("#"))


_LOCATION_RE = re.compile(r"^    location ([^\n{]*)\{\n(.*?)\n    \}$", re.DOTALL | re.MULTILINE)


def _locations(config: dict) -> dict[str, str]:
    conf = render_web_terminals(config)["nginx/nginx.conf"]
    pairs = [(m.group(1).strip(), m.group(2)) for m in _LOCATION_RE.finditer(_directives(conf))]
    found = dict(pairs)
    assert len(found) == len(pairs), "duplicate location headers"
    return found


def _exempt_locations(locations: dict[str, str]) -> dict[str, str]:
    """Every regex location under a user's mount: the C-ASSETS and C-CAP exemptions."""
    return {h: b for h, b in locations.items() if h.startswith(f"~ ^{URL_MOUNT_ROOT}/")}


def _is_cap_header(header: str) -> bool:
    return f"/{PANEL_CAP_SEGMENT}/" in header


def _asset_locations(locations: dict[str, str]) -> dict[str, str]:
    """The C-ASSETS exemptions."""
    return {h: b for h, b in _exempt_locations(locations).items() if not _is_cap_header(h)}


def _cap_locations(locations: dict[str, str]) -> dict[str, str]:
    """The C-CAP exemptions."""
    return {h: b for h, b in _exempt_locations(locations).items() if _is_cap_header(h)}


def _pattern(header: str) -> re.Pattern[str]:
    assert header.startswith("~ ")
    return re.compile(header[2:])


def _asset_location_for(locations: dict[str, str], user: str) -> tuple[str, str]:
    matches = [
        (h, b)
        for h, b in _asset_locations(locations).items()
        if h.startswith(f"~ ^{URL_MOUNT_ROOT}/{user}/")
    ]
    assert len(matches) == 1, f"expected one asset location for {user}: {matches}"
    return matches[0]


def _c_assets_paths(user: str) -> list[str]:
    root = f"{URL_MOUNT_ROOT}/{user}"
    paths = [
        f"{root}/design-system/js/theme-manager.js",
        f"{root}/design-system/css/tokens.css",
        f"{root}/static/fonts/fonts.css",
        f"{root}/static/fonts/dm-mono-400.ttf",
    ]
    for panel_id in ("artifacts", "my-dash", "jupyter", "channel-finder"):
        paths += [f"{root}/panel/{panel_id}/{sub}/x.js" for sub in PANEL_ASSET_SUBPATHS]
    for panel_id, subs in FIRST_PARTY_PANEL_STATIC.items():
        paths += [f"{root}/panel/{panel_id}/{sub}/vendor/x.min.js" for sub in subs]
    return paths


def _gated_paths(user: str) -> list[str]:
    root = f"{URL_MOUNT_ROOT}/{user}"
    return [
        f"{root}/",
        f"{root}/api/panels",
        f"{root}/files/abc/report.html",
        f"{root}/static/session.html",
        f"{root}/static/index.html",
        f"{root}/panel-static/agent-panel/index.html",
        f"{root}/panel-static/agent-panel/design-system/x.js",
        f"{root}/panel/jupyter/files/out.html",
        f"{root}/panel/artifacts/files/abc/report.html",
        f"{root}/panel/artifacts/api/markdown/abc/rendered",
        f"{root}/panel/my-dash/static/js/app.js",
        f"{root}/panel/my-dash/static/css/app.css",
        f"{root}/panel/my-dash/design-system",
        f"{root}/panel/.x/design-system/x.js",
        f"{root}/panel/jupyter/terminal-static/js/api.js",
        f"{root}/ws/terminal",
    ]


# ---------------------------------------------------------------------------
# Auth on: one exemption per user, matching exactly C-ASSETS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("auth", [_PASSWORD], ids=["password"])
def test_each_user_gets_one_asset_exemption(auth):
    locations = _locations(_config(auth))
    assets = _asset_locations(locations)
    assert len(assets) == len(_USERS)
    for user in ("alice", "bob"):
        _asset_location_for(locations, user)
        # The gated location itself is unchanged: still behind auth_request.
        assert f"auth_request /_osprey_auth/{user};" in locations[f"{URL_MOUNT_ROOT}/{user}/"]


@pytest.mark.parametrize("user", ["alice", "bob"])
def test_the_exemption_matches_every_c_assets_path(user):
    header, _ = _asset_location_for(_locations(_config(_PASSWORD)), user)
    pattern = _pattern(header)
    for path in _c_assets_paths(user):
        assert pattern.search(path), path


@pytest.mark.parametrize("user", ["alice", "bob"])
def test_no_agent_writable_or_relayed_path_is_exempt(user):
    """Artifact files, /panel-static, Jupyter files/, runtime panels stay gated."""
    header, _ = _asset_location_for(_locations(_config(_PASSWORD)), user)
    pattern = _pattern(header)
    for path in _gated_paths(user):
        assert not pattern.search(path), path


def test_one_users_exemption_never_matches_another_users_mount():
    locations = _locations(_config(_PASSWORD))
    alice, _ = _asset_location_for(locations, "alice")
    for path in _c_assets_paths("bob"):
        assert not _pattern(alice).search(path), path


def test_the_exemption_strips_only_the_mount_and_targets_the_users_own_upstream():
    locations = _locations(_config(_PASSWORD))
    upstreams = set()
    for user in ("alice", "bob"):
        _, body = _asset_location_for(locations, user)
        gated = locations[f"{URL_MOUNT_ROOT}/{user}/"]
        upstream = re.search(r"proxy_pass (http://[^/;]+)/;", gated).group(1)
        assert f"proxy_pass {upstream};" in body
        assert f"rewrite ^{URL_MOUNT_ROOT}/{user}(/.*)$ $1 break;" in body
        upstreams.add(upstream)
    assert len(upstreams) == 2


# ---------------------------------------------------------------------------
# Auth on: what the exemption forwards
# ---------------------------------------------------------------------------


def _bodies():
    locations = _locations(_config(_PASSWORD))
    return [_asset_location_for(locations, user) for user in ("alice", "bob")]


@pytest.mark.parametrize("index", [0, 1])
def test_the_exemption_carries_no_auth_request_and_no_secret(index):
    header, body = _bodies()[index]
    assert "auth_request" not in body, header
    assert "include " not in body, header
    assert TERMINAL_SECRET_HEADER not in body, header
    assert "error_page 401" not in body, header


@pytest.mark.parametrize("index", [0, 1])
def test_the_exemption_forwards_none_of_the_clients_headers(index):
    """``proxy_pass_request_headers off`` drops the secret header unspelled."""
    _, body = _bodies()[index]
    assert "proxy_pass_request_headers off;" in body


@pytest.mark.parametrize("index", [0, 1])
def test_the_exemption_claims_every_credential_name_by_clearing_it(index):
    _, body = _bodies()[index]
    for header in (*_IDENTITY_HEADERS, "Authorization", "Cookie"):
        assert body.count(f"proxy_set_header {header} ") == 1, header
        assert f'proxy_set_header {header} "";' in body, header
    assert "$osprey_auth_" not in body
    assert "$cookie_" not in body


@pytest.mark.parametrize("index", [0, 1])
def test_the_exemption_allows_get_only_and_no_upgrade(index):
    _, body = _bodies()[index]
    assert re.search(r"limit_except GET \{\s*deny all;\s*\}", body)
    assert "Upgrade" not in body
    assert "$connection_upgrade" not in body


@pytest.mark.parametrize("index", [0, 1])
def test_the_exemption_names_no_header_twice(index):
    _, body = _bodies()[index]
    names = re.findall(r"proxy_set_header (\S+) ", body)
    assert len(names) == len({n.lower() for n in names}), names


# ---------------------------------------------------------------------------
# Auth on: the C-CAP exemption
# ---------------------------------------------------------------------------

_CAP = "19a3c4f20.AbC-dEf_gHiJkLmN.Zz09-_aBcDeFgHiJkLmNoPqRsTuVwXyZ01"


def _cap_location_for(locations: dict[str, str], user: str) -> tuple[str, str]:
    matches = [
        (h, b)
        for h, b in _cap_locations(locations).items()
        if h.startswith(f"~ ^{URL_MOUNT_ROOT}/{user}/")
    ]
    assert len(matches) == 1, f"expected one cap location for {user}: {matches}"
    return matches[0]


def _cap_paths(user: str) -> list[str]:
    root = f"{URL_MOUNT_ROOT}/{user}"
    return [
        f"{root}/panel-static/agent-panel/{PANEL_CAP_SEGMENT}/{_CAP}/",
        f"{root}/panel-static/agent-panel/{PANEL_CAP_SEGMENT}/{_CAP}/js/app.js",
        f"{root}/panel-static/agent-panel/{PANEL_CAP_SEGMENT}/{_CAP}/data.json",
        f"{root}/panel/agent-dash/{PANEL_CAP_SEGMENT}/{_CAP}/",
        f"{root}/panel/agent-dash/{PANEL_CAP_SEGMENT}/{_CAP}/api/data",
        f"{root}/panel/agent-dash/{PANEL_CAP_SEGMENT}/{_CAP}/ws",
    ]


#: Every ``/panel*`` shape a browser can ask a user's mount for: the C-ASSETS
#: paths, the C-CAP paths, and everything else, which must stay gated.
def _panel_paths(user: str) -> list[str]:
    root = f"{URL_MOUNT_ROOT}/{user}"
    others = [
        f"{root}/panel",
        f"{root}/panel/",
        f"{root}/panel/agent-dash",
        f"{root}/panel/agent-dash/",
        f"{root}/panel/agent-dash/index.html",
        f"{root}/panel/agent-dash/{PANEL_CAP_SEGMENT}/{_CAP}",
        f"{root}/panel/agent-dash/{PANEL_CAP_SEGMENT}/",
        f"{root}/panel/agent-dash/x/{PANEL_CAP_SEGMENT}/{_CAP}/y",
        f"{root}/panel/.x/{PANEL_CAP_SEGMENT}/{_CAP}/y",
        f"{root}/panel/agent-dash/~d/{_CAP}/y",
        f"{root}/panel-static",
        f"{root}/panel-static/agent-panel/",
        f"{root}/panel-static/agent-panel/index.html",
        f"{root}/panel-static/agent-panel/{PANEL_CAP_SEGMENT}/{_CAP}",
        f"{root}/panel-staticx/agent-panel/{PANEL_CAP_SEGMENT}/{_CAP}/y",
        f"{root}/panels/agent-panel/{PANEL_CAP_SEGMENT}/{_CAP}/y",
        f"{root}/panel/jupyter/files/out.html",
        f"{root}/panel/jupyter/api/kernels",
        f"{root}/panel/artifacts/files/abc/report.html",
        f"{root}/panel/artifacts/api/markdown/abc/rendered",
        f"{root}/panel/my-dash/static/js/app.js",
    ]
    return [*_c_assets_paths(user), *_cap_paths(user), *_gated_paths(user), *others]


def _app_path(path: str, user: str) -> str:
    """What the app sees for ``path`` once the location strips the user's mount."""
    return path.removeprefix(f"{URL_MOUNT_ROOT}/{user}")


@pytest.mark.parametrize("user", ["alice", "bob"])
def test_every_panel_path_is_exempt_only_where_the_app_exempts_it(user):
    """No ``/panel*`` path skips ``auth_request`` unless C-ASSETS or C-CAP covers it.

    The app's own predicates decide: a path the C-ASSETS location matches is one
    :func:`is_panel_asset_path` exempts, a path the C-CAP location matches is
    one :func:`is_panel_cap_path` exempts, and every other path falls to the
    gated location.
    """
    locations = _locations(_config(_PASSWORD))
    assets = _pattern(_asset_location_for(locations, user)[0])
    caps = _pattern(_cap_location_for(locations, user)[0])
    for path in _panel_paths(user):
        app_path = _app_path(path, user)
        assert bool(assets.search(path)) == is_panel_asset_path(app_path) or app_path.startswith(
            ("/design-system/", "/static/fonts/")
        ), path
        assert bool(caps.search(path)) == is_panel_cap_path(app_path), path
        assert not (assets.search(path) and caps.search(path)), path


@pytest.mark.parametrize("user", ["alice", "bob"])
def test_the_cap_exemption_matches_every_cap_path_and_no_gated_one(user):
    header, _ = _cap_location_for(_locations(_config(_PASSWORD)), user)
    pattern = _pattern(header)
    for path in _cap_paths(user):
        assert pattern.search(path), path
    for path in [*_gated_paths(user), *_c_assets_paths(user)]:
        assert not pattern.search(path), path


def test_one_users_cap_exemption_never_matches_another_users_mount():
    locations = _locations(_config(_PASSWORD))
    alice, _ = _cap_location_for(locations, "alice")
    for path in _cap_paths("bob"):
        assert not _pattern(alice).search(path), path


def test_the_cap_exemption_targets_the_users_own_upstream():
    locations = _locations(_config(_PASSWORD))
    for user in ("alice", "bob"):
        _, body = _cap_location_for(locations, user)
        gated = locations[f"{URL_MOUNT_ROOT}/{user}/"]
        upstream = re.search(r"proxy_pass (http://[^/;]+)/;", gated).group(1)
        assert f"proxy_pass {upstream};" in body
        assert f"rewrite ^{URL_MOUNT_ROOT}/{user}(/.*)$ $1 break;" in body


@pytest.mark.parametrize("user", ["alice", "bob"])
def test_the_cap_exemption_carries_no_credential_upstream(user):
    header, body = _cap_location_for(_locations(_config(_PASSWORD)), user)
    assert "auth_request" not in body, header
    assert "include " not in body, header
    assert "error_page 401" not in body, header
    assert TERMINAL_SECRET_HEADER not in body, header
    for name in (*_IDENTITY_HEADERS, "Authorization", "Cookie"):
        assert body.count(f"proxy_set_header {name} ") == 1, name
        assert f'proxy_set_header {name} "";' in body, name
    assert "$osprey_auth_" not in body
    assert "$cookie_" not in body


@pytest.mark.parametrize("user", ["alice", "bob"])
def test_the_cap_exemption_forwards_only_the_headers_it_names(user):
    """``proxy_pass_request_headers off`` drops the secret header unspelled."""
    _, body = _cap_location_for(_locations(_config(_PASSWORD)), user)
    assert "proxy_pass_request_headers off;" in body
    for name in ("Content-Type", "Accept", "Range", "If-None-Match", "If-Modified-Since"):
        assert f"proxy_set_header {name} $" in body, name


@pytest.mark.parametrize("user", ["alice", "bob"])
def test_the_cap_exemption_relays_websockets(user):
    """A runtime panel's websocket to its own sidecar travels under its cap."""
    _, body = _cap_location_for(_locations(_config(_PASSWORD)), user)
    assert "proxy_set_header Upgrade $http_upgrade;" in body
    assert "proxy_set_header Connection $connection_upgrade;" in body
    for name in ("Key", "Version", "Protocol", "Extensions"):
        assert f"proxy_set_header Sec-WebSocket-{name} $http_sec_websocket_{name.lower()};" in body
    assert "limit_except" not in body
    names = re.findall(r"proxy_set_header (\S+) ", body)
    assert len(names) == len({n.lower() for n in names}), names


# ---------------------------------------------------------------------------
# Auth off: nothing to be exempt from, nothing rendered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("auth", [None, _OPEN], ids=["token", "open"])
def test_no_exemption_without_an_auth_request(auth):
    locations = _locations(_config(auth))
    assert _exempt_locations(locations) == {}
    assert not any("auth_request" in body for body in locations.values())
