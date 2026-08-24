"""Tests for multi-user web-terminal artifact rendering (osprey.deployment.web_terminals.render)."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from jinja2 import Environment, Template

from osprey.deployment.web_terminals.artifacts import web_artifacts_dir
from osprey.deployment.web_terminals.auth_credentials import AUTH_ENV_FILENAME
from osprey.deployment.web_terminals.personas import env_var_suffix
from osprey.deployment.web_terminals.ports import (
    PANEL_ENV_VARS,
    allocate_ports,
    base_ports_from_config,
)
from osprey.deployment.web_terminals.render import (
    AUTH_ENV_DIGEST_LABEL,
    TERMINAL_SECRET_HEADER,
    _auth_tls_context,
    clear_nginx_templates_dir,
    deployment_external_origin,
    render_web_terminals,
    terminal_secret_env_var,
)

# The sidecar's own env-var names, imported rather than spelled out: the compose
# service and the app factory are the two ends of one contract, and a rename on
# either side must fail here instead of silently rendering a variable nothing reads.
from osprey.services.auth_sidecar.app import (
    ENV_EXTERNAL_ORIGIN,
    ENV_METHOD,
    ENV_OIDC_CLIENT_ID_ENV,
    ENV_OIDC_CLIENT_SECRET_ENV,
    ENV_OIDC_ISSUER,
    ENV_OIDC_SUBJECT_PREFIX,
    ENV_PW_HASH_PREFIX,
    ENV_SESSION_LIFETIME,
    ENV_SESSION_SECRET,
    ENV_STATE_SECRET,
    ENV_TLS_ENABLED,
    ENV_USERS,
)
from osprey.utils.workspace import agent_data_base_dir

# The four classic config-set families; the effective per-family base set the
# render actually allocates from also carries every registry default
# (channel_finder, okf, ...) — resolved once here so `allocate_ports(_BASE_PORTS,
# i)` calls throughout this module see the same full set the render does.
_CONFIGURED_BASE_PORTS = {"web": 9091, "artifact": 9291, "ariel": 9391, "lattice": 9491}
_BASE_PORTS = base_ports_from_config(
    {f"{family}_base_port": port for family, port in _CONFIGURED_BASE_PORTS.items()}
)


def _config(users: list[str], groups: list[dict] | None = None) -> dict:
    """Build a minimal-but-complete facility config exercising every field
    render_web_terminals() reads."""
    web_terminals: dict = {
        "enabled": True,
        "nginx_port": 9080,
        "web_base_port": _CONFIGURED_BASE_PORTS["web"],
        "artifact_base_port": _CONFIGURED_BASE_PORTS["artifact"],
        "ariel_base_port": _CONFIGURED_BASE_PORTS["ariel"],
        "lattice_base_port": _CONFIGURED_BASE_PORTS["lattice"],
        "users": users,
    }
    if groups is not None:
        web_terminals["landing"] = {"groups": groups}
    return {
        "facility": {
            "name": "Demo Light Source",
            "prefix": "dls",
            "timezone": "America/Los_Angeles",
        },
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {"web_terminals": web_terminals},
    }


_MULTI_USER_CONFIG = _config(["alice", "bob", "carol"])


def test_render_returns_exactly_three_artifacts() -> None:
    """render_web_terminals() produces the compose overlay, nginx fragment, and landing page."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


def test_render_is_deterministic() -> None:
    """Same config in -> byte-identical output out, twice."""
    # Arrange
    config_a = copy.deepcopy(_MULTI_USER_CONFIG)
    config_b = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    first = render_web_terminals(config_a)
    second = render_web_terminals(config_b)

    # Assert
    assert first == second


def test_compose_parses_as_yaml_with_one_service_and_one_volume_pair_per_user() -> None:
    """The compose overlay is valid YAML: one web-<user> service + nginx, one volume pair per user."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert set(compose["services"].keys()) == {"nginx", *(f"web-{u}" for u in users)}
    assert set(compose["volumes"].keys()) == {
        volume for u in users for volume in (f"{u}-claude-config", f"{u}-agent-data")
    }


def test_nginx_image_defaults_when_config_omits_it() -> None:
    """With no `nginx_image` in config, the nginx service uses the public default tag."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "nginx_image" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert compose["services"]["nginx"]["image"] == "nginx:1.27-alpine"


def test_nginx_image_custom_value_lands_on_the_nginx_service() -> None:
    """A configured `nginx_image` overrides the nginx service image; hosts that pull
    only from a private mirror point it at their own registry."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    custom = "registry.example.com:5050/mirrors/nginx:1.27-alpine"
    config["modules"]["web_terminals"]["nginx_image"] = custom

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert compose["services"]["nginx"]["image"] == custom


def test_compose_service_env_has_terminal_user_and_8087_constant_per_service() -> None:
    """Each per-user service sets OSPREY_TERMINAL_USER, and the 8087 web-internal
    default is referenced (in the healthcheck commentary) once per service."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    compose_text = artifacts["docker-compose.web.yml"]
    compose = yaml.safe_load(compose_text)

    # Assert
    for user in users:
        env = compose["services"][f"web-{user}"]["environment"]
        assert f"OSPREY_TERMINAL_USER={user}" in env
    # The healthcheck commentary mentions the fixed 8087 default twice per
    # per-user service block (docker-compose.web.yml.j2's healthcheck comment).
    assert compose_text.count("8087") == 2 * len(users)


def test_compose_ports_are_non_colliding_families_matching_allocate_ports() -> None:
    """Every user's ports — one per derived family — must match allocate_ports()
    exactly and never collide. Iterates PANEL_ENV_VARS (registry-derived), so a
    newly registered companion server is asserted here without editing this test."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]
    effective_base_ports = base_ports_from_config(config["modules"]["web_terminals"])

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    seen_ports: set[int] = set()
    for index, user in enumerate(users):
        expected = allocate_ports(effective_base_ports, index)
        env = compose["services"][f"web-{user}"]["environment"]
        env_map = dict(item.split("=", 1) for item in env if "=" in item)
        actual = {"web": int(env_map["OSPREY_WEB_PORT"])}
        for family, env_var in PANEL_ENV_VARS.items():
            actual[family] = int(env_map[env_var])
        assert actual == expected
        for port in actual.values():
            assert port not in seen_ports, f"port {port} collides across users"
            seen_ports.add(port)


def test_per_user_services_bind_loopback_not_0_0_0_0() -> None:
    """C3: each per-user service's four app processes bind 127.0.0.1 (loopback),
    never 0.0.0.0 — nginx (bind-nginx-reverse-proxy) becomes the only off-host
    path. `network_mode: host` is retained (EPICS CA UDP broadcast)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    compose_text = artifacts["docker-compose.web.yml"]
    compose = yaml.safe_load(compose_text)

    # Assert
    assert "0.0.0.0" not in compose_text
    assert compose["services"]["nginx"]["network_mode"] == "host"
    for user in users:
        service = compose["services"][f"web-{user}"]
        assert service["network_mode"] == "host"
        env = dict(item.split("=", 1) for item in service["environment"] if "=" in item)
        assert env["OSPREY_TERMINAL_BIND_HOST"] == "127.0.0.1"


def test_nginx_fragment_has_one_routing_location_per_user() -> None:
    """nginx.conf.j2 emits one `location /u/<user>/` proxy block per configured user."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "server {" in nginx_conf
    assert f"listen {config['modules']['web_terminals']['nginx_port']};" in nginx_conf
    for user in users:
        assert f"location /u/{user}/ {{" in nginx_conf


def test_nginx_fragment_has_proxy_pass_to_each_users_loopback_web_upstream() -> None:
    """Each `location /u/<user>/` block proxy_passes to that user's own loopback
    `web` port (bind-per-user-apps-loopback) — not a redirect to a distinct origin."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "return 302" not in nginx_conf  # Phase-1 redirect scheme is gone
    for index in range(len(users)):
        expected_port = allocate_ports(_BASE_PORTS, index)["web"]
        assert f"proxy_pass http://127.0.0.1:{expected_port}/;" in nginx_conf


def test_nginx_fragment_has_websocket_upgrade_machinery() -> None:
    """The `/ws/terminal` and panel WebSocket handshakes need proxy_http_version 1.1
    plus the Upgrade/Connection headers, backed by a `map $http_upgrade
    $connection_upgrade` directive declared ONCE at http/top-level (NOT nested
    inside `server{}` — a `map` directive there would fail `nginx -t`)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert nginx_conf.count("map $http_upgrade $connection_upgrade") == 1
    map_index = nginx_conf.index("map $http_upgrade $connection_upgrade")
    server_index = nginx_conf.index("server {")
    assert map_index < server_index, "map{} must sit at http/top-level, before server{}"

    assert "proxy_http_version 1.1;" in nginx_conf
    assert "proxy_set_header Upgrade $http_upgrade;" in nginx_conf
    assert "proxy_set_header Connection $connection_upgrade;" in nginx_conf


def test_nginx_fragment_proxy_disables_buffering_and_raises_read_timeout_for_sse() -> None:
    """Every request under `/u/<user>/` flows through nginx as a proxy, not a
    redirect, so nginx is genuinely in the data path and its
    DEFAULT proxy buffering and 60s read timeout would apply to the app's
    heartbeat-less Server-Sent-Events streams (`/api/files/events`, chat SSE) too
    — batching/delaying `data:` lines behind nginx's buffer and tearing down an
    idle SSE connection every 60s. Each proxied `/u/<user>/` location must
    disable buffering and raise the read timeout well above that default.
    WebSocket traffic is unaffected (it bypasses proxy buffering once the
    Upgrade handshake completes), so this doesn't undo the map/upgrade-header
    machinery covered by the sibling test above."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert — one occurrence per proxied user location (not a stray global
    # setting), and the timeout is meaningfully raised, not left at nginx's
    # 60s default.
    assert nginx_conf.count("proxy_buffering off;") == len(users)
    timeout_matches = re.findall(r"proxy_read_timeout (\d+)s;", nginx_conf)
    assert len(timeout_matches) == len(users)
    for value in timeout_matches:
        assert int(value) > 60, "proxy_read_timeout must be raised above nginx's 60s default"


def test_nginx_fragment_has_trailing_slash_redirect_per_user() -> None:
    """A no-trailing-slash `/u/<user>` bookmark 301-redirects into `/u/<user>/`
    rather than silently falling through to the landing page at `/`."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    for user in users:
        assert f"location = /u/{user} {{" in nginx_conf
        assert f"return 301 /u/{user}/;" in nginx_conf


def test_landing_renders_one_card_per_user_in_users_group() -> None:
    """The auto-populated `users` group renders one landing card per configured user."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert landing_html.count('class="landing-card-label"') == len(users)
    for user in users:
        assert f">{user}<" in landing_html
        assert f'href="/u/{user}/"' in landing_html


def test_landing_zero_users_suppresses_users_group_and_shows_empty_state() -> None:
    """0-user config: no cards render, and the generic empty-state message shows."""
    # Arrange
    config = copy.deepcopy(_config([]))

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert 'class="landing-card-label"' not in landing_html
    assert "No terminals or links are configured yet." in landing_html


def test_landing_single_user_renders_exactly_one_card() -> None:
    """1-user config: exactly one card renders, no empty-state message."""
    # Arrange
    config = copy.deepcopy(_config(["solo"]))

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert landing_html.count('class="landing-card-label"') == 1
    assert "No terminals or links are configured yet." not in landing_html


def test_landing_escapes_html_special_characters_in_user_and_link_labels() -> None:
    """Unsafe characters in a user name or link label must be escaped, never raw."""
    # Arrange
    unsafe_user = "<script>alert('u')</script>"
    groups = [
        {"type": "users"},
        {
            "type": "links",
            "label": "Facility <Tools> & More",
            "links": [{"label": "Elog & Status", "url": "https://elog.example.org"}],
        },
    ]
    config = copy.deepcopy(_config([unsafe_user], groups=groups))

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert unsafe_user not in landing_html
    assert "&lt;script&gt;" in landing_html
    assert "Facility <Tools> & More" not in landing_html
    assert "Facility &lt;Tools&gt; &amp; More" in landing_html
    assert "Elog &amp; Status" in landing_html


def test_landing_transform_renders_links_group_items() -> None:
    """A config `links` group's items must appear as landing cards with their own URLs."""
    # Arrange
    groups = [
        {"type": "users"},
        {
            "type": "links",
            "label": "Facility Tools",
            "links": [
                {"label": "Elog", "url": "https://elog.example.org"},
                {"label": "Status Page", "url": "https://status.example.org"},
            ],
        },
    ]
    config = copy.deepcopy(_config(["alice"], groups=groups))

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert "Facility Tools" in landing_html
    assert ">Elog<" in landing_html
    assert 'href="https://elog.example.org"' in landing_html
    assert ">Status Page<" in landing_html
    assert 'href="https://status.example.org"' in landing_html
    # one users-group card (alice) + two links-group cards
    assert landing_html.count('class="landing-card-label"') == 3


# ---------------------------------------------------------------------------
# Landing sections for standalone-service personas (`landing_group`)
#
# A roster is not always a list of people: a deployment may run a whole second
# product behind its own login (the shipped control-assistant stack runs the
# ARIEL logbook assistant beside its two operator tiers). A persona declaring
# `landing_group` files its users under their own landing heading, drawn as a
# `variant: "tray"` section, so the page reads people first and services after.
# Presentation only — nothing about the container, its ports or its routes
# moves.
# ---------------------------------------------------------------------------


def _body(landing_html: str) -> str:
    """The page below its inline stylesheet.

    Class-absence assertions must not be answered by the stylesheet, which
    always defines every rule the page could use regardless of what this render
    actually emitted.
    """
    return landing_html.split("</style>", 1)[1]


def _sections(landing_html: str) -> list[tuple[str, str]]:
    """Every rendered landing section as ``(class attribute, heading text)``.

    Parsed out of the markup rather than asserted against ``_build_groups``'s
    return value, so these tests cover the template's own use of ``variant``
    (the class it chooses) and not just the dict that feeds it.
    """
    return re.findall(
        r'<section class="([^"]+)">\s*<h2 class="landing-group-label">([^<]*)</h2>',
        landing_html,
    )


def _cards_in_section(landing_html: str, heading: str) -> list[str]:
    """The card labels under one section heading, in render order."""
    body = landing_html.split(f">{heading}</h2>", 1)[1].split("</section>", 1)[0]
    return re.findall(r'<span class="landing-card-label">([^<]*)</span>', body)


def _roster_config(users: list[dict], personas: dict, groups: list[dict] | None = None) -> dict:
    """A config whose roster resolves against a persona catalog.

    ``image_source`` stays at its registry default so the catalog needs no
    per-persona project paths on disk — these tests are about the landing page,
    not about image resolution.
    """
    config = copy.deepcopy(_config(users, groups=groups))
    config["modules"]["web_terminals"]["personas"] = personas
    return config


_OPERATOR_CATALOG = {
    "readonly": {"project": "dls-readonly", "build_profile": "readonly"},
    "readwrite": {"project": "dls-readwrite", "build_profile": "readwrite"},
    "ariel": {
        "project": "dls-ariel",
        "build_profile": "ariel",
        "landing_group": "Standalone deployments",
    },
}

_OPERATOR_ROSTER = [
    {"name": "alice", "index": 0, "persona": "readwrite"},
    {"name": "bob", "index": 1, "persona": "readonly"},
    {"name": "ariel", "index": 2, "persona": "ariel"},
]


def test_landing_group_persona_lifts_its_users_into_their_own_tray_section() -> None:
    """The declaring persona's users leave the roster section for a tray of their own.

    This is the whole feature in one assertion: alice and bob stay under the
    roster heading, ariel moves below into a section headed by the persona's
    `landing_group`, and only that second section carries the tray class.
    """
    # Arrange
    config = _roster_config(_OPERATOR_ROSTER, _OPERATOR_CATALOG)

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    assert _sections(landing_html) == [
        ("landing-group", "Terminals"),
        ("landing-group landing-tray", "Standalone deployments"),
    ]
    assert _cards_in_section(landing_html, "Terminals") == ["alice", "bob"]
    assert _cards_in_section(landing_html, "Standalone deployments") == ["ariel"]


def test_landing_group_changes_nothing_but_the_landing_page() -> None:
    """`landing_group` is presentation: the compose overlay and nginx fragment
    for the same roster are byte-identical with and without it.

    The key names a landing section, and a reader could reasonably fear it also
    moves the container somewhere — this pins that it does not touch the image,
    the ports, the volumes or the proxy route.
    """
    # Arrange
    plain_catalog = copy.deepcopy(_OPERATOR_CATALOG)
    del plain_catalog["ariel"]["landing_group"]
    with_group = _roster_config(_OPERATOR_ROSTER, _OPERATOR_CATALOG)
    without_group = _roster_config(_OPERATOR_ROSTER, plain_catalog)

    # Act
    grouped = render_web_terminals(with_group)
    ungrouped = render_web_terminals(without_group)

    # Assert
    assert grouped["docker-compose.web.yml"] == ungrouped["docker-compose.web.yml"]
    assert grouped["nginx/nginx.conf"] == ungrouped["nginx/nginx.conf"]
    # ...and the landing page is the ONLY artifact that did move.
    assert grouped["nginx/landing.html"] != ungrouped["nginx/landing.html"]


def test_landing_without_any_landing_group_renders_no_tray() -> None:
    """A catalog declaring none renders exactly one flat section, as before.

    The additivity guard: every deployment predating this key must keep its
    current landing page, so the tray class must not appear anywhere.
    """
    # Arrange
    catalog = copy.deepcopy(_OPERATOR_CATALOG)
    del catalog["ariel"]["landing_group"]
    config = _roster_config(_OPERATOR_ROSTER, catalog)

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    assert _sections(landing_html) == [("landing-group", "Terminals")]
    # Body only: the stylesheet always carries the `.landing-tray` RULE; what
    # must be absent is any element wearing the class.
    assert "landing-tray" not in _body(landing_html)


def test_landing_users_section_label_is_configurable() -> None:
    """`{type: users, label: ...}` renames the section holding the ungrouped users."""
    # Arrange
    config = _roster_config(
        _OPERATOR_ROSTER, _OPERATOR_CATALOG, groups=[{"type": "users", "label": "Users"}]
    )

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    assert _sections(landing_html) == [
        ("landing-group", "Users"),
        ("landing-group landing-tray", "Standalone deployments"),
    ]


def test_landing_tray_sections_follow_first_appearance_roster_order() -> None:
    """Two distinct `landing_group`s order by where each is first seen in the roster.

    Ordering has to come from the config rather than from dict iteration
    accident, so an operator can predict the page from the roster alone. The
    catalog below deliberately declares the two personas in the OPPOSITE order
    to the roster.
    """
    # Arrange
    catalog = {
        "dashboards": {"project": "dls-dash", "landing_group": "Dashboards"},
        "ariel": {"project": "dls-ariel", "landing_group": "Standalone deployments"},
        "readonly": {"project": "dls-readonly"},
    }
    roster = [
        {"name": "bob", "index": 0, "persona": "readonly"},
        {"name": "ariel", "index": 1, "persona": "ariel"},
        {"name": "lattice", "index": 2, "persona": "dashboards"},
    ]
    config = _roster_config(roster, catalog)

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    assert _sections(landing_html) == [
        ("landing-group", "Terminals"),
        ("landing-group landing-tray", "Standalone deployments"),
        ("landing-group landing-tray", "Dashboards"),
    ]


def test_landing_two_users_of_one_grouped_persona_share_a_single_tray() -> None:
    """A `landing_group` is a section, not a card: two users of the same persona
    land in one tray rather than opening two identically-headed ones."""
    # Arrange
    roster = [
        {"name": "ariel", "index": 0, "persona": "ariel"},
        {"name": "ariel-archive", "index": 1, "persona": "ariel"},
    ]
    config = _roster_config(roster, _OPERATOR_CATALOG)

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    assert [cls for cls, _ in _sections(landing_html)].count("landing-group landing-tray") == 1
    assert _cards_in_section(landing_html, "Standalone deployments") == [
        "ariel",
        "ariel-archive",
    ]


def test_landing_group_heading_is_html_escaped() -> None:
    """The heading is operator-supplied config, so it is escaped like every other
    interpolated string on this page — it reaches an element body, not an
    attribute, but it reaches the document either way."""
    # Arrange
    catalog = copy.deepcopy(_OPERATOR_CATALOG)
    catalog["ariel"]["landing_group"] = "<script>alert('g')</script>"
    config = _roster_config(_OPERATOR_ROSTER, catalog)

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    assert "<script>alert" not in landing_html
    assert "&lt;script&gt;" in landing_html


def test_landing_persona_badge_is_dropped_when_it_only_repeats_the_user_name() -> None:
    """A card reading `ariel` over a badge reading `ARIEL` says nothing twice.

    The badge names which tier a login belongs to, so it earns its place only
    when it differs from the label — the case a single-tenant service persona
    (roster entry and persona named after the same thing) never hits, and a
    people roster always does.
    """
    # Arrange
    config = _roster_config(_OPERATOR_ROSTER, _OPERATOR_CATALOG)

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    badges = re.findall(r'<span class="landing-card-sublabel">([^<]*)</span>', landing_html)
    assert badges == ["readwrite", "readonly"]


def test_landing_persona_badge_is_dropped_case_insensitively() -> None:
    """The badge renders uppercase, so `Ariel`/`ariel` would read as duplicated too."""
    # Arrange
    catalog = {"Ariel": {"project": "dls-ariel", "landing_group": "Standalone deployments"}}
    config = _roster_config([{"name": "ariel", "index": 0, "persona": "Ariel"}], catalog)

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    assert 'class="landing-card-sublabel"' not in landing_html


def test_landing_unrecognized_variant_falls_back_to_the_plain_section() -> None:
    """The template matches `variant` against the literal "tray", not truthiness.

    `_build_groups` emits only "tray" today, so this drives the template
    directly: a future caller adding a second variant must add its stylesheet
    rules deliberately rather than inherit the tray's by accident.
    """
    # Arrange
    groups = [{"label": "Odd", "items": [{"label": "x", "url": "/u/x/"}], "variant": "carousel"}]

    # Act
    landing_html = _render_landing_template(groups)

    # Assert
    assert _sections(landing_html) == [("landing-group", "Odd")]
    assert "landing-tray" not in _body(landing_html)


def _render_landing_template(groups: list[dict]) -> str:
    """Render landing.html.j2 with an arbitrary ``groups`` context.

    Loads the template through the same package path and autoescape setting
    ``render_web_terminals`` uses, so what is exercised is the shipped file
    under its real escaping rules — the only way to reach a ``groups`` shape
    ``_build_groups`` cannot currently produce.
    """
    from importlib.resources import as_file, files

    from jinja2 import Environment, FileSystemLoader

    from osprey.deployment.web_terminals.render import (
        _LANDING_TEMPLATE,
        _TEMPLATE_PACKAGE_PATH,
        _landing_theme_blocks,
    )

    template_dir = files("osprey").joinpath(_TEMPLATE_PACKAGE_PATH)
    with as_file(template_dir) as template_path:
        env = Environment(loader=FileSystemLoader(str(template_path)), autoescape=True)
        return env.get_template(_LANDING_TEMPLATE).render(
            facility_name="Demo Light Source",
            groups=groups,
            theme_blocks=_landing_theme_blocks({}),
        )


def test_render_missing_deploy_fqdn_raises_when_users_configured() -> None:
    """landing_url can't be resolved without deploy.fqdn once at least one user exists."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    del config["deploy"]["fqdn"]

    # Act / Assert
    with pytest.raises(ValueError, match="deploy.fqdn"):
        render_web_terminals(config)


def test_render_missing_deploy_fqdn_is_fine_with_zero_users() -> None:
    """landing_url is never needed when there are no per-user services to bake it into."""
    # Arrange
    config = copy.deepcopy(_config([]))
    del config["deploy"]["fqdn"]

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "docker-compose.web.yml" in artifacts


# ---------------------------------------------------------------------------
# Task 1.3: one external-origin derivation behind landing_url and OIDC
# ---------------------------------------------------------------------------


def _tls_config(users: list[str] | None = None) -> dict:
    """A config with TLS terminated at nginx (cert/key set, as the render demands)."""
    config = copy.deepcopy(_config(users) if users is not None else _MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }
    return config


def _capture_render_contexts(config: dict) -> dict[str, dict]:
    """Render `config`, capturing the Jinja context each template was given.

    `external_origin` is exported for templates that don't consume it yet (the
    nginx TLS-redirect block and the compose sidecar service land later), so the
    context itself is the only place its export can be pinned today.
    """
    captured: dict[str, dict] = {}

    def _record(self, **context):  # type: ignore[no-untyped-def]
        captured[self.name] = context
        return ""

    with patch.object(Template, "render", _record):
        render_web_terminals(config)
    return captured


def test_external_origin_without_tls_is_plain_http_on_the_published_nginx_port() -> None:
    """No TLS: nginx publishes one plain port, and that port is part of the origin."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    contexts = _capture_render_contexts(config)

    # Assert
    assert contexts["nginx.conf.j2"]["external_origin"] == "http://dls-deploy.dls.example.org:9080"


def test_external_origin_with_tls_is_https_with_443_left_implicit() -> None:
    """TLS on: the 443 server is the sole content server, and 443 is left implicit —
    the canonical origin serialization, and the form an OIDC client registration is
    normally written in (an IdP compares `redirect_uri` character for character)."""
    # Arrange
    config = _tls_config()

    # Act
    contexts = _capture_render_contexts(config)

    # Assert — the nginx_port never appears in a TLS origin
    assert contexts["nginx.conf.j2"]["external_origin"] == "https://dls-deploy.dls.example.org"


def test_external_origin_is_exported_to_both_the_nginx_and_compose_contexts() -> None:
    """One derivation, exported to every consumer: the nginx server blocks and the
    compose overlay (whose auth sidecar builds its OIDC redirect_uri from it)."""
    # Arrange
    config = _tls_config()

    # Act
    contexts = _capture_render_contexts(config)

    # Assert
    origin = "https://dls-deploy.dls.example.org"
    assert contexts["nginx.conf.j2"]["external_origin"] == origin
    assert contexts["docker-compose.web.yml.j2"]["external_origin"] == origin


def test_landing_url_is_the_external_origin_verbatim_under_tls() -> None:
    """The divergence guard this task exists for: `landing_url` is not assembled
    separately, so it cannot drift from the origin the OIDC redirect_uri uses."""
    # Arrange
    config = _tls_config()

    # Act
    contexts = _capture_render_contexts(config)
    compose_ctx = contexts["docker-compose.web.yml.j2"]

    # Assert
    assert compose_ctx["landing_url"] == compose_ctx["external_origin"]


def test_landing_url_baked_into_containers_follows_tls_into_https() -> None:
    """Regression: hardcoding OSPREY_TERMINAL_LANDING_URL to
    `http://<fqdn>:<nginx_port>` regardless of TLS ships every container in a TLS
    deployment a "back to landing" link on the plain origin — which under the TLS
    posture only ever redirects, and would never carry a Secure session cookie."""
    # Arrange
    config = _tls_config()

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    landing_env = [
        env
        for env in compose["services"]["web-alice"]["environment"]
        if env.startswith("OSPREY_TERMINAL_LANDING_URL=")
    ]
    assert landing_env == ["OSPREY_TERMINAL_LANDING_URL=https://dls-deploy.dls.example.org"]


def test_landing_url_without_tls_is_unchanged_from_today() -> None:
    """The plain-HTTP deployment (every current facility) renders byte-identically."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    assert (
        "OSPREY_TERMINAL_LANDING_URL=http://dls-deploy.dls.example.org:9080"
        in compose["services"]["web-alice"]["environment"]
    )


def test_external_origin_is_required_by_auth_even_with_an_empty_roster() -> None:
    """`deploy.fqdn` stops being users-only once auth is on: the sidecar needs an
    origin to build its redirect_uri from, so a roster-less auth deployment must fail
    loudly here rather than at the IdP with a redirect_uri mismatch."""
    # Arrange
    config = _tls_config(users=[])
    del config["deploy"]["fqdn"]
    config["modules"]["web_terminals"]["auth"] = {"method": "oidc"}

    # Act / Assert
    with pytest.raises(ValueError, match="deploy.fqdn"):
        render_web_terminals(config)


def test_render_missing_web_base_port_raises_value_error() -> None:
    """A user list that can't fully resolve allocate_ports() must fail loudly, not
    silently. Only `web` can be missing — every companion family carries a
    registry default (see test_render_missing_companion_base_port_uses_default)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    del config["modules"]["web_terminals"]["web_base_port"]

    # Act / Assert
    with pytest.raises(ValueError, match="web"):
        render_web_terminals(config)


def test_render_missing_companion_base_port_uses_registry_default() -> None:
    """A config omitting a companion family's base port (e.g. written before that
    panel existed) must render with the registry default, not fail — the
    zero-migration guarantee that keeps feature parity from breaking old configs."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    del config["modules"]["web_terminals"]["lattice_base_port"]

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert — lattice family renders from its registry default for every user.
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    default_base = FRAMEWORK_WEB_SERVERS["lattice_dashboard"].multi_user_base_port
    for index, user in enumerate(["alice", "bob", "carol"]):
        env = compose["services"][f"web-{user}"]["environment"]
        assert f"OSPREY_LATTICE_DASHBOARD_PORT={default_base + index}" in env


def test_removing_user_regenerates_without_their_service_route_and_volumes() -> None:
    """C8: dropping a user from users[] regenerates without their service/route/card/volume,
    while other users are unaffected. Phase-1 generation must never emit a directive that
    tears down a removed user's data — volume teardown is an explicit, documented Phase-3 step.
    """
    # Arrange — render the full roster, then regenerate with 'bob' removed
    full = render_web_terminals(_config(["alice", "bob", "carol"]))
    reduced = render_web_terminals(_config(["alice", "carol"]))

    full_compose = yaml.safe_load(full["docker-compose.web.yml"])
    reduced_compose = yaml.safe_load(reduced["docker-compose.web.yml"])

    # Assert — the removal is real (sanity: bob WAS present in the full render)
    assert "web-bob" in full_compose["services"]
    assert "bob-claude-config" in full_compose["volumes"]

    # bob is gone from every generated consumer (service, route, landing card, volumes)
    for path, content in reduced.items():
        assert "bob" not in content, f"removed user 'bob' still present in {path}"
    assert "web-bob" not in reduced_compose["services"]
    assert "bob-claude-config" not in reduced_compose.get("volumes", {})
    assert "bob-agent-data" not in reduced_compose.get("volumes", {})

    # the remaining users are untouched across all three artifacts
    for user in ("alice", "carol"):
        assert f"web-{user}" in reduced_compose["services"]
        assert f"{user}-claude-config" in reduced_compose["volumes"]
        assert f"/{user}" in reduced["nginx/nginx.conf"]

    # generation is pure artifact emission — it never issues a volume-destroying directive
    for content in reduced.values():
        assert "down -v" not in content
        assert "volume rm" not in content
        assert "volume prune" not in content


def test_object_form_users_render_with_explicit_index() -> None:
    """Object-form users (`{"name": ..., "index": ...}`) drive port allocation from
    their own `index` field, not their position in the list."""
    # Arrange
    config = copy.deepcopy(_config([{"name": "alice", "index": 5}]))

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    expected = allocate_ports(_BASE_PORTS, 5)
    env = dict(
        item.split("=", 1)
        for item in compose["services"]["web-alice"]["environment"]
        if "=" in item
    )
    assert int(env["OSPREY_WEB_PORT"]) == expected["web"]
    assert int(env["OSPREY_ARTIFACT_SERVER_PORT"]) == expected["artifact"]


def test_mixed_object_and_bare_users_render_together() -> None:
    """A roster mixing object-form and legacy bare-string entries renders both:
    the object-form entry uses its explicit index, the bare entry falls back to
    its position in the list."""
    # Arrange
    config = copy.deepcopy(_config([{"name": "alice", "index": 5}, "bob"]))

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert set(compose["services"].keys()) == {"nginx", "web-alice", "web-bob"}
    alice_env = dict(
        item.split("=", 1)
        for item in compose["services"]["web-alice"]["environment"]
        if "=" in item
    )
    bob_env = dict(
        item.split("=", 1) for item in compose["services"]["web-bob"]["environment"] if "=" in item
    )
    assert int(alice_env["OSPREY_WEB_PORT"]) == allocate_ports(_BASE_PORTS, 5)["web"]
    # 'bob' is the second list entry (position 1) -> falls back to positional index 1.
    assert int(bob_env["OSPREY_WEB_PORT"]) == allocate_ports(_BASE_PORTS, 1)["web"]
    assert ">alice<" in landing_html
    assert ">bob<" in landing_html


def test_legacy_bare_users_render_identically_to_before() -> None:
    """A bare-string-only roster is unaffected by the object-form parsing path:
    ports and landing cards still come from positional index, byte-for-byte the
    same as the pre-existing behavior covered elsewhere in this file."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    for index, user in enumerate(("alice", "bob", "carol")):
        expected = allocate_ports(_BASE_PORTS, index)
        env = dict(
            item.split("=", 1)
            for item in compose["services"][f"web-{user}"]["environment"]
            if "=" in item
        )
        assert int(env["OSPREY_WEB_PORT"]) == expected["web"]


def test_malformed_user_entries_are_dropped() -> None:
    """A non-string, non-well-formed-object entry is dropped rather than raising
    (well-formedness is lint's job, not render's) -- mirrors the prior silent-drop
    behavior for non-string entries."""
    # Arrange
    config = copy.deepcopy(
        _config([{"name": "alice", "index": 0}, {"name": "no-index"}, 42, None, "bob"])
    )

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert set(compose["services"].keys()) == {"nginx", "web-alice", "web-bob"}
    # 'bob' is the 5th raw list entry (position 4). A bare entry keeps its RAW
    # position rather than compacting over the dropped junk ahead of it -- pinned
    # here so the semantic is locked (raw position also prevents a bare fallback
    # from colliding with an object entry's explicit index in a mixed roster).
    bob_env = dict(
        item.split("=", 1) for item in compose["services"]["web-bob"]["environment"] if "=" in item
    )
    assert int(bob_env["OSPREY_WEB_PORT"]) == allocate_ports(_BASE_PORTS, 4)["web"]


# ---------------------------------------------------------------------------
# display_name -> OSPREY_WEB_APP_NAME (per-user window/tab title seam)
# ---------------------------------------------------------------------------


def _service_env(compose: dict, service: str) -> dict[str, str]:
    """Parse a compose service's `- KEY=value` environment list into a dict."""
    return dict(
        item.split("=", 1) for item in compose["services"][service]["environment"] if "=" in item
    )


def test_display_name_emits_app_name_env_line_only_for_the_user_that_sets_it() -> None:
    """A user's `display_name` renders an `OSPREY_WEB_APP_NAME` env line for that
    service; a user without one omits the line entirely (app falls back to config
    web.app_name)."""
    # Arrange
    config = copy.deepcopy(
        _config([{"name": "alice", "index": 0, "display_name": "Operations"}, "bob"])
    )

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert _service_env(compose, "web-alice")["OSPREY_WEB_APP_NAME"] == "Operations"
    assert "OSPREY_WEB_APP_NAME" not in _service_env(compose, "web-bob")


def test_no_display_name_anywhere_emits_no_app_name_env_line() -> None:
    """The common (bare-string) roster emits no OSPREY_WEB_APP_NAME line at all —
    the seam is inert until a user opts in."""
    # Act
    artifacts = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "OSPREY_WEB_APP_NAME" not in artifacts["docker-compose.web.yml"]


def test_display_name_with_spaces_and_colon_is_yaml_quoted_and_round_trips() -> None:
    """A `display_name` containing spaces and a `": "` (which would derail an
    unquoted compose `- KEY=value` scalar into a mapping) is safely quoted so the
    parsed env value is the exact original string."""
    # Arrange
    tricky = "Operations: Control Room"
    config = copy.deepcopy(_config([{"name": "alice", "index": 0, "display_name": tricky}]))

    # Act
    artifacts = render_web_terminals(config)
    compose_text = artifacts["docker-compose.web.yml"]
    compose = yaml.safe_load(compose_text)

    # Assert — the whole KEY=value is a single double-quoted YAML scalar, and it
    # parses back to the exact display_name (no embedded quotes, no split mapping).
    assert '- "OSPREY_WEB_APP_NAME=Operations: Control Room"' in compose_text
    assert _service_env(compose, "web-alice")["OSPREY_WEB_APP_NAME"] == tricky


def test_display_name_with_double_quote_is_escaped_and_round_trips() -> None:
    """A `display_name` containing a double quote is escaped inside the quoted YAML
    scalar rather than prematurely closing it."""
    # Arrange
    tricky = 'The "Main" Console'
    config = copy.deepcopy(_config([{"name": "alice", "index": 0, "display_name": tricky}]))

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert _service_env(compose, "web-alice")["OSPREY_WEB_APP_NAME"] == tricky


# ---------------------------------------------------------------------------
# Landing page: design-system token values baked from web.theme
# ---------------------------------------------------------------------------


def _landing_style(config: dict) -> str:
    """The landing page's inline <style> block, up to the first static rule."""
    html = render_web_terminals(config)["nginx/landing.html"]
    return html[html.index("<style>") : html.index("* { box-sizing")]


def _themed_config(theme: str | None) -> dict:
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    if theme is not None:
        config.setdefault("web", {})["theme"] = theme
    return config


def test_landing_bakes_the_configured_theme_family_in_both_modes() -> None:
    """A family-valued web.theme yields dark under :root plus a light media query,
    so the page follows the viewer's OS — the same thing the terminals behind it do
    when configured with a bare family."""
    # Act
    style = _landing_style(_themed_config("desy"))

    # Assert — DESY dark canvas under :root, DESY light behind the query
    assert "--bg-primary: #091720;" in style
    assert "@media (prefers-color-scheme: light)" in style
    assert "--bg-primary: #f3f8fc;" in style
    assert "color-scheme: dark;" in style
    assert "color-scheme: light;" in style


def test_landing_honors_a_pinned_mode_with_no_media_query() -> None:
    """A concrete-id web.theme pins the mode: one block, and no OS-following."""
    # Act
    style = _landing_style(_themed_config("desy-light"))

    # Assert
    assert "--bg-primary: #f3f8fc;" in style
    assert "color-scheme: light;" in style
    assert "prefers-color-scheme" not in style
    assert "color-scheme: dark;" not in style


def test_landing_defaults_to_the_main_family_when_web_theme_is_absent() -> None:
    """No `web` section at all still themes the page — from the main family."""
    # Act
    style = _landing_style(_themed_config(None))

    # Assert — main family's canvases, and still both modes
    assert "--bg-primary: #111217;" in style
    assert "--bg-primary: #f4f5f5;" in style


def test_landing_falls_back_on_an_unknown_web_theme() -> None:
    """A typo must not leave the page unstyled — it resolves like the terminals do,
    to the main family, and pins nothing."""
    # Act
    style = _landing_style(_themed_config("no-such-theme"))

    # Assert
    assert "--bg-primary: #111217;" in style
    assert "@media (prefers-color-scheme: light)" in style


def test_landing_carries_no_hardcoded_pre_token_palette() -> None:
    """The hand-written hexes this page used before the token system are gone.

    Guards the actual regression risk: a later edit reintroducing a literal
    colour would look fine on the default theme and be wrong on every other one.
    """
    # Act
    style = _landing_style(_themed_config("desy"))

    # Assert — the old teal accent and slate canvas
    assert "#4fd1c5" not in style
    assert "#0a0f1a" not in style


def test_landing_defines_every_variable_its_stylesheet_uses() -> None:
    """Every `var(--x)` in the page must be baked; a standalone file has no
    stylesheet to fall back to, so an undefined one renders as nothing."""
    # Arrange
    html = render_web_terminals(_themed_config("desy"))["nginx/landing.html"]

    # Act
    used = set(re.findall(r"var\((--[a-z0-9-]+)\)", html))
    defined = set(re.findall(r"^\s+(--[a-z0-9-]+):", html, re.MULTILINE))

    # Assert
    assert used, "expected the landing stylesheet to use custom properties"
    assert used <= defined, f"undefined custom properties: {sorted(used - defined)}"


def test_landing_theme_values_are_pattern_checked_before_inlining() -> None:
    """A value that could break out of <style> fails the render rather than
    shipping. HTML escaping would not catch a `</style>` sequence."""
    # Arrange
    config = _themed_config("desy")

    # Act / Assert
    with patch(
        "osprey.interfaces.design_system.theme_config.theme_css_variables",
        return_value={"--bg-primary": "#000</style><script>alert(1)</script>"},
    ):
        with pytest.raises(ValueError, match="not safe to inline"):
            render_web_terminals(config)


def test_landing_render_fails_loudly_on_a_renamed_token() -> None:
    """A token the page needs but the theme no longer defines must break the
    render, not leave a deployed page quietly off-palette."""
    # Arrange
    from osprey.interfaces.design_system.theme_config import MissingThemeVariableError

    config = _themed_config("desy")

    # Act / Assert
    with patch(
        "osprey.deployment.web_terminals.render._LANDING_THEME_VARIABLES",
        ("--bg-primary", "--no-such-token"),
    ):
        with pytest.raises(MissingThemeVariableError, match="no-such-token"):
            render_web_terminals(config)


# ---------------------------------------------------------------------------
# theme -> OSPREY_WEB_THEME (per-user default theme seam)
# ---------------------------------------------------------------------------


def test_theme_emits_theme_env_line_only_for_the_user_that_sets_it() -> None:
    """A user's `theme` renders an `OSPREY_WEB_THEME` env line for that service; a
    user without one omits the line entirely (app falls back to config web.theme)."""
    # Arrange
    config = copy.deepcopy(_config([{"name": "alice", "index": 0, "theme": "desy-light"}, "bob"]))

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert _service_env(compose, "web-alice")["OSPREY_WEB_THEME"] == "desy-light"
    assert "OSPREY_WEB_THEME" not in _service_env(compose, "web-bob")


def test_no_theme_anywhere_emits_no_theme_env_line() -> None:
    """The common (bare-string) roster emits no OSPREY_WEB_THEME line at all — the
    seam is inert until a user opts in."""
    # Act
    artifacts = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "OSPREY_WEB_THEME" not in artifacts["docker-compose.web.yml"]


def test_theme_and_display_name_are_independent_per_user() -> None:
    """Each optional per-user field is emitted on its own; setting one must not
    drag the other in or leave it out."""
    # Arrange
    config = copy.deepcopy(
        _config(
            [
                {"name": "alice", "index": 0, "theme": "desy"},
                {"name": "bob", "index": 1, "display_name": "Operations"},
            ]
        )
    )

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    alice, bob = _service_env(compose, "web-alice"), _service_env(compose, "web-bob")
    assert alice["OSPREY_WEB_THEME"] == "desy"
    assert "OSPREY_WEB_APP_NAME" not in alice
    assert bob["OSPREY_WEB_APP_NAME"] == "Operations"
    assert "OSPREY_WEB_THEME" not in bob


def test_theme_value_is_yaml_quoted_and_round_trips() -> None:
    """Quoted the same way as OSPREY_WEB_APP_NAME: a value containing a `": "`
    would otherwise derail the unquoted compose `- KEY=value` scalar into a
    mapping. Theme ids never contain one today — this guards the quoting itself,
    so a hand-written config can't break the compose file."""
    # Arrange
    tricky = "not: a theme"
    config = copy.deepcopy(_config([{"name": "alice", "index": 0, "theme": tricky}]))

    # Act
    artifacts = render_web_terminals(config)
    compose_text = artifacts["docker-compose.web.yml"]
    compose = yaml.safe_load(compose_text)

    # Assert
    assert '- "OSPREY_WEB_THEME=not: a theme"' in compose_text
    assert _service_env(compose, "web-alice")["OSPREY_WEB_THEME"] == tricky


def test_catalog_present_all_users_on_default_persona_is_byte_identical_image_and_mount() -> None:
    """A `personas` catalog can exist without moving any user off the default persona
    (no roster entry sets `persona:`, so every entry falls back to `default_persona`).
    resolve_personas()'s registry-mode default-persona branch must still produce the
    SAME unsuffixed `<registry_url>/web-terminal:latest` image and the SAME
    `/app/<facility_prefix>-assistant` agent-data mount root a no-catalog config
    produces — introducing a catalog is zero-migration until a user is actually
    reassigned to a non-default persona."""
    # Arrange
    baseline_config = copy.deepcopy(_MULTI_USER_CONFIG)
    catalog_config = copy.deepcopy(_MULTI_USER_CONFIG)
    catalog_config["modules"]["web_terminals"]["default_persona"] = "assistant"
    catalog_config["modules"]["web_terminals"]["personas"] = {
        "assistant": {
            "project": "dls-assistant",
            "project_path": "../dls-assistant",
            "build_profile": "profiles/assistant.yml",
        },
    }
    users = baseline_config["modules"]["web_terminals"]["users"]

    # Act
    baseline = render_web_terminals(baseline_config)
    catalog = render_web_terminals(catalog_config)
    baseline_compose = yaml.safe_load(baseline["docker-compose.web.yml"])
    catalog_compose = yaml.safe_load(catalog["docker-compose.web.yml"])

    # Assert
    for user in users:
        baseline_svc = baseline_compose["services"][f"web-{user}"]
        catalog_svc = catalog_compose["services"][f"web-{user}"]
        assert catalog_svc["image"] == baseline_svc["image"]
        assert catalog_svc["volumes"] == baseline_svc["volumes"]
        assert (
            catalog_svc["image"]
            == "git.dls.example.org:5050/physics/production/dls-profiles/web-terminal:latest"
        )
        assert f"{user}-agent-data:/app/dls-assistant/var/agent_data" in catalog_svc["volumes"]


def test_persona_extra_mounts_render_as_extra_per_user_volume_lines() -> None:
    """A persona's `extra_mounts` render as additional `volumes:` entries on every
    user of that persona, after the two default (claude-config, agent-data) mounts."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["default_persona"] = "assistant"
    web_terminals["personas"] = {
        "assistant": {
            "project": "dls-assistant",
            "project_path": "../dls-assistant",
            "build_profile": "profiles/assistant.yml",
        },
        "gui": {
            "project": "dls-gui",
            "project_path": "../dls-gui",
            "build_profile": "profiles/gui.yml",
            "extra_mounts": ["/opt/site-data:/app/site-data:ro", "shared-cache:/app/cache"],
        },
    }
    web_terminals["users"] = [
        {"name": "alice", "index": 0},  # default persona, no extra_mounts
        {"name": "bob", "index": 1, "persona": "gui"},
    ]

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert — bob (gui) carries the three default mounts plus the persona's two
    bob_volumes = compose["services"]["web-bob"]["volumes"]
    assert bob_volumes == [
        "bob-claude-config:/data/claude-config",
        "bob-agent-data:/app/dls-gui/var/agent_data",
        "./var/audit/bob:/app/dls-gui/var/audit",
        "/opt/site-data:/app/site-data:ro",
        "shared-cache:/app/cache",
    ]
    # alice (default persona, no extra_mounts) keeps exactly the three default mounts
    assert compose["services"]["web-alice"]["volumes"] == [
        "alice-claude-config:/data/claude-config",
        "alice-agent-data:/app/dls-assistant/var/agent_data",
        "./var/audit/alice:/app/dls-assistant/var/audit",
    ]


def test_no_extra_mounts_leaves_only_the_default_volume_lines() -> None:
    """A no-personas config (the zero-migration default) emits exactly the three
    default per-user volume lines (claude-config, agent-data, the per-user audit
    zone) — the extra_mounts loop adds nothing."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    for user in config["modules"]["web_terminals"]["users"]:
        assert compose["services"][f"web-{user}"]["volumes"] == [
            f"{user}-claude-config:/data/claude-config",
            f"{user}-agent-data:/app/dls-assistant/var/agent_data",
            f"./var/audit/{user}:/app/dls-assistant/var/audit",
        ]


def test_agent_data_volume_mounts_where_the_container_resolves_its_agent_data_root() -> None:
    """The mount target is read from `agent_data.base_dir`, not spelled as a literal.

    The container resolves its own agent-data root from that key (see
    ``workspace.resolve_agent_data_root``, anchored on the config's
    ``project_root`` — in-container, ``container_project_dir``). If the mount
    names a different path, the volume is mounted over an empty directory while
    memory, sessions and artifacts are written into the container's writable
    layer, and every user's data is discarded the next time the container is
    recreated. Nothing fails at mount time and nothing appears in any log, so
    this coupling is asserted against the shared reader's own answer rather
    than against a path repeated here.
    """
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    expected = f"/app/dls-assistant/{agent_data_base_dir(config)}"
    for user in config["modules"]["web_terminals"]["users"]:
        assert f"{user}-agent-data:{expected}" in compose["services"][f"web-{user}"]["volumes"]


def test_a_relocated_agent_data_base_dir_moves_the_volume_mount_with_it() -> None:
    """Relocating `agent_data.base_dir` moves the mount, because the agent moves too.

    The failure this guards is silent in exactly one direction: a project that
    relocates its agent-data root keeps rendering a mount at the OLD path, so
    the volume survives while holding nothing and the live data has no volume
    behind it.
    """
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["agent_data"] = {"base_dir": "state/agent"}

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    for user in config["modules"]["web_terminals"]["users"]:
        volumes = compose["services"][f"web-{user}"]["volumes"]
        assert f"{user}-agent-data:/app/dls-assistant/state/agent" in volumes
        assert not any("var/agent_data" in mount for mount in volumes)


def test_an_absolute_agent_data_base_dir_is_not_re_anchored_under_the_project_dir() -> None:
    """An absolute `base_dir` names the same absolute path on both sides.

    ``anchored_path`` returns an absolute configured path unchanged, so the
    container writes to it verbatim; concatenating it after the project
    directory would mount the volume at a path nothing writes to.
    """
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["agent_data"] = {"base_dir": "/srv/osprey-agent-data"}

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    for user in config["modules"]["web_terminals"]["users"]:
        volumes = compose["services"][f"web-{user}"]["volumes"]
        assert f"{user}-agent-data:/srv/osprey-agent-data" in volumes
        assert not any("/app/dls-assistant/srv" in mount for mount in volumes)


def test_nginx_mounts_resolve_into_the_repos_build_zone(tmp_path) -> None:
    """nginx's two mounts address the render output, under the pinned project directory.

    Every relative path in the rendered compose file resolves against the
    compose PROJECT DIRECTORY — the repo root (``compose_base_cmd`` pins
    ``--project-directory <repo>``) — and never against the file's own location,
    which is ``build/``. The nginx config and landing page are render output, so
    their mounts have to carry the ``build/`` prefix themselves; a bare
    ``./nginx/...`` would resolve to ``<repo>/nginx/``, where nothing is ever
    written, and compose would create empty directories there and serve nginx a
    directory in place of its config.

    Asserted against the path the writer actually produces rather than a repeated
    literal, so the mount and the render cannot drift apart.
    """
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert — bind mounts only. The envsubst output directory is a tmpfs and has
    # no host source to resolve, so it is declared on the service's own `tmpfs:`
    # key rather than in this list.
    sources = [
        mount.split(":")[0]
        for mount in compose["services"]["nginx"]["volumes"]
        if isinstance(mount, str)
    ]
    resolved = [(tmp_path / source).resolve() for source in sources[:2]]
    assert resolved == [
        web_artifacts_dir(tmp_path) / "nginx" / "nginx.conf",
        web_artifacts_dir(tmp_path) / "nginx" / "landing.html",
    ]
    # The same two relative paths are the keys the render emits, so what compose
    # resolves is exactly what write_web_terminal_artifacts writes into build/.
    assert [source.removeprefix("./build/") for source in sources[:2]] == [
        "nginx/nginx.conf",
        "nginx/landing.html",
    ]
    assert {"nginx/nginx.conf", "nginx/landing.html"} <= set(artifacts)


def test_auth_default_none() -> None:
    """No `web_terminals.auth` stanza -> auth_method defaults to 'none' (v1 has no auth)."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "none"


def test_auth_default_none_reads_configured_method() -> None:
    """A configured `web_terminals.auth.method` is read through, not clobbered by the default."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {"method": "password"}

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "password"


def test_auth_method_non_string_falls_back_to_none() -> None:
    """A malformed (non-str) `auth.method` (well-formedness is lint's job, not this
    function's) must not leak into the rendered seam as-is — it falls back to the
    same inert "none" default as an absent/empty method."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {"method": {"nested": "not-a-string"}}

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "none"


def test_tls_default_off() -> None:
    """No `web_terminals.tls` stanza -> tls_enabled defaults to False (v1 stays http)."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["tls_enabled"] is False
    assert context["tls_cert"] is None
    assert context["tls_key"] is None


def test_tls_default_off_reads_configured_stanza() -> None:
    """A configured `web_terminals.tls` stanza is read through untouched."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["tls_enabled"] is True
    assert context["tls_cert"] == "/etc/nginx/certs/dls.crt"
    assert context["tls_key"] == "/etc/nginx/certs/dls.key"


# ---------------------------------------------------------------------------
# Task 1.2: the full auth stanza — `_auth_tls_context` is the single parser
# ---------------------------------------------------------------------------


def test_auth_context_defaults_are_inert_without_an_auth_stanza() -> None:
    """No `auth` stanza -> every auth key resolves to a default that renders nothing:
    no method, no image pin, no OIDC issuer. The two knobs that must still hold a
    usable value (the sidecar's port and the session lifetime) carry the documented
    defaults so a facility enabling auth needs only `method`."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "none"
    assert context["auth_allow_insecure_http"] is False
    assert context["auth_image"] is None
    assert context["auth_oidc_issuer"] is None
    assert context["auth_port"] == 9070
    assert context["auth_session_lifetime"] == 12 * 60 * 60


def test_auth_context_reads_every_configured_scalar() -> None:
    """A fully specified `auth` stanza is read through field for field."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "method": "password",
        "port": 9401,
        "session_lifetime": 3600,
        "allow_insecure_http": True,
        "image": "git.dls.example.org:5050/physics/production/dls-auth:2026.8.1",
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "password"
    assert context["auth_port"] == 9401
    assert context["auth_session_lifetime"] == 3600
    assert context["auth_allow_insecure_http"] is True
    assert context["auth_image"].endswith("dls-auth:2026.8.1")


def test_auth_context_reads_oidc_issuer_and_credential_env_var_names() -> None:
    """`oidc` mode names its issuer inline and its client credentials only by env-var
    name — the credentials themselves live in `.env.auth`, never in config.yml."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "method": "oidc",
        "allow_insecure_http": True,
        "oidc": {
            "issuer": "https://sso.dls.example.org/realms/facility",
            "client_id_env": "DLS_SSO_CLIENT_ID",
            "client_secret_env": "DLS_SSO_CLIENT_SECRET",
        },
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "oidc"
    assert context["auth_oidc_issuer"] == "https://sso.dls.example.org/realms/facility"
    assert context["auth_oidc_client_id_env"] == "DLS_SSO_CLIENT_ID"
    assert context["auth_oidc_client_secret_env"] == "DLS_SSO_CLIENT_SECRET"


def test_auth_context_oidc_credential_env_var_names_have_defaults() -> None:
    """A facility that doesn't rename them gets the framework's own env-var names, so
    an `oidc` stanza only has to carry the issuer."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "method": "oidc",
        "allow_insecure_http": True,
        "oidc": {"issuer": "https://sso.dls.example.org/realms/facility"},
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_oidc_client_id_env"] == "OSPREY_AUTH_OIDC_CLIENT_ID"
    assert context["auth_oidc_client_secret_env"] == "OSPREY_AUTH_OIDC_CLIENT_SECRET"


def test_auth_context_unknown_method_raises_value_error() -> None:
    """An `auth.method` naming a method no sidecar implements is a hard error, not a
    forward-compatible passthrough: rendering it would emit an `auth_request` seam
    that nothing can answer, locking the deployment out with an unactionable 403."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {"method": "oauth2_proxy"}

    # Act / Assert
    with pytest.raises(ValueError, match="auth.method") as exc_info:
        _auth_tls_context(web_terminals)
    message = str(exc_info.value)
    assert "oauth2_proxy" in message
    # The message names the alternatives, so the fix is obvious from the error alone.
    assert "password" in message
    assert "oidc" in message


def test_auth_context_malformed_scalars_fall_back_to_defaults() -> None:
    """Wrong-typed values follow this module's defensive-read convention (lint is the
    authoritative gate on well-formedness) rather than crashing the render. `bool` is
    called out explicitly: it satisfies `isinstance(x, int)`, so an unguarded read
    would turn `port: true` into port 1."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "port": True,
        "session_lifetime": -1,
        "image": "",
        "oidc": "not-a-dict",
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_port"] == 9070
    assert context["auth_session_lifetime"] == 12 * 60 * 60
    assert context["auth_image"] is None
    assert context["auth_oidc_issuer"] is None
    assert context["auth_oidc_client_id_env"] == "OSPREY_AUTH_OIDC_CLIENT_ID"


def test_auth_context_never_reads_a_credential_out_of_config() -> None:
    """Secrets belong in `.env.auth`; the parser reads env-var *names* only. A config
    that pastes a literal client secret (or password hash) in must not carry it into
    any render context, where it would land in a committed compose/nginx artifact."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "method": "oidc",
        "allow_insecure_http": True,
        "oidc": {
            "issuer": "https://sso.dls.example.org/realms/facility",
            "client_secret": "s3cr3t-should-never-render",
        },
        "password_hash": "scrypt.16384.8.1.salt.hash",
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    rendered_values = " ".join(str(value) for value in context.values())
    assert "s3cr3t-should-never-render" not in rendered_values
    assert "scrypt." not in rendered_values


def test_render_auth_context_gate_method_without_tls_raises() -> None:
    """Fail-closed render gate: authentication over cleartext HTTP would ship session
    cookies in the open, so the render refuses rather than emitting it — the same
    posture as the tls cert/key fail-fast right beside it."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = {"method": "password"}

    # Act / Assert
    with pytest.raises(ValueError, match="allow_insecure_http") as exc_info:
        render_web_terminals(config)
    assert "tls.enabled" in str(exc_info.value)


def test_render_auth_context_gate_allow_insecure_http_permits_the_render() -> None:
    """The escape hatch is explicit and per-deployment: a facility terminating TLS at
    an upstream proxy opts in by name."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = {
        "method": "password",
        "allow_insecure_http": True,
    }

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "auth_request" in artifacts["nginx/nginx.conf"]


def test_render_auth_context_gate_is_satisfied_by_tls_enabled() -> None:
    """The ordinary way to pass the gate is to actually enable TLS."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = {"method": "oidc"}
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "auth_request" in artifacts["nginx/nginx.conf"]


def test_render_auth_context_gate_does_not_fire_for_method_none() -> None:
    """The gate is about protecting session cookies; with no authentication there are
    none, so plain-HTTP deployments keep rendering exactly as they do today."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = {"method": "none"}

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "auth_request" not in artifacts["nginx/nginx.conf"]


def test_render_succeeds_with_auth_default_none_and_tls_default_off() -> None:
    """A config with no auth/tls stanzas at all still renders the full artifact set — 1.4 only
    establishes the config contract, it must not change Phase-1 render behavior."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "auth" not in config["modules"]["web_terminals"]
    assert "tls" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)

    # Assert — round-trips unchanged (defaults are inert; no seam rendering here)
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }
    yaml.safe_load(artifacts["docker-compose.web.yml"])


def test_render_tolerates_non_dict_auth_and_tls_stanzas() -> None:
    """`_as_dict`-style defensive reads: a malformed (non-dict) auth/tls value must not raise —
    it falls back to the same defaults as an absent stanza. Well-formedness is lint's job."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = "not-a-dict"
    config["modules"]["web_terminals"]["tls"] = ["also", "not", "a", "dict"]

    # Act
    artifacts = render_web_terminals(config)
    context = _auth_tls_context(config["modules"]["web_terminals"])

    # Assert
    assert context["auth_method"] == "none"
    assert context["tls_enabled"] is False
    assert "docker-compose.web.yml" in artifacts


def test_nginx_seam_default_gated_off_emits_no_ssl_listen_or_auth_request() -> None:
    """C1: with no `auth`/`tls` stanzas (the v1 default posture), the rendered nginx
    fragment must carry NEITHER a `listen 443 ssl` block NOR an `auth_request`
    directive — the seam is fully inert until a facility config opts in."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "auth" not in config["modules"]["web_terminals"]
    assert "tls" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "listen 443 ssl" not in nginx_conf
    assert "ssl_certificate" not in nginx_conf
    assert "auth_request" not in nginx_conf


def test_nginx_seam_tls_enabled_emits_ssl_listen_and_configured_cert_paths() -> None:
    """C2 (render half): `tls.enabled: true` emits `listen 443 ssl` plus
    `ssl_certificate`/`ssl_certificate_key` referencing the configured paths,
    inside the `server{}` block (not at http/top-level)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
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
    assert "ssl_certificate /etc/nginx/certs/dls.crt;" in nginx_conf
    assert "ssl_certificate_key /etc/nginx/certs/dls.key;" in nginx_conf
    server_index = nginx_conf.index("server {")
    ssl_index = nginx_conf.index("listen 443 ssl;")
    assert server_index < ssl_index, "the ssl listener must sit inside server{}, not above it"


def test_nginx_seam_tls_enabled_without_cert_or_key_raises_value_error() -> None:
    """`tls.enabled: true` without both `cert` and `key` can't render a coherent
    `ssl_certificate`/`ssl_certificate_key` pair — fail loudly at render time rather
    than emit a directive pointing at a missing path (which `nginx -t` would reject
    anyway, but with a far less actionable error)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["tls"] = {"enabled": True}

    # Act / Assert
    with pytest.raises(ValueError, match="tls"):
        render_web_terminals(config)


def test_nginx_seam_auth_method_set_gates_every_proxied_user_location() -> None:
    """C2 (render half): a configured `auth.method` (anything but the "none" default)
    leaves NO proxied `/u/<user>/` location ungated.

    The per-user target shape each `auth_request` names is pinned by the
    `auth_location` tests below; this one guards the coverage property they
    can't — that the count of gated locations equals the count of proxied ones,
    so a user added through a code path that forgets the gate fails here.
    Fail-closed is asserted alongside it: nothing in the render authorizes on
    its own, so a sidecar that is down or unconfigured denies rather than
    admits.
    """
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]
    config["modules"]["web_terminals"]["auth"] = {
        "method": "password",
        "allow_insecure_http": True,
    }

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    directives = _directives(nginx_conf)
    assert directives.count("auth_request ") == len(users)
    assert directives.count("    location /u/") == len(users)
    assert "return 200;" not in directives


def test_nginx_seam_auth_none_omits_auth_request_even_with_tls_enabled() -> None:
    """The two seams are independently gated: TLS on with `auth.method` left at its
    "none" default must still omit `auth_request` entirely."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
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


# ---------------------------------------------------------------------------
# Task 2.5: mcp.topology fail-closed
# ---------------------------------------------------------------------------


def test_mcp_topology_omitted_renders_unchanged() -> None:
    """No `web_terminals.mcp` stanza at all -> today's behavior, byte-identical
    to a config that has never heard of the topology key (zero migration)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "mcp" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


def test_mcp_topology_explicit_per_container_stdio_renders_unchanged() -> None:
    """Explicitly spelling out the default value is equally inert."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = {"topology": "per_container_stdio"}

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


def test_mcp_topology_shared_http_raises_value_error_scoped_to_shared_tier() -> None:
    """`shared_http` is a recognized-but-rejected value (Task 2.4 owns the lint
    ERROR twin of this check): render must fail closed, and the message must be
    scoped to the shared framework-MCP tier — it must never read as though it
    were rejecting HTTP MCP transport in general, since a facility's own
    `claude_code.servers` custom `url` entries are a separate, unaffected,
    already-supported path."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = {"topology": "shared_http"}

    # Act / Assert
    with pytest.raises(ValueError, match="mcp.topology") as exc_info:
        render_web_terminals(config)
    message = str(exc_info.value)
    assert "shared_http" in message
    assert "per_container_stdio" in message
    # Scoped to the shared framework-MCP tier, not a blanket HTTP-transport ban.
    assert "claude_code.servers" in message
    assert "unaffected" in message


def test_mcp_topology_unknown_value_raises_value_error() -> None:
    """Any value other than `per_container_stdio` is fail-closed, not just the one
    named-in-the-schema `shared_http` alternative."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = {"topology": "some_future_value"}

    # Act / Assert
    with pytest.raises(ValueError, match="mcp.topology"):
        render_web_terminals(config)


def test_mcp_topology_malformed_mcp_stanza_falls_back_to_default() -> None:
    """A non-dict `mcp` stanza (well-formedness is lint's job, not this
    function's) must not raise here — it falls back to the same default as an
    absent stanza, matching the `_as_dict`-style defensive-read convention used
    throughout this module (see `_auth_tls_context`)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = "not-a-dict"

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "docker-compose.web.yml" in artifacts


def test_mcp_topology_custom_url_server_config_is_a_separate_namespace() -> None:
    """Regression guard against over-broad validation: a facility config that
    also carries a project-level `claude_code.servers` custom `url` entry
    alongside `modules.web_terminals` must render exactly as it would without
    that entry — the topology gate reads only `modules.web_terminals.mcp`, and
    must never inspect, walk, or reject on the unrelated `claude_code.servers`
    key. The corresponding `.mcp.json` rendering assertion (that the url-server
    entry itself renders unchanged) lives in
    `tests/registry/test_mcp.py::test_topology_default_leaves_custom_url_server_untouched`.
    """
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["claude_code"] = {
        "servers": {
            "remote-api": {"url": "http://remote:8001/sse"},
        }
    }
    assert "mcp" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)

    # Assert — unaffected by the sibling claude_code.servers stanza
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


# ---------------------------------------------------------------------------
# Task 3.1: per-user internal auth locations
# ---------------------------------------------------------------------------


def _auth_config(users: list[str] | None = None, **auth: object) -> dict:
    """A rendering config with authentication on.

    `allow_insecure_http` is set unless the caller enables TLS, because the
    render refuses `auth.method` over cleartext otherwise — a gate these tests
    are not about (it has its own coverage above).
    """
    config = _config(users if users is not None else ["alice", "bob", "carol"])
    stanza: dict = {"method": "password", "allow_insecure_http": True}
    stanza.update(auth)
    config["modules"]["web_terminals"]["auth"] = stanza
    return config


def _render_nginx(config: dict) -> str:
    return render_web_terminals(config)["nginx/nginx.conf"]


def _directives(conf: str) -> str:
    """`conf` with its comment lines dropped.

    Counting and absence assertions have to ignore comment prose, which in this
    template frequently names the very construct being counted or ruled out —
    the comment above `error_page` explains what the `auth_request` above it
    does, and would otherwise be counted as a second `auth_request`.
    """
    return "\n".join(line for line in conf.splitlines() if not line.lstrip().startswith("#"))


def test_auth_location_is_rendered_once_per_roster_user() -> None:
    """Each user gets their OWN internal auth target, not a shared one.

    A single shared target could only learn which user is being authorized from
    the request itself (`$request_uri`/`X-Original-URI`) — client-controlled
    input on the authorization path. Per-user locations make the identity a
    property of which location matched, which the client cannot influence.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        assert f"location = /_osprey_auth/{user} {{" in nginx_conf
    assert nginx_conf.count("location = /_osprey_auth/") == len(users)
    # The pre-sidecar shared stub is gone, not merely supplemented.
    assert "location = /_osprey_auth {" not in nginx_conf


def test_auth_location_is_named_by_each_users_own_auth_request() -> None:
    """`/u/<user>/` subrequests `<user>`'s target — never another user's.

    Pinning the pairing (not just the presence of both directives) is what
    catches a loop that renders every `auth_request` against the first user.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        location = re.search(rf"location /u/{user}/ \{{(.*?)\n    \}}", nginx_conf, flags=re.DOTALL)
        assert location is not None, f"no proxied location rendered for {user}"
        assert f"auth_request /_osprey_auth/{user};" in location.group(1)
    assert _directives(nginx_conf).count("auth_request ") == len(users)


def test_auth_location_verifies_against_the_sidecar_with_a_render_time_username() -> None:
    """The target proxies to the sidecar's `/verify`, with `?user=` baked in.

    The username is a literal from the roster, so the sidecar is asked about
    exactly the user whose prefix was requested — there is no code path where a
    crafted URL, header, or cookie chooses a different one.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice", "bob"]))

    # Assert
    assert "proxy_pass http://127.0.0.1:9070/verify?user=alice;" in nginx_conf
    assert "proxy_pass http://127.0.0.1:9070/verify?user=bob;" in nginx_conf


def test_auth_location_follows_the_configured_auth_port() -> None:
    """A facility that moves the sidecar off 9070 is routed to the new port —
    the template reads `auth_port`, it never hardcodes the default."""
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice"], port=9471))

    # Assert
    assert "proxy_pass http://127.0.0.1:9471/verify?user=alice;" in nginx_conf
    assert ":9070/verify" not in nginx_conf


def test_auth_location_is_internal_only() -> None:
    """Every auth target carries `internal;`, so a browser cannot request one
    directly — the subrequest is the only way in."""
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        target = re.search(
            rf"location = /_osprey_auth/{user} \{{(.*?)\n    \}}", nginx_conf, flags=re.DOTALL
        )
        assert target is not None, f"no auth target rendered for {user}"
        assert "internal;" in target.group(1)
    assert nginx_conf.count("internal;") == len(users)


def test_auth_location_suppresses_the_subrequest_body() -> None:
    """`proxy_pass_request_body off` + a cleared `Content-Length` on every auth
    target.

    Without both, nginx announces the parent POST's body to a sidecar that
    never reads it, and the subrequest hangs until it times out — every
    authenticated POST in the deployment stalls. This assertion is the only
    cheap guard for that; the failure mode is invisible to a syntax check.
    """
    # Arrange
    users = ["alice", "bob"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    assert nginx_conf.count("proxy_pass_request_body off;") == len(users)
    assert nginx_conf.count('proxy_set_header Content-Length "";') == len(users)


def test_auth_location_verify_subrequest_fails_fast() -> None:
    """Every verify target states short timeouts instead of inheriting 60s.

    This subrequest sits in front of every gated request, so its worst case is
    the deployment's worst case: on nginx's defaults, one wedged sidecar holds
    each open terminal's next request for a minute before failing it. A
    loopback signature check has no legitimate reason to take that long.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        body = _location_body(nginx_conf, f"location = /_osprey_auth/{user}")
        assert "proxy_connect_timeout 2s;" in body
        assert "proxy_read_timeout 5s;" in body
    # The gated terminal locations keep the long SSE/WebSocket read timeout —
    # the short one belongs to the subrequest only.
    assert "proxy_read_timeout 3600s;" in _location_body(nginx_conf, "location /u/alice/")


def test_auth_location_absent_when_method_is_none() -> None:
    """`auth.method: none` renders no auth target, no `auth_request`, and no
    reference to the sidecar's port — byte-for-byte the pre-auth output.

    (The wider byte-equality guarantee is pinned by the golden fixtures in
    test_golden_render.py; this asserts the property directly for the explicit
    `method: none` spelling as well as the absent-stanza default.)
    """
    # Arrange
    default_config = copy.deepcopy(_MULTI_USER_CONFIG)
    explicit_none = copy.deepcopy(_MULTI_USER_CONFIG)
    explicit_none["modules"]["web_terminals"]["auth"] = {"method": "none"}

    # Act
    default_conf = _render_nginx(default_config)
    explicit_conf = _render_nginx(explicit_none)

    # Assert
    assert explicit_conf == default_conf
    assert "_osprey_auth" not in default_conf
    assert "auth_request" not in default_conf
    assert "9070" not in default_conf


def test_auth_location_zero_users_renders_no_auth_targets() -> None:
    """An empty roster with auth on has nobody to authorize: no targets, and —
    critically — no location that answers 200 in their absence."""
    # Act
    nginx_conf = _render_nginx(_auth_config([]))

    # Assert
    assert "_osprey_auth" not in nginx_conf
    assert "return 200;" not in nginx_conf


# ---------------------------------------------------------------------------
# Task 3.2: public /auth locations
# ---------------------------------------------------------------------------


def _location_body(nginx_conf: str, header: str) -> str:
    """The directives inside a top-level `location …` block, by its header line."""
    match = re.search(rf"{re.escape(header)} \{{(.*?)\n    \}}", nginx_conf, flags=re.DOTALL)
    assert match is not None, f"no `{header}` block in the rendered fragment"
    return match.group(1)


def test_auth_public_location_proxies_the_login_surface_to_the_sidecar() -> None:
    """`/auth/` reaches the sidecar's own `/auth/` prefix — the login page, the
    credential POST, the OIDC callback, logout, and the page's assets all ride
    this one prefix location rather than an enumeration that drifts from the
    sidecar's router."""
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "    location /auth/ {" in nginx_conf
    assert "proxy_pass http://127.0.0.1:9070/auth/;" in _location_body(
        nginx_conf, "location /auth/"
    )


def test_auth_public_location_follows_the_configured_auth_port() -> None:
    """The public surface and the internal verify targets are pointed at the same
    configured sidecar port — neither hardcodes the 9070 default."""
    # Act
    nginx_conf = _render_nginx(_auth_config(port=9471))

    # Assert
    assert "proxy_pass http://127.0.0.1:9471/auth/;" in nginx_conf
    assert ":9070" not in nginx_conf


def test_auth_public_location_is_not_behind_auth_request() -> None:
    """The login flow must answer without a session — gating it behind one is a
    locked door whose key is inside the room. Same for the landing page at `= /`,
    which is both the identity chooser and what the nginx healthcheck probes."""
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "auth_request" not in _location_body(nginx_conf, "location /auth/")
    assert "auth_request" not in _location_body(nginx_conf, "location = /")


def test_auth_public_location_is_not_internal() -> None:
    """`/auth/` is browser-reachable: an `internal` marker here would 404 every
    login attempt (the exact failure a copy-paste from the verify targets makes,
    and one no syntax check catches)."""
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "internal;" not in _location_body(nginx_conf, "location /auth/")


def test_auth_public_location_sets_explicit_short_proxy_timeouts() -> None:
    """The login surface states its own timeouts instead of taking whatever the
    terminal locations need.

    The `/u/<user>/` locations run `proxy_read_timeout 3600s` to keep SSE and
    WebSocket streams alive; the login flow is a few short round trips to a
    loopback service, so a wedged sidecar should surface in seconds rather than
    hang a browser tab.
    """
    # Act
    body = _location_body(_render_nginx(_auth_config()), "location /auth/")

    # Assert
    assert "proxy_connect_timeout 5s;" in body
    assert "proxy_send_timeout 15s;" in body
    assert "proxy_read_timeout 15s;" in body
    assert "3600s" not in body


def test_auth_public_location_does_not_expose_the_verify_endpoint() -> None:
    """`/verify` is reachable only through the `internal` per-user targets. It
    must not be proxied under the public prefix, where a browser could ask the
    sidecar about any username it liked."""
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "location /verify" not in nginx_conf
    assert "location = /verify" not in nginx_conf
    # Every directive (as opposed to comment prose) naming `/verify` is the
    # proxy_pass target of an internal per-user location.
    verify_lines = [line for line in _directives(nginx_conf).splitlines() if "/verify" in line]
    assert verify_lines, "the verify targets themselves should still be rendered"
    assert all("/verify?user=" in line for line in verify_lines)


def test_auth_public_location_absent_when_method_is_none() -> None:
    """No sidecar, no public login surface — a `none` deployment renders no
    `/auth/` location at all."""
    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "location /auth/" not in nginx_conf


def test_auth_public_location_renders_for_an_empty_roster() -> None:
    """The login surface is a property of the sidecar, not of the roster: it is
    rendered even with zero users, so a roster added later needs no second
    render path (and an empty deployment still serves a coherent origin)."""
    # Act
    nginx_conf = _render_nginx(_auth_config([]))

    # Assert
    assert "    location /auth/ {" in nginx_conf


# ---------------------------------------------------------------------------
# Task 3.5: upstream cookie stripping
# ---------------------------------------------------------------------------


def test_cookie_strip_on_every_proxied_user_location() -> None:
    """No browser cookie reaches a per-user container.

    One cookie jar serves the whole origin, so an operator with several users
    unlocked would otherwise hand each container a session cookie naming all of
    them — and these containers run agent-generated code, so anything readable
    inside one has effectively left the perimeter.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        assert 'proxy_set_header Cookie "";' in _location_body(nginx_conf, f"location /u/{user}/")


def test_cookie_strip_spares_the_sidecar_locations() -> None:
    """The sidecar is the one upstream that MUST see the cookie — it is the
    thing that reads the session. Stripping it there would deny every request
    and lock the deployment out."""
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice"]))

    # Assert
    assert "Cookie" not in _location_body(nginx_conf, "location /auth/")
    assert "Cookie" not in _directives(_location_body(nginx_conf, "location = /_osprey_auth/alice"))


def test_cookie_strip_absent_when_method_is_none() -> None:
    """Without authentication the cookie is not CUT — it is the only gate left.

    There is no sidecar session to keep out of a container here, but there is
    still a shared cookie jar: one origin serves every terminal, so the browser's
    raw `Cookie` header names every OTHER user's session this browser holds. So
    the auth-off location forwards exactly one cookie — the one this user's own
    app issued — and nothing else.
    """
    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    for user, port in (("alice", 9091), ("bob", 9092), ("carol", 9093)):
        body = _directives(_location_body(nginx_conf, f"location /u/{user}/"))
        cookie = f"osprey_terminal_session_{port}"
        assert f'proxy_set_header Cookie "{cookie}=$cookie_{cookie}";' in body
        # Never the blanket strip, which would leave this location with no gate
        # at all, and never a bare forward of the whole jar.
        assert 'proxy_set_header Cookie "";' not in body
        assert "$http_cookie" not in body


# ---------------------------------------------------------------------------
# Task 3.3: content-negotiated 401
# ---------------------------------------------------------------------------


def test_negotiated_401_classifies_requests_by_accept_and_upgrade() -> None:
    """A `map` decides navigation-vs-program from the Accept header, with the
    Upgrade header concatenated in front of it.

    The concatenation is what excludes WebSocket handshakes: an upgrade token
    in front pushes `text/html` off the `^` anchor, so a handshake scores 0
    whatever its Accept header claims.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "map $http_upgrade$http_accept $osprey_auth_wants_login_page {" in nginx_conf
    map_body = re.search(
        r"map \$http_upgrade\$http_accept \$osprey_auth_wants_login_page \{(.*?)\n\}",
        nginx_conf,
        flags=re.DOTALL,
    )
    assert map_body is not None
    assert "default 0;" in map_body.group(1)
    assert '"~*^text/html" 1;' in map_body.group(1)


def test_negotiated_401_every_user_location_intercepts_with_the_shared_handler() -> None:
    """Each `/u/<user>/` sends its 401 to the named handler and states, as a
    render-time literal, which user the handler should prompt for.

    `error_page 401 = @…` (with the `=`) is what lets the handler's own status
    replace the 401 — without it a browser would receive a 401 carrying a
    Location header, which it does not follow.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        body = _location_body(nginx_conf, f"location /u/{user}/")
        assert f"set $osprey_auth_user {user};" in body
        assert "error_page 401 = @osprey_auth_401;" in body


def test_negotiated_401_handler_redirects_navigations_and_bares_everything_else() -> None:
    """The handler redirects only when the map said "navigation", and answers a
    bare 401 otherwise.

    A blanket redirect would hand a JSON caller a page of login HTML and leave
    an EventSource reconnecting forever against a login page that never becomes
    the stream it asked for.
    """
    # Act
    body = _location_body(_render_nginx(_auth_config()), "location @osprey_auth_401")

    # Assert
    assert "if ($osprey_auth_wants_login_page) {" in body
    assert "return 302 /auth/login?user=$osprey_auth_user&next=$osprey_auth_next;" in body
    # The unconditional fallthrough, outside the `if`.
    assert re.search(r"\n        return 401;", body)


def test_negotiated_401_return_to_is_allowlisted_never_a_raw_path_variable() -> None:
    """The return-to emits `$osprey_auth_next` — and neither raw path variable.

    `$uri` is decoded by the time it is read, so emitting it copies a decoded
    CR LF straight into the Location header (response-header injection).
    `$request_uri` is injection-safe but still client-controlled, and a leading
    `//` in it reads as an absolute URL that walks the browser off-site. Only
    the allowlist map produces a value that is safe to emit AND safe to trust.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config())
    body = _location_body(nginx_conf, "location @osprey_auth_401")

    # Assert
    assert "next=$osprey_auth_next;" in body
    directives = _directives(body)
    assert "next=$uri" not in directives
    assert "$request_uri" not in directives


def test_negotiated_401_return_to_allowlist_rejects_injection_and_keeps_deep_links() -> None:
    """The allowlist regex itself, exercised against what an attacker sends.

    This runs the pattern the template actually emits rather than restating it,
    so a widened character class fails here. nginx decodes percent-escapes
    before this map is consulted, hence the decoded forms below: `%0d%0a`
    arrives as a real CR LF, `%26` as a real `&`.
    """
    # Arrange
    nginx_conf = _render_nginx(_auth_config())
    pattern = re.search(r'\n    "~(\S+)" \$uri;', nginx_conf)
    assert pattern is not None, "no allowlist entry in the $osprey_auth_next map"
    # PCRE's `\z` is spelled `\Z` in Python; the anchor semantics are the same
    # (end of subject, NOT "before a trailing newline" the way `$` behaves).
    allowlist = re.compile(pattern.group(1).replace(r"\z", r"\Z"))

    # Act / Assert — legitimate deep links survive
    for path in (
        "/u/alice/",
        "/u/alice/files/notes_v2.1-final.txt",
        "/u/alice-b/panel/~cache/x",
    ):
        assert allowlist.match(path), f"deep link {path!r} should round-trip"

    # Act / Assert — every hostile decoded form collapses to the empty default
    for path in (
        "/u/alice/x\r\nX-Injected: yes",  # header injection
        "/u/alice/x\nSet-Cookie: evil=1",  # session fixation on our own origin
        "/u/alice/x\n",  # trailing bare LF: why the anchor is \z, not $
        "/u/alice/x&user=bob",  # second `user` param wins in most parsers
        "/u/alice/x?next=/u/bob/",  # smuggled query
        "//evil.example.org/",  # protocol-relative open redirect
        "/auth/login",  # outside every user prefix
        "/u/alice/x y",  # raw space splits the header value
    ):
        assert not allowlist.match(path), f"{path!r} must not be reflected"


def test_negotiated_401_every_redirect_is_relative_to_the_clients_own_origin() -> None:
    """`absolute_redirect off` sits at SERVER level, not in one location.

    Left on, nginx rebuilds a relative redirect target into an absolute URL
    from `$host` plus its OWN listening port — wrong for the very topology
    `auth.allow_insecure_http` exists to serve: behind a facility TLS
    terminator, `https://facility/u/alice` becomes `http://facility:9080/u/alice/`,
    downgrading the scheme and advertising the internal port. Scoping it to the
    login handler alone would leave the `/u/<user>` bookmark 301s — a link an
    operator is far more likely to have saved — still absolute.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice"]))
    content = _server_blocks(nginx_conf)[0]

    # Assert — declared before the first location, so every one inherits it
    server_preamble = _directives(content).split("location", 1)[0]
    assert "absolute_redirect off;" in server_preamble
    assert content.count("absolute_redirect off;") == 1
    assert "return 302 http" not in content
    # The bookmark redirect is the one this placement exists to cover.
    assert "return 301 /u/alice/;" in _location_body(nginx_conf, "location = /u/alice")


def test_negotiated_401_login_target_is_the_public_auth_prefix() -> None:
    """The place denied users are sent must itself be reachable without a
    session — the redirect target lives under the ungated `/auth/` prefix."""
    # Act
    nginx_conf = _render_nginx(_auth_config())
    body = _location_body(nginx_conf, "location @osprey_auth_401")

    # Assert
    assert "/auth/login" in body
    assert "    location /auth/ {" in nginx_conf
    assert "auth_request" not in _location_body(nginx_conf, "location /auth/")


def test_negotiated_401_absent_when_method_is_none() -> None:
    """Nothing to negotiate without authentication: no map, no `error_page`,
    no handler."""
    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "$osprey_auth_wants_login_page" not in nginx_conf
    assert "error_page" not in nginx_conf
    assert "@osprey_auth_401" not in nginx_conf


# ---------------------------------------------------------------------------
# Task 3.4: TLS redirect server block
# ---------------------------------------------------------------------------


def _server_blocks(nginx_conf: str) -> list[str]:
    """The top-level `server { … }` blocks, in render order.

    Split on the closing brace in column 0, which only a top-level block has
    (every nested `location` closes at an indent).
    """
    blocks = re.findall(r"^server \{\n(.*?)^\}$", nginx_conf, flags=re.DOTALL | re.MULTILINE)
    assert blocks, "no top-level server block in the rendered fragment"
    return blocks


def test_tls_redirect_off_renders_exactly_one_server_block() -> None:
    """Without TLS there is nothing to redirect to: one server, on the plain
    port, serving everything — unchanged from before TLS was renderable."""
    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert len(_server_blocks(nginx_conf)) == 1
    assert "return 301 https://" not in nginx_conf
    assert "listen 9080;" in nginx_conf


def test_tls_redirect_splits_into_a_redirect_server_and_a_content_server() -> None:
    """With TLS on, the plain port and 443 become two separate servers.

    One server listening on both ports would answer the same content over
    cleartext, and any session cookie it set would cross the network in the
    clear — the exact risk TLS was enabled to remove.
    """
    # Act
    blocks = _server_blocks(_render_nginx(_tls_config()))

    # Assert
    assert len(blocks) == 2
    redirect, content = blocks
    assert "listen 9080;" in redirect
    assert "listen [::]:9080;" in redirect
    assert "listen 443 ssl;" in content
    assert "listen [::]:443 ssl;" in content
    # Neither listener answers on the other's port.
    assert "443" not in redirect
    assert "listen 9080;" not in content


def test_tls_redirect_server_serves_nothing_but_the_redirect() -> None:
    """The plain-port server has no locations, no root, and no proxying — its
    only content is the 301. Anything else served there would be served over
    cleartext."""
    # Act
    redirect = _server_blocks(_render_nginx(_tls_config()))[0]

    # Assert
    assert "return 301 https://$host$request_uri;" in redirect
    assert "location" not in redirect
    assert "root " not in redirect
    assert "proxy_pass" not in redirect
    assert "auth_request" not in redirect


def test_tls_redirect_target_is_the_requested_host_not_the_render_time_origin() -> None:
    """The 301 goes to `$host` — the name the client actually used.

    A deployment answers to more names than config knows (aliases, internal DNS
    names, a bare IP); rewriting the host at render time would bounce those
    clients to a name they may not resolve. `external_origin` is for values
    that must be fixed at render time, like an IdP-registered redirect_uri.
    """
    # Arrange
    config = _tls_config()
    fqdn = config["deploy"]["fqdn"]

    # Act
    redirect = _server_blocks(_render_nginx(config))[0]

    # Assert
    assert "https://$host$request_uri" in redirect
    assert fqdn not in redirect


def test_tls_redirect_content_server_holds_the_cert_and_every_user_route() -> None:
    """The 443 server is the sole content server: cert pair, landing page, and
    every per-user route live there, not in the redirect server."""
    # Arrange
    config = _tls_config(["alice", "bob"])

    # Act
    content = _server_blocks(_render_nginx(config))[1]

    # Assert
    assert "ssl_certificate /etc/nginx/certs/dls.crt;" in content
    assert "ssl_certificate_key /etc/nginx/certs/dls.key;" in content
    assert "location = / {" in content
    assert "location /u/alice/ {" in content
    assert "location /u/bob/ {" in content


def test_tls_redirect_leaves_the_auth_surface_only_on_the_secure_server() -> None:
    """With auth and TLS both on, the login flow and the per-user gates exist
    only behind TLS — there is no plain-http path to a login form, so no cookie
    can be issued on the cleartext origin."""
    # Arrange
    config = _auth_config()
    config["modules"]["web_terminals"]["auth"].pop("allow_insecure_http")
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    redirect, content = _server_blocks(_render_nginx(config))

    # Assert
    assert "location /auth/" not in redirect
    assert "_osprey_auth" not in redirect
    assert "location /auth/ {" in content
    assert "auth_request /_osprey_auth/alice;" in content
    # Count-equality across the two-server matrix: every gated location and
    # every auth target lives in the 443 server and nowhere else, so the split
    # cannot silently leave a duplicate of one on the cleartext listener.
    users = config["modules"]["web_terminals"]["users"]
    assert _directives(content).count("auth_request ") == len(users)
    assert _directives(redirect).count("auth_request ") == 0
    assert content.count("location = /_osprey_auth/") == len(users)
    assert redirect.count("location") == 0


def test_nginx_landing_location_is_exact_match_only() -> None:
    """The landing page is served for `/` ONLY — every other unmatched path 404s.

    A prefix `location /` catch-all would answer stray API calls (e.g. a panel
    fetch that lost its `/u/<user>` prefix) with 200 + landing HTML, hiding the
    bug behind a downstream JSON parse error instead of a visible 404.
    """
    # Act
    artifacts = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "location = / {" in nginx_conf
    assert not re.search(r"location / \{", nginx_conf)


# ---------------------------------------------------------------------------
# Task 4.4: the auth sidecar's compose service, and the secrets isolation
# between it and the per-user containers
# ---------------------------------------------------------------------------


def _compose(config: dict) -> dict:
    """The rendered compose overlay, parsed."""
    return yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])


def _env_names(service: dict) -> list[str]:
    """The env-var names a compose service declares inline, in rendered order."""
    return [line.split("=", 1)[0] for line in service.get("environment", [])]


def test_auth_sidecar_service_is_absent_for_method_none() -> None:
    """The default posture renders the overlay it renders today: no sidecar service,
    no auth environment, no reference to the credential file. Every deployment that
    has not opted into authentication must be untouched by this feature."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    rendered = render_web_terminals(config)["docker-compose.web.yml"]

    # Assert
    assert set(yaml.safe_load(rendered)["services"]) == {
        "nginx",
        "web-alice",
        "web-bob",
        "web-carol",
    }
    assert "OSPREY_AUTH" not in rendered
    assert AUTH_ENV_FILENAME not in rendered


def test_auth_sidecar_service_is_the_only_service_authentication_adds() -> None:
    """Turning auth on adds exactly one service — `auth` — and changes nothing about
    the per-user set. The name is deliberately not `web-auth`: usernames render as
    `web-<user>`, so a roster user named "auth" would collide with that spelling but
    cannot collide with this one."""
    # Act
    services = set(_compose(_auth_config())["services"])

    # Assert
    assert services == {"auth", "nginx", "web-alice", "web-bob", "web-carol"}


def test_auth_sidecar_service_serves_a_single_uvicorn_bound_to_loopback() -> None:
    """The sidecar binds the loopback address on `auth.port` under host networking,
    exactly like the per-user apps: nginx shares the netns and is the only intended
    caller, so the login surface is unreachable off-host on its own port.

    Single process, not a worker pool — the revocation list and the brute-force
    throttle are in-memory, so a second worker would answer from a different view of
    both (a logged-out session still valid on whichever worker missed the logout).
    """
    # Act
    auth = _compose(_auth_config(port=9411))["services"]["auth"]

    # Assert
    assert auth["command"] == [
        "uvicorn",
        "osprey.services.auth_sidecar.app:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        "9411",
    ]
    assert "--workers" not in auth["command"]
    assert auth["network_mode"] == "host"


def test_auth_sidecar_service_image_defaults_to_the_local_build_tag() -> None:
    """Local mode: no `auth.image`, so the service names the tag the runtime build
    produces — the persona-image pattern (`<project>-<name>:local`) applied to the
    sidecar, whose project is resolve_personas()'s default `<prefix>-assistant`."""
    # Act
    auth = _compose(_auth_config())["services"]["auth"]

    # Assert
    assert auth["image"] == "dls-assistant-auth:local"


def test_auth_sidecar_service_image_pins_a_configured_registry_image() -> None:
    """Registry mode: `auth.image` is used verbatim — publishing it is the facility
    CI's job, and nothing is built on the deploy host."""
    # Arrange
    image = "git.dls.example.org:5050/physics/production/dls-auth:2026.8.1"

    # Act
    auth = _compose(_auth_config(image=image))["services"]["auth"]

    # Assert
    assert auth["image"] == image


def test_auth_sidecar_service_carries_the_roster_in_declaration_order() -> None:
    """The sidecar answers for exactly the users nginx routes to, from the same
    resolved `services` list — a user cannot exist in one and be missing from the
    other. Order is the roster's declaration order, which the login page reuses."""
    # Act
    auth = _compose(_auth_config(["carol", "alice"]))["services"]["auth"]

    # Assert
    assert f"{ENV_USERS}=carol,alice" in auth["environment"]


def test_auth_sidecar_service_environment_is_exactly_the_non_secret_settings() -> None:
    """The inline `environment:` block is pinned key for key, and every key in it is
    non-secret. This is the assertion that fails if a later task reaches for a secret
    here instead of `.env.auth`: compose interpolates `environment:` values (a stored
    `scrypt$...` hash would be eaten by `$` expansion) and, unlike an env_file, the
    values land in the committed compose artifact."""
    # Act
    auth = _compose(_auth_config())["services"]["auth"]

    # Assert
    assert _env_names(auth) == [
        "TZ",
        ENV_METHOD,
        ENV_SESSION_LIFETIME,
        ENV_TLS_ENABLED,
        ENV_EXTERNAL_ORIGIN,
        ENV_USERS,
    ]


def test_auth_sidecar_service_tls_flag_and_origin_follow_the_deployment() -> None:
    """`OSPREY_AUTH_TLS_ENABLED` is not about the sidecar's own listener (nginx
    terminates TLS; nothing is mounted here) — it decides the `Secure` attribute on
    the cookies it issues, so it must track the public origin, not the local bind."""
    # Arrange
    plain = _auth_config()
    secured = _auth_config()
    secured["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    plain_env = _compose(plain)["services"]["auth"]["environment"]
    secured_env = _compose(secured)["services"]["auth"]["environment"]

    # Assert
    assert f"{ENV_TLS_ENABLED}=false" in plain_env
    assert f"{ENV_EXTERNAL_ORIGIN}=http://dls-deploy.dls.example.org:9080" in plain_env
    assert f"{ENV_TLS_ENABLED}=true" in secured_env
    assert f"{ENV_EXTERNAL_ORIGIN}=https://dls-deploy.dls.example.org" in secured_env


def test_auth_sidecar_service_names_oidc_credential_variables_only_in_oidc_mode() -> None:
    """`oidc` mode adds the issuer plus the NAMES of the two variables holding the
    client credentials — one level of indirection, so the credentials themselves stay
    in `.env.auth` under whatever names the facility already uses.

    Password mode emits none of it rather than an inert empty issuer, and the whole
    OIDC group is gated on the mode: the password roster here still declares an
    `oidc_subject` left over from an earlier posture, and none of it renders.
    """
    # Arrange
    issuer = "https://sso.dls.example.org/realms/facility"
    password_roster = [{"name": "alice", "index": 0, "oidc_subject": "alice@dls.example.org"}]

    # Act
    password_env = _compose(_auth_config(password_roster))["services"]["auth"]["environment"]
    oidc_env = _compose(
        _auth_config(
            method="oidc",
            oidc={
                "issuer": issuer,
                "client_id_env": "DLS_SSO_CLIENT_ID",
                "client_secret_env": "DLS_SSO_CLIENT_SECRET",
            },
        )
    )["services"]["auth"]["environment"]

    # Assert
    assert not [line for line in password_env if "OIDC" in line]
    assert f"{ENV_OIDC_ISSUER}={issuer}" in oidc_env
    assert f"{ENV_OIDC_CLIENT_ID_ENV}=DLS_SSO_CLIENT_ID" in oidc_env
    assert f"{ENV_OIDC_CLIENT_SECRET_ENV}=DLS_SSO_CLIENT_SECRET" in oidc_env
    # The variable names are named; the credentials behind them never appear.
    assert not [line for line in oidc_env if line.startswith("DLS_SSO_CLIENT")]


def test_auth_sidecar_service_maps_each_users_oidc_subject_from_the_roster() -> None:
    """The non-secret half of the roster->identity mapping travels from the roster
    entry to the sidecar's environment under the name `env_var_suffix()` gives it —
    asserted against that function rather than a hardcoded literal, so this test
    cannot quietly disagree with the mapping that keys the password hashes.

    The value is quoted as a whole `KEY=value` scalar because a subject is typically
    an email or a DN, and a DN's `": "` would otherwise derail the YAML parse. A
    roster user who declares no subject gets no line at all, rather than an empty one
    that would map every unidentified caller onto them.
    """
    # Arrange
    config = _auth_config(
        [
            {"name": "alice", "index": 0, "oidc_subject": "cn=Alice Smith, ou=Users"},
            {"name": "dana-b", "index": 1, "oidc_subject": "dana@dls.example.org"},
            "bob",
        ],
        method="oidc",
        oidc={"issuer": "https://sso.dls.example.org"},
    )

    # Act
    auth_env = _compose(config)["services"]["auth"]["environment"]

    # Assert — the suffix is whatever env_var_suffix() says it is, including for
    # the hyphenated name, which is where a hand-rolled second mapping would drift.
    assert (
        f"{ENV_OIDC_SUBJECT_PREFIX}{env_var_suffix('alice')}=cn=Alice Smith, ou=Users" in auth_env
    )
    assert f"{ENV_OIDC_SUBJECT_PREFIX}{env_var_suffix('dana-b')}=dana@dls.example.org" in auth_env
    assert not [
        line
        for line in auth_env
        if line.startswith(f"{ENV_OIDC_SUBJECT_PREFIX}{env_var_suffix('bob')}")
    ]


def test_auth_sidecar_service_omits_the_oidc_claim_unless_the_facility_names_one() -> None:
    """No configured claim means no env line, so the sidecar applies its own
    documented default. Restating that default in the renderer would give the
    deployment two places to change it and one of them would eventually be wrong."""
    # Act
    auth_env = _compose(
        _auth_config(method="oidc", oidc={"issuer": "https://sso.dls.example.org"})
    )["services"]["auth"]["environment"]

    # Assert
    assert not [line for line in auth_env if line.startswith("OSPREY_AUTH_OIDC_CLAIM")]


def test_auth_sidecar_service_names_a_configured_non_default_oidc_claim() -> None:
    """An IdP that carries the mappable identity somewhere other than `sub` (a
    facility SSO keyed on `email` or `preferred_username`) is configured by name, and
    that name reaches the sidecar."""
    # Act
    auth_env = _compose(
        _auth_config(
            method="oidc",
            oidc={"issuer": "https://sso.dls.example.org", "claim": "preferred_username"},
        )
    )["services"]["auth"]["environment"]

    # Assert
    assert "OSPREY_AUTH_OIDC_CLAIM=preferred_username" in auth_env


def test_auth_context_reads_the_oidc_claim_and_leaves_it_none_when_unusable() -> None:
    """The parse side of the claim: configured strings come through, and anything
    unusable (absent, blank, wrong-typed) resolves to None — this module's defensive
    -read convention, and the value that makes the sidecar's default apply."""
    # Arrange
    configured = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    configured["auth"] = {"method": "oidc", "oidc": {"claim": "email"}}
    malformed = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    malformed["auth"] = {"method": "oidc", "oidc": {"claim": ["not", "a", "string"]}}
    blank = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    blank["auth"] = {"method": "oidc", "oidc": {"claim": "   "}}

    # Act / Assert
    assert _auth_tls_context(configured)["auth_oidc_claim"] == "email"
    assert _auth_tls_context(malformed)["auth_oidc_claim"] is None
    assert _auth_tls_context(blank)["auth_oidc_claim"] is None
    assert (
        _auth_tls_context(copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"])[
            "auth_oidc_claim"
        ]
        is None
    )


def test_auth_sidecar_service_healthcheck_probes_its_own_health_route() -> None:
    """`/health` answers 200 even when the sidecar is unconfigured and refusing
    everything, so the container stays up to explain the lockout instead of
    crash-looping. Probed in-image with python: `curl` is not in an OSPREY service
    image (the nginx service's curl probe runs in the nginx image)."""
    # Act
    auth = _compose(_auth_config(port=9411))["services"]["auth"]

    # Assert
    probe = auth["healthcheck"]["test"]
    assert probe[0] == "CMD-SHELL"
    assert "http://127.0.0.1:9411/health" in probe[1]
    assert "curl" not in probe[1]


def test_auth_env_isolation_secrets_reach_the_sidecar_only_through_env_file() -> None:
    """`.env.auth` (0600, sidecar-only) is the sidecar's whole secret surface, and it
    arrives as a service-level `env_file`. That is the only path that survives the
    values intact: compose interpolates `environment:` entries, and a stored scrypt
    hash (`scrypt$16384$8$1$salt$key`) is full of `$`."""
    # Act
    rendered = render_web_terminals(_auth_config())["docker-compose.web.yml"]
    auth = yaml.safe_load(rendered)["services"]["auth"]

    # Assert
    assert auth["env_file"] == AUTH_ENV_FILENAME
    for secret_var in (ENV_SESSION_SECRET, ENV_STATE_SECRET, ENV_PW_HASH_PREFIX):
        assert secret_var not in rendered


def test_auth_env_isolation_sidecar_never_reads_env_production() -> None:
    """The inverse of the isolation rule: `.env.users` is the per-user
    containers' shared env file (provider keys, facility credentials). The login
    surface has no business reading it, and pulling it in would also drag the agent
    tier's variables into the process that mints sessions."""
    # Act
    auth = _compose(_auth_config())["services"]["auth"]

    # Assert
    assert ".env.users" not in str(auth.get("env_file"))
    assert not [name for name in _env_names(auth) if name.startswith("ANTHROPIC")]


def test_auth_env_isolation_no_web_service_carries_an_auth_variable() -> None:
    """The proposal's central isolation assertion. Every per-user container runs the
    agent; the session signing key would let any user's agent mint a valid cookie for
    every other user, which is exactly the isolation this feature establishes. So no
    `web-*` service may carry an `OSPREY_AUTH_*` variable inline or reach `.env.auth`
    through an env_file — while keeping the `.env.users` it has today."""
    # Act
    compose = _compose(_auth_config())

    # Assert
    for name, service in compose["services"].items():
        if not name.startswith("web-"):
            continue
        assert service["env_file"] == ".env.users"
        assert AUTH_ENV_FILENAME not in str(service["env_file"])
        assert not [env for env in _env_names(service) if env.startswith("OSPREY_AUTH")]


def test_auth_sidecar_carries_the_env_auth_digest_as_a_label() -> None:
    """The digest of `.env.auth` is rendered INTO the sidecar's service definition.

    Compose implementations bake env_file content into a container at creation
    time and only reliably recreate on a service-DEFINITION change — an edit to
    the file alone recreates nothing on podman-compose. Stamping the content
    digest as a label turns a file edit into a definition change, the one
    recreate trigger every implementation honours."""
    # Act
    auth = yaml.safe_load(
        render_web_terminals(_auth_config(), auth_env_digest="a" * 64)["docker-compose.web.yml"]
    )["services"]["auth"]

    # Assert
    assert auth["labels"][AUTH_ENV_DIGEST_LABEL] == "a" * 64


def test_auth_sidecar_digest_label_is_the_only_thing_a_digest_change_touches() -> None:
    """Two renders differing only in the digest must differ only in the label:
    the recreate the label triggers has to be scoped to the sidecar, and any
    other drift would bounce services that cannot see `.env.auth` at all."""
    # Act
    first = yaml.safe_load(
        render_web_terminals(_auth_config(), auth_env_digest="a" * 64)["docker-compose.web.yml"]
    )
    second = yaml.safe_load(
        render_web_terminals(_auth_config(), auth_env_digest="b" * 64)["docker-compose.web.yml"]
    )

    # Assert
    assert first["services"]["auth"] != second["services"]["auth"]
    first["services"]["auth"].pop("labels")
    second["services"]["auth"].pop("labels")
    assert first == second


def test_auth_sidecar_digest_label_is_omitted_when_no_digest_is_supplied() -> None:
    """`render_web_terminals` without a digest (the `osprey scaffold
    web-terminals render` path, which has no project root to digest) emits no
    DIGEST label — the deploy path re-renders with a current digest on every
    `osprey up`, so a digest-less scaffold render can never reach a running
    stack with a stale one.

    The sidecar does still carry a `labels:` block: the repo-identity label is
    unconditional (it says which checkout the container belongs to, which is
    true whether or not a digest was supplied). So the assertion is about the
    digest key's absence, not about the block's."""
    # Act
    auth = _compose(_auth_config())["services"]["auth"]

    # Assert
    assert AUTH_ENV_DIGEST_LABEL not in (auth.get("labels") or {})


def test_method_none_render_is_byte_identical_with_and_without_a_digest() -> None:
    """With authentication off there is no sidecar and no env_file to track:
    a passed digest must leave the render untouched (and must not leak into
    any other service's definition)."""
    # Act
    plain = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))
    digested = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG), auth_env_digest="c" * 64)

    # Assert
    assert plain == digested
    assert "c" * 64 not in digested["docker-compose.web.yml"]


def test_render_refuses_a_non_hex_auth_env_digest() -> None:
    """The digest is templated verbatim into compose YAML (inside quotes, no
    escaping), so anything but lowercase sha256 hex could smuggle YAML
    structure into the service definition. Unreachable from today's producers
    (hashlib hexdigest or None), but render_web_terminals is a public seam."""
    with pytest.raises(ValueError, match="sha256 hex digest"):
        render_web_terminals(_auth_config(), auth_env_digest='x"\n    privileged: true\n    y: "')


def test_render_auth_on_refuses_a_roster_name_outside_the_username_charset() -> None:
    """Defense in depth on the render path itself. The charset is enforced by lint
    (skippable with `--no-lint`) and by credential provisioning (password mode only),
    so a malformed name could reach the renderer with neither having run — and the
    name is written literally into an nginx location key and the verify subrequest's
    `?user=` value. `bob&user=alice` would render a query carrying two user
    parameters, making the authorized identity a function of how the sidecar's query
    parser breaks the tie. With auth on, that is a refusal, not a render."""
    # Arrange
    injected = _auth_config(["alice", "bob&user=alice"])
    uppercase = _auth_config(["Alice"])

    # Act / Assert
    with pytest.raises(ValueError, match="modules.web_terminals.users") as exc_info:
        render_web_terminals(injected)
    message = str(exc_info.value)
    # The offender is named, and only the offender — a message that just says
    # "a username is invalid" leaves the operator grepping a 40-user roster.
    assert "bob&user=alice" in message
    assert "'alice'" not in message
    assert "auth.method" in message
    with pytest.raises(ValueError, match="Alice"):
        render_web_terminals(uppercase)


def test_render_auth_off_still_renders_a_roster_name_outside_the_charset() -> None:
    """The gate is scoped to authentication being on. With `auth.method: none` the
    username is a routing label rather than an authorization identity, and every
    deployment rendering such a name today must keep rendering — lint still reports
    it. Tightening this would be a silent breaking change for no security gain."""
    # Arrange
    config = _config(["alice", "bob.jones"])

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    assert "web-bob.jones" in compose["services"]


def test_render_auth_on_charset_gate_rejects_a_trailing_newline_in_a_roster_name() -> None:
    """`$` in the charset pattern also matches *before* a trailing newline, so a
    `.match()` gate would accept `"alice\\n"` — a name that renders into an nginx
    location key mid-directive. The gate uses `.fullmatch()`; this pins that, because
    the difference is invisible in every other test's inputs."""
    # Act / Assert
    with pytest.raises(ValueError, match="modules.web_terminals.users"):
        render_web_terminals(_auth_config(["alice\n"]))


def test_render_auth_on_refuses_roster_names_that_collide_on_their_env_var_suffix() -> None:
    """`env_var_suffix()` is lossy: `alice-b` and `alice_b` both key `..._ALICE_B`, so
    the pair would share ONE `OSPREY_AUTH_PW_HASH_ALICE_B` — one user's password
    opening the other's terminal, the exact isolation failure this feature exists to
    prevent.

    Both names pass the charset check, so this needs its own gate. Downstream it is
    caught only on the password-mode deploy path; in `oidc` mode it surfaces as a
    whole-deployment 503 that names neither user. Render time knows both names, so it
    says both.
    """
    # Act / Assert
    with pytest.raises(ValueError, match="env-var suffix") as exc_info:
        render_web_terminals(_auth_config(["alice-b", "alice_b"], method="oidc"))
    message = str(exc_info.value)
    assert "alice-b" in message
    assert "alice_b" in message
    assert env_var_suffix("alice-b") in message


def test_render_auth_off_still_renders_roster_names_that_collide_on_their_suffix() -> None:
    """Scoped like the charset gate: with no authentication there are no per-user
    credential variables to collide over, so a roster that renders today keeps
    rendering. Lint still reports the collision."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_config(["alice-b", "alice_b"]))["docker-compose.web.yml"]
    )

    # Assert
    assert {"web-alice-b", "web-alice_b"} <= set(compose["services"])


# ---------------------------------------------------------------------------
# EVENTS panel -> per-user dispatcher credential
#
# The dispatcher's bearer is minted into the deploy `.env` and gates every
# dashboard DATA endpoint. It reaches a web terminal through the PER-USER
# `environment:` block rather than the shared `env_file: .env.production`,
# because that one file is handed to EVERY user: a token placed there would give
# a read-only persona a credential that can fire triggers. Gating on the
# persona's declared EVENTS panel is what keeps the tier boundary intact.
# ---------------------------------------------------------------------------

_DISPATCHER_TOKEN_LINE = "EVENT_DISPATCHER_TOKEN=${EVENT_DISPATCHER_TOKEN:-}"


def _events_persona_config() -> dict:
    """Two personas; only `readwrite` declares the EVENTS panel."""
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["default_persona"] = "readonly"
    web_terminals["personas"] = {
        "readonly": {"project": "dls-readonly", "project_path": "../dls-readonly"},
        "readwrite": {"project": "dls-readwrite", "project_path": "../dls-readwrite"},
    }
    web_terminals["users"] = [
        {"name": "alice", "index": 0, "persona": "readwrite"},
        {"name": "bob", "index": 1, "persona": "readonly"},
    ]
    return config


def test_events_panel_persona_gets_the_dispatcher_token() -> None:
    """A user whose persona declares the EVENTS panel gets the dispatcher bearer in
    its own `environment:` block, interpolated from the deploy `.env` at compose
    time so the secret is never written into a rendered artifact."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config(), dispatcher_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    assert _DISPATCHER_TOKEN_LINE in compose["services"]["web-alice"]["environment"]


def test_persona_without_events_panel_gets_no_dispatcher_token() -> None:
    """The tier boundary: a persona declaring no EVENTS panel must not receive a
    credential that can fire triggers -- which is exactly why this cannot live in
    the shared .env.production that every user's container reads."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config(), dispatcher_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    bob_env = compose["services"]["web-bob"]["environment"]
    assert not any("EVENT_DISPATCHER_TOKEN" in line for line in bob_env)


def test_render_without_dispatcher_personas_emits_no_token_line() -> None:
    """The no-project-root render path (`osprey scaffold web-terminals render`)
    passes no persona set, the same way it passes no auth digest -- so it emits no
    credential. Harmless: every `osprey up` re-renders through the artifacts seam,
    which resolves the set from disk."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config())["docker-compose.web.yml"]
    )

    # Assert
    for service in ("web-alice", "web-bob"):
        env = compose["services"][service]["environment"]
        assert not any("EVENT_DISPATCHER_TOKEN" in line for line in env)


# ---------------------------------------------------------------------------
# ARIEL config -> per-user database password
#
# `osprey up` mints a strong ARIEL_DB_PASSWORD into the deploy `.env`, and
# Postgres initializes its volume with it. Inside a web terminal, BOTH ARIEL
# consumers -- the panel's own server and the `ariel` MCP server -- derive their
# DSN through `resolve_ariel_dsn`, which reads that variable from the
# environment and otherwise falls back to the shipped default. A container that
# never receives it therefore authenticates with the wrong password and the
# ARIEL tab reports the database as unavailable.
#
# Per-user, not `.env.production`, for the same reason as the dispatcher token:
# that file is handed to every user alike and cannot express "the personas that
# configure ARIEL".
# ---------------------------------------------------------------------------

_ARIEL_PASSWORD_LINE = "ARIEL_DB_PASSWORD=${ARIEL_DB_PASSWORD:-ariel}"


def test_ariel_persona_gets_the_database_password() -> None:
    """A user whose persona configures ARIEL gets the minted Postgres password in
    its own `environment:` block, interpolated from the deploy `.env` at compose
    time so the secret is never written into a rendered artifact."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config(), ariel_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    assert _ARIEL_PASSWORD_LINE in compose["services"]["web-alice"]["environment"]


def test_persona_without_ariel_gets_no_database_password() -> None:
    """A persona that configures no logbook needs no logbook credential."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config(), ariel_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    bob_env = compose["services"]["web-bob"]["environment"]
    assert not any("ARIEL_DB_PASSWORD" in line for line in bob_env)


def test_render_without_ariel_personas_emits_no_password_line() -> None:
    """The no-project-root render path passes no persona set and so emits no
    credential, exactly as it does for the dispatcher token."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config())["docker-compose.web.yml"]
    )

    # Assert
    for service in ("web-alice", "web-bob"):
        env = compose["services"][service]["environment"]
        assert not any("ARIEL_DB_PASSWORD" in line for line in env)


# ---------------------------------------------------------------------------
# Write entitlement -> per-user Bluesky launch token
#
# `BLUESKY_LAUNCH_TOKEN` is what lets the `bluesky` MCP server arm a queue start
# on the operator's chat approval alone. Per-user for the same reason as the two
# credentials above, and more sharply: the shared `.env.production` is handed to
# every user's container alike, so a launch token placed there would hand
# queue-start capability -- physical hardware motion -- to the read-only
# personas that must never hold it.
# ---------------------------------------------------------------------------

_LAUNCH_TOKEN_LINE = "BLUESKY_LAUNCH_TOKEN=${BLUESKY_LAUNCH_TOKEN:-}"


# ---------------------------------------------------------------------------
# Archiver connector -> per-user store password
#
# `osprey up` mints the archiver store's password into the deploy `.env` under
# the name the connector block reads (`archiver.<type>.password_env`, which the
# control-assistant preset spells MONGO_ROOT_PASSWORD). The agent inside a web
# terminal authenticates with exactly that variable, and `.env.users` excludes
# service tokens by design -- so a container that is not handed it per-user
# reports "Environment variable 'MONGO_ROOT_PASSWORD' is not set" on every
# archiver read, while the single-user host path (which reads the whole `.env`)
# works. The grant carries the configured NAME, so a facility-run store under
# another variable is granted the same way.
# ---------------------------------------------------------------------------

_ARCHIVER_PASSWORD_LINE = "MONGO_ROOT_PASSWORD=${MONGO_ROOT_PASSWORD:-}"


def test_archiver_persona_gets_the_store_password() -> None:
    """A user whose persona reads a store password gets it in its own `environment:`
    block, interpolated from the deploy `.env` at compose time so the secret is
    never written into a rendered artifact."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(
            _events_persona_config(),
            archiver_password_personas={"readwrite": "MONGO_ROOT_PASSWORD"},
        )["docker-compose.web.yml"]
    )

    # Assert
    assert _ARCHIVER_PASSWORD_LINE in compose["services"]["web-alice"]["environment"]


def test_archiver_grant_carries_the_configured_variable_name() -> None:
    """A persona whose connector names another variable is handed THAT one."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(
            _events_persona_config(), archiver_password_personas={"readwrite": "FACILITY_DB_PW"}
        )["docker-compose.web.yml"]
    )

    # Assert
    alice_env = compose["services"]["web-alice"]["environment"]
    assert "FACILITY_DB_PW=${FACILITY_DB_PW:-}" in alice_env
    assert not any("MONGO_ROOT_PASSWORD" in line for line in alice_env)


def test_persona_without_an_archiver_password_gets_none() -> None:
    """A persona whose archiver reads no password needs no store credential."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(
            _events_persona_config(),
            archiver_password_personas={"readwrite": "MONGO_ROOT_PASSWORD"},
        )["docker-compose.web.yml"]
    )

    # Assert
    bob_env = compose["services"]["web-bob"]["environment"]
    assert not any("MONGO_ROOT_PASSWORD" in line for line in bob_env)


def test_render_without_archiver_personas_emits_no_password_line() -> None:
    """The no-project-root render path passes no persona map and so emits no
    credential, exactly as it does for the other three."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config())["docker-compose.web.yml"]
    )

    # Assert
    for service in ("web-alice", "web-bob"):
        env = compose["services"][service]["environment"]
        assert not any("MONGO_ROOT_PASSWORD" in line for line in env)


def test_persona_less_roster_entry_is_answered_from_the_deploy_config() -> None:
    """The zero-migration path (no personas; the web image IS the deploy project)
    reads the grant straight from the deploy config, as the other grants do."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["archiver"] = {
        "type": "mongodb_archiver",
        "mongodb_archiver": {"host": "localhost", "password_env": "MONGO_ROOT_PASSWORD"},
    }

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    assert _ARCHIVER_PASSWORD_LINE in compose["services"]["web-alice"]["environment"]


def test_entitled_persona_gets_the_launch_token() -> None:
    """A user whose persona is entitled to arm a queue start gets the launch token
    in its own `environment:` block, interpolated from the deploy `.env` at compose
    time so the secret is never written into a rendered artifact."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config(), launch_token_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    assert _LAUNCH_TOKEN_LINE in compose["services"]["web-alice"]["environment"]


def test_unentitled_persona_gets_no_launch_token() -> None:
    """The tier boundary: a persona outside the entitled set must not receive the
    credential that arms hardware motion -- which is exactly why this cannot live
    in the shared .env.production that every user's container reads."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config(), launch_token_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    bob_env = compose["services"]["web-bob"]["environment"]
    assert not any("BLUESKY_LAUNCH_TOKEN" in line for line in bob_env)


# A deployment that opted into a second plan lane (`bluesky.second_lane`) renders
# a second bridge with its own launch token, because each lane is a whole stack
# armed separately -- one shared token would let a launch approved against one
# machine be replayed against the other. An entitled persona is told both lanes
# and so is handed both tokens; the tier boundary is the same one.

_SECOND_LANE_TOKEN_LINE = "BLUESKY_LIVE_LAUNCH_TOKEN=${BLUESKY_LIVE_LAUNCH_TOKEN:-}"


def _two_lane_persona_config() -> dict:
    config = _events_persona_config()
    config["services"] = {
        "bluesky": {"port": 8090, "target": "va"},
        "bluesky_live": {"port": 8190, "target": "live"},
    }
    return config


def test_entitled_persona_on_a_two_lane_deployment_gets_both_launch_tokens() -> None:
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_two_lane_persona_config(), launch_token_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    alice_env = compose["services"]["web-alice"]["environment"]
    assert _LAUNCH_TOKEN_LINE in alice_env
    assert _SECOND_LANE_TOKEN_LINE in alice_env


def test_unentitled_persona_on_a_two_lane_deployment_gets_neither_token() -> None:
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_two_lane_persona_config(), launch_token_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    bob_env = compose["services"]["web-bob"]["environment"]
    assert not any("LAUNCH_TOKEN" in line for line in bob_env)


def test_single_lane_deployment_grants_lane_ones_token_alone() -> None:
    """The pre-lane render exactly: one token line, the historical name."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config(), launch_token_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    alice_env = compose["services"]["web-alice"]["environment"]
    assert [line for line in alice_env if "LAUNCH_TOKEN" in line] == [_LAUNCH_TOKEN_LINE]


def test_render_without_launch_token_personas_emits_no_token_line() -> None:
    """The no-project-root render path (`osprey scaffold web-terminals render`)
    passes no persona set, so no user is armed. Harmless: every `osprey up`
    re-renders through the artifacts seam, which resolves the set from disk."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config())["docker-compose.web.yml"]
    )

    # Assert
    for service in ("web-alice", "web-bob"):
        env = compose["services"][service]["environment"]
        assert not any("BLUESKY_LAUNCH_TOKEN" in line for line in env)


def test_persona_less_roster_is_armed_from_the_config_itself() -> None:
    """The zero-migration path -- roster entries that name no persona, where the web
    image IS the deploy project -- has no persona to look up, so entitlement is read
    from this same config with no disk read. That keeps the determinism contract
    while still arming a deployment that never adopted personas.
    """
    # Arrange
    config = copy.deepcopy(_config(["alice"]))
    config["control_system"] = {"writes_enabled": True}

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    assert _LAUNCH_TOKEN_LINE in compose["services"]["web-alice"]["environment"]


def test_persona_less_roster_without_writes_is_not_armed() -> None:
    """The same path, read-only: writes were never granted, so there is nothing to
    arm and no token is emitted."""
    # Act
    compose = yaml.safe_load(render_web_terminals(_config(["alice"]))["docker-compose.web.yml"])

    # Assert
    env = compose["services"]["web-alice"]["environment"]
    assert not any("BLUESKY_LAUNCH_TOKEN" in line for line in env)


# ---------------------------------------------------------------------------
# Graph store config -> per-user Neo4j password
#
# `osprey up` mints a strong GRAPHDB_PASSWORD into the deploy `.env`, and Neo4j
# initializes its volume with it. Inside a web terminal the consumer is the
# `graph` MCP server, which resolves what to dial and with which password
# through `resolve_graphdb_connection` -- that reads the variable from the
# environment and otherwise falls back to the shipped default, so a container
# that never receives it authenticates with the wrong password and every graph
# query fails.
#
# Per-user, not `.env.users`, for the same reason as the credentials above: that
# file is handed to every user alike and cannot say "the personas that configure
# a graph store". Note this grant crosses the tier boundary on purpose -- Neo4j
# has one write-capable account, and read-only-ness is enforced by the graph
# server's read transactions, not by withholding the password.
# ---------------------------------------------------------------------------

_GRAPHDB_PASSWORD_LINE = "GRAPHDB_PASSWORD=${GRAPHDB_PASSWORD:-ospreygraph}"


def test_graphdb_persona_gets_the_store_password() -> None:
    """A user whose persona configures a graph store gets the minted Neo4j password
    in its own `environment:` block, interpolated from the deploy `.env` at compose
    time so the secret is never written into a rendered artifact."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config(), graphdb_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    assert _GRAPHDB_PASSWORD_LINE in compose["services"]["web-alice"]["environment"]


def test_persona_without_graphdb_gets_no_store_password() -> None:
    """A persona that configures no graph store needs no graph-store credential."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config(), graphdb_personas={"readwrite"})[
            "docker-compose.web.yml"
        ]
    )

    # Assert
    bob_env = compose["services"]["web-bob"]["environment"]
    assert not any("GRAPHDB_PASSWORD" in line for line in bob_env)


def test_render_without_graphdb_personas_emits_no_store_password_line() -> None:
    """The no-project-root render path passes no persona set and so emits no
    credential, exactly as it does for the dispatcher token."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config())["docker-compose.web.yml"]
    )

    # Assert
    for service in ("web-alice", "web-bob"):
        env = compose["services"][service]["environment"]
        assert not any("GRAPHDB_PASSWORD" in line for line in env)


def test_persona_less_roster_gets_graphdb_password_from_the_config_itself() -> None:
    """The zero-migration path -- roster entries that name no persona, where the web
    image IS the deploy project -- has no persona to look up, so entitlement is read
    from this same config with no disk read."""
    # Arrange
    config = copy.deepcopy(_config(["alice"]))
    config["services"] = {"graphdb": {}}

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    assert _GRAPHDB_PASSWORD_LINE in compose["services"]["web-alice"]["environment"]


def test_persona_less_roster_without_graphdb_gets_no_password() -> None:
    """The same path with no graph store configured: nothing to dial, so nothing to
    authenticate with."""
    # Act
    compose = yaml.safe_load(render_web_terminals(_config(["alice"]))["docker-compose.web.yml"])

    # Assert
    env = compose["services"]["web-alice"]["environment"]
    assert not any("GRAPHDB_PASSWORD" in line for line in env)


# ---------------------------------------------------------------------------
# Telemetry ingest token -> EVERY per-user container
#
# The one telemetry credential a web terminal must hold, and the one place it
# can arrive. Each persona's rendered config.yml carries the ingest token
# reference verbatim -- the build defers it, because the STORE mints the value
# -- and the agent re-resolves it against the container's own environment at
# spawn. `.env.users` deliberately does not carry it
# (`env_production._build_env_production_subset`), so without this line the
# placeholder is still a placeholder inside the container and the terminal's
# strict startup check refuses to run: a crash-loop behind the proxy, on every
# deploy, for every user.
#
# Unconditional, unlike the four credentials above: every terminal runs an
# agent that emits telemetry as the same account, so there is no tier to
# express -- gating it on a persona set would only produce terminals that
# cannot start.
# ---------------------------------------------------------------------------

_INGEST_TOKEN_LINE = "ZO_INGEST_SA_TOKEN=${ZO_INGEST_SA_TOKEN:-}"


def test_every_user_container_receives_the_telemetry_ingest_token() -> None:
    """Both users, whatever their persona: the token is not an entitlement."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_events_persona_config())["docker-compose.web.yml"]
    )

    # Assert
    for service in ("web-alice", "web-bob"):
        assert _INGEST_TOKEN_LINE in compose["services"][service]["environment"]


def test_the_ingest_token_is_emitted_without_any_persona_set() -> None:
    """The no-project-root render path (`osprey scaffold web-terminals render`)
    passes no persona sets at all. The four entitlement credentials are omitted
    there; this one must not be, or that render produces a stack whose terminals
    cannot start."""
    # Act
    compose = yaml.safe_load(render_web_terminals(_config(["alice"]))["docker-compose.web.yml"])

    # Assert
    assert _INGEST_TOKEN_LINE in compose["services"]["web-alice"]["environment"]


def test_the_ingest_token_is_interpolated_never_written_into_the_artifact() -> None:
    """The secret itself never lands in a rendered file: compose resolves the
    reference at container-create time from the deploy `.env`, which is where
    `osprey up` harvests the token to."""
    # Act
    rendered = render_web_terminals(_config(["alice"]))["docker-compose.web.yml"]

    # Assert
    assert "${ZO_INGEST_SA_TOKEN" in rendered


def test_the_shipped_config_and_the_container_env_name_the_same_variable() -> None:
    """The end of the chain, pinned across the two files that have to agree.

    A persona's telemetry block names the variable; this compose service is the
    only thing that puts it in the container. Repoint one without the other --
    exactly what happened when the shipped configs moved off the store's root
    credential -- and every web terminal starts naming a variable nothing sets.
    """
    # Arrange
    import osprey

    shipped_config = Path(osprey.__file__).parent / "templates" / "project" / "config.yml.j2"
    shipped = shipped_config.read_text(encoding="utf-8")
    declared = re.search(r"^ *password: \$\{([A-Za-z_][A-Za-z0-9_]*)\}$", shipped, re.MULTILINE)
    assert declared, "the shipped telemetry block no longer names a bare ${VAR} password"

    # Act
    compose = yaml.safe_load(render_web_terminals(_config(["alice"]))["docker-compose.web.yml"])

    # Assert
    environment = compose["services"]["web-alice"]["environment"]
    delivered = {entry.split("=", 1)[0] for entry in environment}
    assert declared.group(1) in delivered


# ---------------------------------------------------------------------------
# Per-user operator secret carrier (PLAN Task 5.1)
#
# nginx is the only thing that may hold a user's operator secret: it injects it
# as a request header on the way to that user's terminal, from a snippet whose
# value envsubst fills in at container start. Two properties are load-bearing
# and pinned below: NO secret value may reach a rendered artifact, and a render
# with no secrets in hand (`osprey scaffold web-terminals render`) must be
# byte-identical to what it produced before this carrier existed.
# ---------------------------------------------------------------------------


def _capture_compose_context(config: dict, **kwargs) -> dict:
    """Render *config* and return the context handed to docker-compose.web.yml.j2.

    The per-service `terminal_secret_env`/`external_origin` keys are consumed by
    the compose template (Task 5.3), so the context is the seam where this task's
    contract is observable. Captured by wrapping the real template's `render`,
    so the real templates still render and the real output is still produced.
    """
    captured: dict = {}
    real_get_template = Environment.get_template

    def spy(self, name, *args, **kw):  # type: ignore[no-untyped-def]
        template = real_get_template(self, name, *args, **kw)
        if name == "docker-compose.web.yml.j2":
            real_render = template.render

            def capture(**ctx):  # type: ignore[no-untyped-def]
                captured.update(ctx)
                return real_render(**ctx)

            template.render = capture
        return template

    with patch.object(Environment, "get_template", spy):
        render_web_terminals(config, **kwargs)
    assert captured, "the compose template was never rendered"
    return captured


def _secrets_for(users: list[str]) -> dict[str, str]:
    """A plausible `{username: secret}` mapping, with per-user distinct values."""
    return {user: f"s3cr3t-value-for-{user}" for user in users}


def test_render_without_terminal_secrets_returns_exactly_the_three_artifacts() -> None:
    """The no-secrets render path emits no nginx template snippets at all."""
    # Act
    artifacts = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert set(artifacts) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


def test_render_is_byte_identical_with_and_without_terminal_secrets() -> None:
    """Handing the render the secrets changes nothing about the three artifacts:
    the values live only in the deploy `.env`, and the snippets are additive."""
    # Arrange — auth on, so there are snippets for the addition to be additive to
    config = _auth_config(["alice", "bob", "carol"])

    # Act
    plain = render_web_terminals(copy.deepcopy(config))
    with_secrets = render_web_terminals(
        copy.deepcopy(config), terminal_secrets=_secrets_for(["alice", "bob", "carol"])
    )

    # Assert
    for artifact in ("docker-compose.web.yml", "nginx/nginx.conf", "nginx/landing.html"):
        assert with_secrets[artifact] == plain[artifact]


def test_terminal_secrets_render_one_snippet_per_gated_roster_user() -> None:
    """One `secret-<user>.conf.template` per GATED roster user, in the templates
    directory the nginx service bind-mounts at /etc/nginx/templates."""
    # Act
    artifacts = render_web_terminals(
        _auth_config(["alice", "bob", "carol"]),
        terminal_secrets=_secrets_for(["alice", "bob", "carol"]),
    )

    # Assert
    assert set(artifacts) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
        "nginx/templates/secret-alice.conf.template",
        "nginx/templates/secret-bob.conf.template",
        "nginx/templates/secret-carol.conf.template",
    }


def test_no_secret_snippet_is_written_for_a_location_nothing_gates() -> None:
    """A snippet nothing `include`s is a plaintext secret in the nginx container
    for no reader.

    nginx emits the include under exactly one predicate — `auth_method != "none"
    and not svc.login_exempt` — so with authentication off nothing includes
    anything, and a `login: false` entry is served ungated with the header
    CLEARED. Neither gets a file. The roster REFUSALS still cover everyone: each
    user's own container reads its own secret whatever the perimeter does.
    """
    # Arrange
    secrets = _secrets_for(["alice", "bob", "carol"])

    # Act
    auth_off = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG), terminal_secrets=secrets)
    mixed = render_web_terminals(
        _mixed_login_config("kiosk"), terminal_secrets=_secrets_for(["alice", "kiosk"])
    )

    # Assert
    assert [path for path in auth_off if path.startswith("nginx/templates/")] == []
    assert [path for path in mixed if path.startswith("nginx/templates/")] == [
        "nginx/templates/secret-alice.conf.template"
    ]
    # The ungated user's secret is nowhere in the render — not as a file, and not
    # as the variable an include would have referenced.
    for content in mixed.values():
        assert "secret-kiosk.conf" not in content


def test_every_secret_snippet_is_included_by_exactly_one_location() -> None:
    """The set of snippets equals the set of includes, on a mixed roster.

    Stated as an equality rather than as two containments: an include with no
    file crash-loops nginx at start, and a file with no include is the dead
    secret above. Both directions have to hold at once.
    """
    # Arrange
    config = _mixed_login_config("kiosk")

    # Act
    artifacts = render_web_terminals(config, terminal_secrets=_secrets_for(["alice", "kiosk"]))

    # Assert
    materialized = {
        Path(path).name.removesuffix(".template")
        for path in artifacts
        if path.startswith("nginx/templates/")
    }
    included = {
        Path(path).name
        for path in re.findall(
            r"^\s*include (/etc/nginx/osprey/\S+);",
            artifacts["nginx/nginx.conf"],
            flags=re.MULTILINE,
        )
    }
    assert materialized == included == {"secret-alice.conf"}


def test_a_login_exempt_user_still_needs_a_minted_secret() -> None:
    """Not writing the snippet does not relax the roster gate.

    The ungated user's own container authenticates every request against that
    secret — the `?token=` exchange is the only way into it — so a roster where
    it is missing is still refused, snippet or no snippet.
    """
    # Act / Assert
    with pytest.raises(ValueError, match="OSPREY_TERMINAL_SECRET_KIOSK"):
        render_web_terminals(_mixed_login_config("kiosk"), terminal_secrets=_secrets_for(["alice"]))


def test_terminal_secret_snippet_names_the_variable_never_the_value() -> None:
    """The snippet is a single `proxy_set_header` naming the per-user variable.

    envsubst substitutes it at nginx start from the deploy `.env`; the value
    itself must not appear in ANY rendered artifact, which is the whole reason
    the carrier is a template rather than a rendered conf.
    """
    # Arrange
    secrets = _secrets_for(["alice", "bob", "carol"])

    # Act
    artifacts = render_web_terminals(
        _auth_config(["alice", "bob", "carol"]), terminal_secrets=secrets
    )

    # Assert
    snippet = artifacts["nginx/templates/secret-alice.conf.template"]
    # QUOTED: a present-but-empty variable (which the compose carrier's
    # `${VAR:-}` guarantees for a blank .env entry) then substitutes to an empty
    # header value, which only that user's app refuses. Unquoted it would emit
    # `proxy_set_header X-Osprey-Terminal-Secret ;`, which nginx will not parse
    # — one blank variable taking every terminal in the deployment down.
    assert 'proxy_set_header X-Osprey-Terminal-Secret "${OSPREY_TERMINAL_SECRET_ALICE}";' in snippet
    for content in artifacts.values():
        for secret in secrets.values():
            assert secret not in content


def test_terminal_secret_snippet_carries_only_that_users_variable() -> None:
    """No user's snippet may name another user's variable: the whole point of a
    per-location include is that alice's terminal never sees bob's secret."""
    # Act
    artifacts = render_web_terminals(
        _auth_config(["alice", "bob", "carol"]),
        terminal_secrets=_secrets_for(["alice", "bob", "carol"]),
    )

    # Assert
    snippet = artifacts["nginx/templates/secret-bob.conf.template"]
    assert "OSPREY_TERMINAL_SECRET_ALICE" not in snippet
    assert "OSPREY_TERMINAL_SECRET_CAROL" not in snippet
    assert snippet.count("proxy_set_header") == 1


def test_terminal_secret_variable_follows_the_shared_env_var_suffix_mapping() -> None:
    """The suffix comes from `env_var_suffix`, the one definition minting,
    rendering and the sidecar's own lookup all share — never a second
    `upper|replace` free to drift from it."""
    # Arrange
    config = _auth_config(["alice-b"])

    # Act
    artifacts = render_web_terminals(config, terminal_secrets=_secrets_for(["alice-b"]))

    # Assert: the FILE is keyed by the username, the VARIABLE by its suffix
    snippet = artifacts["nginx/templates/secret-alice-b.conf.template"]
    assert f"${{OSPREY_TERMINAL_SECRET_{env_var_suffix('alice-b')}}}" in snippet


def test_render_refuses_a_roster_user_with_no_terminal_secret() -> None:
    """Fail closed: a user whose secret is missing would get an empty header
    injected and a terminal that refuses every request behind the proxy."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act / Assert
    with pytest.raises(ValueError, match="OSPREY_TERMINAL_SECRET_BOB"):
        render_web_terminals(config, terminal_secrets=_secrets_for(["alice", "carol"]))


def test_render_refuses_a_blank_terminal_secret() -> None:
    """An empty value is absence, not a secret — the same reading the app's own
    credential holder applies to `OSPREY_TERMINAL_SECRET`."""
    # Arrange
    secrets = _secrets_for(["alice", "bob", "carol"]) | {"bob": "   "}

    # Act / Assert
    with pytest.raises(ValueError, match="OSPREY_TERMINAL_SECRET_BOB"):
        render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG), terminal_secrets=secrets)


def test_render_ignores_secrets_for_users_no_longer_on_the_roster() -> None:
    """A decommissioned user's secret may still sit in the deploy `.env`; the
    roster decides who gets a snippet, so no header can be injected for them."""
    # Arrange
    secrets = _secrets_for(["alice", "bob", "carol", "dave"])

    # Act
    artifacts = render_web_terminals(
        _auth_config(["alice", "bob", "carol"]), terminal_secrets=secrets
    )

    # Assert
    assert "nginx/templates/secret-dave.conf.template" not in artifacts
    for content in artifacts.values():
        assert "OSPREY_TERMINAL_SECRET_DAVE" not in content


def test_empty_terminal_secrets_on_an_empty_roster_render_no_snippets() -> None:
    """A roster-less deployment has nothing to inject and must still render."""
    # Act
    artifacts = render_web_terminals(_config([]), terminal_secrets={})

    # Assert
    assert not [path for path in artifacts if path.startswith("nginx/templates/")]


def test_each_service_names_its_own_terminal_secret_variable() -> None:
    """The compose template (Task 5.3) puts `OSPREY_TERMINAL_SECRET` into each
    user's container from this per-service variable name."""
    # Act
    context = _capture_compose_context(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert [service["terminal_secret_env"] for service in context["services"]] == [
        f"OSPREY_TERMINAL_SECRET_{env_var_suffix(name)}" for name in ("alice", "bob", "carol")
    ]


def test_service_terminal_secret_variable_is_present_without_any_secrets() -> None:
    """The variable NAME is not a secret, so it is threaded on every render —
    the compose reference resolves to empty when the deploy `.env` has none."""
    # Act
    context = _capture_compose_context(_config(["alice"]))

    # Assert
    assert context["services"][0]["terminal_secret_env"] == "OSPREY_TERMINAL_SECRET_ALICE"


def test_each_service_carries_the_deployments_external_origin() -> None:
    """The app checks a mutating request's `Origin` against this value, so it
    must be the origin a browser actually sends — port-full over plain HTTP."""
    # Act
    context = _capture_compose_context(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    for service in context["services"]:
        assert service["external_origin"] == "http://dls-deploy.dls.example.org:9080"


def test_service_external_origin_drops_the_port_under_tls() -> None:
    """With TLS on, 443 is implicit — the canonical origin serialization, and
    the one a browser sends. A port-full value would fail every Origin check."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/site.crt",
        "key": "/etc/nginx/certs/site.key",
    }

    # Act
    context = _capture_compose_context(config)

    # Assert
    for service in context["services"]:
        assert service["external_origin"] == "https://dls-deploy.dls.example.org"


def test_service_external_origin_is_the_one_the_landing_link_uses() -> None:
    """One origin for the whole deployment: a per-service value that differed
    from `landing_url` would make a link off the landing page fail the app's
    own Origin check."""
    # Act
    context = _capture_compose_context(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert {service["external_origin"] for service in context["services"]} == {
        context["landing_url"]
    }


def test_clear_nginx_templates_dir_removes_stale_snippets(tmp_path: Path) -> None:
    """A decommissioned user's snippet must not survive a re-render: nginx's
    entrypoint envsubsts every file in the mounted templates directory, so a
    leftover would keep injecting a header for a user who no longer exists."""
    # Arrange
    templates = tmp_path / "nginx" / "templates"
    templates.mkdir(parents=True)
    stale = templates / "secret-dave.conf.template"
    stale.write_text("proxy_set_header X-Osprey-Terminal-Secret ${OSPREY_TERMINAL_SECRET_DAVE};\n")
    sibling = tmp_path / "nginx" / "nginx.conf"
    sibling.write_text("server {}\n")

    # Act
    removed = clear_nginx_templates_dir(tmp_path)

    # Assert
    assert removed == [stale]
    assert not stale.exists()
    assert sibling.exists(), "only the templates directory is cleared"


def test_clear_nginx_templates_dir_tolerates_a_first_render(tmp_path: Path) -> None:
    """Nothing has been written yet on a fresh checkout; that is not an error."""
    # Act
    removed = clear_nginx_templates_dir(tmp_path)

    # Assert
    assert removed == []


def test_render_refuses_a_roster_name_that_is_not_one_snippet_filename() -> None:
    """The name becomes a filename and an nginx `include` argument, so a name
    carrying a path separator would write outside the cleared templates
    directory. Refused whatever the auth method: this carrier is rendered even
    with `auth.method: none`, where the strict username charset is not enforced.
    """
    # Arrange
    config = _config(["alice", "../../etc/nginx/conf.d/evil"])

    # Act / Assert
    with pytest.raises(ValueError, match="cannot carry an operator secret"):
        render_web_terminals(
            config, terminal_secrets={"alice": "a", "../../etc/nginx/conf.d/evil": "b"}
        )


def test_terminal_secret_snippets_stay_inside_the_templates_directory() -> None:
    """Every emitted path is one file directly under the directory the write
    seam clears — nothing a re-render would leave behind somewhere else."""
    # Act
    artifacts = render_web_terminals(
        _auth_config(["alice", "bob", "carol"]),
        terminal_secrets=_secrets_for(["alice", "bob", "carol"]),
    )

    # Assert
    snippets = [path for path in artifacts if path.startswith("nginx/templates/")]
    assert len(snippets) == 3
    for path in snippets:
        assert Path(path).parent.as_posix() == "nginx/templates"


def test_render_refuses_a_dotted_roster_name_that_derives_an_illegal_variable() -> None:
    """`env_var_suffix` maps only '-' to '_', so a dot survives into the
    variable. envsubst does not recognize `${OSPREY_TERMINAL_SECRET_ALICE.B}`,
    leaves it verbatim, and nginx dies at start with `invalid variable name` —
    an opaque crash loop the render can refuse instead."""
    # Arrange
    config = _config(["alice.b"])

    # Act / Assert
    with pytest.raises(ValueError, match=r"OSPREY_TERMINAL_SECRET_ALICE\.B"):
        render_web_terminals(config, terminal_secrets={"alice.b": "a-secret"})


def test_a_dotted_roster_name_still_renders_without_terminal_secrets() -> None:
    """The refusal is scoped to the carrier: a name that renders today must keep
    rendering on the path that carries no secrets."""
    # Act
    artifacts = render_web_terminals(_config(["alice.b"]))

    # Assert
    assert "web-alice.b" in yaml.safe_load(artifacts["docker-compose.web.yml"])["services"]


def test_render_refuses_roster_names_that_collide_on_their_secret_variable() -> None:
    """`alice-b` and `alice_b` key one variable, so both terminals would be
    handed the SAME secret — the isolation this carrier exists to establish.
    Refused with authentication OFF too, where the sidecar's own collision gate
    (`_check_roster_env_var_collisions`) never runs."""
    # Arrange
    config = _config(["alice-b", "alice_b"])
    secrets = _secrets_for(["alice-b", "alice_b"])

    # Act / Assert
    with pytest.raises(ValueError, match="OSPREY_TERMINAL_SECRET_ALICE_B"):
        render_web_terminals(config, terminal_secrets=secrets)


def test_case_colliding_roster_names_are_refused_too() -> None:
    """`Alice` and `alice` are two roster entries and one variable."""
    # Act / Assert
    with pytest.raises(ValueError, match="OSPREY_TERMINAL_SECRET_ALICE"):
        render_web_terminals(
            _config(["Alice", "alice"]), terminal_secrets=_secrets_for(["Alice", "alice"])
        )


def test_clear_nginx_templates_dir_removes_a_nested_stale_snippet(tmp_path: Path) -> None:
    """nginx's entrypoint walks the mounted directory recursively, so a snippet
    one level down is exactly as live as one at the top."""
    # Arrange
    nested = tmp_path / "nginx" / "templates" / "old" / "secret-dave.conf.template"
    nested.parent.mkdir(parents=True)
    nested.write_text("proxy_set_header X-Osprey-Terminal-Secret ${OSPREY_TERMINAL_SECRET_DAVE};\n")

    # Act
    removed = clear_nginx_templates_dir(tmp_path)

    # Assert
    assert removed == [nested]
    assert not nested.exists()
    assert not nested.parent.exists()


def test_clear_nginx_templates_dir_leaves_an_empty_directory_behind(tmp_path: Path) -> None:
    """The nginx service bind-mounts this directory: a missing source is
    materialized by the runtime as a root-owned directory (or refused), which is
    a confusing failure for a deployment that simply has no snippets yet."""
    # Act
    clear_nginx_templates_dir(tmp_path)

    # Assert
    assert (tmp_path / "nginx" / "templates").is_dir()


# ---------------------------------------------------------------------------
# Task 5.2: the gate-bound operator-secret include
# ---------------------------------------------------------------------------


def _secret_include(user: str) -> str:
    """The include directive one user's gated location is expected to carry."""
    return f"include /etc/nginx/osprey/secret-{user}.conf;"


def _mixed_login_config(exempt: str) -> dict:
    """An auth-on roster where one entry declared `login: false`."""
    return _auth_config(
        [
            {"name": "alice", "index": 0},
            {"name": exempt, "index": 1, "login": False},
        ]
    )


def test_nginx_gated_location_includes_only_its_own_users_secret_snippet() -> None:
    """Each authenticated location pulls in that user's snippet and no other's.

    The snippet is what sets the operator-secret header, so a location that
    named a second user's file (or a second user's variable) would make one
    terminal reachable with another terminal's credential — the isolation this
    carrier exists to establish.
    """
    # Arrange
    users = ["alice", "bob"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        body = _location_body(nginx_conf, f"location /u/{user}/")
        assert _secret_include(user) in body
        assert terminal_secret_env_var(user) in body
        for other in users:
            if other != user:
                assert _secret_include(other) not in body
                assert terminal_secret_env_var(other) not in body


def test_nginx_secret_include_sits_inside_the_auth_guarded_branch() -> None:
    """The include is emitted between the `auth_request` and the `proxy_pass`.

    That ordering is the whole point of the task: the directive exists only in
    a location that gates, and a denied subrequest ends the request before the
    upstream connection is ever opened — so nothing that failed the gate can
    carry the header to a terminal.
    """
    # Act
    body = _directives(_location_body(_render_nginx(_auth_config(["alice"])), "location /u/alice/"))

    # Assert
    assert body.index("auth_request /_osprey_auth/alice;") < body.index(_secret_include("alice"))
    assert body.index(_secret_include("alice")) < body.index("proxy_pass ")


def test_nginx_secret_include_absent_when_auth_method_is_none() -> None:
    """With no gate in front of the app there is no authorized request for
    nginx to vouch for, so it injects nothing and the app's own token->cookie
    session is the only gate — the posture the render had before this carrier
    existed."""
    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "include " not in _directives(nginx_conf)
    assert "/etc/nginx/osprey/" not in nginx_conf


def test_nginx_login_exempt_location_gets_no_secret_and_keeps_its_cookie() -> None:
    """A `login: false` entry is served ungated, so it is treated exactly like
    an auth-off deployment: no header injected, and the cookie NOT stripped.

    Stripping it there would break the only gate that entry still has — the
    app's own token->cookie session, which cannot read a cookie nginx cut.
    """
    # Act
    nginx_conf = _render_nginx(_mixed_login_config("kiosk"))
    exempt = _location_body(nginx_conf, "location /u/kiosk/")
    gated = _location_body(nginx_conf, "location /u/alice/")

    # Assert — the exempt location carries neither half
    assert _secret_include("kiosk") not in exempt
    assert 'proxy_set_header Cookie "";' not in exempt
    # Assert — its gated peer in the same render carries both
    assert _secret_include("alice") in gated
    assert 'proxy_set_header Cookie "";' in gated


def test_nginx_secret_include_and_cookie_strip_share_one_predicate() -> None:
    """The two effects are gated on one condition and must never diverge.

    A location with the strip but no header authenticates nothing and 401s
    every request; a location with the header but no strip lets an
    unauthenticated request carry an operator's session cookie into a container
    that runs agent-generated code. Asserted across every render shape rather
    than on one, because the divergence is what a later edit would reintroduce.
    """
    # Arrange
    shapes = {
        "auth-off": (copy.deepcopy(_MULTI_USER_CONFIG), ["alice", "bob", "carol"]),
        "auth-on": (_auth_config(["alice", "bob"]), ["alice", "bob"]),
        "mixed": (_mixed_login_config("kiosk"), ["alice", "kiosk"]),
    }

    # Act / Assert
    for shape, (config, users) in shapes.items():
        nginx_conf = _render_nginx(config)
        for user in users:
            body = _location_body(nginx_conf, f"location /u/{user}/")
            has_secret = _secret_include(user) in body
            has_strip = 'proxy_set_header Cookie "";' in body
            assert has_secret == has_strip, f"{shape}/{user}: {has_secret=} {has_strip=}"


def test_nginx_secret_include_target_is_outside_the_globbed_conf_d() -> None:
    """The snippets are envsubst'ed into /etc/nginx/osprey, never conf.d.

    The base image globs `conf.d/*.conf` into `http{}`, where a
    `proxy_set_header` would apply to EVERY upstream — the auth sidecar
    included — instead of to one user's location.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice"]))

    # Assert
    includes = re.findall(r"^\s*include (\S+);", nginx_conf, flags=re.MULTILINE)
    assert includes == ["/etc/nginx/osprey/secret-alice.conf"]
    assert "conf.d" not in includes[0]


def test_nginx_secret_include_matches_the_filename_the_renderer_writes() -> None:
    """The include and the snippet are two halves of one contract.

    nginx refuses to start on a missing include, so a rename on either side
    must fail here rather than as a crash-looping proxy whose error names
    neither the user nor the roster.
    """
    # Arrange
    users = ["alice", "bob"]
    config = _auth_config(users)

    # Act
    artifacts = render_web_terminals(config, terminal_secrets=_secrets_for(users))
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    snippets = [path for path in artifacts if path.startswith("nginx/templates/")]
    assert len(snippets) == len(users)
    for path in snippets:
        materialized = Path(path).name.removesuffix(".template")
        assert f"include /etc/nginx/osprey/{materialized};" in nginx_conf


def test_nginx_conf_never_spells_the_secret_header_itself() -> None:
    """The header directive lives in the per-user snippet, not in this conf.

    That is what keeps the secret VALUE out of every rendered artifact: the
    snippet is an nginx *template* naming a variable, substituted at container
    start, while this conf is mounted verbatim and would carry whatever it
    spelled into build/, into git status, and into an image layer.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice"]))

    # Assert
    assert TERMINAL_SECRET_HEADER not in nginx_conf


# ---------------------------------------------------------------------------
# Remediation round: the perimeter's header contract, and one derivation of the
# per-user secret variable
# ---------------------------------------------------------------------------


def test_terminal_secret_env_var_is_the_minting_sides_definition() -> None:
    """M1: the render side and the mint side derive ONE variable name.

    `render.terminal_secret_env_var` is an alias of
    `auth_credentials.terminal_secret_var`. Two independent derivations could be
    minted one way and referenced another, and the failure is invisible until a
    terminal refuses every request behind a healthy-looking proxy.
    """
    from osprey.deployment.web_terminals.auth_credentials import (
        TERMINAL_SECRET_VAR_PREFIX,
        terminal_secret_var,
    )
    from osprey.deployment.web_terminals.render import TERMINAL_SECRET_ENV_PREFIX

    assert TERMINAL_SECRET_ENV_PREFIX == TERMINAL_SECRET_VAR_PREFIX
    for name in ("alice", "alice-b", "Bob", "carol_2", "d"):
        assert terminal_secret_env_var(name) == terminal_secret_var(name)


def test_every_terminal_location_clears_the_authorization_header() -> None:
    """M4: a client-supplied `Authorization` never reaches a terminal.

    The app does accept an `Authorization: Bearer` — the panel token, a narrow
    tier its own tooling uses — but that traffic is loopback inside the
    container and never crosses nginx, so clearing the header at the perimeter
    costs nothing and closes the one route by which a caller could present a
    credential this proxy did not vouch for. Unconditional, unlike the cookie
    and secret headers: it is never legitimate here in any auth mode.
    """
    # Arrange
    shapes = {
        "auth-off": (copy.deepcopy(_MULTI_USER_CONFIG), ["alice", "bob", "carol"]),
        "auth-on": (_auth_config(["alice", "bob"]), ["alice", "bob"]),
        "mixed": (_mixed_login_config("kiosk"), ["alice", "kiosk"]),
    }

    # Act / Assert
    for shape, (config, users) in shapes.items():
        nginx_conf = _render_nginx(config)
        for user in users:
            body = _directives(_location_body(nginx_conf, f"location /u/{user}/"))
            assert 'proxy_set_header Authorization "";' in body, f"{shape}/{user}"


def test_ungated_locations_clear_the_operator_secret_header() -> None:
    """M4: where nginx injects no operator secret, it clears the header instead.

    A value arriving in `X-Osprey-Terminal-Secret` on an ungated location came
    from the client, and the app trusts that header absolutely — forwarding it
    would let anyone who can reach the port authenticate as the operator.
    """
    # Arrange / Act
    auth_off = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))
    mixed = _render_nginx(_mixed_login_config("kiosk"))
    cleared = 'proxy_set_header X-Osprey-Terminal-Secret "";'

    # Assert — every auth-off location clears it
    for user in ("alice", "bob", "carol"):
        assert cleared in _directives(_location_body(auth_off, f"location /u/{user}/"))

    # Assert — with auth on, only the exempt location does; the gated one has
    # the include that SETS it and must not clear it afterwards.
    assert cleared in _directives(_location_body(mixed, "location /u/kiosk/"))
    gated = _directives(_location_body(mixed, "location /u/alice/"))
    assert cleared not in gated
    assert _secret_include("alice") in gated


def test_ungated_locations_forward_only_that_users_own_session_cookie() -> None:
    """I1: an ungated location narrows the cookie jar to one cookie.

    Cookies ignore ports, so one jar serves this whole origin: the browser's raw
    `Cookie` header names every OTHER terminal it has unlocked and, with auth
    on, the sidecar's origin-wide `osprey_auth_session`. These containers run
    agent-generated code. The cookie cannot be cut here — it is the only gate
    left in front of the terminal — so exactly one is forwarded.
    """
    # Arrange
    mixed = _render_nginx(_mixed_login_config("kiosk"))
    exempt = _directives(_location_body(mixed, "location /u/kiosk/"))
    kiosk_port = allocate_ports(_BASE_PORTS, 1)["web"]

    # Assert
    cookie = f"osprey_terminal_session_{kiosk_port}"
    assert f'proxy_set_header Cookie "{cookie}=$cookie_{cookie}";' in exempt
    assert "$http_cookie" not in exempt
    # The sidecar's own origin-wide cookie is not among what travels.
    assert "osprey_auth_session" not in exempt


def test_forwarded_cookie_name_comes_from_the_apps_own_helper() -> None:
    """I1: the perimeter and the app spell the cookie name from one definition.

    A name assembled independently in the template would forward a cookie the
    app does not read — a terminal that cannot hold a session, with nothing in
    either end's config to show why.
    """
    from osprey.interfaces.common_middleware import session_cookie_name

    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    for index, user in enumerate(("alice", "bob", "carol")):
        expected = session_cookie_name(allocate_ports(_BASE_PORTS, index)["web"])
        body = _location_body(nginx_conf, f"location /u/{user}/")
        assert f'proxy_set_header Cookie "{expected}=$cookie_{expected}";' in body


# ---------------------------------------------------------------------------
# modules.web_terminals.external_origin — the front-terminator escape hatch
# ---------------------------------------------------------------------------


def _external_origin_of(config: dict) -> str:
    """The origin the render bakes into every per-user container."""
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])
    values = [
        entry.split("=", 1)[1]
        for entry in compose["services"]["web-alice"]["environment"]
        if entry.startswith("OSPREY_TERMINAL_EXTERNAL_ORIGIN=")
    ]
    assert len(values) == 1, compose["services"]["web-alice"]["environment"]
    return values[0]


def test_external_origin_defaults_to_the_derivation_from_deploy_fqdn() -> None:
    """Unset, nothing changes: the origin is this nginx's own address."""
    # Act / Assert
    assert _external_origin_of(_config(["alice"])) == "http://dls-deploy.dls.example.org:9080"


def test_a_configured_external_origin_is_what_every_container_checks_against() -> None:
    """A load balancer terminating TLS in front of this nginx is what the
    browser actually reaches, so its address is the origin.

    The browser sends `Origin: https://<facility-host>`; the derivation below
    would have produced this nginx's own address instead, which no browser in
    that topology ever sends — so every write, approval and chat message would
    be refused while every page still loaded.
    """
    # Arrange
    config = _config(["alice"])
    config["modules"]["web_terminals"]["external_origin"] = "https://terminals.example.org"

    # Act
    origin = _external_origin_of(config)

    # Assert — verbatim, and everywhere the deployment states an origin.
    assert origin == "https://terminals.example.org"
    artifacts = render_web_terminals(config)
    assert deployment_external_origin(config) == "https://terminals.example.org"
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])
    landing = [
        entry
        for entry in compose["services"]["web-alice"]["environment"]
        if entry.startswith("OSPREY_TERMINAL_LANDING_URL=")
    ]
    assert landing == ["OSPREY_TERMINAL_LANDING_URL=https://terminals.example.org"]


def test_a_configured_external_origin_removes_the_need_for_deploy_fqdn() -> None:
    """The override IS the origin, so the value it replaces is not required.

    A facility whose browsers only ever reach the load balancer has no reason to
    record a browser-reachable name for the deploy host itself.
    """
    # Arrange
    config = _config(["alice"])
    config["deploy"] = {"host": "dls-deploy"}
    config["modules"]["web_terminals"]["external_origin"] = "https://terminals.example.org"

    # Act / Assert
    assert _external_origin_of(config) == "https://terminals.example.org"


def test_the_missing_fqdn_refusal_names_the_override_as_the_other_way_out() -> None:
    """The operator hitting this is often the one who needs the override."""
    # Arrange
    config = _config(["alice"])
    config["deploy"] = {"host": "dls-deploy"}

    # Act / Assert
    with pytest.raises(ValueError, match="external_origin"):
        render_web_terminals(config)


@pytest.mark.parametrize(
    "value",
    [
        "https://terminals.example.org/",  # trailing slash
        "https://terminals.example.org/terminals",  # a path
        "terminals.example.org",  # no scheme
        "ftp://terminals.example.org",  # not a scheme this perimeter serves
        "https://terminals.example.org?x=1",  # a query
        "https://user:pw@terminals.example.org",  # credentials
        "https://terminals.example.org:notaport",
        9080,
    ],
)
def test_render_refuses_an_external_origin_that_is_not_an_origin(value: object) -> None:
    """Refused at render, not trusted.

    The value is compared against the browser's `Origin` header as a whole
    string, and a browser writes none of these. Anything but scheme://host[:port]
    renders a deployment whose pages all load and whose every write answers 403 —
    a failure with no signal outside a container log.
    """
    # Arrange
    config = _config(["alice"])
    config["modules"]["web_terminals"]["external_origin"] = value

    # Act / Assert
    with pytest.raises(ValueError, match="external_origin"):
        render_web_terminals(config)


def test_a_blank_external_origin_falls_back_to_the_derivation() -> None:
    """Absence spelled as an empty string is still absence — the same reading
    every other optional string key in this module gets."""
    # Arrange
    config = _config(["alice"])
    config["modules"]["web_terminals"]["external_origin"] = "   "

    # Act / Assert
    assert _external_origin_of(config) == "http://dls-deploy.dls.example.org:9080"


def test_envsubst_output_tmpfs_is_not_world_readable() -> None:
    """M5: the directory the operator secrets are substituted into is 0700.

    A tmpfs mounted with no mode lands at 1777 — world-writable, and
    world-READABLE around files the entrypoint writes at 0644, which are every
    roster user's operator secret in plaintext. Root-owned 0700 still serves
    both readers, since the entrypoint writes as root and nginx's master
    process reads the includes as root.

    The EXACT string is asserted, not a parsed mode, because the spelling is the
    property under test. The mode has to reach the kernel as octal on both
    compose providers OSPREY supports: the long-form `volumes:` entry spells it
    as YAML octal, which podman-compose forwards as the decimal `mode=448` and
    the kernel rejects. `<path>:mode=0700` on the service-level `tmpfs:` key is
    passed through unchanged by both.
    """
    # Act
    compose = yaml.safe_load(
        render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))["docker-compose.web.yml"]
    )

    # Assert
    assert compose["services"]["nginx"]["tmpfs"] == ["/etc/nginx/osprey:mode=0700"]
    # The long form is gone, so the volume list is bind mounts alone — a mapping
    # there would carry the mode in the unportable spelling.
    assert all(isinstance(volume, str) for volume in compose["services"]["nginx"]["volumes"]), (
        compose["services"]["nginx"]["volumes"]
    )
