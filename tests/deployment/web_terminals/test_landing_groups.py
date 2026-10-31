"""The landing page's groups, and the one title a shared card carries everywhere.

`landing.groups` lists sections of three types — `users` (the people), `shared`
(one card per `shared:` login; the retired `apps` spelling reads the same) and
`links` — rendered in the order the operator wrote them. Omitted, it is people
then shared logins. A shared card is titled by its login file's `name:` in
three places that must agree: its landing card, its browser tab
(`OSPREY_WEB_APP_NAME` on its container) and the sidecar's chooser (the
`OSPREY_AUTH_ROSTER_LABEL_<SUFFIX>` line read back into `AuthSettings`). A
person's card keeps its roster name in all three.

The roster here has three people and three apps. Every env name is imported
from the sidecar, so a rename on either end fails here.
"""

from __future__ import annotations

import re

import pytest
import yaml

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.auth_sidecar.app import ENV_USERS, AuthSettings
from osprey.services.auth_sidecar.roster_env import ROSTER_LABEL_VAR_PREFIX, env_var_suffix
from osprey.services.auth_sidecar.routes.entry import OpenedCard, password_cards

PEOPLE = ["alice", "bob", "carol"]
APPS = ["logbook", "knowledge", "channel-finder"]
TITLES = {
    "logbook": "Logbook Research",
    "knowledge": "Facility Knowledge",
    "channel-finder": "Channel Finder",
}
PROJECT = "dls-profiles"


def _config(groups: list[dict] | None = None, *, shared: list[str] | None = None) -> dict:
    """A built ``config.yml`` for three people on one tier and (by default) three apps.

    Shaped as ``osprey build`` writes it: the project's name, and the
    ``renders:`` projection naming every render the roster runs, the tier and
    one ``build/<project>-<app>`` per shared app.
    """
    apps = list(APPS) if shared is None else shared
    renders: dict = {
        "assistant": {
            "kind": "tier",
            "project": f"{PROJECT}-assistant",
            "project_path": f"build/{PROJECT}-assistant",
        }
    }
    for app in apps:
        renders[app] = {
            "kind": "tier",
            "project": f"{PROJECT}-{app}",
            "project_path": f"build/{PROJECT}-{app}",
        }
    web_terminals: dict = {
        "enabled": True,
        "users": [
            {"name": person, "index": index, "tier": "assistant"}
            for index, person in enumerate(PEOPLE)
        ],
        "shared": apps,
        "default_tier": "assistant",
        "renders": renders,
        "auth": {"method": "password", "allow_insecure_http": True},
    }
    if groups is not None:
        web_terminals["landing"] = {"groups": groups}
    return {
        "project_name": PROJECT,
        "facility": {"prefix": "dls"},
        "system": {"timezone": "America/Los_Angeles"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {"web_terminals": web_terminals},
    }


def _sections(landing_html: str) -> list[tuple[str, str]]:
    """Every rendered landing section as ``(class attribute, heading text)``."""
    return re.findall(
        r'<section class="([^"]+)">\s*<h2 class="landing-group-label">([^<]*)</h2>',
        landing_html,
    )


def _cards_in_section(landing_html: str, heading: str) -> list[str]:
    """The card labels under one section heading, in render order."""
    body = landing_html.split(f">{heading}</h2>", 1)[1].split("</section>", 1)[0]
    return re.findall(r'<span class="landing-card-label">([^<]*)</span>', body)


def _render(config: dict) -> dict[str, str]:
    return render_web_terminals(config, shared_titles=TITLES)


def _landing(config: dict) -> str:
    return _render(config)["nginx/landing.html"]


def _compose(config: dict) -> dict:
    return yaml.safe_load(_render(config)["docker-compose.web.yml"])


def _service_env(compose: dict, service: str) -> dict[str, str]:
    return dict(
        item.split("=", 1) for item in compose["services"][service]["environment"] if "=" in item
    )


def _sidecar_settings(config: dict) -> AuthSettings:
    """The sidecar's own reading of the env the render gave it."""
    return AuthSettings.from_env(_service_env(_compose(config), "auth"))


# ── landing groups ───────────────────────────────────────────────────────────


def test_default_groups_are_people_then_shared_logins_in_a_tray() -> None:
    # Act
    html = _landing(_config())

    # Assert
    assert _sections(html) == [
        ("landing-group", "Terminals"),
        ("landing-group landing-tray", "Shared logins"),
    ]
    assert _cards_in_section(html, "Terminals") == PEOPLE
    assert _cards_in_section(html, "Shared logins") == [TITLES[app] for app in APPS]


def test_the_retired_apps_group_type_reads_as_shared() -> None:
    # Act
    retired = _landing(_config([{"type": "users"}, {"type": "apps"}]))
    current = _landing(_config([{"type": "users"}, {"type": "shared"}]))

    # Assert
    assert _sections(retired) == _sections(current)
    assert _cards_in_section(retired, "Shared logins") == [TITLES[app] for app in APPS]


def test_groups_render_in_operator_order_with_their_own_labels() -> None:
    # Arrange
    groups = [
        {"type": "links", "label": "Dashboards", "links": [{"label": "Grafana", "url": "/g/"}]},
        {"type": "shared", "label": "Shared services"},
        {"type": "users", "label": "Operators"},
    ]

    # Act
    html = _landing(_config(groups))

    # Assert
    assert [heading for _, heading in _sections(html)] == [
        "Dashboards",
        "Shared services",
        "Operators",
    ]
    assert _cards_in_section(html, "Shared services") == [TITLES[app] for app in APPS]
    assert _cards_in_section(html, "Operators") == PEOPLE


def test_app_cards_link_to_their_own_mount() -> None:
    # Act
    html = _landing(_config())

    # Assert
    for app in APPS:
        assert f'href="/u/{app}/"' in html


def test_no_shared_apps_suppresses_the_apps_heading() -> None:
    # Act
    html = _landing(_config(shared=[]))

    # Assert
    assert _sections(html) == [("landing-group", "Terminals")]


def test_a_users_only_layout_lists_no_app_among_the_people() -> None:
    # Act
    html = _landing(_config([{"type": "users"}]))

    # Assert
    assert _sections(html) == [("landing-group", "Terminals")]
    assert _cards_in_section(html, "Terminals") == PEOPLE


def test_hidden_names_hide_the_people_but_not_the_shared_logins() -> None:
    # Act
    html = _landing(_config([{"type": "users", "names": "hidden"}, {"type": "shared"}]))

    # Assert
    assert _cards_in_section(html, "Shared logins") == [TITLES[app] for app in APPS]
    for person in PEOPLE:
        assert f'href="/u/{person}/"' not in html


def test_landing_group_is_not_read_anywhere_in_the_render() -> None:
    # Arrange
    config = _config([{"type": "users"}])
    config["modules"]["web_terminals"]["renders"]["assistant"]["landing_group"] = "Elsewhere"

    # Act
    html = _landing(config)

    # Assert
    assert "Elsewhere" not in html
    assert _cards_in_section(html, "Terminals") == PEOPLE


# ── one title in three places ────────────────────────────────────────────────


def test_app_title_is_the_browser_tab_title() -> None:
    # Act
    compose = _compose(_config())

    # Assert
    for app in APPS:
        assert _service_env(compose, f"web-{app}")["OSPREY_WEB_APP_NAME"] == TITLES[app]
    for person in PEOPLE:
        assert "OSPREY_WEB_APP_NAME" not in _service_env(compose, f"web-{person}")


def test_sidecar_env_carries_a_label_line_for_each_app_and_none_for_people() -> None:
    # Act
    env = _service_env(_compose(_config()), "auth")

    # Assert
    for app in APPS:
        assert env[f"{ROSTER_LABEL_VAR_PREFIX}{env_var_suffix(app)}"] == TITLES[app]
    for person in PEOPLE:
        assert f"{ROSTER_LABEL_VAR_PREFIX}{env_var_suffix(person)}" not in env
    assert env[ENV_USERS].split(",") == PEOPLE + APPS


def test_sidecar_reads_app_titles_and_defaults_people_to_their_name() -> None:
    # Act
    settings = _sidecar_settings(_config())

    # Assert
    assert {app: settings.label(app) for app in APPS} == TITLES
    assert [settings.label(person) for person in PEOPLE] == PEOPLE
    assert settings.label("not-on-the-roster") == "not-on-the-roster"


def test_chooser_cards_carry_the_app_title() -> None:
    # Arrange
    settings = _sidecar_settings(_config())

    # Act
    cards = password_cards(settings, "alice")

    # Assert
    assert cards[0] == OpenedCard(name="alice", own=True, label="alice")
    assert {card.name: card.label for card in cards[1:]} == TITLES


def test_chooser_page_lists_the_app_title(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    from starlette.requests import Request

    from osprey.services.auth_sidecar.routes import entry

    settings = _sidecar_settings(_config())
    monkeypatch.setattr(entry, "get_settings", lambda request: settings)
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    cards = (OpenedCard(name="alice", own=True), OpenedCard(name="logbook", own=False))

    # Act
    page = entry.render(request, shape="cards", status_code=200, cards=cards).body.decode()

    # Assert
    assert '<a href="/u/logbook/">Logbook Research</a>' in page
    assert '<a href="/u/alice/">alice</a>' in page


# ── token-login hint, escaping ───────────────────────────────────────────────


def _token_config() -> dict:
    config = _config()
    config["modules"]["web_terminals"]["auth"] = {"method": "token", "allow_insecure_http": True}
    return config


def test_token_login_hint_names_people_and_never_a_shared_title() -> None:
    # Act
    html = _landing(_token_config())

    # Assert
    hints = re.findall(r"osprey users login-url ([^<]*)</code>", html)
    assert hints == PEOPLE
    apps_body = html.split(">Shared logins</h2>", 1)[1].split("</section>", 1)[0]
    assert apps_body.count('<span class="landing-card-token">login link</span>') == len(APPS)
    assert "landing-card-token-hint" not in apps_body


def test_operator_markup_in_app_label_and_title_is_escaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    import sys

    from starlette.requests import Request

    from osprey.services.auth_sidecar.routes import entry

    monkeypatch.setattr(sys.modules[__name__], "TITLES", {**TITLES, "logbook": "<b>Logbook</b>"})
    config = _config([{"type": "users"}, {"type": "shared", "label": "<script>x</script>"}])
    settings = _sidecar_settings(config)
    monkeypatch.setattr(entry, "get_settings", lambda request: settings)
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})

    # Act
    html = _landing(config)
    page = entry.render(
        request, shape="cards", status_code=200, cards=(OpenedCard(name="logbook", own=False),)
    ).body.decode()

    # Assert
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;x&lt;/script&gt;" in html
    assert "<b>Logbook</b>" not in html
    assert "&lt;b&gt;Logbook&lt;/b&gt;" in html
    assert "<b>Logbook</b>" not in page
    assert "&lt;b&gt;Logbook&lt;/b&gt;" in page
