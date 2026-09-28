"""A ``users`` section set to ``names: hidden`` publishes no roster name.

The landing page is a static file nginx serves to anyone who can reach it. A
deployment behind a login wall can therefore choose to keep its roster off that
page: ``names: hidden`` on a ``type: users`` section replaces the section's name
cards with one sign-in button that opens the card-less sign-in route. The names
must be absent from the file, not hidden by a stylesheet.

Everything else about the page keeps its meaning: tray sections lifted out of
the same entry and ``links`` sections render as before, and a config that sets
``names: shown`` or leaves it out renders byte-identically to one that never
knew the key.
"""

from __future__ import annotations

import copy
import re

import pytest

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.auth_sidecar.routes.entry import ENTRY_PATH

from .test_golden_render import EXAMPLE_CONFIG
from .test_render import (
    _OPERATOR_CATALOG,
    _OPERATOR_ROSTER,
    _body,
    _config,
    _roster_config,
    _sections,
)

_SIGN_IN_RE = re.compile(r'<a class="landing-sign-in" href="([^"]*)">([^<]*)</a>')

_LOGIN_WALL = {"method": "password", "allow_insecure_http": True}


def _walled(config: dict) -> dict:
    """``config`` behind a password login, the posture ``names: hidden`` needs."""
    config = copy.deepcopy(config)
    config["modules"]["web_terminals"]["auth"] = dict(_LOGIN_WALL)
    return config


def _landing(config: dict) -> str:
    return render_web_terminals(config)["nginx/landing.html"]


def _style(landing_html: str) -> str:
    return landing_html.split("<style>", 1)[1].split("</style>", 1)[0]


def _section(landing_html: str, heading: str) -> str:
    """The markup of the section headed ``heading``, from ``<section`` to ``</section>``."""
    body = _body(landing_html)
    head = body.index(f">{heading}</h2>")
    start = body.rindex("<section", 0, head)
    return body[start : body.index("</section>", head) + len("</section>")]


def test_hidden_names_render_one_sign_in_button_and_no_roster_name() -> None:
    """The section carries one button and no card, label, badge or user path."""
    # Arrange
    config = _walled(_config(["alice", "bob", "carol"], [{"type": "users", "names": "hidden"}]))

    # Act
    body = _body(_landing(config))

    # Assert
    buttons = _SIGN_IN_RE.findall(body)
    assert len(buttons) == 1
    assert buttons[0][1] == "Log in to your terminal"
    assert "landing-card-label" not in body
    for leaked in ("alice", "bob", "carol", "/u/"):
        assert leaked not in body


def test_sign_in_button_opens_the_card_less_sign_in_route() -> None:
    """The href is the sidecar's own constant, so page and route cannot drift apart."""
    # Arrange
    config = _walled(_config(["alice", "bob"], [{"type": "users", "names": "hidden"}]))

    # Act
    ((href, _label),) = _SIGN_IN_RE.findall(_body(_landing(config)))

    # Assert
    assert href == ENTRY_PATH


def test_hidden_names_leave_tray_sections_as_they_are() -> None:
    """A persona's ``landing_group`` tray still renders its service card."""
    # Arrange
    config = _walled(
        _roster_config(_OPERATOR_ROSTER, _OPERATOR_CATALOG, [{"type": "users", "names": "hidden"}])
    )

    # Act
    landing_html = _landing(config)
    body = _body(landing_html)

    # Assert
    assert _sections(landing_html) == [
        ("landing-group", "Terminals"),
        ("landing-group landing-tray", "Standalone deployments"),
    ]
    assert '<span class="landing-card-label">ariel</span>' in _section(
        landing_html, "Standalone deployments"
    )
    assert "alice" not in body
    assert "bob" not in body


def test_hidden_names_leave_links_sections_as_they_are() -> None:
    """A ``links`` section beside a hidden roster renders exactly as without the switch."""
    # Arrange
    links = {
        "type": "links",
        "label": "Facility Tools",
        "links": [{"label": "Elog", "url": "https://elog.dls.example.org"}],
    }
    hidden = _walled(_config(["alice"], [{"type": "users", "names": "hidden"}, links]))
    absent = _walled(_config(["alice"], [{"type": "users"}, links]))

    # Act
    hidden_links = _section(_landing(hidden), "Facility Tools")
    absent_links = _section(_landing(absent), "Facility Tools")

    # Assert
    assert hidden_links == absent_links
    assert "Elog" in hidden_links


def test_hidden_names_with_every_user_in_a_tray_render_no_button() -> None:
    """A button that hides nobody is never shown; the empty default section drops as before."""
    # Arrange
    roster = [{"name": "ariel", "index": 0, "persona": "ariel"}]
    config = _walled(
        _roster_config(roster, _OPERATOR_CATALOG, [{"type": "users", "names": "hidden"}])
    )

    # Act
    landing_html = _landing(config)

    # Assert
    assert "landing-sign-in" not in _body(landing_html)
    assert _sections(landing_html) == [("landing-group landing-tray", "Standalone deployments")]


def _with_names(config: dict, names: str | None) -> dict:
    """``config`` with ``names`` set on every ``users`` group (removed when ``None``)."""
    config = copy.deepcopy(config)
    landing = config["modules"]["web_terminals"].setdefault("landing", {})
    groups = landing.setdefault("groups", [{"type": "users"}])
    for group in groups:
        if group.get("type") == "users":
            group.pop("names", None)
            if names is not None:
                group["names"] = names
    return config


@pytest.mark.parametrize(
    ("base", "names"),
    [
        (_config(["alice", "bob"], [{"type": "users"}]), "shown"),
        (_config(["alice", "bob"], [{"type": "users"}]), None),
        (EXAMPLE_CONFIG, "shown"),
    ],
    ids=["shown", "absent", "example-config-shown"],
)
def test_shown_and_absent_names_render_identical_pages(base: dict, names: str | None) -> None:
    """``shown`` and absent change no byte of any artifact."""
    # Arrange
    reference = render_web_terminals(_with_names(base, None))

    # Act
    rendered = render_web_terminals(_with_names(base, names))

    # Assert
    assert rendered == reference


def test_a_page_without_a_button_carries_no_sign_in_style() -> None:
    """The ``.landing-sign-in`` rule is emitted only when a button is on the page."""
    # Act
    landing_html = _landing(_config(["alice", "bob"], [{"type": "users"}]))

    # Assert
    assert "landing-sign-in" not in _style(landing_html)


def test_a_page_with_the_button_defines_every_variable_its_stylesheet_uses() -> None:
    """The button's rule uses only custom properties the page bakes."""
    # Arrange
    config = _walled(_config(["alice", "bob"], [{"type": "users", "names": "hidden"}]))
    config["web"] = {"theme": "desy"}

    # Act
    html = _landing(config)
    used = set(re.findall(r"var\((--[a-z0-9-]+)\)", html))
    defined = set(re.findall(r"^\s+(--[a-z0-9-]+):", html, re.MULTILINE))

    # Assert
    assert ".landing-sign-in" in _style(html)
    assert used <= defined, f"undefined custom properties: {sorted(used - defined)}"
