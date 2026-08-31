"""The landing page says which terminals are entered through a login URL.

A card whose user reaches their terminal by opening that user's own ``?token=``
URL looks, on the page alone, exactly like a card behind a login wall: same
chip, same link, same hover. It is not the same thing. Nobody can open it from
the landing page cold — the URL carries the credential, and it is minted by
``osprey users login-url <name>``, a verb an operator has no reason to guess.

So a token-login card carries two honesty marks: a badge naming the posture,
and one muted line naming the verb. It KEEPS its href — the ``?token=``
exchange leaves a session cookie behind, so the card is a real return path for
a browser that has opened the URL once, and disabling it would be a second lie
in the other direction.

These tests pin the two postures the marks distinguish (``password`` with one
login-exempt entry, and ``token`` where the whole roster is exempt by
construction), that the hint names the RIGHT user per card, and the one
precondition the derivation carries (see the scaffold test at the bottom).
"""

from __future__ import annotations

import copy
import re

from osprey.deployment.web_terminals.render import render_web_terminals

from .test_render import _MULTI_USER_CONFIG, _body, _config

#: One navigable card: its href and everything between the anchor tags.
_CARD_RE = re.compile(r'<a class="landing-card" href="([^"]*)">(.*?)</a>', re.DOTALL)

#: The chip and the hint line the badge renders as.
_BADGE_RE = re.compile(r'<span class="landing-card-token">([^<]*)</span>')
_HINT_RE = re.compile(r'<span class="landing-card-token-hint">(.*?)</span>', re.DOTALL)


def _cards(landing_html: str) -> dict[str, tuple[str, str]]:
    """Every navigable card on the page, keyed by its label.

    Parsed out of the markup (below the stylesheet, which defines every rule the
    page could use whatever this render emitted) rather than read off the
    landing context, because what an operator is misled by is the rendered page.

    Returns:
        ``{label: (href, inner markup)}``.
    """
    cards = {}
    for href, inner in _CARD_RE.findall(_body(landing_html)):
        label = re.search(r'<span class="landing-card-label">([^<]*)</span>', inner)
        assert label is not None, inner
        cards[label.group(1)] = (href, inner)
    return cards


def _badged(landing_html: str) -> set[str]:
    """The labels of the cards carrying the token-login badge."""
    return {label for label, (_, inner) in _cards(landing_html).items() if _BADGE_RE.search(inner)}


def _hint_of(landing_html: str, label: str) -> str:
    """The hint line on one card, as rendered."""
    _, inner = _cards(landing_html)[label]
    hint = _HINT_RE.search(inner)
    assert hint is not None, f"no token-login hint on card {label!r}: {inner}"
    return hint.group(1)


def _password_config() -> dict:
    """A password-walled roster with exactly one login-exempt entry.

    The shipped control-assistant shape in miniature: operators sign in, and the
    standalone research service beside them (`login: false`) does not.
    """
    config = copy.deepcopy(
        _config([{"name": "alice", "index": 0}, {"name": "ariel", "index": 1, "login": False}])
    )
    config["modules"]["web_terminals"]["auth"] = {
        "method": "password",
        "allow_insecure_http": True,
    }
    return config


def test_only_the_login_exempt_card_is_badged_behind_a_password_wall() -> None:
    """Exactly the entry nginx does not vouch for gets the marks.

    Badging the signed-in card too would make the page say every terminal needs
    a URL nobody has, which is the same failure this badge exists to fix, only
    inverted.
    """
    # Act
    landing_html = render_web_terminals(_password_config())["nginx/landing.html"]

    # Assert
    assert _badged(landing_html) == {"ariel"}


def test_the_badged_card_keeps_its_href_behind_a_password_wall() -> None:
    """The badge labels the card; it does not disable it.

    The `?token=` exchange mints a session cookie, so an operator who has opened
    the login URL once returns through this very link. A card rendered inert
    would strand them.
    """
    # Act
    landing_html = render_web_terminals(_password_config())["nginx/landing.html"]

    # Assert
    href, _ = _cards(landing_html)["ariel"]
    assert href.endswith("/u/ariel/"), href


def test_every_card_is_badged_and_navigable_under_the_token_posture() -> None:
    """With `auth.method` at its `token` default nginx vouches for nobody, so the
    whole roster is entered through its own URL — and every card says so while
    staying a link."""
    # Act
    landing_html = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))["nginx/landing.html"]

    # Assert
    cards = _cards(landing_html)
    assert _badged(landing_html) == {"alice", "bob", "carol"}
    assert all(href.endswith(f"/u/{label}/") for label, (href, _) in cards.items()), cards


def test_the_hint_names_the_verb_and_this_card_s_own_user() -> None:
    """The hint's whole point is the command an operator can run. A hint that
    named the wrong user would send them to another person's terminal with
    another person's live credential in the URL."""
    # Act
    landing_html = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))["nginx/landing.html"]

    # Assert
    for name in ("alice", "bob", "carol"):
        hint = _hint_of(landing_html, name)
        assert f"osprey users login-url {name}" in hint, hint
        for other in {"alice", "bob", "carol"} - {name}:
            assert other not in hint, hint


def test_a_card_behind_the_login_wall_carries_neither_mark() -> None:
    """Neither class appears on a card nginx vouches for — asserted on the body,
    not the whole page, since the stylesheet always defines both rules."""
    # Act
    landing_html = render_web_terminals(_password_config())["nginx/landing.html"]

    # Assert
    _, alice = _cards(landing_html)["alice"]
    assert "landing-card-token" not in alice


def test_an_operator_supplied_link_card_is_never_badged() -> None:
    """Only roster cards carry the key at all: a `landing.groups` entry is a
    `{label, url}` dict the operator wrote, and `item["token_login"]` is Jinja
    `Undefined` there. The badge must not leak onto it under the posture where
    every roster card has one."""
    # Arrange
    config = copy.deepcopy(
        _config(
            ["alice"],
            groups=[
                {"type": "users"},
                {
                    "type": "links",
                    "label": "Dashboards",
                    "links": [{"label": "Archiver", "url": "/archiver"}],
                },
            ],
        )
    )

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    assert set(_cards(landing_html)) == {"alice", "Archiver"}
    assert _badged(landing_html) == {"alice"}


def test_a_scaffold_render_of_a_disabled_module_badges_nothing() -> None:
    """The derivation's one precondition, documented rather than worked around.

    `token_login_users` answers `[]` for a config whose
    `modules.web_terminals.enabled` is falsy — it is the deploy summary's
    reader, and a module nobody enabled deploys nothing. `render_web_terminals`
    has no such gate (`osprey scaffold web-terminals render` renders a
    not-yet-enabled config on purpose, so an operator can look at the artifacts
    before turning it on), so that render shows an unbadged page for a
    deployment that, once enabled, is entirely token-login.

    Pinned as the behaviour it is: a scaffold preview is not the deployed page,
    and every fixture that asks this module about postures enables it. If the
    preview is ever made to badge, this test is the one that says so.
    """
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["enabled"] = False

    # Act
    landing_html = render_web_terminals(config)["nginx/landing.html"]

    # Assert
    assert _badged(landing_html) == set()
