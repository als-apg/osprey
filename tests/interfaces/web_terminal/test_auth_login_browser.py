"""Browser test: landing card -> password prompt -> the user's own terminal.

The composed-stack serving test
(``tests/deployment/web_terminals/test_auth_serving.py``) drives this same
perimeter with httpx, which is the right instrument for headers, cookies and
status codes. It cannot see the part a person actually performs: that the
landing page's cards are real links a click follows, that the redirect a denied
``auth_request`` emits lands a *browser* on the prompt, that the prompt asks for
a password and never for a name, and that submitting the form — with the
``Origin`` a browser sets unprompted and nothing else — walks back to the
terminal the click was aimed at.

That round trip is the deployment's front door, and every hop in it is a
different component: nginx's ``error_page``/``map`` pair chooses the redirect,
the sidecar renders the prompt and evaluates the credential, and the browser's
own cookie jar carries the result back through nginx's ``auth_request``. The
in-process web-terminal browser suites in this directory cannot reach it —
they run the app behind an ASGI prefix-stripping stand-in, with no nginx and no
sidecar in the picture at all.

So this suite drives the REAL containerised stack, reusing
:func:`~tests.deployment.web_terminals.test_auth_serving.serving_stack` rather
than standing up a second copy of it. It is marked both ``browser`` and
``dockerbuild`` and skips cleanly when either chromium or docker is missing.

WHAT THE DESTINATION IS. The per-user terminal app cannot run in this harness —
it needs the ``claude`` binary and a provider — so the upstream behind
``/u/<user>/`` is the serving test's stub, which answers the app root with a
page carrying
:data:`~tests.deployment.web_terminals.test_auth_serving.TERMINAL_STAND_IN_MARKER`.
Arriving there proves the perimeter delivered the browser to *that user's own
upstream* with a session; it does not prove anything about the terminal UI,
which the in-process suites in this directory cover.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from osprey.services.auth_sidecar.passwords import generation_tag
from osprey.services.auth_sidecar.sessions import SESSION_COOKIE_NAME
from tests.deployment.web_terminals.test_auth_serving import (
    _PASSWORDS,
    TERMINAL_STAND_IN_MARKER,
    serving_stack,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page

    from tests.deployment.web_terminals.test_auth_serving import Stack


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.browser,
    pytest.mark.dockerbuild,
    pytest.mark.slow,
    pytest.mark.skipif(not _docker_available(), reason="docker not available"),
]


def _require_chromium() -> None:
    """Skip the module unless a chromium binary is really launchable.

    Checked *here*, in the fixture that starts the containers, rather than left
    to ``chromium_browser``: pytest builds module-scoped fixtures before
    function-scoped ones, so without this the ordinary unit lane — which has
    docker but no chromium — would compose the whole stack and tear it down
    again for a module in which every test then skips.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover
        pytest.skip("playwright package not installed")

    playwright = sync_playwright().start()
    try:
        playwright.chromium.launch(headless=True).close()
    except Exception as exc:  # pragma: no cover - the usual CI unit-lane path
        pytest.skip(f"Chromium binary not available: {exc}")
    finally:
        # Always stopped: a leaked playwright loop makes every later
        # asyncio.run()/pytest-asyncio test in the worker raise.
        playwright.stop()


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """The composed perimeter, shared by every test in this module.

    Module-scoped for the same reason the serving test's is: three containers
    per test would dominate the runtime. Each test opens its own browser
    context, so no cookie jar is shared between them.
    """
    _require_chromium()
    with serving_stack(tmp_path_factory.mktemp("auth-login-browser")) as running:
        yield running


@pytest.fixture
def page(chromium_browser: Browser) -> Iterator[Page]:
    """A page in a fresh, empty context — a browser that has never logged in."""
    context = chromium_browser.new_context()
    try:
        yield context.new_page()
    finally:
        context.close()


def _unlock(page: Page, user: str) -> None:
    """Type the password and submit, the way the prompt asks to be used.

    Deliberately a real form submission rather than a scripted POST: the
    ``Origin`` header the sidecar requires is set by the browser and cannot be
    set by page script, so this is the only way to exercise that check as a
    person meets it.
    """
    page.fill("#password", _PASSWORDS[user])
    page.click("button.login-submit")


def test_landing_card_leads_to_a_password_prompt_for_that_user(stack: Stack, page: Page) -> None:
    """Clicking alice's card sends this browser to alice's prompt.

    Three separate things have to hold for this to work, and no unit test sees
    any of them together: the landing card is a link to ``/u/alice/``, nginx's
    denied ``auth_request`` turns a *navigation* into a 302 rather than a bare
    401, and the return-to it carries survives into the prompt.
    """
    # Act
    page.goto(f"{stack.base_url}/")
    page.click('a.landing-card[href="/u/alice/"]')

    # Assert
    page.wait_for_selector(".login-card")
    assert "/auth/login" in page.url
    assert page.locator("span.login-user").inner_text().strip() == "alice"
    # The return-to came through the whole chain: card href -> nginx's
    # allowlist map -> the sidecar's own validation -> the form's hidden field.
    assert page.locator('input[name="next"]').input_value() == "/u/alice/"


def test_the_prompt_asks_for_a_password_and_never_for_a_name(stack: Stack, page: Page) -> None:
    """No username field: identity comes from the card, not from typing.

    A name the operator types is a name an attacker can suggest, and it would
    also make the prompt a place to enumerate the roster. The user is fixed by
    the link that got here and travels in a hidden field; the only thing a
    person supplies is the secret.
    """
    # Act
    page.goto(f"{stack.base_url}/u/alice/")
    page.wait_for_selector(".login-card")

    # Assert — a hidden field is fine; a typeable one is the failure.
    assert page.locator('input[name="user"]').get_attribute("type") == "hidden"
    assert page.locator('input[name="user"]').input_value() == "alice"
    assert page.locator('input[type="text"]').count() == 0
    assert page.locator("input:visible").count() == 1
    assert page.locator("#password").get_attribute("type") == "password"


def test_password_unlocks_and_returns_to_the_users_own_terminal(stack: Stack, page: Page) -> None:
    """The whole front door, end to end, as a person walks through it.

    Card -> prompt -> password -> back at ``/u/alice/`` with the terminal
    behind it. The session cookie is set by the sidecar on the deployment
    origin and spent by nginx's ``auth_request`` on the very next request,
    which is the handoff this feature exists to make work.
    """
    # Arrange
    page.goto(f"{stack.base_url}/")
    page.click('a.landing-card[href="/u/alice/"]')
    page.wait_for_selector(".login-card")

    # Act
    _unlock(page, "alice")

    # Assert
    page.wait_for_selector(f"#{TERMINAL_STAND_IN_MARKER}")
    assert page.url == f"{stack.base_url}/u/alice/"
    cookies = {cookie["name"]: cookie for cookie in page.context.cookies()}
    assert SESSION_COOKIE_NAME in cookies
    assert cookies[SESSION_COOKIE_NAME]["httpOnly"] is True


def test_an_unlocked_browser_is_still_refused_at_another_users_terminal(
    stack: Stack, page: Page
) -> None:
    """Unlocking alice does not open bob, and the prompt says whose it is.

    The perimeter's whole point, seen the way an operator would meet it: the
    same browser, one hop later, is asked for a *different* user's password
    rather than let through.
    """
    # Arrange
    page.goto(f"{stack.base_url}/u/alice/")
    page.wait_for_selector(".login-card")
    _unlock(page, "alice")
    page.wait_for_selector(f"#{TERMINAL_STAND_IN_MARKER}")

    # Act
    page.goto(f"{stack.base_url}/u/bob/")

    # Assert
    page.wait_for_selector(".login-card")
    assert page.locator("span.login-user").inner_text().strip() == "bob"


def test_a_wrong_password_re_prompts_without_naming_the_reason(stack: Stack, page: Page) -> None:
    """A refusal returns the prompt with one fixed message and no session.

    One message for every reason — unknown user, no credential, wrong secret —
    is what stops the prompt from being an oracle. ``carol`` is used because
    her attempt window is nothing else's business.
    """
    # Arrange
    page.goto(f"{stack.base_url}/u/carol/")
    page.wait_for_selector(".login-card")

    # Act
    page.fill("#password", "not-carols-password")
    page.click("button.login-submit")

    # Assert
    page.wait_for_selector("p.login-error")
    assert page.locator("p.login-error").get_attribute("role") == "alert"
    assert "carol" not in page.locator("p.login-error").inner_text()
    assert SESSION_COOKIE_NAME not in {cookie["name"] for cookie in page.context.cookies()}


def test_an_expired_session_sends_a_reload_back_to_the_prompt(stack: Stack, page: Page) -> None:
    """A tab left open overnight reloads into the login page, not an error.

    Forged rather than waited for — the rendered lifetime is twelve hours — and
    forged with the sidecar's own signing secret, so what the perimeter refuses
    is a cookie whose signature it accepts. That is the case a browser actually
    meets: the cookie is still there and still valid-looking, and the entry
    inside it has lapsed.
    """
    # Arrange — a real unlock first, so the reload is a genuine "this tab was
    # working a moment ago" and not a browser that never had a session.
    page.goto(f"{stack.base_url}/u/alice/")
    page.wait_for_selector(".login-card")
    _unlock(page, "alice")
    page.wait_for_selector(f"#{TERMINAL_STAND_IN_MARKER}")

    codec = stack.codec()
    lapsed = codec.new_state().with_user(
        "alice",
        expires_at=codec.now() - 1,
        generation_tag=generation_tag(stack.stored_hashes["alice"]),
    )
    page.context.clear_cookies()
    page.context.add_cookies(
        [
            {
                "name": SESSION_COOKIE_NAME,
                "value": codec.encode(lapsed),
                "domain": "127.0.0.1",
                "path": "/",
            }
        ]
    )

    # Act
    page.reload()

    # Assert
    page.wait_for_selector(".login-card")
    assert page.locator("span.login-user").inner_text().strip() == "alice"
    assert page.locator(f"#{TERMINAL_STAND_IN_MARKER}").count() == 0
