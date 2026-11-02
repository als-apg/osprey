"""Browser tests: the watchdog dialog against the real routes, record and event stream.

Flags live in a real hold record on disk and are read through the real
``GET /api/watchdog/flags``; frames travel the real broadcaster and SSE socket.
Only a Stop's turn interrupt is replaced (at the page's answer request): whether
a turn stops is the route's suite, what the dialog does with the answer is this
one's.

Run:
    uv run pytest tests/interfaces/web_terminal/test_watchdog_dialog_browser.py -m browser -q

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import contextlib
import json
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import Request
from fastapi.routing import APIRoute

from osprey_connectors import session_hold
from tests.interfaces.web_terminal._posture_chip import (
    CHIP_STATE,
    TIMEOUT,
    chip_hub,
    settled_chip,
)
from tests.interfaces.web_terminal._surface import back_to_chat

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from playwright.sync_api import Browser, Page

with contextlib.suppress(ImportError):
    from playwright.sync_api import expect

pytestmark = [pytest.mark.browser, pytest.mark.slow]

DIALOG = ".watchdog-alert-dialog"
FLAG = f"{DIALOG} .watchdog-alert-flag"
STOP = f"{DIALOG} .watchdog-alert-stop"
CONTINUE = f"{DIALOG} .watchdog-alert-continue"

#: WATCHDOG_POLL_MS in the shipped module, plus room for one slow read.
POLL_WAIT_MS = 15_000
#: Long enough that a poll, had one been armed, would have fired.
QUIET_MS = 12_000

_FRAME_ROUTE = "/__test/watchdog-frame"
_SETTLE_TWO_FRAMES = (
    "() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))"
)


def _add_frame_route(app: Any) -> None:
    """Broadcast from the server's own loop: the broadcaster's queues belong to it."""

    async def send_frame(request: Request) -> dict[str, bool]:
        app.state.broadcaster.broadcast(await request.json())
        return {"sent": True}

    app.router.routes.insert(0, APIRoute(_FRAME_ROUTE, send_frame, methods=["POST"]))


@contextmanager
def _watchdog_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
) -> Iterator[tuple[str, Any, Path]]:
    with chip_hub(tmp_path, monkeypatch) as (base_url, app, _):
        root = tmp_path / "holds"
        root.mkdir()
        if enabled:
            app.state.session_hold_root = root
        _add_frame_route(app)
        yield base_url, app, root


def _session_key(page: Page) -> str:
    page.wait_for_function(
        "async () => (await import('/static/js/terminal.js')).getCurrentSessionId() !== null",
        timeout=TIMEOUT,
    )
    return page.evaluate(
        "async () => (await import('/static/js/terminal.js')).getCurrentSessionId()"
    )


def _open_flag(root: Path, key: str, name: str, *, reason: str = "it moves a magnet") -> str:
    flag_id = f"flag-{name}"
    assert session_hold.open_flag(
        key,
        flag_id=flag_id,
        source=name,
        reason=reason,
        prompt_ref=f"ref-{name}",
        prompt_excerpt=f"prompt seen by {name}",
        root=root,
    )
    return flag_id


def _send_frame(page: Page, frame: dict[str, Any]) -> None:
    page.evaluate(
        "async ([url, body]) => { await fetch(url, {method: 'POST', "
        "headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}); }",
        [_FRAME_ROUTE, frame],
    )


def _flag_frame(key: str, name: str) -> dict[str, str]:
    return {
        "type": "watchdog_flag",
        "session_key": key,
        "flag_id": f"flag-{name}",
        "watchdog": name,
    }


def _open_page(browser: Browser, base_url: str) -> tuple[Page, str]:
    page, _ = settled_chip(browser, base_url)
    # The SSE socket is what the frame helper needs; a real frame proves it below.
    page.wait_for_function("() => document.readyState === 'complete'", timeout=TIMEOUT)
    return page, _session_key(page)


def _assert_never_says_safe(page: Page) -> None:
    assert not re.search(r"\bsafe\b", page.inner_text("body"), re.IGNORECASE)


def test_two_watchdogs_share_one_dialog_naming_watchdog_prompt_and_reason(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, key = _open_page(chromium_browser, base_url)
        _open_flag(root, key, "alpha", reason="alpha reason")
        _open_flag(root, key, "beta", reason="beta reason")
        page.reload()
        expect(page.locator(FLAG)).to_have_count(2, timeout=TIMEOUT)
        expect(page.locator(DIALOG)).to_have_count(1)
        for name in ("alpha", "beta"):
            item = page.locator(f'{FLAG}[data-flag-id="flag-{name}"]')
            expect(item).to_contain_text(name)
            expect(item).to_contain_text(f"prompt seen by {name}")
            expect(item).to_contain_text(f"{name} reason")
        _assert_never_says_safe(page)


def test_a_flag_joins_an_open_dialog_on_its_frame_and_chip_shows_held(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, key = _open_page(chromium_browser, base_url)
        _open_flag(root, key, "alpha")
        _send_frame(page, _flag_frame(key, "alpha"))
        expect(page.locator(FLAG)).to_have_count(1, timeout=TIMEOUT)
        expect(page.locator(CHIP_STATE)).to_have_text("held", timeout=TIMEOUT)

        _open_flag(root, key, "beta")
        _send_frame(page, _flag_frame(key, "beta"))
        expect(page.locator(FLAG)).to_have_count(2, timeout=3_000)


def test_escape_scrim_and_another_confirm_leave_it_open(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, key = _open_page(chromium_browser, base_url)
        _open_flag(root, key, "alpha")
        _send_frame(page, _flag_frame(key, "alpha"))
        expect(page.locator(FLAG)).to_have_count(1, timeout=TIMEOUT)

        page.keyboard.press("Escape")
        page.keyboard.press("Enter")
        page.mouse.click(2, 2)
        page.evaluate(_SETTLE_TWO_FRAMES)
        expect(page.locator(FLAG)).to_have_count(1)

        page.evaluate(
            """async () => {
                const m = await import('/static/js/posture-confirm.js');
                void m.showConfirm({title: 'Another', body: 'x', confirmLabel: 'Yes'});
                m.dismissConfirm();
                void m.showConfirm({title: 'Another', body: 'x', confirmLabel: 'Yes'});
            }"""
        )
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)
        expect(page.locator(FLAG)).to_have_count(1)
        assert session_hold.is_held(key, root=root)


def test_stop_and_continue_answer_the_displayed_flags_and_close(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, key = _open_page(chromium_browser, base_url)
        ids = [_open_flag(root, key, "alpha"), _open_flag(root, key, "beta")]
        _send_frame(page, _flag_frame(key, "alpha"))
        expect(page.locator(FLAG)).to_have_count(2, timeout=TIMEOUT)

        # A flag the screen does not show yet is not answered by this click.
        _open_flag(root, key, "late")
        with page.expect_response("**/api/watchdog/flags/answer") as answered:
            page.locator(CONTINUE).click()
        assert answered.value.ok
        assert json.loads(answered.value.request.post_data or "{}")["flag_ids"] == ids
        expect(page.locator(FLAG)).to_have_count(1, timeout=TIMEOUT)
        expect(page.locator(FLAG)).to_contain_text("late")

        def stopped(route: Any) -> None:
            released = session_hold.answer_flags(
                key, ["flag-late"], answer="stop", operator="t", root=root
            )
            route.fulfill(json={"released": list(released), "stop_unconfirmed": []})

        page.route("**/api/watchdog/flags/answer", stopped)
        page.locator(STOP).click()
        expect(page.locator(DIALOG)).to_have_count(0, timeout=TIMEOUT)
        assert not session_hold.is_held(key, root=root)


def test_stop_unconfirmed_says_the_turn_did_not_stop_and_stays_open(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, key = _open_page(chromium_browser, base_url)
        flag_id = _open_flag(root, key, "alpha")
        _send_frame(page, _flag_frame(key, "alpha"))
        expect(page.locator(FLAG)).to_have_count(1, timeout=TIMEOUT)

        def unconfirmed(route: Any) -> None:
            marked = session_hold.mark_stop_unconfirmed(
                key, [flag_id], operator="t", started=("sleep 900",), root=root
            )
            route.fulfill(json={"released": [], "stop_unconfirmed": list(marked)})

        page.route("**/api/watchdog/flags/answer", unconfirmed)
        page.locator(STOP).click()
        expect(page.locator(FLAG)).to_contain_text("turn did not stop", timeout=TIMEOUT)
        expect(page.locator(FLAG)).to_contain_text("sleep 900")
        expect(page.locator(CONTINUE)).to_be_enabled()


def test_a_dropped_first_frame_is_recovered_by_the_poll(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, key = _open_page(chromium_browser, base_url)
        expect(page.locator(DIALOG)).to_have_count(0)
        _open_flag(root, key, "alpha")  # no frame is ever sent
        expect(page.locator(FLAG)).to_have_count(1, timeout=POLL_WAIT_MS)


def test_disabled_watchdog_does_not_poll(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch, enabled=False) as (base_url, _app, _root):
        reads: list[str] = []
        page = chromium_browser.new_page()
        page.on(
            "request",
            lambda request: (
                reads.append(request.url) if "/api/watchdog/flags" in request.url else None
            ),
        )
        page.goto(base_url)
        page.wait_for_function(
            "async () => (await import('/static/js/terminal.js')).getCurrentSessionId() !== null",
            timeout=TIMEOUT,
        )
        page.wait_for_timeout(2_000)
        assert reads, "the load read never happened"
        settled = len(reads)
        page.wait_for_timeout(QUIET_MS)
        assert len(reads) == settled
        expect(page.locator(DIALOG)).to_have_count(0)


def test_unavailable_is_a_per_watchdog_indicator_with_no_dialog(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, key = _open_page(chromium_browser, base_url)
        _send_frame(
            page,
            {
                "type": "watchdog_status",
                "session_key": key,
                "watchdog": "alpha",
                "status": "unavailable",
            },
        )
        line = page.locator('.watchdog-status-line[data-watchdog="alpha"]')
        expect(line).to_contain_text("unavailable", timeout=TIMEOUT)
        expect(page.locator(DIALOG)).to_have_count(0)
        _assert_never_says_safe(page)


def test_a_held_refusal_in_the_conversation_rereads_the_flags_at_once(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, key = _open_page(chromium_browser, base_url)
        _open_flag(root, key, "alpha")
        page.evaluate(
            """() => {
                const line = document.createElement('p');
                line.className = 'op-tool-refusal';
                line.dataset.reason = 'session_held';
                document.body.append(line);
            }"""
        )
        expect(page.locator(FLAG)).to_have_count(1, timeout=3_000)


def test_the_chat_raises_the_same_dialog(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _watchdog_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, key = _open_page(chromium_browser, base_url)
        back_to_chat(page)
        _open_flag(root, key, "alpha")
        _send_frame(page, _flag_frame(key, "alpha"))
        expect(page.locator(FLAG)).to_contain_text("prompt seen by alpha", timeout=TIMEOUT)
        page.locator(CONTINUE).click()
        expect(page.locator(DIALOG)).to_have_count(0, timeout=TIMEOUT)
