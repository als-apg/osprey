"""Unit tests for the demo-video overlay init script and its Python helpers.

CI-safe: the helpers are exercised against a fake page that records calls, and the
init script is checked as text (plus a syntax check when ``node`` is on PATH). No
browser is launched.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from docs.screenshots import video_overlay
from docs.screenshots.video_overlay import OVERLAY_JS, move_cursor, set_caption


class _FakeMouse:
    def __init__(self, log: list) -> None:
        self._log = log

    def move(self, x: float, y: float, **kwargs) -> None:
        self._log.append(("mouse", x, y))


class _FakePage:
    def __init__(self) -> None:
        self.log: list = []
        self.mouse = _FakeMouse(self.log)

    def evaluate(self, expression: str, arg=None):
        self.log.append(("eval", expression, arg))

    def wait_for_timeout(self, ms) -> None:
        self.log.append(("wait", ms))
        raise AssertionError("move_cursor must not wait")


@pytest.fixture(autouse=True)
def _reset_positions(monkeypatch):
    monkeypatch.setattr(video_overlay, "_cursor_positions", {})


# ---------------------------------------------------------------------------
# OVERLAY_JS text
# ---------------------------------------------------------------------------


def _first_statement(js: str) -> str:
    body = js.strip()
    assert body.startswith("(() => {"), "init script must be a self-invoking wrapper"
    return body[len("(() => {") :].lstrip().splitlines()[0]


def test_overlay_js_begins_with_top_frame_guard():
    assert _first_statement(OVERLAY_JS) == "if (window.top !== window) return;"


def test_overlay_js_mounts_on_domcontentloaded_or_immediately():
    assert 'addEventListener("DOMContentLoaded", mount' in OVERLAY_JS
    assert 'document.readyState === "loading"' in OVERLAY_JS


def test_overlay_js_defines_functions_before_mounting():
    define_at = OVERLAY_JS.index("window.__demoCursor =")
    assert define_at < OVERLAY_JS.index('addEventListener("DOMContentLoaded"')
    for name in ("window.__demoCursor", "window.__demoCaption"):
        assert f"{name} =" in OVERLAY_JS


def test_overlay_js_queues_calls_until_mounted():
    assert "queue.push(call)" in OVERLAY_JS
    assert "while (queue.length) apply(queue.shift())" in OVERLAY_JS


def test_overlay_js_elements_are_fixed_topmost_and_click_through():
    for token in ("position:fixed", "z-index:2147483647", "pointer-events:none"):
        assert token in OVERLAY_JS


def test_the_page_carries_no_speed_badge():
    # The speed-up badge is burned in at encode time, the only place that
    # knows the factor; the page shows none of its own.
    assert "__demoBadge" not in OVERLAY_JS
    assert "fast-forward" not in OVERLAY_JS
    assert "\\u23e9" not in OVERLAY_JS


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_overlay_js_is_valid_javascript(tmp_path):
    script = tmp_path / "overlay.js"
    script.write_text(OVERLAY_JS)
    result = subprocess.run(
        ["node", "--check", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_set_caption_passes_text():
    page = _FakePage()
    set_caption(page, "Asking the agent")
    [(kind, expr, arg)] = page.log
    assert kind == "eval" and "__demoCaption" in expr and arg == "Asking the agent"


def test_set_caption_none_hides():
    page = _FakePage()
    set_caption(page, None)
    assert page.log == [("eval", "(t) => window.__demoCaption(t)", None)]


def test_move_cursor_interpolates_and_draws_each_step():
    page = _FakePage()
    move_cursor(page, 100, 50, steps=4)
    mouse = [e for e in page.log if e[0] == "mouse"]
    evals = [e for e in page.log if e[0] == "eval"]
    assert mouse == [
        ("mouse", 25, 12.5),
        ("mouse", 50, 25),
        ("mouse", 75, 37.5),
        ("mouse", 100, 50),
    ]
    assert [e[2] for e in evals] == [[25, 12.5], [50, 25], [75, 37.5], [100, 50]]
    assert all("__demoCursor" in e[1] for e in evals)
    # Each mouse move is followed by its dot redraw.
    assert [e[0] for e in page.log] == ["mouse", "eval"] * 4


def test_move_cursor_never_waits_between_steps():
    # Each step is a round trip played in real time; the glide adds no waits.
    page = _FakePage()
    move_cursor(page, 100, 50, steps=3)
    assert [e[0] for e in page.log] == ["mouse", "eval"] * 3


def test_move_cursor_continues_from_last_position():
    page = _FakePage()
    move_cursor(page, 100, 100, steps=1)
    page.log.clear()
    move_cursor(page, 200, 0, steps=2)
    assert [e for e in page.log if e[0] == "mouse"] == [("mouse", 150, 50), ("mouse", 200, 0)]


def test_move_cursor_positions_are_per_page():
    a, b = _FakePage(), _FakePage()
    move_cursor(a, 100, 100, steps=1)
    move_cursor(b, 10, 10, steps=2)
    assert [e for e in b.log if e[0] == "mouse"] == [("mouse", 5, 5), ("mouse", 10, 10)]


@pytest.mark.parametrize("steps", [0, -3])
def test_move_cursor_nonpositive_steps_moves_once(steps):
    page = _FakePage()
    move_cursor(page, 40, 20, steps=steps)
    assert [e for e in page.log if e[0] == "mouse"] == [("mouse", 40, 20)]


def test_overlay_module_never_sleeps():
    assert "time.sleep" not in open(video_overlay.__file__).read()
