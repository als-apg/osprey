"""Unit tests for the storyboard beat primitives, on a minimal fake page and take."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from docs.screenshots import video_beats, video_probes
from docs.screenshots.video_beats import (
    Beat,
    click,
    focus,
    hold,
    prompt,
    spotlight,
    wait_for,
    wait_gone,
)
from docs.screenshots.video_probes import ProbeFailed


class FakeLocator:
    def __init__(self, page, selector):
        self.page = page
        self.selector = selector
        self.first = self

    def filter(self, has_text=None):  # noqa: ARG002 - the text match is shown()'s job
        return self

    def scroll_into_view_if_needed(self):
        self.page.scrolled.append(self.selector)

    def bounding_box(self):
        return self.page.boxes.get(self.selector)


class FakePage:
    """``visible`` maps a selector to its text; ``schedule`` maps a tick to a change."""

    def __init__(self, visible=None, schedule=None):
        self.visible = dict(visible or {})
        self.schedule = dict(schedule or {})
        self.ticks = 0
        self.waits: list[int] = []
        self.scrolled: list[str] = []
        self.boxes: dict[str, dict] = {}
        self.spotlights: list = []

    def evaluate(self, js, arg=None):
        if "__demoSpotlight" in js:
            self.spotlights.append(arg)
            return None
        if js == video_beats._SHOWN_JS:
            selector, text = arg
            return selector in self.visible and (not text or text in self.visible[selector])
        return "line one\nline two"  # the terminal text

    def wait_for_timeout(self, ms):
        self.waits.append(ms)
        self.ticks += 1
        for selector, text in self.schedule.pop(self.ticks, {}).items():
            if text is None:
                self.visible.pop(selector, None)
            else:
                self.visible[selector] = text

    def locator(self, selector):
        return FakeLocator(self, selector)


class FakeTake:
    def __init__(self, page=None, turn="running"):
        self.page = page or FakePage()
        self.state = turn
        self.log: list = []

    @contextmanager
    def step(self, name, caption, fast_forward=False):
        self.log.append(("enter", name, caption, fast_forward))
        yield
        self.log.append(("exit", name))

    def watch(self, _name, _dialogs):
        return lambda: None

    def turn(self):
        return self.state

    def tail(self):
        # The engine's Take reads the console of its view; this fake's is the terminal.
        return video_probes.terminal_tail(self.page)

    def submit(self, name, text):
        self.log.append(("submit", name, text))

    def click(self, locator):
        self.log.append(("click", locator.selector))

    def focus_terminal(self):
        self.log.append(("focus",))


def _failure(beat, take) -> ProbeFailed:
    with pytest.raises(ProbeFailed) as exc:
        beat(take)
    return exc.value


# --- each primitive's failure names its own beat ------------------------------


def test_click_with_nothing_visible_fails_its_beat() -> None:
    take = FakeTake()
    failed = _failure(click("continue", None, ".go"), take)
    assert failed.step == "continue" and ".go" in failed.detail
    assert not [e for e in take.log if e[0] == "click"]


def test_wait_for_timeout_fails_its_beat_with_the_terminal_tail() -> None:
    take = FakeTake()
    failed = _failure(wait_for("judge", None, ".alert", budget_ms=1_000), take)
    assert failed.step == "judge"
    assert "1000 ms" in failed.detail and "line two" in failed.detail


def test_wait_for_fails_at_once_on_a_refuse_selector() -> None:
    take = FakeTake(FakePage(visible={".status": "unavailable"}))
    beat = wait_for("judge", None, ".alert", budget_ms=60_000, refuse=((".status", "down"),))
    failed = _failure(beat, take)
    assert (failed.step, failed.detail) == ("judge", "down")
    assert take.page.ticks == 0


def test_wait_for_fails_at_once_on_an_api_error_turn() -> None:
    take = FakeTake(turn="errored")
    failed = _failure(wait_for("judge", None, ".alert", budget_ms=60_000), take)
    assert failed.step == "judge" and "API error" in failed.detail
    assert take.page.ticks == 0


def test_wait_gone_timeout_fails_its_beat() -> None:
    take = FakeTake(FakePage(visible={".dialog": ""}))
    failed = _failure(wait_gone("released", None, ".dialog", budget_ms=1_000), take)
    assert failed.step == "released" and ".dialog still on screen" in failed.detail


# --- waits and beats ----------------------------------------------------------


def test_wait_for_returns_as_soon_as_the_selector_shows() -> None:
    page = FakePage(schedule={3: {".alert": "flagged"}})
    take = FakeTake(page)
    wait_for("judge", None, ".alert", text="flagged", budget_ms=60_000)(take)
    assert page.ticks == 3
    assert page.waits == [video_probes.POLL_MS] * 3


def test_wait_for_ignores_the_selector_without_its_text() -> None:
    take = FakeTake(FakePage(visible={".alert": "other-flag"}))
    failed = _failure(wait_for("judge", None, ".alert", text="guard", budget_ms=600), take)
    assert failed.step == "judge"


def test_wait_gone_returns_once_the_selector_leaves() -> None:
    page = FakePage(visible={".dialog": ""}, schedule={2: {".dialog": None}})
    wait_gone("released", None, ".dialog", budget_ms=10_000)(FakeTake(page))
    assert page.ticks == 2


def test_a_beat_runs_inside_a_step_with_its_name_caption_and_fast_forward() -> None:
    take = FakeTake()
    Beat("judge", "The agent judges", lambda t: t.log.append(("run",)), True)(take)
    assert take.log == [
        ("enter", "judge", "The agent judges", True),
        ("run",),
        ("exit", "judge"),
    ]


def test_primitives_carry_their_fast_forward_flag() -> None:
    assert wait_for("w", None, ".x", budget_ms=1, fast_forward=True).fast_forward
    assert not wait_gone("g", None, ".x", budget_ms=1).fast_forward
    assert not hold("h", None, 1).fast_forward


def test_prompt_submits_its_text_under_its_beat_name() -> None:
    take = FakeTake()
    prompt("ask", "An operator asks", "loosen the guard")(take)
    assert ("submit", "ask", "loosen the guard") in take.log


def test_focus_hold_and_click_act_on_the_take() -> None:
    page = FakePage(visible={".go": "Continue"})
    take = FakeTake(page)
    focus("focus", None)(take)
    hold("hold", None, 7_000)(take)
    click("continue", None, ".go", text="Continue")(take)
    assert ("focus",) in take.log and ("click", ".go") in take.log
    assert page.waits == [7_000]


# --- spotlight ----------------------------------------------------------------

BOX = {"x": 10.0, "y": 20.0, "width": 300.0, "height": 40.0}


def test_spotlight_outlines_the_element_for_its_hold_then_clears() -> None:
    page = FakePage(visible={"li": "read SR:BEAM:USER_MODE"})
    page.boxes["li"] = BOX
    take = FakeTake(page)
    spotlight("rule", "The rule", "li", text="SR:BEAM:USER_MODE", ms=4_000)(take)
    assert page.scrolled == ["li"]
    assert page.spotlights == [BOX, None]
    assert page.waits == [4_000]
    assert take.log[0] == ("enter", "rule", "The rule", False)


def test_spotlight_with_nothing_visible_fails_its_beat_and_draws_nothing() -> None:
    page = FakePage(visible={"li": "another rule"})
    failed = _failure(spotlight("rule", None, "li", text="USER_MODE", ms=1), FakeTake(page))
    assert failed.step == "rule" and "li" in failed.detail
    assert page.spotlights == []


def test_spotlight_without_a_box_fails_its_beat() -> None:
    page = FakePage(visible={"li": "x"})
    failed = _failure(spotlight("rule", None, "li", ms=1), FakeTake(page))
    assert failed.step == "rule" and page.spotlights == []


def test_spotlight_clears_even_when_the_hold_is_interrupted() -> None:
    page = FakePage(visible={"li": "x"})
    page.boxes["li"] = BOX

    def interrupted(_ms):
        raise RuntimeError("take aborted")

    page.wait_for_timeout = interrupted
    with pytest.raises(RuntimeError):
        spotlight("rule", None, "li", ms=1)(FakeTake(page))
    assert page.spotlights == [BOX, None]
