"""The code review video's Settings drawer beats, on a fake page that slides the drawer."""

from __future__ import annotations

import pytest
from docs.screenshots import video_beats, video_code_review
from docs.screenshots.video_code_review import DRAWER_JS, PROCEED
from docs.screenshots.video_probes import ProbeFailed

STORYBOARD = {beat.name: beat for beat in video_code_review.CODE_REVIEW}


class FakeLocator:
    def __init__(self, selector):
        self.selector = selector
        self.first = self


class FakePage:
    """``states`` are the drawer readings in order; the last one repeats."""

    def __init__(self, states, warning=False):
        self.states = list(states)
        self.warning = warning
        self.waits = 0

    def evaluate(self, js, arg=None):
        if js == DRAWER_JS:
            return self.states[0] if len(self.states) == 1 else self.states.pop(0)
        if js == video_beats._SHOWN_JS:
            return arg[0] == PROCEED and self.warning
        raise AssertionError(js)

    def locator(self, selector):
        return FakeLocator(selector)

    def wait_for_timeout(self, _ms):
        self.waits += 1


class FakeTake:
    def __init__(self, page):
        self.page = page
        self.clicked: list[str] = []

    def click(self, locator):
        self.clicked.append(locator.selector)
        self.page.warning = False


CLOSED = {"open": False, "shown": False, "gone": True}
SLIDING_IN = {"open": True, "shown": False, "gone": False}
OPEN = {"open": True, "shown": True, "gone": False}
SLIDING_OUT = {"open": False, "shown": False, "gone": False}


def test_settings_open_proceeds_past_the_warning_once_and_waits_for_the_slide() -> None:
    page = FakePage([CLOSED, CLOSED, SLIDING_IN, OPEN], warning=True)
    take = FakeTake(page)
    STORYBOARD["settings-open"].run(take)
    assert take.clicked == [PROCEED]
    assert page.waits == 3


def test_settings_open_without_the_warning_clicks_nothing() -> None:
    take = FakeTake(FakePage([SLIDING_IN, OPEN]))
    STORYBOARD["settings-open"].run(take)
    assert take.clicked == []


def test_settings_closed_waits_until_the_drawer_is_off_screen() -> None:
    page = FakePage([{**OPEN, "open": False}, SLIDING_OUT, CLOSED])
    STORYBOARD["settings-closed"].run(FakeTake(page))
    assert page.waits == 2


def test_a_drawer_that_never_opens_fails_its_beat() -> None:
    with pytest.raises(ProbeFailed) as failed:
        STORYBOARD["settings-open"].run(FakeTake(FakePage([CLOSED])))
    assert failed.value.step == "settings-open" and "shown" in failed.value.detail
