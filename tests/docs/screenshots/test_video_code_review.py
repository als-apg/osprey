"""The code review video's own beats: the ⌘K route to the rules and the Settings drawer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from docs.screenshots import video_beats, video_code_review
from docs.screenshots.video_code_review import (
    ACKNOWLEDGE_JS,
    ACTIVE_JS,
    CHORD_JS,
    DRAWER_JS,
    PALETTE_FOCUSED_JS,
    ROW_TEXT,
    SEARCH,
    WARNING,
)
from docs.screenshots.video_probes import ProbeFailed

STORYBOARD = {beat.name: beat for beat in video_code_review.CODE_REVIEW}


class FakeKeyboard:
    def __init__(self, page):
        self.page = page

    def press(self, key):
        self.page.keys.append(key)
        if key == "ArrowDown":
            self.page.active = (self.page.active + 1) % len(self.page.rows)

    def type(self, text, delay):
        self.page.keys.append(("type", text, delay))


class FakeLocator:
    def __init__(self, page, selector):
        self.page = page
        self.selector = selector
        self.first = self

    def count(self):
        return len(self.page.rows)


class FakePage:
    """``states`` are the drawer readings in order; the last one repeats."""

    def __init__(self, states=(), warning=False, rows=(), platform="mac", focus_after=0):
        self.states = list(states)
        self.warning = warning
        self.rows = list(rows)
        self.active = 0
        self.platform = platform
        self.focus_after = focus_after
        self.keys: list = []
        self.waits = 0
        self.keyboard = FakeKeyboard(self)

    def evaluate(self, js, arg=None):
        if js == DRAWER_JS:
            return self.states[0] if len(self.states) == 1 else self.states.pop(0)
        if js == video_beats._SHOWN_JS:
            return arg[0] == WARNING and self.warning
        if js == CHORD_JS:
            return "Meta+k" if self.platform == "mac" else "Control+k"
        if js == PALETTE_FOCUSED_JS:
            return self.waits >= self.focus_after
        if js == ACTIVE_JS:
            return self.rows[self.active] if self.rows else None
        raise AssertionError(js)

    def locator(self, selector):
        return FakeLocator(self, selector)

    def wait_for_timeout(self, _ms):
        self.waits += 1


def _take(page):
    return SimpleNamespace(page=page)


CLOSED = {"open": False, "shown": False, "gone": True}
SLIDING_IN = {"open": True, "shown": False, "gone": False}
OPEN = {"open": True, "shown": True, "gone": False}
SLIDING_OUT = {"open": False, "shown": False, "gone": False}


def test_settings_open_waits_for_the_slide() -> None:
    page = FakePage([CLOSED, SLIDING_IN, OPEN])
    STORYBOARD["settings-open"].run(_take(page))
    assert page.waits == 2


def test_the_settings_warning_on_camera_fails_the_take() -> None:
    with pytest.raises(ProbeFailed) as failed:
        STORYBOARD["settings-open"].run(_take(FakePage([CLOSED], warning=True)))
    assert failed.value.step == "settings-open" and "warning" in failed.value.detail


def test_settings_closed_waits_until_the_drawer_is_off_screen() -> None:
    page = FakePage([{**OPEN, "open": False}, SLIDING_OUT, CLOSED])
    STORYBOARD["settings-closed"].run(_take(page))
    assert page.waits == 2


def test_a_drawer_that_never_opens_fails_its_beat() -> None:
    with pytest.raises(ProbeFailed) as failed:
        STORYBOARD["settings-open"].run(_take(FakePage([CLOSED])))
    assert failed.value.step == "settings-open" and "shown" in failed.value.detail


@pytest.mark.parametrize(("platform", "chord"), [("mac", "Meta+k"), ("linux", "Control+k")])
def test_the_palette_opens_on_the_pages_own_chord(platform, chord) -> None:
    page = FakePage(platform=platform, focus_after=2)
    STORYBOARD["palette"].run(_take(page))
    assert page.keys == [chord]
    assert page.waits == 2


def test_a_palette_that_never_takes_the_keyboard_fails_its_beat() -> None:
    with pytest.raises(ProbeFailed) as failed:
        STORYBOARD["palette"].run(_take(FakePage(focus_after=10**6)))
    assert failed.value.step == "palette"


def test_the_search_is_typed_at_reading_speed() -> None:
    page = FakePage()
    STORYBOARD["search"].run(_take(page))
    ((kind, text, delay),) = page.keys
    assert (kind, text) == ("type", SEARCH) and delay > 0


def test_the_reveal_arrows_down_to_the_rules_row_and_presses_enter() -> None:
    page = FakePage(rows=["Code review: level", f"{ROW_TEXT} code-review/rules", "Other"])
    STORYBOARD["reveal"].run(_take(page))
    assert page.keys == ["ArrowDown", "Enter"]


def test_the_reveal_without_the_rules_row_fails_and_presses_no_enter() -> None:
    page = FakePage(rows=["Code review: level", "Other"])
    with pytest.raises(ProbeFailed) as failed:
        STORYBOARD["reveal"].run(_take(page))
    assert failed.value.step == "reveal" and "Enter" not in page.keys


def test_the_warning_is_acknowledged_off_camera_for_the_server_session() -> None:
    page = SimpleNamespace(evaluate=lambda js: "server-session" if js == ACKNOWLEDGE_JS else None)
    video_code_review.acknowledge_settings(SimpleNamespace(page=page))
    assert "osprey-settings-warning-ack" in ACKNOWLEDGE_JS
    assert "scopedStorageKey" in ACKNOWLEDGE_JS


def test_no_server_session_to_acknowledge_fails_before_recording() -> None:
    page = SimpleNamespace(evaluate=lambda _js: None)
    with pytest.raises(ProbeFailed) as failed:
        video_code_review.acknowledge_settings(SimpleNamespace(page=page))
    assert failed.value.step == "acknowledge"


def test_every_row_of_the_script_has_its_own_subtitle() -> None:
    captions = [b.caption for b in video_code_review.CODE_REVIEW]
    assert None not in captions
    shown = [c for i, c in enumerate(captions) if i == 0 or captions[i - 1] != c]
    assert shown == list(video_code_review.CAPTIONS)
