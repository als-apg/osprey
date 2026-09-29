"""Unit tests for the demo-video page probes.

CI-safe: every probe runs against small fake page and frame objects that
answer ``evaluate`` from canned values, so no browser is launched.
"""

from __future__ import annotations

import math

import pytest
from docs.screenshots import video_probes
from docs.screenshots.video_probes import (
    REPL_READY_MARKERS,
    TRUST_MARKERS,
    ProbeFailed,
    arm_relayout,
    await_relayout,
    await_repl_ready,
    azimuth_change,
    camera_azimuth,
    draft_ready,
    plot_frame,
    terminal_focused,
    xterm_text,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeKeyboard:
    def __init__(self) -> None:
        self.presses: list[str] = []

    def press(self, key: str) -> None:
        self.presses.append(key)


class FakeHandle:
    def __init__(self, frame: object | None) -> None:
        self._frame = frame

    def content_frame(self) -> object | None:
        return self._frame


class FakeLocator:
    def __init__(self, frame: object | None, present: bool) -> None:
        self._frame = frame
        self._present = present

    @property
    def first(self) -> FakeLocator:
        return self

    def element_handle(self) -> FakeHandle | None:
        return FakeHandle(self._frame) if self._present else None


class FakePage:
    """Answers ``evaluate`` by matching a substring of the script."""

    def __init__(
        self,
        *,
        texts: list[str] | None = None,
        focused: bool = False,
        children: dict[str, object] | None = None,
        values: dict[str, object] | None = None,
    ) -> None:
        self.texts = list(texts or [""])
        self.focused = focused
        self.children = children or {}
        self.values = values or {}
        self.keyboard = FakeKeyboard()
        self.waits: list[int] = []
        self.evaluated: list[tuple[str, object]] = []
        self.text_reads = 0

    def evaluate(self, script: str, arg: object = None) -> object:
        self.evaluated.append((script, arg))
        if ".xterm-rows" in script:
            i = min(self.text_reads, len(self.texts) - 1)
            self.text_reads += 1
            return self.texts[i]
        if "xterm-helper-textarea" in script:
            return self.focused
        for key, value in self.values.items():
            if key in script:
                return value
        raise AssertionError(f"unexpected script: {script[:60]}")

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self.children.get(selector), selector in self.children)

    def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def test_markers_match_the_proposal():
    assert TRUST_MARKERS == ("Yes, I trust this folder", "Quick safety check")
    assert REPL_READY_MARKERS == ("? for shortcuts",)


def test_xterm_text_turns_non_breaking_spaces_into_spaces():
    # xterm's DOM renderer writes some spaces as U+00A0, which no marker spells.
    rule = "\u2500" * 20
    page = FakePage(texts=[f'{rule}\n\u276f\u00a0Try "how do I log an error?"'])
    assert video_probes.xterm_text(page) == f'{rule}\n\u276f Try "how do I log an error?"'
    await_repl_ready(page)
    assert page.keyboard.presses == []


def test_terminal_tail_is_the_last_five_non_blank_lines():
    page = FakePage(texts=["a\n\nb\nc\nd\ne\nDo you want to proceed?\n\u00a0"])
    assert video_probes.terminal_tail(page) == "b\nc\nd\ne\nDo you want to proceed?"


def test_terminal_tail_is_empty_when_the_page_cannot_answer():
    class Broken:
        def evaluate(self, *_a):
            raise RuntimeError("page closed")

    assert video_probes.terminal_tail(Broken()) == ""


def test_repl_ready_accepts_an_empty_prompt_box_under_a_custom_status_line():
    idle = "─────────────\n❯ \n─────────────\n  Opus | 0% 0k/200K | build\n  ⏸ manual mode on"
    page = FakePage(texts=[idle])
    await_repl_ready(page)
    assert page.keyboard.presses == []


def test_repl_ready_does_not_mistake_the_trust_menu_for_the_prompt():
    trust = "Security guide\n❯ No, exit\n  Yes, I trust this folder"
    page = FakePage(texts=[trust])
    with pytest.raises(ProbeFailed):
        await_repl_ready(page, budget_s=1)


def test_repl_ready_accepts_the_idle_prompt_hint_under_a_custom_status_line():
    # A deployment's status line replaces Claude Code's "? for shortcuts" footer;
    # the empty prompt's placeholder hint still marks the idle REPL.
    rule = "─" * 20
    idle = f'{rule}\n❯ Try "how does <filepath> work?"\n{rule}\n  Opus | 0% 0k/200K | build'
    page = FakePage(texts=[idle])
    await_repl_ready(page)
    assert page.keyboard.presses == []


def test_terminal_focused_true_and_false():
    assert terminal_focused(FakePage(focused=True)) is True
    assert terminal_focused(FakePage(focused=False)) is False


def test_terminal_focused_checks_helper_textarea():
    page = FakePage(focused=True)
    terminal_focused(page)
    script, _ = page.evaluated[0]
    assert "document.activeElement" in script
    assert "xterm-helper-textarea" in script


def test_xterm_text_reads_rows():
    page = FakePage(texts=["line one\nline two"])
    assert xterm_text(page) == "line one\nline two"
    assert ".xterm-rows" in page.evaluated[0][0]


def test_xterm_text_none_is_empty_string():
    assert xterm_text(FakePage(texts=[None])) == ""  # type: ignore[list-item]


def test_repl_ready_immediately_presses_nothing():
    page = FakePage(texts=["> \n? for shortcuts"])
    await_repl_ready(page)
    assert page.keyboard.presses == []
    assert page.waits == []


def test_repl_ready_late_trust_dialog_presses_enter_exactly_once():
    trust = "Quick safety check\n❯ 1. Yes, I trust this folder"
    page = FakePage(texts=["", "", trust, trust, trust, "? for shortcuts"])
    await_repl_ready(page)
    assert page.keyboard.presses == ["Enter"]
    assert page.text_reads == 6
    assert set(page.waits) == {video_probes.POLL_MS}


def test_repl_ready_moves_the_cursor_onto_yes_before_confirming():
    # A Claude Code that lists "No, exit" first puts the cursor on it; Enter
    # there would decline and end the session.
    on_no = "Security guide\n❯ No, exit\n  Yes, I trust this folder\nEnter to confirm"
    on_yes = "Security guide\n  No, exit\n❯ Yes, I trust this folder\nEnter to confirm"
    page = FakePage(texts=["", on_no, on_yes, "", "? for shortcuts"])
    await_repl_ready(page)
    assert page.keyboard.presses == ["ArrowDown", "Enter"]


def test_repl_ready_moves_the_cursor_up_when_yes_is_listed_first():
    on_no = "  Yes, I trust this folder\n❯ No, exit"
    on_yes = "❯ Yes, I trust this folder\n  No, exit"
    page = FakePage(texts=[on_no, on_yes, "? for shortcuts"])
    await_repl_ready(page)
    assert page.keyboard.presses == ["ArrowUp", "Enter"]


def test_repl_ready_trust_dialog_without_repl_times_out():
    page = FakePage(texts=["Yes, I trust this folder"])
    with pytest.raises(ProbeFailed) as exc:
        await_repl_ready(page, budget_s=1)
    assert exc.value.step == "repl-ready"
    assert page.keyboard.presses == ["Enter"]


def test_repl_ready_polls_with_wait_for_timeout_within_budget():
    page = FakePage(texts=["booting"])
    with pytest.raises(ProbeFailed):
        await_repl_ready(page, budget_s=2)
    assert page.waits == [200] * 10
    assert page.text_reads == 10


def test_repl_ready_timeout_reports_last_five_lines():
    text = "\n".join(f"line {i}" for i in range(1, 9)) + "\n\n   \n"
    page = FakePage(texts=[text])
    with pytest.raises(ProbeFailed) as exc:
        await_repl_ready(page, budget_s=0.2)
    err = exc.value
    assert err.step == "repl-ready"
    assert "line 8" in err.detail and "line 4" in err.detail
    assert "line 3" not in err.detail
    assert "repl-ready" in str(err)
    assert page.keyboard.presses == []


def test_probe_modules_use_no_sleep():
    import inspect

    assert "time.sleep" not in inspect.getsource(video_probes)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _eye_frame(x: float, y: float) -> FakePage:
    return FakePage(values={"camera.eye": {"x": x, "y": y, "z": 0.5}})


def test_plot_frame_follows_iframe_chain():
    plot = FakePage()
    panel = FakePage(children={video_probes.PLOT_IFRAME: plot})
    page = FakePage(children={video_probes.ARTIFACTS_IFRAME: panel})
    assert plot_frame(page) is plot
    assert video_probes.ARTIFACTS_IFRAME == 'iframe.panel-iframe[data-panel-id="artifacts"]'
    assert video_probes.PLOT_IFRAME == "iframe.preview-iframe-light"


def test_plot_frame_missing_panel_fails_rotate():
    with pytest.raises(ProbeFailed) as exc:
        plot_frame(FakePage())
    assert exc.value.step == "rotate"


def test_plot_frame_missing_preview_fails_rotate():
    page = FakePage(children={video_probes.ARTIFACTS_IFRAME: FakePage()})
    with pytest.raises(ProbeFailed) as exc:
        plot_frame(page)
    assert exc.value.step == "rotate"


def test_camera_azimuth_is_atan2_y_x_in_degrees():
    assert camera_azimuth(_eye_frame(1.25, 1.25)) == pytest.approx(45.0)
    assert camera_azimuth(_eye_frame(-1.0, 0.0)) == pytest.approx(180.0)
    assert camera_azimuth(_eye_frame(0.0, -2.0)) == pytest.approx(-90.0)
    assert camera_azimuth(_eye_frame(1.0, 2.0)) == pytest.approx(math.degrees(math.atan2(2, 1)))


def test_camera_azimuth_reads_full_layout_scene_camera():
    frame = _eye_frame(1.0, 0.0)
    camera_azimuth(frame)
    assert "_fullLayout" in frame.evaluated[0][0]
    assert "scene" in frame.evaluated[0][0]


def test_camera_azimuth_prefers_the_live_scene_camera():
    frame = _eye_frame(1.0, 0.0)
    camera_azimuth(frame)
    script = frame.evaluated[0][0]
    assert script.index("getCamera") < script.index("s.camera.eye")


def test_camera_azimuth_without_scene_fails_rotate():
    frame = FakePage(values={"camera.eye": None})
    with pytest.raises(ProbeFailed) as exc:
        camera_azimuth(frame)
    assert exc.value.step == "rotate"


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [(0, 90, 90), (170, -170, 20), (-45, 135, 180), (10, 10, 0), (-90, 45, 135)],
)
def test_azimuth_change_wraps(before, after, expected):
    assert azimuth_change(before, after) == pytest.approx(expected)


def test_relayout_waiter_arms_once_and_awaits_with_timeout():
    frame = FakePage(values={"__videoRelayout = new": True, "Promise.race": True})
    assert arm_relayout(frame) is True
    assert "plotly_relayout" in frame.evaluated[0][0]
    assert await_relayout(frame) is True
    script, arg = frame.evaluated[1]
    assert arg == video_probes.RELAYOUT_TIMEOUT_MS == 2000
    assert "window.__videoRelayout = null" in script


def test_relayout_waiter_reports_timeout():
    frame = FakePage(values={"Promise.race": False})
    assert await_relayout(frame, timeout_ms=50) is False
    assert frame.evaluated[0][1] == 50


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------


def _draft_page(state: object) -> FakePage:
    ariel = FakePage(values={"#draft-banner[data-draft-id]": state})
    return FakePage(children={video_probes.ARIEL_IFRAME: ariel})


def test_draft_ready_when_banner_image_and_contrast():
    assert draft_ready(_draft_page({"banner": True, "image": True, "greyLevels": 2}))


@pytest.mark.parametrize(
    "state",
    [
        {"banner": False, "image": True, "greyLevels": 2},
        {"banner": True, "image": False, "greyLevels": 0},
        {"banner": True, "image": True, "greyLevels": 1},
        {"banner": True, "image": True, "greyLevels": 0},
        None,
    ],
)
def test_draft_not_ready(state):
    assert not draft_ready(_draft_page(state))


def test_draft_ready_without_ariel_panel_is_false():
    assert draft_ready(FakePage()) is False


def test_draft_script_checks_image_and_canvas():
    page = _draft_page({"banner": True, "image": True, "greyLevels": 3})
    draft_ready(page)
    ariel = page.children[video_probes.ARIEL_IFRAME]
    script = ariel.evaluated[0][0]  # type: ignore[attr-defined]
    assert "#file-preview img" in script
    assert "naturalWidth > 0" in script
    assert "getImageData" in script
