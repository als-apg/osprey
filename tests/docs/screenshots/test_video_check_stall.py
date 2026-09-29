"""Unit tests for the demo-video stall watchdog.

CI-safe: the clock is injected and transcripts are small files in ``tmp_path``,
so nothing waits in real time.
"""

from __future__ import annotations

import pytest
from docs.screenshots.video_check_stall import (
    PLOT_DIALOGS,
    STALL_S,
    StallWatch,
    dialog_on_screen,
    screen_fingerprint,
    transcript_bytes,
)


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_the_spinner_timer_alone_is_not_progress() -> None:
    before = "✻ Pondering… (12s · esc to interrupt)\n❯ "
    after = "✶ Pondering… (1m 57s · esc to interrupt)\n❯ "
    assert screen_fingerprint(before) == screen_fingerprint(after)


def test_streamed_tokens_and_new_text_are_progress() -> None:
    base = "✻ Pondering… (12s · ↓ 300 tokens · esc to interrupt)"
    assert screen_fingerprint(base) != screen_fingerprint(base.replace("300", "1.2k"))
    assert screen_fingerprint("⏺ Read(a.py)") != screen_fingerprint("⏺ Read(b.py)")


@pytest.mark.parametrize(
    "screen",
    [
        "Auto mode is now billed … Enter to continue · Esc to cancel",
        " Do you want to proceed?\n ❯ 1. Yes",
    ],
)
def test_a_blocking_dialog_is_named(screen) -> None:
    assert dialog_on_screen(screen, PLOT_DIALOGS) is not None


def test_ordinary_output_is_no_dialog() -> None:
    assert dialog_on_screen("⏺ Plotted SR01C:BPM1..3 over 24 h\n❯ ", PLOT_DIALOGS) is None


def test_transcript_bytes_counts_the_session_and_its_subagents(tmp_path) -> None:
    transcript = tmp_path / "sess.jsonl"
    transcript.write_text("x" * 10)
    sub = tmp_path / "sess" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-a.jsonl").write_text("y" * 5)
    (sub / "notes.txt").write_text("ignored")
    assert transcript_bytes(transcript) == 15


def test_transcript_bytes_without_a_transcript_is_zero(tmp_path) -> None:
    assert transcript_bytes(None) == 0
    assert transcript_bytes(tmp_path / "missing.jsonl") == 0


def test_a_quiet_session_stalls_after_the_threshold() -> None:
    clock = Clock()
    watch = StallWatch(clock)
    watch.observe("⏺ working", 10)
    clock.t += STALL_S - 1
    assert watch.observe("⏺ working", 10) == pytest.approx(STALL_S - 1)
    assert not watch.stalled
    clock.t += 2
    watch.observe("⏺ working", 10)
    assert watch.stalled


@pytest.mark.parametrize(("screen", "size"), [("⏺ working more", 10), ("⏺ working", 11)])
def test_either_signal_moving_resets_the_watch(screen, size) -> None:
    clock = Clock()
    watch = StallWatch(clock)
    watch.observe("⏺ working", 10)
    clock.t += STALL_S - 1
    watch.observe(screen, size)
    clock.t += STALL_S - 1
    assert watch.observe(screen, size) == pytest.approx(STALL_S - 1)
    assert not watch.stalled
