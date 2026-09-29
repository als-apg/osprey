"""Unit tests for the demo-video encoder helpers.

CI-safe: no ffmpeg, no browser. Binary lookups and ``subprocess.run`` are
monkeypatched, so only the command lines and the parsing are exercised.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from docs.screenshots import capture, video_encode
from docs.screenshots.video_encode import (
    SIZE_BUDGET_BYTES,
    TARGET_S,
    build_ffconcat,
    encode_commands,
    encode_to_budget,
    poster_command,
    probe,
    speed_factor,
)
from docs.screenshots.video_timeline import Step, Timeline


def test_the_target_length_lies_inside_the_review_window() -> None:
    from docs.screenshots import __main__ as cli

    assert cli.VIDEO_MIN_S < TARGET_S < cli.VIDEO_MAX_S


def test_speed_factor_clamps_below_one_to_one() -> None:
    # 10 s of fast-forward footage into 47.5 s of spare budget would slow it down.
    assert speed_factor(10.0, 10.0) == 1.0


def test_speed_factor_inside_range_is_exact_ratio() -> None:
    # 190 s of fast-forward footage must fit into 57.5 - 10 = 47.5 s.
    assert speed_factor(190.0, 10.0) == pytest.approx(4.0)


def test_speed_factor_clamps_above_sixteen_to_sixteen() -> None:
    assert speed_factor(1000.0, 5.0) == 16.0


@pytest.mark.parametrize("realtime_total", [57.5, 70.0])
def test_speed_factor_non_positive_denominator_returns_sixteen(realtime_total: float) -> None:
    assert speed_factor(30.0, realtime_total) == 16.0


def test_speed_factor_honours_custom_target() -> None:
    assert speed_factor(40.0, 10.0, target=20.0) == pytest.approx(4.0)


# -- build_ffconcat ----------------------------------------------------------


def _timeline(*spans: tuple[float, float, bool]) -> Timeline:
    steps = [
        Step(name=f"s{i}", fast_forward=ff, start=start, end=end)
        for i, (start, end, ff) in enumerate(spans)
    ]
    return Timeline(theme="dark", version="x", recorded_at=0.0, steps=steps)


def _frames(*times: float) -> list[tuple[Path, float]]:
    return [(Path(f"/frames/frame_{i:06d}.jpg"), t) for i, t in enumerate(times)]


def _entries(text: str) -> list[tuple[str, float | None]]:
    """Parse ffconcat text into (file, duration-or-None) pairs."""
    lines = text.strip().splitlines()
    assert lines[0] == "ffconcat version 1.0"
    entries: list[tuple[str, float | None]] = []
    for line in lines[1:]:
        if line.startswith("file "):
            entries.append((line[len("file ") :].strip("'"), None))
        elif line.startswith("duration "):
            name, _ = entries[-1]
            entries[-1] = (name, float(line[len("duration ") :]))
        else:
            raise AssertionError(f"unexpected line {line!r}")
    return entries


def test_ffconcat_idle_gap_holds_frame_in_real_time() -> None:
    # A real-time step with a 5 s idle stretch where the screen did not change.
    timeline = _timeline((100.0, 110.0, False))
    text, total = build_ffconcat(_frames(100.0, 101.0, 106.0), timeline)
    entries = _entries(text)
    assert [d for _, d in entries[:-1]] == pytest.approx([1.0, 5.0, 4.0])
    assert total == pytest.approx(10.0)


def test_ffconcat_idle_gap_between_steps_plays_in_real_time() -> None:
    # Gap between two real-time steps is outside any span: speed 1.
    timeline = _timeline((0.0, 2.0, False), (5.0, 6.0, False))
    text, total = build_ffconcat(_frames(0.0, 1.0), timeline)
    assert [d for _, d in _entries(text)[:-1]] == pytest.approx([1.0, 5.0])
    assert total == pytest.approx(6.0)


def test_ffconcat_restamps_last_frame_before_t0() -> None:
    timeline = _timeline((10.0, 14.0, False))
    frames = _frames(2.0, 7.0, 12.0)
    text, total = build_ffconcat(frames, timeline)
    entries = _entries(text)
    names = [n for n, _ in entries]
    # frame 0 dropped; frame 1 kept and re-stamped to t0 = 10.
    assert names == [str(frames[1][0]), str(frames[2][0]), str(frames[2][0])]
    assert [d for _, d in entries[:-1]] == pytest.approx([2.0, 2.0])
    assert total == pytest.approx(4.0)


def test_ffconcat_frame_exactly_at_t0_drops_earlier_frames() -> None:
    timeline = _timeline((10.0, 12.0, False))
    frames = _frames(5.0, 10.0)
    text, total = build_ffconcat(frames, timeline)
    names = [n for n, _ in _entries(text)]
    assert names == [str(frames[1][0]), str(frames[1][0])]
    assert total == pytest.approx(2.0)


def test_ffconcat_closing_hold_after_fast_forward_wait() -> None:
    # 1000 s fast-forward wait, then a 3 s real-time hold where the screen
    # changes once at 1001 and then stays still. speed = 1000 / 54.5 clamps to 16.
    timeline = _timeline((0.0, 1000.0, True), (1000.0, 1003.0, False))
    speed = 16.0
    frames = _frames(0.0, 500.0, 1001.0)
    text, total = build_ffconcat(frames, timeline)
    entries = _entries(text)
    assert [d for _, d in entries[:-1]] == pytest.approx([500.0 / speed, 500.0 / speed + 1.0, 2.0])
    # The last file is repeated with no duration.
    assert entries[-1] == (str(frames[-1][0]), None)
    assert total == pytest.approx(1000.0 / speed + 3.0)


def test_ffconcat_still_frame_crossing_fast_forward_to_real_time() -> None:
    # One still frame from mid-wait through the whole hold: the wait part is
    # sped up, the hold part plays in real time.
    timeline = _timeline((0.0, 160.0, True), (160.0, 164.0, False))
    speed = 160.0 / (TARGET_S - 4.0)
    text, total = build_ffconcat(_frames(0.0, 80.0), timeline)
    durations = [d for _, d in _entries(text)[:-1]]
    assert durations == pytest.approx([80.0 / speed, 80.0 / speed + 4.0], abs=1e-5)
    assert total == pytest.approx(160.0 / speed + 4.0, abs=1e-5)


def test_ffconcat_span_assignment_across_boundary() -> None:
    # Real-time step, then fast-forward step; one frame straddles the boundary.
    timeline = _timeline((0.0, 5.0, False), (5.0, 325.0, True))
    speed = 320.0 / (TARGET_S - 5.0)
    text, total = build_ffconcat(_frames(0.0, 3.0, 45.0), timeline)
    durations = [d for _, d in _entries(text)[:-1]]
    assert durations == pytest.approx([3.0, 2.0 + 40.0 / speed, 280.0 / speed])
    assert total == pytest.approx(5.0 + 320.0 / speed)


def test_ffconcat_total_is_sum_of_written_durations() -> None:
    timeline = _timeline((0.0, 7.3, False), (7.3, 300.1, True), (300.1, 303.0, False))
    text, total = build_ffconcat(_frames(0.0, 0.7, 3.33, 150.0, 301.0), timeline)
    durations = [d for _, d in _entries(text)[:-1]]
    assert total == pytest.approx(sum(durations), abs=1e-9)


def test_ffconcat_drops_frames_after_last_step_end() -> None:
    timeline = _timeline((0.0, 4.0, False))
    frames = _frames(0.0, 2.0, 9.0)
    text, total = build_ffconcat(frames, timeline)
    names = [n for n, _ in _entries(text)]
    assert names == [str(frames[0][0]), str(frames[1][0]), str(frames[1][0])]
    assert total == pytest.approx(4.0)


def test_ffconcat_quotes_paths_with_single_quotes() -> None:
    timeline = _timeline((0.0, 1.0, False))
    text, _ = build_ffconcat([(Path("/tmp/it's/f.jpg"), 0.0)], timeline)
    assert "file '/tmp/it'\\''s/f.jpg'" in text


def test_ffconcat_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        build_ffconcat([], _timeline((0.0, 1.0, False)))
    with pytest.raises(ValueError):
        build_ffconcat(_frames(0.0), _timeline())
    with pytest.raises(ValueError):
        # every frame falls after the last step end
        build_ffconcat(_frames(5.0), _timeline((0.0, 1.0, False)))


# -- encode_commands / poster_command / probe ---------------------------------


@pytest.fixture
def fake_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_encode.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_encode.shutil, "which", lambda name: None)


def _value(cmd: list[str], flag: str) -> str:
    assert cmd.count(flag) == 1, f"{flag} must appear exactly once in {cmd}"
    return cmd[cmd.index(flag) + 1]


@pytest.mark.usefixtures("fake_which")
def test_encode_returns_two_passes(tmp_path: Path) -> None:
    cmds = encode_commands(tmp_path / "list.ffconcat", tmp_path / "out.mp4", 40.0)
    assert len(cmds) == 2
    assert all(isinstance(c, list) and all(isinstance(a, str) for a in c) for c in cmds)
    assert all(c[0] == "/usr/bin/ffmpeg" for c in cmds)
    assert [_value(c, "-pass") for c in cmds] == ["1", "2"]


@pytest.mark.usefixtures("fake_which")
def test_encode_both_passes_pin_duration(tmp_path: Path) -> None:
    # Without -t the fps filter doubles the closing hold, so both passes must cut.
    cmds = encode_commands(tmp_path / "list.ffconcat", tmp_path / "out.mp4", 42.12345)
    assert [_value(c, "-t") for c in cmds] == ["42.123", "42.123"]
    # -t is an output option: it sits after the input.
    for c in cmds:
        assert c.index("-t") > c.index("-i")


@pytest.mark.usefixtures("fake_which")
def test_encode_concat_input_and_filters(tmp_path: Path) -> None:
    concat = tmp_path / "list.ffconcat"
    for c in encode_commands(concat, tmp_path / "out.mp4", 40.0):
        i = c.index("-i")
        assert c[i - 4 : i] == ["-f", "concat", "-safe", "0"]
        assert c[i + 1] == str(concat)
        assert _value(c, "-vf") == "fps=25,scale=1920:1080:out_range=tv,format=yuv420p"
        assert _value(c, "-c:v") == "libx264"


@pytest.mark.usefixtures("fake_which")
def test_encode_bitrate_fits_eight_megabytes(tmp_path: Path) -> None:
    cmds = encode_commands(tmp_path / "list.ffconcat", tmp_path / "out.mp4", 40.0)
    expected = int(8_000_000 * 8 * 0.95 / 40.0)
    assert [_value(c, "-b:v") for c in cmds] == [str(expected)] * 2


@pytest.mark.usefixtures("fake_which")
def test_encode_second_pass_writes_faststart_mp4(tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    first, second = encode_commands(tmp_path / "list.ffconcat", out, 40.0)
    assert second[-1] == str(out)
    assert _value(second, "-movflags") == "+faststart"
    assert "-an" in first
    assert first[-3:] == ["-f", "null", os.devnull]


@pytest.mark.usefixtures("fake_which")
def test_encode_pass_log_beside_the_frames_not_the_video(tmp_path: Path) -> None:
    # The pass logs are scratch: they stay in the take's work dir, which the
    # run deletes, and never land in the folder the videos are published from.
    out = tmp_path / "videos" / "out.mp4"
    concat = tmp_path / "work" / "list.ffconcat"
    cmds = encode_commands(concat, out, 40.0)
    logs = [_value(c, "-passlogfile") for c in cmds]
    assert logs[0] == logs[1]
    assert Path(logs[0]).parent == concat.parent
    assert Path(logs[0]).name.startswith("out-")


@pytest.mark.usefixtures("fake_which")
def test_encode_rejects_non_positive_duration(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        encode_commands(tmp_path / "list.ffconcat", tmp_path / "out.mp4", 0.0)


@pytest.mark.usefixtures("no_ffmpeg")
def test_encode_missing_ffmpeg_skips(tmp_path: Path) -> None:
    with pytest.raises(capture.ScreenshotSkip, match="ffmpeg"):
        encode_commands(tmp_path / "list.ffconcat", tmp_path / "out.mp4", 40.0)


@pytest.mark.usefixtures("fake_which")
def test_poster_extracts_last_frame(tmp_path: Path) -> None:
    out, poster = tmp_path / "out.mp4", tmp_path / "poster.jpg"
    cmd = poster_command(out, poster)
    assert cmd[0] == "/usr/bin/ffmpeg"
    # Seek near the end, then keep overwriting one image: the last frame wins.
    assert cmd.index("-sseof") < cmd.index("-i")
    assert float(_value(cmd, "-sseof")) < 0
    assert _value(cmd, "-i") == str(out)
    assert _value(cmd, "-update") == "1"
    assert "-frames:v" not in cmd
    assert cmd[-1] == str(poster)


def _rotating() -> Timeline:
    steps = [
        Step(name="prompt-1", fast_forward=False, start=0.0, end=10.0),
        Step(name="plot", fast_forward=True, start=10.0, end=110.0),
        Step(name="rotate", fast_forward=False, start=110.0, end=114.0),
        Step(name="close", fast_forward=False, start=114.0, end=117.0),
    ]
    return Timeline(theme="dark", version="x", recorded_at=0.0, steps=steps)


def test_the_poster_is_the_last_frame_of_the_rotate_beat() -> None:
    # The still shown before playback: the 3D plot on screen after the agent
    # has worked, taken where the rotation ends.
    timeline = _rotating()
    speed = video_encode.timeline_speed(timeline)
    rotate_end = 10.0 + 100.0 / speed + 4.0
    expected = math.ceil(rotate_end * video_encode.OUTPUT_FPS - 1e-6) - 1
    assert video_encode.poster_frame(timeline) == expected
    assert expected / video_encode.OUTPUT_FPS < rotate_end


def test_without_a_rotate_beat_the_poster_is_the_last_frame() -> None:
    assert video_encode.poster_frame(_timeline((0.0, 5.0, False))) is None


@pytest.mark.usefixtures("fake_which")
def test_poster_extracts_the_chosen_frame(tmp_path: Path) -> None:
    out, poster = tmp_path / "out.mp4", tmp_path / "poster.jpg"
    cmd = poster_command(out, poster, frame=812)
    assert _value(cmd, "-i") == str(out)
    assert _value(cmd, "-vf") == "select=eq(n\\,812)"
    assert _value(cmd, "-frames:v") == "1"
    assert "-sseof" not in cmd
    assert cmd[-1] == str(poster)


@pytest.mark.usefixtures("no_ffmpeg")
def test_poster_missing_ffmpeg_skips(tmp_path: Path) -> None:
    with pytest.raises(capture.ScreenshotSkip, match="ffmpeg"):
        poster_command(tmp_path / "out.mp4", tmp_path / "poster.jpg")


_PROBE_JSON = {
    "streams": [
        {
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "25/1",
            "pix_fmt": "yuv420p",
            "color_range": "tv",
        }
    ],
    "format": {"duration": "44.960000", "size": "7340032"},
}


@pytest.mark.usefixtures("fake_which")
def test_probe_parses_ffprobe_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_PROBE_JSON), stderr="")

    monkeypatch.setattr(video_encode.subprocess, "run", fake_run)
    out = tmp_path / "out.mp4"
    info = probe(out)
    assert info == {
        "width": 1920,
        "height": 1080,
        "fps": 25.0,
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "duration": pytest.approx(44.96),
        "size": 7340032,
    }
    ((cmd, kwargs),) = calls
    assert isinstance(cmd, list)
    assert cmd[0] == "/usr/bin/ffprobe"
    assert cmd[-1] == str(out)
    assert kwargs.get("shell") is not True


@pytest.mark.usefixtures("fake_which")
def test_probe_fractional_frame_rate_and_missing_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = json.loads(json.dumps(_PROBE_JSON))
    data["streams"][0]["r_frame_rate"] = "30000/1001"
    del data["streams"][0]["color_range"]

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data), stderr="")

    monkeypatch.setattr(video_encode.subprocess, "run", fake_run)
    info = probe(tmp_path / "out.mp4")
    assert info["fps"] == pytest.approx(29.97, abs=1e-2)
    assert info["color_range"] is None


@pytest.mark.usefixtures("fake_which")
def test_probe_failure_raises_with_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="junk\nout.mp4: No such file")

    monkeypatch.setattr(video_encode.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="No such file"):
        probe(tmp_path / "out.mp4")


@pytest.mark.usefixtures("no_ffmpeg")
def test_probe_missing_ffprobe_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called without ffprobe")

    monkeypatch.setattr(video_encode.subprocess, "run", fail_run)
    with pytest.raises(capture.ScreenshotSkip, match="ffprobe"):
        probe(tmp_path / "out.mp4")


# --- the size budget ------------------------------------------------------------


class _Encoder:
    """Records each encode's bitrate and reports the next scripted file size."""

    def __init__(self, sizes: list[int]) -> None:
        self.sizes = sizes
        self.bitrates: list[int] = []
        self.runs = 0

    def run(self, cmd: list[str], check: bool) -> None:
        assert check
        self.runs += 1
        if "-pass" in cmd and _value(cmd, "-pass") == "2":
            self.bitrates.append(int(_value(cmd, "-b:v")))

    def size_of(self, _path: Path) -> int:
        return self.sizes[len(self.bitrates) - 1]


@pytest.mark.usefixtures("fake_which")
def test_an_encode_inside_the_budget_runs_once(tmp_path: Path) -> None:
    enc = _Encoder([7_500_000])
    size = encode_to_budget(tmp_path / "l.ffconcat", tmp_path / "o.mp4", 45.0, enc.run, enc.size_of)
    assert size == 7_500_000
    assert enc.runs == 2 and len(enc.bitrates) == 1


@pytest.mark.usefixtures("fake_which")
def test_an_overshooting_encode_is_redone_at_a_proportionally_lower_bitrate(
    tmp_path: Path,
) -> None:
    # Two-pass rate control can land past its target; the file must still fit.
    enc = _Encoder([8_530_000, 7_700_000])
    size = encode_to_budget(tmp_path / "l.ffconcat", tmp_path / "o.mp4", 45.0, enc.run, enc.size_of)
    assert size == 7_700_000
    first, second = enc.bitrates
    assert second < first
    assert second == pytest.approx(first * SIZE_BUDGET_BYTES * 0.95 / 8_530_000, rel=0.01)


@pytest.mark.usefixtures("fake_which")
def test_an_encode_that_never_fits_fails_loudly(tmp_path: Path) -> None:
    enc = _Encoder([9_000_000, 8_900_000, 8_800_000])
    with pytest.raises(RuntimeError, match="8000000"):
        encode_to_budget(tmp_path / "l.ffconcat", tmp_path / "o.mp4", 45.0, enc.run, enc.size_of)
    assert len(enc.bitrates) == 3


# --- the burned-in real-time clock ---------------------------------------------


def _story() -> Timeline:
    # focus (real time), prompt-1 typed from t=1, a 100 s fast-forward wait,
    # an open beat, a second 60 s wait, then the closing hold.
    steps = [
        Step(name="focus-1", fast_forward=False, start=0.0, end=1.0),
        Step(name="prompt-1", fast_forward=False, start=1.0, end=4.0),
        Step(name="plot", fast_forward=True, start=4.0, end=104.0),
        Step(name="open-artifact", fast_forward=False, start=104.0, end=106.0),
        Step(name="correlate", fast_forward=True, start=106.0, end=166.0),
        Step(name="close", fast_forward=False, start=166.0, end=169.0),
    ]
    return Timeline(theme="dark", version="x", recorded_at=0.0, steps=steps)


class _Recorder:
    """An annotate callback that records what each ffconcat piece is labelled."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, int, float | None]] = []

    def __call__(self, path: Path, clock_s: int, speed: float | None) -> Path:
        self.calls.append((path, clock_s, speed))
        return Path(f"/burned/{len(self.calls):05d}_{clock_s}_{speed}.jpg")


def _labelled(text: str) -> list[tuple[float, int, float | None]]:
    """(output start time, clock, speed) of each written piece, from the file names."""
    out, t = [], 0.0
    for name, duration in _entries(text)[:-1]:
        _, clock, speed = Path(name).stem.split("_")
        out.append((t, int(clock), None if speed == "None" else float(speed)))
        t += duration or 0.0
    return out


def test_burned_pieces_keep_the_video_length() -> None:
    frames = _frames(0.0, 2.0, 50.0, 105.0, 120.0)
    plain, total = build_ffconcat(frames, _story())
    burned, burned_total = build_ffconcat(frames, _story(), annotate=_Recorder())
    # Burned pieces sit on the output frame grid: the length rounds up to a frame.
    assert total <= burned_total < total + 1 / video_encode.OUTPUT_FPS + 1e-6
    assert len(_entries(burned)) > len(_entries(plain))


def test_the_clock_counts_every_real_second_from_the_first_prompt() -> None:
    rec = _Recorder()
    build_ffconcat(_frames(0.0, 50.0, 120.0), _story(), annotate=rec)
    clocks = [c for _, c, _ in rec.calls]
    assert clocks == sorted(clocks)
    # Zero until the prompt, then every whole second of the 168 s after it.
    assert set(clocks) == set(range(0, 168))


def test_the_clock_at_a_step_start_is_that_step_start_minus_the_first_prompt() -> None:
    timeline = _story()
    text, _ = build_ffconcat(_frames(0.0, 50.0, 120.0), timeline, annotate=_Recorder())
    pieces = _labelled(text)
    speed = speed_factor(timeline.ff_total, timeline.realtime_total)
    ff = [(s.start, s.end) for s in timeline.steps if s.fast_forward]
    for step in timeline.steps[2:]:
        at = sum(
            (min(step.start, b) - a) / (speed if any(x <= a < y for x, y in ff) else 1.0)
            for a, b in _cuts(timeline)
            if a < step.start
        )
        # The first output frame at or after the step's start.
        frame_at = math.ceil(at * video_encode.OUTPUT_FPS - 1e-6) / video_encode.OUTPUT_FPS
        shown = next(c for t, c, _ in reversed(pieces) if t <= frame_at + 1e-6)
        assert shown == int(step.start - 1.0), step.name


def _cuts(timeline: Timeline) -> list[tuple[float, float]]:
    edges = sorted({s.start for s in timeline.steps} | {s.end for s in timeline.steps})
    return list(zip(edges, edges[1:], strict=False))


def test_the_speed_is_passed_only_while_sped_up_and_is_the_factor_used() -> None:
    timeline = _story()
    rec = _Recorder()
    build_ffconcat(_frames(0.0, 50.0, 105.0, 120.0), timeline, annotate=rec)
    speed = speed_factor(timeline.ff_total, timeline.realtime_total)
    used = {s for _, _, s in rec.calls}
    assert used == {None, speed}
    # Real-time beats carry no badge: the open beat's seconds 103 and 104.
    assert all(s is None for _, c, s in rec.calls if c in (103, 104))
    assert all(s == speed for _, c, s in rec.calls if 10 <= c <= 100)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize("prompt_at", [0.23, 0.3756, 0.3799])
def test_every_output_frame_shows_the_real_time_of_that_frame(tmp_path: Path, prompt_at) -> None:
    # Each piece of the list is a flat grey whose level encodes its clock value;
    # the real encoder filter picks the frames, and every output frame n must
    # show floor(real time at n / fps - first prompt). At 16x one output frame
    # spans 0.64 s of real time, so a half-frame rounding error shows up here;
    # the still-image demuxer stamps each piece on the output frame grid, so a
    # piece that starts between two frames must not appear on the earlier one.
    import subprocess

    from PIL import Image

    # The first prompt starts off the frame grid, as in a real take, so the
    # clock ticks fall between output frames.
    steps = [
        Step(name="focus-1", fast_forward=False, start=0.0, end=prompt_at),
        Step(name="prompt-1", fast_forward=False, start=prompt_at, end=2.3),
        Step(name="plot", fast_forward=True, start=2.3, end=18.3),
        Step(name="close", fast_forward=False, start=18.3, end=21.0),
    ]
    timeline = Timeline(theme="dark", version="x", recorded_at=0.0, steps=steps)
    source = tmp_path / "src.png"
    Image.new("L", (64, 36), 0).save(source)

    def level(clock: int) -> int:
        return 20 + 10 * clock

    def annotate(_path: Path, clock: int, _speed: float | None) -> Path:
        out = tmp_path / f"c{clock:03d}.png"
        if not out.exists():
            Image.new("L", (64, 36), level(clock)).save(out)
        return out

    text, total = build_ffconcat([(source, 0.0)], timeline, annotate=annotate)
    concat = tmp_path / "list.ffconcat"
    concat.write_text(text)
    fps_filter = video_encode.VIDEO_FILTER.split(",scale=")[0]
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-t", f"{total:.3f}", "-vf", f"{fps_filter},format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    size = 64 * 36
    speed = speed_factor(timeline.ff_total, timeline.realtime_total)
    ff = [(2.3, 18.3)]

    def real_at(t: float) -> float:
        lo, hi = 0.0, 21.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if video_encode._playback(0.0, mid, ff, speed) < t else (lo, mid)
        return lo

    levels = {level(c): c for c in range(0, 25)}
    wrong = []
    for n in range(len(raw) // size):
        value = raw[n * size + size // 2]
        shown = levels[min(levels, key=lambda lv: abs(lv - value))]
        expected = max(0, math.floor(real_at(n / video_encode.OUTPUT_FPS) - prompt_at + 1e-6))
        if shown != expected:
            wrong.append((n, expected, shown))
    assert wrong == []


def test_the_encoded_length_counts_the_gaps_between_steps() -> None:
    # Gaps between steps play in real time too; the speed-up must leave room
    # for them, so an unclamped video lands on the target length exactly.
    timeline = _timeline((0.0, 10.0, False), (15.0, 215.0, True), (215.0, 220.0, False))
    assert video_encode.timeline_speed(timeline) == pytest.approx(200.0 / (TARGET_S - 20.0))
    _, total = build_ffconcat(_frames(0.0, 100.0), timeline)
    assert total == pytest.approx(TARGET_S, abs=1e-3)
