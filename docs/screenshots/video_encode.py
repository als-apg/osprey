"""Encoding helpers for the landing-page demo video.

The recorder captures some steps in real time and others (long agent turns) as
fast-forward footage. :func:`timeline_speed` decides how much the fast-forward
footage is sped up so the finished video lands near :data:`TARGET_S` (every
second outside the fast-forward steps, gaps included, plays in real time), and
:func:`build_ffconcat` turns the captured frames into an ffconcat list whose
per-frame durations apply that speed-up; with an ``annotate`` callback it also
splits the list at every real-clock second, so a burned-in clock is exact on
every output frame. :func:`encode_to_budget` encodes the list within the size
budget, :func:`poster_command` extracts the poster (the last frame of the
rotate beat, :func:`poster_frame`), and :func:`probe` reads the finished file
back with ffprobe. Every external process is an argument list run without a
shell.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docs.screenshots import capture

if TYPE_CHECKING:
    from docs.screenshots.video_timeline import Timeline

MIN_SPEED = 1.0
MAX_SPEED = 16.0

# The finished MP4 must stay under 8 MB (decimal); 5 % is left for the container
# and for the rate control overshooting its target.
SIZE_BUDGET_BYTES = 8_000_000
SIZE_HEADROOM = 0.95
OUTPUT_FPS = 25
VIDEO_FILTER = f"fps={OUTPUT_FPS},scale=1920:1080:out_range=tv,format=yuv420p"
# How far before the end the poster extraction starts decoding; long enough to
# contain at least one frame at OUTPUT_FPS.
POSTER_SEEK_S = 1.0


# The length the fast-forward footage is fitted to, inside the 50-80 s window
# the review listing checks.
TARGET_S = 57.5


def speed_factor(ff_total: float, realtime_total: float, target: float = TARGET_S) -> float:
    """Return the playback multiplier for the fast-forward footage.

    The real-time steps play unchanged, so the fast-forward footage must fit
    into what remains of *target*. The result is clamped to ``[1, 16]``: never
    slowed down, and never sped up past the point where it stops being legible.
    When the real-time steps alone already fill the target, the maximum is used.
    """
    budget = target - realtime_total
    if budget <= 0:
        return MAX_SPEED
    return min(max(ff_total / budget, MIN_SPEED), MAX_SPEED)


def _quote(path: Path | str) -> str:
    # ffconcat quoting: a single quote closes the string, is escaped, and reopens it.
    return "'" + str(path).replace("'", "'\\''") + "'"


def timeline_speed(timeline: Timeline) -> float:
    """The fast-forward speed-up the encoder uses for *timeline*.

    Everything outside the fast-forward steps plays in real time, the gaps
    between steps included, so the fast-forward footage fits into what they
    leave of the target length.
    """
    spans = [s for s in timeline.steps if s.start is not None and s.end is not None]
    if not spans:
        raise ValueError("timeline has no completed steps")
    recorded = max(s.end for s in spans) - min(s.start for s in spans)
    return speed_factor(timeline.ff_total, recorded - timeline.ff_total)


def _playback(
    start: float, end: float, ff_spans: Sequence[tuple[float, float]], speed: float
) -> float:
    """Playback seconds for the recorded interval ``[start, end]``.

    Overlap with a fast-forward span plays at *speed*; everything else, real-time
    steps and gaps between steps alike, plays at 1.
    """
    length = end - start
    for span_start, span_end in ff_spans:
        overlap = min(end, span_end) - max(start, span_start)
        if overlap > 0:
            length -= overlap - overlap / speed
    return length


# The step whose start the burned-in clock counts from: the first prompt.
CLOCK_ORIGIN_STEP = "prompt-1"

# Labels a frame for its place in the video: the whole real seconds since the
# first prompt, and the speed-up while the footage is sped up (else None).
Annotate = Callable[[Path, int, "float | None"], Path]


def real_session(timeline: Timeline) -> float:
    """Seconds of real session the video covers, from the first prompt to the end."""
    spans = [s for s in timeline.steps if s.start is not None and s.end is not None]
    if not spans:
        raise ValueError("timeline has no completed steps")
    t0 = min(s.start for s in spans)
    origin = next((s.start for s in spans if s.name == CLOCK_ORIGIN_STEP), t0)
    return max(s.end for s in spans) - origin


def _clock_cuts(start: float, end: float, origin: float, edges: Sequence[float]) -> list[float]:
    """Where the interval ``[start, end]`` must be split: clock ticks and span edges."""
    first_tick = math.floor(start - origin) + 1
    ticks = [origin + k for k in range(max(first_tick, 1), math.ceil(end - origin))]
    inner = sorted({t for t in [*ticks, *edges] if start < t < end})
    return [start, *inner, end]


def build_ffconcat(
    frames: Sequence[tuple[Path, float]],
    timeline: Timeline,
    annotate: Annotate | None = None,
) -> tuple[str, float]:
    """Return the ffconcat text for *frames* and the total output length in seconds.

    *frames* are ``(path, epoch_time)`` pairs in capture order. The video starts at
    the first step's start: the last frame captured before it is re-stamped to that
    instant (it is what the screen showed) and anything earlier is dropped. Frames
    captured after the last step ends are dropped too. Each frame lasts until the
    next one, and the final frame until the last step ends, so a still screen holds
    for as long as it was recorded. The file is repeated at the end because the
    concat demuxer ignores the duration of the last entry.

    The returned total is the sum of the durations as written, which is the exact
    length ffmpeg produces from the list.

    With ``annotate``, each frame's interval is also split at every whole second
    of real time since the first prompt and at the fast-forward edges, and each
    piece is written as the frame ``annotate`` returns for its clock value and
    speed-up, so a burned-in clock is exact on every output frame.
    """
    spans = [s for s in timeline.steps if s.start is not None and s.end is not None]
    if not spans:
        raise ValueError("timeline has no completed steps")
    t0 = min(s.start for s in spans)
    t_end = max(s.end for s in spans)
    speed = timeline_speed(timeline)
    ff_spans = [(s.start, s.end) for s in spans if s.fast_forward]

    ordered = sorted(frames, key=lambda f: f[1])
    before = [f for f in ordered if f[1] < t0]
    kept = [f for f in ordered if t0 <= f[1] <= t_end]
    if before and (not kept or kept[0][1] > t0):
        kept.insert(0, (before[-1][0], t0))
    if not kept:
        raise ValueError("no frames fall inside the recorded steps")

    origin = next((s.start for s in spans if s.name == CLOCK_ORIGIN_STEP), t0)
    edges = sorted({t for span in ff_spans for t in span})

    lines = ["ffconcat version 1.0"]
    total = 0.0
    exact = 0.0
    last = kept[-1][0]
    bounds = [t for _, t in kept[1:]] + [t_end]
    for (path, start), end in zip(kept, bounds, strict=True):
        if annotate is None:
            pieces = [(start, end)]
        else:
            cuts = _clock_cuts(start, end, origin, edges)
            pieces = list(zip(cuts, cuts[1:], strict=False)) or [(start, end)]
        for a, b in pieces:
            length = _playback(a, b, ff_spans, speed)
            shown = path
            if annotate is None:
                duration = round(length, 6)
            else:
                # The still-image demuxer stamps every piece on the output
                # frame grid; a piece that starts between two frames first
                # shows on the later one, so each output frame shows the piece
                # live at its own instant. Pieces no frame samples drop out.
                start_out = math.ceil(exact * OUTPUT_FPS - 1e-6) / OUTPUT_FPS
                exact += length
                end_out = math.ceil(exact * OUTPUT_FPS - 1e-6) / OUTPUT_FPS
                duration = round(end_out - start_out, 6)
                if duration <= 0:
                    continue
                mid = (a + b) / 2
                fast = any(x <= mid < y for x, y in ff_spans)
                clock = max(0, math.floor(a - origin + 1e-9))
                shown = annotate(path, clock, speed if fast else None)
            total += duration
            last = shown
            lines.append(f"file {_quote(shown)}")
            lines.append(f"duration {duration:.6f}")
    lines.append(f"file {_quote(last)}")
    return "\n".join(lines) + "\n", total


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise capture.ScreenshotSkip(f"{name} not found on PATH; the demo video needs it")
    return path


def encode_commands(
    concat_path: Path, out_mp4: Path, duration: float, bitrate_scale: float = 1.0
) -> list[list[str]]:
    """Return the two-pass H.264 command lines that encode *concat_path* to *out_mp4*.

    *duration* is the total returned by :func:`build_ffconcat`. It sets the
    bitrate that keeps the file under the size budget, and both passes cut the
    output to it with ``-t``: the ``fps`` filter otherwise repeats the final
    frame for the length of the closing hold a second time. The pass-log files
    are written next to *concat_path*, in the scratch dir with the frames.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    ffmpeg = _binary("ffmpeg")
    out_mp4 = Path(out_mp4)
    bitrate = int(SIZE_BUDGET_BYTES * 8 * SIZE_HEADROOM / duration * bitrate_scale)
    passlog = str(Path(concat_path).parent / f"{out_mp4.stem}-2pass")

    def common(pass_no: int) -> list[str]:
        return [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-t", f"{duration:.3f}",
            "-vf", VIDEO_FILTER,
            "-c:v", "libx264", "-b:v", str(bitrate),
            "-pass", str(pass_no), "-passlogfile", passlog,
        ]  # fmt: skip

    first = [*common(1), "-an", "-f", "null", os.devnull]
    second = [*common(2), "-movflags", "+faststart", str(out_mp4)]
    return [first, second]


# Encodes tried before a file that will not fit the size budget is an error.
BUDGET_ATTEMPTS = 3


def encode_to_budget(
    concat_path: Path,
    out_mp4: Path,
    duration: float,
    run: Callable[..., Any] = subprocess.run,
    size_of: Callable[[Path], int] = lambda path: Path(path).stat().st_size,
) -> int:
    """Encode *concat_path* to *out_mp4* within :data:`SIZE_BUDGET_BYTES`; return its size.

    Two-pass rate control can land past its target. An encode over the budget
    is redone at a bitrate lowered in proportion to the overshoot; after
    :data:`BUDGET_ATTEMPTS` encodes that still do not fit, it raises
    :class:`RuntimeError`.
    """
    scale = 1.0
    size = 0
    for _ in range(BUDGET_ATTEMPTS):
        for cmd in encode_commands(concat_path, out_mp4, duration, scale):
            run(cmd, check=True)
        size = size_of(out_mp4)
        if size <= SIZE_BUDGET_BYTES:
            return size
        scale *= SIZE_BUDGET_BYTES * SIZE_HEADROOM / size
    raise RuntimeError(
        f"{Path(out_mp4).name} is {size} bytes after {BUDGET_ATTEMPTS} encodes; "
        f"the budget is {SIZE_BUDGET_BYTES}"
    )


# The beat whose last frame is the poster: the 3D plot on screen, after the
# agent has worked.
POSTER_STEP = "rotate"


def poster_frame(timeline: Timeline) -> int | None:
    """The output frame index of the poster: the last frame of the rotate beat.

    None when the timeline has no such beat; the poster is then the last frame.
    """
    spans = [s for s in timeline.steps if s.start is not None and s.end is not None]
    step = next((s for s in spans if s.name == POSTER_STEP), None)
    if step is None:
        return None
    t0 = min(s.start for s in spans)
    ff_spans = [(s.start, s.end) for s in spans if s.fast_forward]
    end_out = _playback(t0, step.end, ff_spans, timeline_speed(timeline))
    return max(0, math.ceil(end_out * OUTPUT_FPS - 1e-6) - 1)


def poster_command(out_mp4: Path, poster_jpg: Path, frame: int | None = None) -> list[str]:
    """Return the ffmpeg command line that writes the poster of *out_mp4*.

    With *frame*, that output frame (see :func:`poster_frame`). Without it, the
    last frame: decoding starts :data:`POSTER_SEEK_S` before the end and every
    frame overwrites the same image, so the file left behind is the final one.
    """
    if frame is not None:
        return [
            _binary("ffmpeg"), "-y", "-loglevel", "error", "-i", str(out_mp4),
            "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", "-q:v", "2", str(poster_jpg),
        ]  # fmt: skip
    return [
        _binary("ffmpeg"), "-y", "-loglevel", "error",
        "-sseof", f"-{POSTER_SEEK_S:g}", "-i", str(out_mp4),
        "-update", "1", "-q:v", "2", str(poster_jpg),
    ]  # fmt: skip


def probe(out_mp4: Path) -> dict[str, Any]:
    """Return width, height, fps, pix_fmt, color_range, duration and size of *out_mp4*.

    ``color_range`` is ``None`` when ffprobe does not report one. A failing
    ffprobe raises :class:`RuntimeError` carrying its last stderr line.
    """
    cmd = [
        _binary("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,pix_fmt,color_range:format=duration,size",
        "-of", "json", str(out_mp4),
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        detail = lines[-1] if lines else f"exit {result.returncode}"
        raise RuntimeError(f"ffprobe failed on {out_mp4}: {detail}")
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    fmt = data["format"]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(stream["r_frame_rate"])),
        "pix_fmt": stream["pix_fmt"],
        "color_range": stream.get("color_range"),
        "duration": float(fmt["duration"]),
        "size": int(fmt["size"]),
    }
