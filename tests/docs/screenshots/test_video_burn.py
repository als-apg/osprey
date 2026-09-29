"""Unit tests for the burned-in real-time clock and speed badge.

CI-safe: images are small synthetic JPEGs in ``tmp_path``; no browser, no ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docs.screenshots.video_burn import (
    BADGE_BOX,
    CLOCK_BOX,
    FRAME_SIZE,
    Burner,
    clock_label,
    speed_label,
)
from PIL import Image, ImageChops


@pytest.mark.parametrize(
    ("seconds", "label"),
    [
        (0.0, "real time 0:00"),
        (59.99, "real time 0:59"),
        (167.9, "real time 2:47"),
        (3725.0, "real time 1:02:05"),
    ],
)
def test_the_clock_reads_whole_elapsed_seconds(seconds, label) -> None:
    assert clock_label(seconds) == label


@pytest.mark.parametrize(
    ("speed", "label"), [(16.0, "16×"), (6.0, "6×"), (5.73, "5.7×"), (1.0, "1×")]
)
def test_the_badge_names_the_factor_the_video_used(speed, label) -> None:
    assert speed_label(speed) == label


def _frame(tmp_path: Path, name: str = "f.jpg", shade: int = 128) -> Path:
    path = tmp_path / name
    Image.new("RGB", FRAME_SIZE, (shade, shade, shade)).save(path, quality=95)
    return path


def _changed(a: Path, b: Path, box: tuple[int, int, int, int]) -> bool:
    with Image.open(a) as ia, Image.open(b) as ib:
        diff = ImageChops.difference(ia.convert("RGB").crop(box), ib.convert("RGB").crop(box))
        return diff.getbbox() is not None and max(diff.getextrema()[0][1:]) > 24


def test_the_clock_is_drawn_bottom_right_and_nothing_else_without_speed(tmp_path) -> None:
    src = _frame(tmp_path)
    out = Burner(tmp_path / "burned")(src, 167, None)
    assert out != src and out.is_file()
    assert _changed(src, out, CLOCK_BOX)
    assert not _changed(src, out, BADGE_BOX)
    # The terminal's text area above the footer strip is untouched.
    assert not _changed(src, out, (1100, 100, 1900, 1040))


def test_the_badge_is_drawn_top_right_while_sped_up(tmp_path) -> None:
    src = _frame(tmp_path)
    out = Burner(tmp_path / "burned")(src, 12, 16.0)
    assert _changed(src, out, BADGE_BOX)
    assert _changed(src, out, CLOCK_BOX)


def test_the_boxes_sit_in_the_corners_clear_of_the_panels() -> None:
    width, height = FRAME_SIZE
    # Clock: inside the bottom footer strip, at the right edge.
    assert CLOCK_BOX[1] >= height - 34 and CLOCK_BOX[3] <= height
    assert CLOCK_BOX[2] <= width and CLOCK_BOX[0] > width - 360
    # Badge: in the top bar, at the right edge.
    assert BADGE_BOX[3] <= 44 and BADGE_BOX[2] <= width


def test_different_clock_values_draw_different_pixels(tmp_path) -> None:
    src = _frame(tmp_path)
    burner = Burner(tmp_path / "burned")
    assert _changed(burner(src, 1, None), burner(src, 2, None), CLOCK_BOX)


def test_each_frame_and_label_is_rendered_once(tmp_path) -> None:
    src = _frame(tmp_path)
    burner = Burner(tmp_path / "burned")
    first = burner(src, 5, 16.0)
    stamp = first.stat().st_mtime_ns
    assert burner(src, 5, 16.0) == first
    assert first.stat().st_mtime_ns == stamp
    assert burner(src, 5, None) != first


@pytest.mark.parametrize("shade", [8, 245])
def test_the_labels_read_on_dark_and_light_screens(tmp_path, shade) -> None:
    src = _frame(tmp_path, f"s{shade}.jpg", shade)
    out = Burner(tmp_path / "burned")(src, 75, 16.0)
    with Image.open(out) as img:
        clock = img.convert("L").crop(CLOCK_BOX)
        lo, hi = clock.getextrema()
    # White text on a dark box, whatever the screen behind it.
    assert hi > 200 and lo < 60
