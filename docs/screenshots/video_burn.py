"""Burn the real elapsed time and the true speed-up into the demo video's frames.

The video plays the agent's working time sped up, so on its own it makes the
agent look faster than it is. Two labels keep it honest:

* a clock in the bottom-right corner, shown on every frame, reading the real
  session time since the first prompt; while the footage is sped up it races;
* a badge in the top-right corner, shown only while sped up, naming the factor
  the encoder used for this video (``⏩ 16×``).

Both are drawn at encode time, because only the encoder knows the factor. The
encoder hands each piece of its frame list to :class:`Burner` with the clock
value and, while sped up, the factor; the burner writes a labelled copy of the
frame once per distinct label and returns its path.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FRAME_SIZE = (1920, 1080)

# Label geometry, in frame pixels. The clock sits in the footer strip under the
# panels (over the UI's own wall clock, so the frame shows one clock); the badge
# sits in the top bar at the right edge.
CLOCK_BOX = (1656, 1051, 1914, 1077)
BADGE_BOX = (1788, 6, 1914, 38)
FONT_SIZE = 17

BOX_FILL = (17, 17, 17, 255)
TEXT_FILL = (255, 255, 255, 255)
JPEG_QUALITY = 92

# Monospaced first, so the clock's digits keep their place as they change.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


def clock_label(seconds: float) -> str:
    """``real time m:ss`` (``h:mm:ss`` past an hour) for whole elapsed seconds."""
    total = max(0, math.floor(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"real time {hours}:{minutes:02d}:{secs:02d}"
    return f"real time {minutes}:{secs:02d}"


def speed_label(speed: float) -> str:
    """The factor as the badge shows it: ``16×``, or ``5.7×`` when not whole."""
    if abs(speed - round(speed)) < 0.05:
        return f"{round(speed)}×"
    return f"{speed:.1f}×"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def _pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    radius = (box[3] - box[1]) // 2
    draw.rounded_rectangle(box, radius=radius, fill=BOX_FILL)


def _centred_text(draw, box, text, font, left: int | None = None) -> None:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    x = left if left is not None else box[0] + (box[2] - box[0] - (x1 - x0)) // 2 - x0
    y = box[1] + (box[3] - box[1] - (y1 - y0)) // 2 - y0
    draw.text((x, y), text, font=font, fill=TEXT_FILL)


# The ⏩ glyph, drawn as two triangles: its height and total width in pixels.
GLYPH_HEIGHT = 14
GLYPH_WIDTH = 2 * int(GLYPH_HEIGHT * 0.62)


def _fast_forward_glyph(draw: ImageDraw.ImageDraw, x: int, y_mid: int) -> int:
    """Draw ⏩ as two triangles; return the x just past it."""
    w, h = GLYPH_WIDTH // 2, GLYPH_HEIGHT
    for i in range(2):
        x0 = x + i * w
        draw.polygon([(x0, y_mid - h // 2), (x0 + w, y_mid), (x0, y_mid + h // 2)], fill=TEXT_FILL)
    return x + 2 * w


class Burner:
    """Writes labelled copies of frames into ``out_dir``, one per distinct label."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.font = _font(FONT_SIZE)
        self._done: dict[tuple[Path, int, float | None], Path] = {}

    def __call__(self, frame: Path, clock_s: int, speed: float | None) -> Path:
        key = (Path(frame), int(clock_s), speed)
        if key in self._done:
            return self._done[key]
        badge = speed_label(speed) if speed is not None else None
        name = f"{Path(frame).stem}_t{int(clock_s):05d}{'_x' + badge[:-1] if badge else ''}.jpg"
        out = self.out_dir / name
        with Image.open(frame) as src:
            img = src.convert("RGB")
        if img.size != FRAME_SIZE:
            img = img.resize(FRAME_SIZE)
        layer = Image.new("RGBA", FRAME_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        _pill(draw, CLOCK_BOX)
        _centred_text(draw, CLOCK_BOX, clock_label(clock_s), self.font)
        if badge is not None:
            _pill(draw, BADGE_BOX)
            text_w = draw.textlength(badge, font=self.font)
            gap = 7
            start = BADGE_BOX[0] + int(
                (BADGE_BOX[2] - BADGE_BOX[0] - GLYPH_WIDTH - gap - text_w) / 2
            )
            y_mid = (BADGE_BOX[1] + BADGE_BOX[3]) // 2
            after = _fast_forward_glyph(draw, start, y_mid)
            _centred_text(draw, BADGE_BOX, badge, self.font, left=after + gap)
        Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB").save(
            out, quality=JPEG_QUALITY
        )
        self._done[key] = out
        return out
