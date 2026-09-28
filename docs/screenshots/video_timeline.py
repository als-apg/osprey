"""Step timeline for the landing-page demo video.

The recorder runs each beat of the storyboard inside :func:`step`, which stamps the
beat's start and end on the epoch clock. The screencast frame timestamps use the same
clock, so the encoder can tell which frames belong to a fast-forward wait and which to
a real-time beat. The :class:`Timeline` is saved next to the video as
``timeline-{theme}.json``: the encoder places the burned-in clock and the
poster from it, and the demo manifest records the OSPREY version and the
recording time it carries.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

Clock = Callable[[], float]
StepCallback = Callable[["Step"], None]


def _epoch() -> float:
    # Looked up on each call so the clock stays the live ``time.time``.
    return time.time()


def _default_version() -> str:
    from docs.screenshots import capture

    return capture.osprey_version()


@dataclass
class Step:
    """One storyboard beat: its span on the epoch clock and how it plays back."""

    name: str
    caption: str | None = None
    fast_forward: bool = False
    start: float | None = None
    end: float | None = None
    error: str | None = None

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return self.end - self.start


@dataclass
class Timeline:
    """The steps of one take, plus the OSPREY version and when it was recorded."""

    theme: str
    clock: Clock = field(default=_epoch, repr=False, compare=False)
    version: str | None = None
    recorded_at: float | None = None
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.version is None:
            self.version = _default_version()
        if self.recorded_at is None:
            self.recorded_at = self.clock()

    @property
    def osprey_version(self) -> str:
        return self.version or "0+unknown"

    @property
    def ff_total(self) -> float:
        """Recorded seconds spent in fast-forward steps."""
        return sum(s.duration for s in self.steps if s.fast_forward)

    @property
    def realtime_total(self) -> float:
        """Recorded seconds spent in real-time steps."""
        return sum(s.duration for s in self.steps if not s.fast_forward)

    @staticmethod
    def path_for(out_dir: Path, theme: str) -> Path:
        return Path(out_dir) / f"timeline-{theme}.json"

    def to_dict(self) -> dict:
        recorded = self.recorded_at if self.recorded_at is not None else 0.0
        return {
            "theme": self.theme,
            "osprey_version": self.osprey_version,
            "recorded_at": self.recorded_at,
            "recorded_at_iso": datetime.fromtimestamp(recorded, UTC).isoformat(),
            "steps": [asdict(s) for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Timeline:
        return cls(
            theme=data["theme"],
            version=data.get("osprey_version") or "0+unknown",
            recorded_at=data.get("recorded_at"),
            steps=[Step(**s) for s in data.get("steps", [])],
        )

    def save(self, out_dir: Path) -> Path:
        path = self.path_for(out_dir, self.theme)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> Timeline:
        return cls.from_dict(json.loads(Path(path).read_text()))


@contextmanager
def step(
    timeline: Timeline,
    name: str,
    caption: str | None = None,
    fast_forward: bool = False,
    on_enter: StepCallback | None = None,
    on_exit: StepCallback | None = None,
) -> Iterator[Step]:
    """Run one beat and record its span in *timeline*.

    *on_enter* runs before the start is stamped, so the time spent showing the caption
    and badge is not part of the span; the end is stamped before *on_exit*, which
    hides them and flushes the frame sink. The step is recorded even when the body
    raises, with the error attached, and the exception propagates.
    """
    current = Step(name=name, caption=caption, fast_forward=fast_forward)
    if on_enter is not None:
        on_enter(current)
    current.start = timeline.clock()
    try:
        yield current
    except BaseException as exc:
        current.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        current.end = timeline.clock()
        timeline.steps.append(current)
        if on_exit is not None:
            on_exit(current)
