"""Declarative registry of documentation videos.

``VIDEOS`` lists every video the docs embed, as :mod:`docs.screenshots.recipes`
lists the screenshots. A :class:`DocVideo` names the file stem, the pages that
embed it, the screen it starts on (its :class:`~docs.screenshots.video_layout.
Layout`: view, active sidebar entry, open tiles), the deployment it is recorded on (``osprey init --set`` values on the
control-assistant tutorial stack, and the environment variables that must
reach it) and its storyboard: beats from :mod:`docs.screenshots.video_beats`,
plus custom beats where a probe is specific to one video.

``python -m docs.screenshots video`` records every entry in both themes, each
take on a fresh stack; ``--video NAME`` records one. Adding a video is one
entry here and its embed on the page.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from docs.screenshots import video_layout
from docs.screenshots.video_beats import Beat
from docs.screenshots.video_landing import LANDING
from docs.screenshots.video_layout import Layout

if TYPE_CHECKING:
    from docs.screenshots.video_take import Take

_ALLOWED_THEMES = frozenset({"dark", "light"})


@dataclass(frozen=True)
class DocVideo:
    """One docs video: what it shows, where it is embedded, what it is recorded on."""

    name: str
    """The registry name, for ``--video`` and the manifest."""

    stem: str
    """File stem: ``<stem>-<theme>.mp4`` and ``<stem>-<theme>-poster.jpg``."""

    storyboard: Sequence[Callable[[Take], None]]
    """The beats, played in order on one take."""

    layout: Layout
    """The view, active sidebar entry and open tiles the take starts on.

    It records what was agreed for the video before its storyboard was
    written; the take's setup applies it and no beat changes it implicitly.
    """

    pages: tuple[str, ...]
    """The pages under ``docs/source/`` that embed the video."""

    poster_step: str
    """The beat whose last frame is the poster."""

    window_s: tuple[float, float]
    """The playback length the storyboard is cut for; outside it only warns."""

    themes: tuple[str, ...] = ("dark", "light")

    init_sets: tuple[str, ...] = ()
    """Extra ``osprey init --set KEY=VALUE`` values for the tutorial stack."""

    needs_env: tuple[str, ...] = ()
    """Variables the deployment needs from the recording shell (names only)."""

    history: tuple[str, ...] = ()
    """Operator prompts played off camera before recording, each to the end of a
    real agent turn, so the video opens on that exchange. Chat only. The
    note under the video must say its first exchange predates the recording."""

    write_switch: tuple[str, str] | None = None
    """(target, position) the login's write switch is moved to before ``history``."""


VIDEOS: tuple[DocVideo, ...] = (
    DocVideo(
        name="landing",
        stem="osprey-demo",
        storyboard=LANDING,
        # What the landing take has always shown: the Claude Code terminal
        # beside the Artifacts tile, with the preset's example plot.
        layout=Layout(surface="terminal", active="artifacts", tiles=("artifacts",)),
        pages=("index.rst",),
        poster_step="rotate",
        window_s=(50.0, 80.0),
    ),
)


def video_named(name: str, videos: Sequence[DocVideo] = VIDEOS) -> DocVideo:
    """The registered video *name*; ``KeyError`` naming the known ones otherwise."""
    for video in videos:
        if video.name == name:
            return video
    raise KeyError(f"no video named {name!r}; known: {', '.join(v.name for v in videos)}")


def validate_videos(videos: Sequence[DocVideo] = VIDEOS) -> None:
    """Raise ``ValueError`` when an entry is malformed or two entries collide.

    Names and stems are unique; themes are a non-empty subset of dark/light;
    the storyboard is not empty and names its poster beat; the layout is a
    :class:`~docs.screenshots.video_layout.Layout` with a known surface; every
    ``init_sets`` value is ``KEY=VALUE``.
    """
    names: set[str] = set()
    stems: set[str] = set()
    for video in videos:
        if video.name in names:
            raise ValueError(f"duplicate video name: {video.name!r}")
        if video.stem in stems:
            raise ValueError(f"duplicate video stem: {video.stem!r}")
        names.add(video.name)
        stems.add(video.stem)
        if not video.themes or not set(video.themes) <= _ALLOWED_THEMES:
            raise ValueError(f"video {video.name!r}: themes must be a subset of dark/light")
        if not video.storyboard:
            raise ValueError(f"video {video.name!r}: empty storyboard")
        problem = video_layout.validate(getattr(video, "layout", None))
        if problem:
            raise ValueError(f"video {video.name!r} {problem}")
        beats = {b.name for b in video.storyboard if isinstance(b, Beat)}
        if video.poster_step not in beats:
            raise ValueError(
                f"video {video.name!r}: poster_step {video.poster_step!r} is not a beat"
            )
        if video.history and video.layout.surface != "chat":
            raise ValueError(f"video {video.name!r}: history needs the chat")
        for pair in video.init_sets:
            if "=" not in pair:
                raise ValueError(f"video {video.name!r}: init_sets value {pair!r} is not KEY=VALUE")
