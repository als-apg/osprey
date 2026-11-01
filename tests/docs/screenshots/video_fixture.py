"""A second video for the engine's tests: the registry ships only the landing demo."""

from __future__ import annotations

from docs.screenshots.video_beats import hold
from docs.screenshots.videos import DocVideo

SECOND = DocVideo(
    name="second",
    stem="osprey-second",
    storyboard=(hold("close", None, 1),),
    pages=("how-to/second.rst",),
    poster_step="close",
    window_s=(15.0, 60.0),
    init_sets=("config.web.ui_mode=simple",),
    needs_env=("DOCS_VIDEO_TEST_KEY",),
)
