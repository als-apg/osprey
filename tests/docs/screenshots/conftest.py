"""Fixtures shared by the docs-video engine's tests."""

from __future__ import annotations

import pytest
from docs.screenshots import __main__ as cli
from docs.screenshots.videos import video_named

from tests.docs.screenshots.video_fixture import SECOND


@pytest.fixture
def second_registered(monkeypatch):
    """The CLI sees the landing demo and :data:`SECOND` as its registry."""
    videos = (video_named("landing"), SECOND)
    monkeypatch.setattr(cli, "VIDEOS", videos)
    monkeypatch.setattr(cli, "video_named", lambda name: video_named(name, videos))
    return SECOND
