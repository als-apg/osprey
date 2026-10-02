"""Shared fixtures for the event-dispatcher tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def facility_zone_config(tmp_path, monkeypatch) -> str:
    """Serve a project config naming the facility zone, the way the dispatcher finds it.

    The containerized dispatcher reads the flattened ``config.yml`` in its working
    directory with no ``CONFIG_FILE`` set; this fixture serves the config exactly
    that way. The zone is nine hours east of UTC with no daylight-saving shift, so
    each instant has one answer.
    """
    (tmp_path / "config.yml").write_text("system:\n  timezone: Asia/Tokyo\n")
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    return "Asia/Tokyo"
