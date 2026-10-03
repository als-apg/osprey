"""Shared fixtures for the ARIEL web interface tests."""

from __future__ import annotations

import pytest

from tests.services.ariel_search.conftest import reset_image_lane_state


@pytest.fixture(autouse=True)
def _reset_image_lane():
    """Every ARIEL interface test starts and ends with a closed, reason-free picture lane.

    Requests no ``monkeypatch``: an autouse fixture that does inverts the
    teardown order ``tests/interfaces/conftest.py`` relies on.
    """
    reset_image_lane_state()
    yield
    reset_image_lane_state()
