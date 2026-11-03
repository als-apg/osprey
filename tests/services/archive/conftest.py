"""Fixtures the archive suites share."""

import os

import pytest


@pytest.fixture
def trees(tmp_path):
    """An empty sources tree and an owner-only destination tree."""
    sources = tmp_path / "sources"
    sources.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    os.chmod(dest, 0o700)
    return sources, dest
