"""Fixtures for MCP python_executor adapter tests."""

import pytest


@pytest.fixture
def execution_folder(tmp_path):
    """Create a temporary execution folder with standard structure."""
    folder = tmp_path / "execution_20240115_120000_abcd1234"
    folder.mkdir(parents=True)
    (folder / "figures").mkdir()
    return folder
