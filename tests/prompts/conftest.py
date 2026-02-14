"""Shared fixtures for prompt builder tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_registry_for_prompt_tests():
    """Mock get_registry() for prompt builder tests that need context_types.

    Several prompt builders (python, time_range_parsing, memory_extraction)
    call get_registry() inside their guide methods to resolve context type
    names. This fixture provides a lightweight mock so these tests don't
    require a full config.yml setup.

    We patch at each module where get_registry was imported at module level.
    """
    mock_context_types = SimpleNamespace(
        PYTHON_RESULTS="PYTHON_RESULTS",
        TIME_RANGE="TIME_RANGE",
        MEMORY_CONTEXT="MEMORY_CONTEXT",
        LOGBOOK_SEARCH_RESULTS="LOGBOOK_SEARCH_RESULTS",
        CHANNEL_ADDRESSES="CHANNEL_ADDRESSES",
        CHANNEL_VALUES="CHANNEL_VALUES",
        CHANNEL_WRITE_RESULTS="CHANNEL_WRITE_RESULTS",
        ARCHIVER_DATA="ARCHIVER_DATA",
    )
    mock_registry = SimpleNamespace(context_types=mock_context_types)

    with (
        patch("osprey.prompts.defaults.python.get_registry", return_value=mock_registry),
        patch(
            "osprey.prompts.defaults.time_range_parsing.get_registry", return_value=mock_registry
        ),
        patch("osprey.prompts.defaults.memory_extraction.get_registry", return_value=mock_registry),
        patch("osprey.registry.get_registry", return_value=mock_registry),
    ):
        yield mock_registry
