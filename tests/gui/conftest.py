"""Shared fixtures for GUI tests."""

from unittest.mock import MagicMock

import pytest

# Import base test class for routing tests
from .routing_test_base import MockHelpers, RoutingTestBase, TestDataGenerator

# Make available to all test modules
__all__ = ["RoutingTestBase", "TestDataGenerator", "MockHelpers"]


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus with emit method."""
    bus = MagicMock()
    bus.emit = MagicMock()
    return bus


@pytest.fixture
def mock_project_manager():
    """Create a mock project manager."""
    return MagicMock()


@pytest.fixture
def mock_routing_cache():
    """Create a mock routing cache."""
    return MagicMock()


@pytest.fixture
def mock_analytics():
    """Create a mock analytics tracker."""
    return MagicMock()


@pytest.fixture
def mock_llm_config():
    """Create mock LLM configuration."""
    return {
        "provider": "anthropic",
        "model_id": "claude-3-sonnet-20240229",
        "api_key": "test-key",
    }
