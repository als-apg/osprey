"""Tests for registry updater functions."""

import pytest

from osprey.generators.registry_updater import (
    add_to_registry,
    get_capability_info,
    is_already_registered,
    remove_from_registry,
)


@pytest.fixture
def sample_registry_content():
    """Sample registry.py content for testing."""
    return """from osprey.registry.base import CapabilityRegistration, ContextClassRegistration

class MyRegistryProvider:
    def get_registry_config(self):
        return dict(
            capabilities=[
                CapabilityRegistration(
                    name="test_capability",
                    module_path="my_project.capabilities.test_capability",
                    class_name="TestCapability",
                    description="Test capability",
                    provides=["TEST_RESULTS"],
                    requires=[]
                ),
                CapabilityRegistration(
                    name="weather_demo",
                    module_path="my_project.capabilities.weather_demo",
                    class_name="WeatherDemoCapability",
                    description="WeatherDemo operations via MCP server",
                    provides=["WEATHERDEMO_RESULTS"],
                    requires=[]
                ),
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="TEST_RESULTS",
                    module_path="my_project.capabilities.test_capability",
                    class_name="TestResultsContext"
                ),
                ContextClassRegistration(
                    context_type="WEATHERDEMO_RESULTS",
                    module_path="my_project.capabilities.weather_demo",
                    class_name="WeatherDemoResultsContext"
                ),
            ]
        )
"""


@pytest.fixture
def temp_registry(tmp_path, sample_registry_content):
    """Create a temporary registry file."""
    registry_file = tmp_path / "registry.py"
    registry_file.write_text(sample_registry_content)
    return registry_file


def test_is_already_registered(temp_registry):
    """Test checking if capability is registered."""
    assert is_already_registered(temp_registry, "weather_demo")
    assert is_already_registered(temp_registry, "test_capability")
    assert not is_already_registered(temp_registry, "nonexistent")


def test_remove_from_registry(temp_registry):
    """Test removing capability from registry."""
    # Remove weather_demo
    new_content, preview, found = remove_from_registry(temp_registry, "weather_demo")

    # Check that it was found
    assert found
    assert "REMOVE" in preview
    assert "WeatherDemoCapability" in preview
    assert "WEATHERDEMO_RESULTS" in preview

    # Check that it's removed from content
    assert "weather_demo" not in new_content
    assert "WeatherDemoCapability" not in new_content
    assert "WEATHERDEMO_RESULTS" not in new_content

    # Check that other capability is still there
    assert "test_capability" in new_content
    assert "TestCapability" in new_content


def test_remove_nonexistent_capability(temp_registry):
    """Test removing a capability that doesn't exist."""
    new_content, preview, found = remove_from_registry(temp_registry, "nonexistent")

    # Should not be found
    assert not found
    assert "found" in preview.lower()  # Message says "No registry entries found"

    # Content should be unchanged
    original_content = temp_registry.read_text()
    assert new_content == original_content


def test_get_capability_info(temp_registry):
    """Test extracting capability information."""
    info = get_capability_info(temp_registry, "weather_demo")

    assert info is not None
    assert info["class_name"] == "WeatherDemoCapability"
    assert info["context_type"] == "WEATHERDEMO_RESULTS"
    assert info["context_class_name"] == "WeatherDemoResultsContext"
    assert info["module_path"] == "my_project.capabilities.weather_demo"


def test_get_capability_info_nonexistent(temp_registry):
    """Test getting info for nonexistent capability."""
    info = get_capability_info(temp_registry, "nonexistent")
    assert info is None


def test_add_then_remove_capability(temp_registry):
    """Test adding a capability and then removing it."""
    # Add new capability
    new_content, add_preview = add_to_registry(
        temp_registry,
        capability_name="slack_mcp",
        class_name="SlackMcpCapability",
        context_type="SLACK_RESULTS",
        context_class_name="SlackMcpResultsContext",
        description="Slack operations via MCP server",
    )

    # Write the new content
    temp_registry.write_text(new_content)

    # Verify it was added
    assert is_already_registered(temp_registry, "slack_mcp")

    # Remove it
    removed_content, remove_preview, found = remove_from_registry(temp_registry, "slack_mcp")

    # Verify it was removed
    assert found
    assert "slack_mcp" not in removed_content
    assert "SlackMcpCapability" not in removed_content

    # Write the removed content
    temp_registry.write_text(removed_content)

    # Verify it's no longer registered
    assert not is_already_registered(temp_registry, "slack_mcp")

    # But original capabilities should still be there
    assert is_already_registered(temp_registry, "weather_demo")
    assert is_already_registered(temp_registry, "test_capability")


def test_remove_preserves_formatting(temp_registry):
    """Test that removal preserves the overall file structure."""
    original_content = temp_registry.read_text()

    # Remove one capability
    new_content, _, _ = remove_from_registry(temp_registry, "weather_demo")

    # Check that the class definition and get_registry_config are still there
    assert "class MyRegistryProvider:" in new_content
    assert "def get_registry_config(self):" in new_content
    assert "return dict(" in new_content
    assert "capabilities=[" in new_content
    assert "context_classes=[" in new_content

    # Check that the other capability is intact
    assert "test_capability" in new_content
    assert "TestCapability" in new_content
