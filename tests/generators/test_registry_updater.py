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
    temp_registry.read_text()

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


# --- Extend-style registry tests ---


@pytest.fixture
def extend_registry_content():
    """Sample extend-style registry.py content (post-migration template format)."""
    return """from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    FrameworkPromptProviderRegistration,
    RegistryConfig
)


class MyProjectRegistryProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return extend_framework_registry(
            framework_prompt_providers=[
                FrameworkPromptProviderRegistration(
                    module_path="my_project.framework_prompts",
                    prompt_builders={
                        "python": "ControlSystemPythonPromptBuilder",
                    }
                )
            ]
        )
"""


@pytest.fixture
def temp_extend_registry(tmp_path, extend_registry_content):
    """Create a temporary extend-style registry file inside a 'my_project' dir.

    The parent directory name is used by get_project_module_name() to derive module paths.
    """
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()
    registry_file = project_dir / "registry.py"
    registry_file.write_text(extend_registry_content)
    return registry_file


def test_add_to_extend_registry_first_capability(temp_extend_registry):
    """Test adding first capability to extend-style registry creates capabilities= and context_classes= params."""
    new_content, preview = add_to_registry(
        temp_extend_registry,
        capability_name="slack_mcp",
        class_name="SlackMcpCapability",
        context_type="SLACK_RESULTS",
        context_class_name="SlackMcpResultsContext",
        description="Slack operations via MCP server",
    )

    # Should contain the new capability registration
    assert 'name="slack_mcp"' in new_content
    assert "SlackMcpCapability" in new_content
    assert "SLACK_RESULTS" in new_content
    assert "SlackMcpResultsContext" in new_content

    # Should have capabilities= and context_classes= parameters
    assert "capabilities=[" in new_content
    assert "context_classes=[" in new_content

    # Should still have the extend_framework_registry call
    assert "extend_framework_registry(" in new_content

    # Preview should mention the new capability
    assert "SlackMcpCapability" in preview


def test_add_to_extend_registry_second_capability(temp_extend_registry):
    """Test adding second capability appends to existing lists."""
    # Add first capability
    first_content, _ = add_to_registry(
        temp_extend_registry,
        capability_name="slack_mcp",
        class_name="SlackMcpCapability",
        context_type="SLACK_RESULTS",
        context_class_name="SlackMcpResultsContext",
        description="Slack operations via MCP server",
    )
    temp_extend_registry.write_text(first_content)

    # Add second capability
    second_content, _ = add_to_registry(
        temp_extend_registry,
        capability_name="github_mcp",
        class_name="GithubMcpCapability",
        context_type="GITHUB_RESULTS",
        context_class_name="GithubMcpResultsContext",
        description="GitHub operations via MCP server",
    )

    # Both capabilities should be present
    assert 'name="slack_mcp"' in second_content
    assert 'name="github_mcp"' in second_content
    assert "SlackMcpCapability" in second_content
    assert "GithubMcpCapability" in second_content
    assert "SLACK_RESULTS" in second_content
    assert "GITHUB_RESULTS" in second_content
    assert "SlackMcpResultsContext" in second_content
    assert "GithubMcpResultsContext" in second_content


def test_add_to_extend_registry_adds_imports(temp_extend_registry):
    """Test that CapabilityRegistration and ContextClassRegistration are added to imports."""
    original_content = temp_extend_registry.read_text()

    # Verify imports don't exist yet
    assert "CapabilityRegistration" not in original_content
    assert "ContextClassRegistration" not in original_content

    new_content, _ = add_to_registry(
        temp_extend_registry,
        capability_name="slack_mcp",
        class_name="SlackMcpCapability",
        context_type="SLACK_RESULTS",
        context_class_name="SlackMcpResultsContext",
        description="Slack operations via MCP server",
    )

    # Both imports should be added
    assert "CapabilityRegistration" in new_content
    assert "ContextClassRegistration" in new_content

    # They should be in the import block
    import_block_end = new_content.index(")")
    import_block = new_content[:import_block_end]
    assert "CapabilityRegistration" in import_block
    assert "ContextClassRegistration" in import_block


def test_add_to_extend_registry_no_duplicate_imports(temp_extend_registry):
    """Test adding second capability doesn't duplicate imports that already exist."""
    # Add first capability (injects imports)
    first_content, _ = add_to_registry(
        temp_extend_registry,
        capability_name="slack_mcp",
        class_name="SlackMcpCapability",
        context_type="SLACK_RESULTS",
        context_class_name="SlackMcpResultsContext",
        description="Slack operations via MCP server",
    )
    temp_extend_registry.write_text(first_content)

    # Add second capability
    second_content, _ = add_to_registry(
        temp_extend_registry,
        capability_name="github_mcp",
        class_name="GithubMcpCapability",
        context_type="GITHUB_RESULTS",
        context_class_name="GithubMcpResultsContext",
        description="GitHub operations via MCP server",
    )

    # Each import should appear exactly once
    assert second_content.count("CapabilityRegistration,") == 1
    assert second_content.count("ContextClassRegistration,") == 1


def test_is_already_registered_extend_style(temp_extend_registry):
    """Test is_already_registered returns True after add, False before."""
    # Before adding: not registered
    assert not is_already_registered(temp_extend_registry, "slack_mcp")

    # Add capability
    new_content, _ = add_to_registry(
        temp_extend_registry,
        capability_name="slack_mcp",
        class_name="SlackMcpCapability",
        context_type="SLACK_RESULTS",
        context_class_name="SlackMcpResultsContext",
        description="Slack operations via MCP server",
    )
    temp_extend_registry.write_text(new_content)

    # After adding: registered
    assert is_already_registered(temp_extend_registry, "slack_mcp")

    # Other names: still not registered
    assert not is_already_registered(temp_extend_registry, "nonexistent")


def test_remove_from_extend_registry(temp_extend_registry):
    """Test remove_from_registry removes entries added via extend-style."""
    # First add a capability
    new_content, _ = add_to_registry(
        temp_extend_registry,
        capability_name="slack_mcp",
        class_name="SlackMcpCapability",
        context_type="SLACK_RESULTS",
        context_class_name="SlackMcpResultsContext",
        description="Slack operations via MCP server",
    )
    temp_extend_registry.write_text(new_content)

    # Now remove it
    removed_content, preview, found = remove_from_registry(temp_extend_registry, "slack_mcp")

    assert found
    assert "REMOVE" in preview
    assert "SlackMcpCapability" in preview

    # Capability and context should be gone
    assert "slack_mcp" not in removed_content
    assert "SlackMcpCapability" not in removed_content
    assert "SLACK_RESULTS" not in removed_content
    assert "SlackMcpResultsContext" not in removed_content

    # The extend_framework_registry structure should still be there
    assert "extend_framework_registry(" in removed_content
    assert "framework_prompt_providers=" in removed_content


def test_get_capability_info_extend_style(temp_extend_registry):
    """Test get_capability_info extracts correct fields from extend-style content."""
    # Add a capability first
    new_content, _ = add_to_registry(
        temp_extend_registry,
        capability_name="slack_mcp",
        class_name="SlackMcpCapability",
        context_type="SLACK_RESULTS",
        context_class_name="SlackMcpResultsContext",
        description="Slack operations via MCP server",
    )
    temp_extend_registry.write_text(new_content)

    # Get info
    info = get_capability_info(temp_extend_registry, "slack_mcp")

    assert info is not None
    assert info["class_name"] == "SlackMcpCapability"
    assert info["context_type"] == "SLACK_RESULTS"
    assert info["context_class_name"] == "SlackMcpResultsContext"
    assert info["module_path"] == "my_project.capabilities.slack_mcp"


def test_extend_registry_full_lifecycle(temp_extend_registry):
    """Test full lifecycle: add → registered → info → remove → not registered."""
    # 1. Not registered initially
    assert not is_already_registered(temp_extend_registry, "slack_mcp")

    # 2. Add capability
    new_content, _ = add_to_registry(
        temp_extend_registry,
        capability_name="slack_mcp",
        class_name="SlackMcpCapability",
        context_type="SLACK_RESULTS",
        context_class_name="SlackMcpResultsContext",
        description="Slack operations via MCP server",
    )
    temp_extend_registry.write_text(new_content)

    # 3. Now registered
    assert is_already_registered(temp_extend_registry, "slack_mcp")

    # 4. Can extract info
    info = get_capability_info(temp_extend_registry, "slack_mcp")
    assert info is not None
    assert info["class_name"] == "SlackMcpCapability"
    assert info["context_type"] == "SLACK_RESULTS"
    assert info["context_class_name"] == "SlackMcpResultsContext"
    assert info["module_path"] == "my_project.capabilities.slack_mcp"

    # 5. Remove capability
    removed_content, preview, found = remove_from_registry(temp_extend_registry, "slack_mcp")
    assert found
    temp_extend_registry.write_text(removed_content)

    # 6. No longer registered
    assert not is_already_registered(temp_extend_registry, "slack_mcp")

    # 7. Info returns None
    assert get_capability_info(temp_extend_registry, "slack_mcp") is None

    # 8. Registry structure intact
    assert "extend_framework_registry(" in removed_content
    assert "framework_prompt_providers=" in removed_content
