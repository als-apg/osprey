"""Tests for config updater functions."""

import pytest

from osprey.generators.config_updater import (
    add_capability_react_to_config,
    get_capability_react_config,
    has_capability_react_model,
    remove_capability_react_from_config,
)


@pytest.fixture
def sample_config_content():
    """Sample config.yml content for testing."""
    return """models:
  orchestrator:
    provider: anthropic
    model_id: claude-sonnet-4
    max_tokens: 4096

  weather_demo_react:
    provider: anthropic
    model_id: claude-haiku-4-5-20251001
    max_tokens: 4096

  slack_mcp_react:
    provider: openai
    model_id: gpt-4o
    max_tokens: 2048

registry_path: src/my_project/registry.py
"""


@pytest.fixture
def temp_config(tmp_path, sample_config_content):
    """Create a temporary config file."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(sample_config_content)
    return config_file


def test_has_capability_react_model(temp_config):
    """Test checking if capability has react model configured."""
    assert has_capability_react_model(temp_config, "weather_demo")
    assert has_capability_react_model(temp_config, "slack_mcp")
    assert not has_capability_react_model(temp_config, "nonexistent")


def test_remove_capability_react_from_config(temp_config):
    """Test removing capability react model from config."""
    # Remove weather_demo_react
    new_content, preview, found = remove_capability_react_from_config(temp_config, "weather_demo")

    # Check that it was found
    assert found
    assert "REMOVE" in preview
    assert "weather_demo_react" in preview

    # Check that it's removed from content
    assert "weather_demo_react" not in new_content

    # Check that other models are still there
    assert "orchestrator:" in new_content
    assert "slack_mcp_react:" in new_content


def test_remove_nonexistent_model(temp_config):
    """Test removing a model that doesn't exist."""
    new_content, preview, found = remove_capability_react_from_config(temp_config, "nonexistent")

    # Should not be found
    assert not found
    assert "found" in preview.lower()  # Message says "No config entry found"

    # Content should be unchanged
    original_content = temp_config.read_text()
    assert new_content == original_content


def test_get_capability_react_config(temp_config):
    """Test getting capability react model configuration."""
    config = get_capability_react_config(temp_config, "weather_demo")

    assert config is not None
    assert config["provider"] == "anthropic"
    assert config["model_id"] == "claude-haiku-4-5-20251001"
    assert config["max_tokens"] == 4096


def test_get_nonexistent_config(temp_config):
    """Test getting config for nonexistent capability."""
    config = get_capability_react_config(temp_config, "nonexistent")
    assert config is None


def test_add_then_remove_model(temp_config):
    """Test adding a model and then removing it."""
    # Add new model
    template_config = {"provider": "anthropic", "model_id": "claude-sonnet-4", "max_tokens": 4096}
    new_content, add_preview = add_capability_react_to_config(
        temp_config, capability_name="jira_mcp", template_config=template_config
    )

    # Write the new content
    temp_config.write_text(new_content)

    # Verify it was added
    assert has_capability_react_model(temp_config, "jira_mcp")

    # Remove it
    removed_content, remove_preview, found = remove_capability_react_from_config(
        temp_config, "jira_mcp"
    )

    # Verify it was removed
    assert found
    assert "jira_mcp_react" not in removed_content

    # Write the removed content
    temp_config.write_text(removed_content)

    # Verify it's no longer there
    assert not has_capability_react_model(temp_config, "jira_mcp")

    # But original models should still be there
    assert has_capability_react_model(temp_config, "weather_demo")
    assert has_capability_react_model(temp_config, "slack_mcp")


def test_remove_preserves_structure(temp_config):
    """Test that removal preserves the overall config structure."""
    original_content = temp_config.read_text()

    # Remove one model
    new_content, _, _ = remove_capability_react_from_config(temp_config, "weather_demo")

    # Check that the models section is still there
    assert "models:" in new_content
    assert "orchestrator:" in new_content

    # Check that other sections are preserved
    assert "registry_path:" in new_content

    # Check that other capability models are intact
    assert "slack_mcp_react:" in new_content


def test_remove_multiple_models_sequentially(temp_config):
    """Test removing multiple models one after another."""
    # Remove first model
    content1, _, found1 = remove_capability_react_from_config(temp_config, "weather_demo")
    assert found1
    temp_config.write_text(content1)

    # Remove second model
    content2, _, found2 = remove_capability_react_from_config(temp_config, "slack_mcp")
    assert found2
    temp_config.write_text(content2)

    # Verify both are removed
    assert not has_capability_react_model(temp_config, "weather_demo")
    assert not has_capability_react_model(temp_config, "slack_mcp")

    # But orchestrator should still be there
    final_content = temp_config.read_text()
    assert "orchestrator:" in final_content
    assert "models:" in final_content
