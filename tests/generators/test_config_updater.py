"""Tests for config updater functions."""

import pytest

from osprey.generators.config_updater import (
    add_capability_react_to_config,
    get_all_model_configs,
    get_capability_react_config,
    has_capability_react_model,
    remove_capability_react_from_config,
    update_all_models,
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
def updater_temp_config(tmp_path, sample_config_content):
    """Create a temporary config file."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(sample_config_content)
    return config_file


def test_has_capability_react_model(updater_temp_config):
    """Test checking if capability has react model configured."""
    assert has_capability_react_model(updater_temp_config, "weather_demo")
    assert has_capability_react_model(updater_temp_config, "slack_mcp")
    assert not has_capability_react_model(updater_temp_config, "nonexistent")


def test_remove_capability_react_from_config(updater_temp_config):
    """Test removing capability react model from config."""
    # Remove weather_demo_react
    new_content, preview, found = remove_capability_react_from_config(
        updater_temp_config, "weather_demo"
    )

    # Check that it was found
    assert found
    assert "REMOVE" in preview
    assert "weather_demo_react" in preview

    # Check that it's removed from content
    assert "weather_demo_react" not in new_content

    # Check that other models are still there
    assert "orchestrator:" in new_content
    assert "slack_mcp_react:" in new_content


def test_remove_nonexistent_model(updater_temp_config):
    """Test removing a model that doesn't exist."""
    new_content, preview, found = remove_capability_react_from_config(
        updater_temp_config, "nonexistent"
    )

    # Should not be found
    assert not found
    assert "found" in preview.lower()  # Message says "No config entry found"

    # Content should be unchanged
    original_content = updater_temp_config.read_text()
    assert new_content == original_content


def test_get_capability_react_config(updater_temp_config):
    """Test getting capability react model configuration."""
    config = get_capability_react_config(updater_temp_config, "weather_demo")

    assert config is not None
    assert config["provider"] == "anthropic"
    assert config["model_id"] == "claude-haiku-4-5-20251001"
    assert config["max_tokens"] == 4096


def test_get_nonexistent_config(updater_temp_config):
    """Test getting config for nonexistent capability."""
    config = get_capability_react_config(updater_temp_config, "nonexistent")
    assert config is None


def test_add_then_remove_model(updater_temp_config):
    """Test adding a model and then removing it."""
    # Add new model
    template_config = {"provider": "anthropic", "model_id": "claude-sonnet-4", "max_tokens": 4096}
    new_content, add_preview = add_capability_react_to_config(
        updater_temp_config, capability_name="jira_mcp", template_config=template_config
    )

    # Write the new content
    updater_temp_config.write_text(new_content)

    # Verify it was added
    assert has_capability_react_model(updater_temp_config, "jira_mcp")

    # Remove it
    removed_content, remove_preview, found = remove_capability_react_from_config(
        updater_temp_config, "jira_mcp"
    )

    # Verify it was removed
    assert found
    assert "jira_mcp_react" not in removed_content

    # Write the removed content
    updater_temp_config.write_text(removed_content)

    # Verify it's no longer there
    assert not has_capability_react_model(updater_temp_config, "jira_mcp")

    # But original models should still be there
    assert has_capability_react_model(updater_temp_config, "weather_demo")
    assert has_capability_react_model(updater_temp_config, "slack_mcp")


def test_remove_preserves_structure(updater_temp_config):
    """Test that removal preserves the overall config structure."""
    updater_temp_config.read_text()

    # Remove one model
    new_content, _, _ = remove_capability_react_from_config(updater_temp_config, "weather_demo")

    # Check that the models section is still there
    assert "models:" in new_content
    assert "orchestrator:" in new_content

    # Check that other sections are preserved
    assert "registry_path:" in new_content

    # Check that other capability models are intact
    assert "slack_mcp_react:" in new_content


def test_remove_multiple_models_sequentially(updater_temp_config):
    """Test removing multiple models one after another."""
    # Remove first model
    content1, _, found1 = remove_capability_react_from_config(updater_temp_config, "weather_demo")
    assert found1
    updater_temp_config.write_text(content1)

    # Remove second model
    content2, _, found2 = remove_capability_react_from_config(updater_temp_config, "slack_mcp")
    assert found2
    updater_temp_config.write_text(content2)

    # Verify both are removed
    assert not has_capability_react_model(updater_temp_config, "weather_demo")
    assert not has_capability_react_model(updater_temp_config, "slack_mcp")

    # But orchestrator should still be there
    final_content = updater_temp_config.read_text()
    assert "orchestrator:" in final_content
    assert "models:" in final_content


# =============================================================================
# Model Configuration Tests (update_all_models and get_all_model_configs)
# =============================================================================


def test_get_all_model_configs(updater_temp_config):
    """Test getting all model configurations from config."""
    models = get_all_model_configs(updater_temp_config)

    assert models is not None
    assert isinstance(models, dict)
    assert "orchestrator" in models
    assert "weather_demo_react" in models
    assert "slack_mcp_react" in models

    # Check orchestrator config
    assert models["orchestrator"]["provider"] == "anthropic"
    assert models["orchestrator"]["model_id"] == "claude-sonnet-4"
    assert models["orchestrator"]["max_tokens"] == 4096


def test_get_all_model_configs_empty():
    """Test getting model configs from invalid file."""
    from pathlib import Path

    invalid_path = Path("/nonexistent/config.yml")
    models = get_all_model_configs(invalid_path)
    assert models is None


def test_update_all_models_basic(updater_temp_config):
    """Test updating all models to a new provider and model."""
    # Update all models to OpenAI GPT-4
    new_content, preview = update_all_models(updater_temp_config, "openai", "gpt-4")

    # Check that content was updated
    assert "provider: openai" in new_content
    assert "model_id: gpt-4" in new_content

    # Check that ALL models were updated (count occurrences)
    assert new_content.count("provider: openai") == 3  # orchestrator, weather_demo, slack_mcp
    assert new_content.count("model_id: gpt-4") == 3

    # Check that old values are gone
    assert "provider: anthropic" not in new_content
    assert "model_id: claude-sonnet-4" not in new_content
    assert "model_id: claude-haiku-4-5-20251001" not in new_content
    assert "model_id: gpt-4o" not in new_content

    # Check preview contains useful information
    assert "openai" in preview
    assert "gpt-4" in preview
    assert "orchestrator" in preview


def test_update_all_models_preserves_max_tokens(updater_temp_config):
    """Test that updating models preserves max_tokens settings."""
    # Update all models
    new_content, _ = update_all_models(updater_temp_config, "anthropic", "claude-opus-4")

    # Write and re-read to verify structure
    updater_temp_config.write_text(new_content)
    models = get_all_model_configs(updater_temp_config)

    # Check that max_tokens were preserved
    assert models["orchestrator"]["max_tokens"] == 4096
    assert models["weather_demo_react"]["max_tokens"] == 4096
    assert models["slack_mcp_react"]["max_tokens"] == 2048  # Original value preserved

    # But providers and models should be updated
    assert models["orchestrator"]["provider"] == "anthropic"
    assert models["orchestrator"]["model_id"] == "claude-opus-4"
    assert models["slack_mcp_react"]["provider"] == "anthropic"
    assert models["slack_mcp_react"]["model_id"] == "claude-opus-4"


def test_update_all_models_preview_shows_changes(updater_temp_config):
    """Test that preview shows what will be changed."""
    # Get initial state
    get_all_model_configs(updater_temp_config)

    # Update to different provider/model
    _, preview = update_all_models(updater_temp_config, "openai", "gpt-4-turbo")

    # Preview should show all models being updated
    assert "orchestrator" in preview
    assert "weather_demo_react" in preview
    assert "slack_mcp_react" in preview

    # Preview should show the new values
    assert "openai" in preview
    assert "gpt-4-turbo" in preview

    # Preview should indicate number of models
    assert "3" in preview or "model" in preview.lower()


def test_update_all_models_with_same_values_shows_no_change(updater_temp_config):
    """Test that updating to same values shows '(no change)' in preview."""
    # Update orchestrator back to its current value
    _, preview = update_all_models(updater_temp_config, "anthropic", "claude-sonnet-4")

    # Preview should indicate no change for orchestrator
    assert "orchestrator" in preview
    # At least one model should show the update (weather_demo and slack_mcp have different models)


def test_update_all_models_preserves_structure(updater_temp_config):
    """Test that updating models preserves overall config structure."""
    original_content = updater_temp_config.read_text()

    # Update all models
    new_content, _ = update_all_models(updater_temp_config, "google", "gemini-pro")

    # Check that models section structure is preserved
    assert "models:" in new_content
    assert "orchestrator:" in new_content
    assert "weather_demo_react:" in new_content
    assert "slack_mcp_react:" in new_content

    # Check that other sections are preserved
    assert "registry_path:" in new_content
    assert new_content.count("registry_path:") == original_content.count("registry_path:")


def test_update_all_models_works_with_added_model(updater_temp_config):
    """Test updating models after adding a new capability model."""
    # First, add a new model
    template_config = {"provider": "anthropic", "model_id": "claude-haiku-4", "max_tokens": 2048}
    new_content, _ = add_capability_react_to_config(
        updater_temp_config, "new_capability", template_config
    )
    updater_temp_config.write_text(new_content)

    # Now update all models
    updated_content, _ = update_all_models(updater_temp_config, "openai", "gpt-3.5-turbo")

    # Verify the new model was also updated
    assert "provider: openai" in updated_content
    assert updated_content.count("provider: openai") == 4  # Including the newly added one

    # Write and verify via get_all_model_configs
    updater_temp_config.write_text(updated_content)
    models = get_all_model_configs(updater_temp_config)

    assert models["new_capability_react"]["provider"] == "openai"
    assert models["new_capability_react"]["model_id"] == "gpt-3.5-turbo"
    assert models["new_capability_react"]["max_tokens"] == 2048  # Preserved


def test_update_all_models_multiple_times(updater_temp_config):
    """Test that models can be updated multiple times sequentially."""
    # First update
    content1, _ = update_all_models(updater_temp_config, "openai", "gpt-4")
    updater_temp_config.write_text(content1)

    models1 = get_all_model_configs(updater_temp_config)
    assert models1["orchestrator"]["provider"] == "openai"
    assert models1["orchestrator"]["model_id"] == "gpt-4"

    # Second update
    content2, _ = update_all_models(updater_temp_config, "anthropic", "claude-opus-4")
    updater_temp_config.write_text(content2)

    models2 = get_all_model_configs(updater_temp_config)
    assert models2["orchestrator"]["provider"] == "anthropic"
    assert models2["orchestrator"]["model_id"] == "claude-opus-4"

    # Third update
    content3, _ = update_all_models(updater_temp_config, "google", "gemini-ultra")
    updater_temp_config.write_text(content3)

    models3 = get_all_model_configs(updater_temp_config)
    assert models3["orchestrator"]["provider"] == "google"
    assert models3["orchestrator"]["model_id"] == "gemini-ultra"

    # Max tokens should still be preserved throughout
    assert models3["orchestrator"]["max_tokens"] == 4096
    assert models3["slack_mcp_react"]["max_tokens"] == 2048


@pytest.fixture
def config_with_channel_finder(tmp_path):
    """Config with channel_finder model for control assistant template."""
    config_content = """models:
  orchestrator:
    provider: cborg
    model_id: anthropic/claude-haiku
    max_tokens: 4096
  channel_finder:
    provider: cborg
    model_id: anthropic/claude-haiku
    max_tokens: 4096
  channel_write:
    provider: cborg
    model_id: anthropic/claude-haiku
    max_tokens: 2048

registry_path: src/my_control_assistant/registry.py
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)
    return config_file


def test_update_all_models_with_channel_finder(config_with_channel_finder):
    """Test updating models in control assistant config with channel_finder."""
    # Update all models to OpenAI
    new_content, preview = update_all_models(config_with_channel_finder, "openai", "gpt-4")

    # Verify all models updated
    config_with_channel_finder.write_text(new_content)
    models = get_all_model_configs(config_with_channel_finder)

    assert models["orchestrator"]["provider"] == "openai"
    assert models["orchestrator"]["model_id"] == "gpt-4"
    assert models["channel_finder"]["provider"] == "openai"
    assert models["channel_finder"]["model_id"] == "gpt-4"
    assert models["channel_write"]["provider"] == "openai"
    assert models["channel_write"]["model_id"] == "gpt-4"

    # Max tokens preserved
    assert models["orchestrator"]["max_tokens"] == 4096
    assert models["channel_finder"]["max_tokens"] == 4096
    assert models["channel_write"]["max_tokens"] == 2048
