"""
Tests for ModelPreferencesStore

Tests the model preferences management functionality including model discovery,
provider configuration, and per-step model preferences.
"""

from unittest.mock import patch

import pytest

from osprey.interfaces.pyqt.model_preferences import ModelPreferencesStore


class TestModelPreferencesStoreInitialization:
    """Test suite for ModelPreferencesStore initialization."""

    def test_init_creates_empty_preferences(self):
        """Test initialization creates empty preferences dict."""
        store = ModelPreferencesStore()

        assert store._preferences == {}

    def test_init_has_infrastructure_steps(self):
        """Test initialization includes infrastructure steps."""
        store = ModelPreferencesStore()

        expected_steps = [
            "orchestrator",
            "router",
            "classifier",
            "task_extractor",
            "clarifier",
            "responder",
        ]

        assert store.INFRASTRUCTURE_STEPS == expected_steps

    def test_init_has_available_models(self):
        """Test initialization includes available models for providers."""
        store = ModelPreferencesStore()

        # Check that common providers are present
        assert "openai" in store._available_models
        assert "anthropic" in store._available_models
        assert "ollama" in store._available_models
        assert "argo" in store._available_models

        # Check that models are lists
        assert isinstance(store._available_models["openai"], list)
        assert len(store._available_models["openai"]) > 0

    def test_init_creates_empty_dynamic_cache(self):
        """Test initialization creates empty dynamic model cache."""
        store = ModelPreferencesStore()

        assert store._dynamic_model_cache == {}


class TestGetProviderFromConfig:
    """Test suite for extracting provider from config."""

    @patch("osprey.interfaces.pyqt.model_preferences.load_config_safe")
    def test_get_provider_from_models_section(self, mock_load):
        """Test extracting provider from models section."""
        mock_load.return_value = {
            "models": {
                "default": {"provider": "anthropic", "model_id": "claude-3-5-sonnet-20241022"}
            }
        }

        store = ModelPreferencesStore()
        provider = store.get_provider_from_config("config.yml")

        assert provider == "anthropic"

    @patch("osprey.interfaces.pyqt.model_preferences.load_config_safe")
    def test_get_provider_from_llm_section(self, mock_load):
        """Test extracting provider from legacy llm section."""
        mock_load.return_value = {"llm": {"provider": "openai"}}

        store = ModelPreferencesStore()
        provider = store.get_provider_from_config("config.yml")

        assert provider == "openai"

    @patch("osprey.interfaces.pyqt.model_preferences.load_config_safe")
    def test_get_provider_from_api_providers(self, mock_load):
        """Test extracting provider from api.providers section."""
        mock_load.return_value = {
            "api": {"providers": {"ollama": {"base_url": "http://localhost:11434"}}}
        }

        store = ModelPreferencesStore()
        provider = store.get_provider_from_config("config.yml")

        assert provider == "ollama"

    @patch("osprey.interfaces.pyqt.model_preferences.load_config_safe")
    def test_get_provider_returns_none_when_not_found(self, mock_load):
        """Test returns None when provider not found."""
        mock_load.return_value = {"other": "config"}

        store = ModelPreferencesStore()
        provider = store.get_provider_from_config("config.yml")

        assert provider is None

    @patch("osprey.interfaces.pyqt.model_preferences.load_config_safe")
    def test_get_provider_handles_empty_config(self, mock_load):
        """Test handles empty config gracefully."""
        mock_load.return_value = None

        store = ModelPreferencesStore()
        provider = store.get_provider_from_config("config.yml")

        assert provider is None

    @patch("osprey.interfaces.pyqt.model_preferences.load_config_safe")
    def test_get_provider_prioritizes_models_section(self, mock_load):
        """Test models section takes priority over other sections."""
        mock_load.return_value = {
            "models": {"default": {"provider": "anthropic"}},
            "llm": {"provider": "openai"},
            "api": {"providers": {"ollama": {}}},
        }

        store = ModelPreferencesStore()
        provider = store.get_provider_from_config("config.yml")

        assert provider == "anthropic"


class TestResolveEnvVar:
    """Test suite for environment variable resolution."""

    @patch.dict("os.environ", {"TEST_VAR": "test_value"})
    def test_resolve_env_var_with_braces(self):
        """Test resolving ${VAR_NAME} syntax."""
        store = ModelPreferencesStore()
        result = store._resolve_env_var("prefix_${TEST_VAR}_suffix")

        assert result == "prefix_test_value_suffix"

    @patch.dict("os.environ", {"TEST_VAR": "test_value"})
    def test_resolve_env_var_without_braces(self):
        """Test resolving $VAR_NAME syntax."""
        store = ModelPreferencesStore()
        result = store._resolve_env_var("prefix_$TEST_VAR")

        assert result == "prefix_test_value"

    @patch.dict("os.environ", {}, clear=True)
    def test_resolve_env_var_missing_variable(self):
        """Test handling missing environment variable."""
        store = ModelPreferencesStore()
        result = store._resolve_env_var("prefix_${MISSING_VAR}_suffix")

        # Should keep original placeholder
        assert result == "prefix_${MISSING_VAR}_suffix"

    def test_resolve_env_var_with_none(self):
        """Test handling None value."""
        store = ModelPreferencesStore()
        result = store._resolve_env_var(None)

        assert result is None

    def test_resolve_env_var_with_non_string(self):
        """Test handling non-string value."""
        store = ModelPreferencesStore()
        result = store._resolve_env_var(123)

        assert result == 123


class TestGetAvailableModels:
    """Test suite for getting available models."""

    def test_get_available_models_static_openai(self):
        """Test getting static OpenAI models."""
        store = ModelPreferencesStore()
        models = store.get_available_models("openai", use_dynamic=False)

        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4" in models

    def test_get_available_models_static_anthropic(self):
        """Test getting static Anthropic models."""
        store = ModelPreferencesStore()
        models = store.get_available_models("anthropic", use_dynamic=False)

        assert isinstance(models, list)
        assert len(models) > 0
        assert any("claude" in m for m in models)

    def test_get_available_models_unknown_provider(self):
        """Test getting models for unknown provider."""
        store = ModelPreferencesStore()
        models = store.get_available_models("unknown_provider", use_dynamic=False)

        assert models == []

    def test_get_available_models_case_insensitive(self):
        """Test provider name is case insensitive."""
        store = ModelPreferencesStore()
        models_lower = store.get_available_models("openai", use_dynamic=False)
        models_upper = store.get_available_models("OPENAI", use_dynamic=False)

        assert models_lower == models_upper

    @patch.object(ModelPreferencesStore, "discover_models_from_provider")
    def test_get_available_models_uses_dynamic_when_enabled(self, mock_discover):
        """Test uses dynamic discovery when enabled."""
        mock_discover.return_value = ["dynamic-model-1", "dynamic-model-2"]

        store = ModelPreferencesStore()
        models = store.get_available_models("openai", config_path="config.yml", use_dynamic=True)

        mock_discover.assert_called_once_with("openai", "config.yml")
        assert models == ["dynamic-model-1", "dynamic-model-2"]

    @patch.object(ModelPreferencesStore, "discover_models_from_provider")
    def test_get_available_models_falls_back_to_static(self, mock_discover):
        """Test falls back to static when dynamic discovery fails."""
        mock_discover.return_value = []

        store = ModelPreferencesStore()
        models = store.get_available_models("openai", config_path="config.yml", use_dynamic=True)

        # Should fall back to static list
        assert len(models) > 0
        assert "gpt-4" in models


class TestDiscoverModelsFromProvider:
    """Test suite for dynamic model discovery."""

    @patch.object(ModelPreferencesStore, "_discover_openai_models")
    def test_discover_openai_models(self, mock_discover):
        """Test discovering OpenAI models."""
        mock_discover.return_value = ["gpt-4", "gpt-3.5-turbo"]

        store = ModelPreferencesStore()
        models = store.discover_models_from_provider("openai", "config.yml")

        mock_discover.assert_called_once_with("config.yml")
        assert models == ["gpt-4", "gpt-3.5-turbo"]

    @patch.object(ModelPreferencesStore, "_discover_ollama_models")
    def test_discover_ollama_models(self, mock_discover):
        """Test discovering Ollama models."""
        mock_discover.return_value = ["llama2", "mistral"]

        store = ModelPreferencesStore()
        models = store.discover_models_from_provider("ollama", "config.yml")

        mock_discover.assert_called_once_with("config.yml")
        assert models == ["llama2", "mistral"]

    @patch.object(ModelPreferencesStore, "_discover_openai_compatible_models")
    def test_discover_argo_models(self, mock_discover):
        """Test discovering Argo models."""
        mock_discover.return_value = ["anthropic/claude-3-5-sonnet-20241022"]

        store = ModelPreferencesStore()
        models = store.discover_models_from_provider("argo", "config.yml")

        mock_discover.assert_called_once_with("config.yml", "argo")
        assert models == ["anthropic/claude-3-5-sonnet-20241022"]

    def test_discover_models_uses_cache(self):
        """Test model discovery uses cache."""
        store = ModelPreferencesStore()

        # Populate cache
        cache_key = "openai:config.yml"
        store._dynamic_model_cache[cache_key] = ["cached-model"]

        # Should return cached value without calling discovery
        models = store.discover_models_from_provider("openai", "config.yml")

        assert models == ["cached-model"]

    @patch.object(ModelPreferencesStore, "_discover_openai_models")
    def test_discover_models_caches_result(self, mock_discover):
        """Test model discovery caches successful results."""
        mock_discover.return_value = ["gpt-4"]

        store = ModelPreferencesStore()
        store.discover_models_from_provider("openai", "config.yml")

        # Check cache was populated
        cache_key = "openai:config.yml"
        assert cache_key in store._dynamic_model_cache
        assert store._dynamic_model_cache[cache_key] == ["gpt-4"]

    @patch.object(ModelPreferencesStore, "_discover_openai_models")
    def test_discover_models_falls_back_on_error(self, mock_discover):
        """Test falls back to static list on discovery error."""
        mock_discover.side_effect = Exception("API error")

        store = ModelPreferencesStore()
        models = store.discover_models_from_provider("openai", "config.yml")

        # Should return static list
        assert len(models) > 0
        assert "gpt-4" in models


class TestSetModelForStep:
    """Test suite for setting model preferences."""

    def test_set_model_for_new_project(self):
        """Test setting model for a new project."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")

        assert store.get_model_for_step("project1", "router") == "gpt-4"

    def test_set_model_for_existing_project(self):
        """Test setting model for existing project."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")
        store.set_model_for_step("project1", "orchestrator", "claude-3-5-sonnet-20241022")

        assert store.get_model_for_step("project1", "router") == "gpt-4"
        assert store.get_model_for_step("project1", "orchestrator") == "claude-3-5-sonnet-20241022"

    def test_set_model_overwrites_existing(self):
        """Test setting model overwrites existing preference."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")
        store.set_model_for_step("project1", "router", "gpt-3.5-turbo")

        assert store.get_model_for_step("project1", "router") == "gpt-3.5-turbo"

    def test_set_model_for_multiple_projects(self):
        """Test setting models for multiple projects."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")
        store.set_model_for_step("project2", "router", "claude-3-5-sonnet-20241022")

        assert store.get_model_for_step("project1", "router") == "gpt-4"
        assert store.get_model_for_step("project2", "router") == "claude-3-5-sonnet-20241022"


class TestGetModelForStep:
    """Test suite for getting model preferences."""

    def test_get_model_for_configured_step(self):
        """Test getting model for configured step."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")

        model = store.get_model_for_step("project1", "router")
        assert model == "gpt-4"

    def test_get_model_for_unconfigured_step(self):
        """Test getting model for unconfigured step returns None."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")

        model = store.get_model_for_step("project1", "orchestrator")
        assert model is None

    def test_get_model_for_nonexistent_project(self):
        """Test getting model for nonexistent project returns None."""
        store = ModelPreferencesStore()

        model = store.get_model_for_step("nonexistent", "router")
        assert model is None


class TestGetAllPreferences:
    """Test suite for getting all preferences."""

    def test_get_all_preferences_for_project(self):
        """Test getting all preferences for a project."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")
        store.set_model_for_step("project1", "orchestrator", "claude-3-5-sonnet-20241022")

        prefs = store.get_all_preferences("project1")

        assert prefs == {"router": "gpt-4", "orchestrator": "claude-3-5-sonnet-20241022"}

    def test_get_all_preferences_returns_copy(self):
        """Test get_all_preferences returns a copy."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")

        prefs = store.get_all_preferences("project1")
        prefs["router"] = "modified"

        # Original should be unchanged
        assert store.get_model_for_step("project1", "router") == "gpt-4"

    def test_get_all_preferences_for_nonexistent_project(self):
        """Test getting preferences for nonexistent project returns empty dict."""
        store = ModelPreferencesStore()

        prefs = store.get_all_preferences("nonexistent")
        assert prefs == {}


class TestSetAllPreferences:
    """Test suite for setting all preferences."""

    def test_set_all_preferences_for_new_project(self):
        """Test setting all preferences for new project."""
        store = ModelPreferencesStore()
        prefs = {"router": "gpt-4", "orchestrator": "claude-3-5-sonnet-20241022"}

        store.set_all_preferences("project1", prefs)

        assert store.get_model_for_step("project1", "router") == "gpt-4"
        assert store.get_model_for_step("project1", "orchestrator") == "claude-3-5-sonnet-20241022"

    def test_set_all_preferences_overwrites_existing(self):
        """Test setting all preferences overwrites existing."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "old-model")

        prefs = {"orchestrator": "new-model"}
        store.set_all_preferences("project1", prefs)

        # Old preference should be gone
        assert store.get_model_for_step("project1", "router") is None
        assert store.get_model_for_step("project1", "orchestrator") == "new-model"

    def test_set_all_preferences_stores_copy(self):
        """Test set_all_preferences stores a copy."""
        store = ModelPreferencesStore()
        prefs = {"router": "gpt-4"}

        store.set_all_preferences("project1", prefs)
        prefs["router"] = "modified"

        # Stored value should be unchanged
        assert store.get_model_for_step("project1", "router") == "gpt-4"


class TestClearPreferences:
    """Test suite for clearing preferences."""

    def test_clear_preferences_removes_project(self):
        """Test clearing preferences removes project."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")

        store.clear_preferences("project1")

        assert store.get_model_for_step("project1", "router") is None
        assert not store.has_preferences("project1")

    def test_clear_preferences_for_nonexistent_project(self):
        """Test clearing preferences for nonexistent project doesn't error."""
        store = ModelPreferencesStore()

        # Should not raise exception
        store.clear_preferences("nonexistent")

    def test_clear_preferences_doesnt_affect_other_projects(self):
        """Test clearing one project doesn't affect others."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")
        store.set_model_for_step("project2", "router", "claude-3-5-sonnet-20241022")

        store.clear_preferences("project1")

        assert store.get_model_for_step("project2", "router") == "claude-3-5-sonnet-20241022"


class TestHasPreferences:
    """Test suite for checking if project has preferences."""

    def test_has_preferences_returns_true_when_configured(self):
        """Test returns True when project has preferences."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")

        assert store.has_preferences("project1") is True

    def test_has_preferences_returns_false_when_not_configured(self):
        """Test returns False when project has no preferences."""
        store = ModelPreferencesStore()

        assert store.has_preferences("project1") is False

    def test_has_preferences_returns_false_after_clear(self):
        """Test returns False after clearing preferences."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")
        store.clear_preferences("project1")

        assert store.has_preferences("project1") is False


class TestGetPreferenceCount:
    """Test suite for getting preference count."""

    def test_get_preference_count_returns_correct_count(self):
        """Test returns correct count of configured steps."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")
        store.set_model_for_step("project1", "orchestrator", "claude-3-5-sonnet-20241022")

        count = store.get_preference_count("project1")
        assert count == 2

    def test_get_preference_count_returns_zero_for_nonexistent(self):
        """Test returns 0 for nonexistent project."""
        store = ModelPreferencesStore()

        count = store.get_preference_count("nonexistent")
        assert count == 0

    def test_get_preference_count_updates_after_changes(self):
        """Test count updates after adding/removing preferences."""
        store = ModelPreferencesStore()

        assert store.get_preference_count("project1") == 0

        store.set_model_for_step("project1", "router", "gpt-4")
        assert store.get_preference_count("project1") == 1

        store.set_model_for_step("project1", "orchestrator", "claude-3-5-sonnet-20241022")
        assert store.get_preference_count("project1") == 2

        store.clear_preferences("project1")
        assert store.get_preference_count("project1") == 0


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_multiple_stores_independent(self):
        """Test multiple store instances are independent."""
        store1 = ModelPreferencesStore()
        store2 = ModelPreferencesStore()

        store1.set_model_for_step("project1", "router", "gpt-4")
        store2.set_model_for_step("project1", "router", "claude-3-5-sonnet-20241022")

        assert store1.get_model_for_step("project1", "router") == "gpt-4"
        assert store2.get_model_for_step("project1", "router") == "claude-3-5-sonnet-20241022"

    def test_set_model_with_empty_string(self):
        """Test setting model with empty string."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "")

        assert store.get_model_for_step("project1", "router") == ""

    def test_set_all_preferences_with_empty_dict(self):
        """Test setting all preferences with empty dict."""
        store = ModelPreferencesStore()
        store.set_model_for_step("project1", "router", "gpt-4")

        store.set_all_preferences("project1", {})

        assert store.get_preference_count("project1") == 0

    def test_infrastructure_steps_constant(self):
        """Test INFRASTRUCTURE_STEPS is a class constant."""
        store1 = ModelPreferencesStore()
        store2 = ModelPreferencesStore()

        assert store1.INFRASTRUCTURE_STEPS is store2.INFRASTRUCTURE_STEPS

    def test_workflow_set_get_clear(self):
        """Test typical workflow of setting, getting, and clearing."""
        store = ModelPreferencesStore()

        # Set preferences
        store.set_model_for_step("project1", "router", "gpt-4")
        store.set_model_for_step("project1", "orchestrator", "claude-3-5-sonnet-20241022")

        # Get preferences
        assert store.has_preferences("project1")
        assert store.get_preference_count("project1") == 2
        prefs = store.get_all_preferences("project1")
        assert len(prefs) == 2

        # Clear preferences
        store.clear_preferences("project1")
        assert not store.has_preferences("project1")
        assert store.get_preference_count("project1") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
