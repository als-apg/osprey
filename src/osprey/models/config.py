"""Model and provider configuration helpers."""

from __future__ import annotations

from typing import Any


def get_model_config(model_name: str, config_path: str | None = None) -> dict[str, Any]:
    """Get model configuration with automatic context detection.

    Works both inside and outside framework contexts.
    All models are configured at the top level in the 'models' section.

    Args:
        model_name: Name of the model (e.g., 'orchestrator', 'classifier', 'time_parsing',
                   'response', 'approval', 'memory', 'task_extraction', 'python_code_generator')
        config_path: Optional explicit path to configuration file for multi-project workflows

    Returns:
        Dictionary with model configuration containing provider, model_id, and optional settings

    Examples:
        Default config (searches current directory):
            >>> get_model_config("orchestrator")
            {'provider': 'anthropic', 'model_id': 'claude-haiku-4-5-20251001', ...}

        Multi-project workflow:
            >>> get_model_config("orchestrator", config_path="~/other-project/config.yml")
            {'provider': 'openai', 'model_id': 'gpt-4o', ...}

    Configuration format (config.yml):
        models:
          orchestrator:
            provider: anthropic
            model_id: claude-haiku-4-5-20251001
          classifier:
            provider: anthropic
            model_id: claude-haiku-4-5-20251001
    """
    from osprey.utils.config import _get_configurable

    configurable = _get_configurable(config_path)
    model_configs = configurable.get("model_configs", {})
    return model_configs.get(model_name, {})


def get_provider_config(provider_name: str, config_path: str | None = None) -> dict[str, Any]:
    """Get API provider configuration with automatic context detection.

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')
        config_path: Optional explicit path to configuration file

    Returns:
        Dictionary with provider configuration
    """
    from osprey.utils.config import _get_configurable

    configurable = _get_configurable(config_path)
    provider_configs = configurable.get("provider_configs", {})
    return provider_configs.get(provider_name, {})
