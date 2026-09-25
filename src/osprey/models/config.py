"""Model and provider configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
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
    from osprey_connectors.config import _get_configurable

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
    from osprey_connectors.config import _get_configurable

    configurable = _get_configurable(config_path)
    provider_configs = configurable.get("provider_configs", {})
    return provider_configs.get(provider_name, {})


def main_model_id(config: Mapping[str, Any], provider: str) -> str:
    """The deployment's main model on *provider*, for a job that names no model of its own.

    ``claude_code.default_model`` answers when the deployment's
    ``claude_code.provider`` is *provider* (or unset); otherwise the provider
    entry's ``api.providers.<provider>.default_model``. The main model is an id
    on the deployment's own provider, so a job routed to another provider takes
    that provider's default instead.

    Args:
        config: The loaded ``config.yml`` mapping.
        provider: The provider the job calls.

    Returns:
        The model id to send.

    Raises:
        ValueError: If neither key names a model.
    """
    claude_code = config.get("claude_code") or {}
    configured = claude_code.get("default_model")
    if configured and claude_code.get("provider") in (None, provider):
        return str(configured)
    entry = ((config.get("api") or {}).get("providers") or {}).get(provider) or {}
    default = entry.get("default_model")
    if default:
        return str(default)
    raise ValueError(
        f"No model named for provider '{provider}': set claude_code.default_model, "
        f"or api.providers.{provider}.default_model in config.yml."
    )
