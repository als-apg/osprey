"""
Model Preferences Manager for PyQt GUI

This module provides in-memory management of per-step model preferences for
discovered projects. Preferences persist only during program execution and
are not saved to disk.
"""

from typing import Dict, List, Optional

from osprey.interfaces.pyqt.gui_utils import load_config_safe


class ModelPreferencesStore:
    """Manages per-step model preferences for projects during runtime."""

    # Infrastructure steps that can have custom models
    INFRASTRUCTURE_STEPS = [
        'orchestrator',
        'router',
        'classifier',
        'task_extractor',
        'clarifier',
        'responder'
    ]

    def __init__(self):
        """Initialize the model preferences manager."""
        # Structure: {project_name: {step_name: model_id}}
        self._preferences: Dict[str, Dict[str, str]] = {}

        # Cache of available models per provider (fallback static list)
        self._available_models: Dict[str, List[str]] = {
            'openai': [
                'gpt-4-turbo-preview',
                'gpt-4',
                'gpt-4-32k',
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-16k'
            ],
            'anthropic': [
                'claude-3-5-sonnet-20241022',
                'claude-3-5-haiku-20241022',
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307',
                'claude-2.1',
                'claude-2.0',
                'claude-instant-1.2'
            ],
            'azure': [
                'gpt-4-turbo',
                'gpt-4',
                'gpt-35-turbo',
                'gpt-35-turbo-16k'
            ],
            'ollama': [
                'llama2',
                'llama2:13b',
                'llama2:70b',
                'llama3.1:8b',
                'mistral',
                'mixtral',
                'codellama',
                'phi'
            ],
            'google': [
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-pro'
            ],
            'cborg': [
                'anthropic/claude-3-5-sonnet-20241022',
                'anthropic/claude-3-5-haiku-20241022',
                'openai/gpt-4-turbo-preview',
                'openai/gpt-4o'
            ],
            'argo': [
                'anthropic/claude-3-5-sonnet-20241022',
                'anthropic/claude-3-5-haiku-20241022',
                'openai/gpt-4-turbo-preview',
                'openai/gpt-4o',
                'openai/gpt-4o-mini'
            ]
        }

        # Cache for dynamically discovered models
        self._dynamic_model_cache: Dict[str, List[str]] = {}

    def get_provider_from_config(self, config_path: str) -> Optional[str]:
        """
        Extract LLM provider from a project's config file.

        Args:
            config_path: Path to the project's config.yml file

        Returns:
            Provider name (e.g., 'openai', 'anthropic') or None if not found
        """
        config = load_config_safe(config_path)
        if not config:
            return None

        try:

            # Try different config structures (in priority order)

            # 1. Check models section first (most specific)
            if 'models' in config:
                models = config['models']
                if models and isinstance(models, dict):
                    # Get provider from first model config
                    first_model = next(iter(models.values()))
                    if isinstance(first_model, dict) and 'provider' in first_model:
                        return first_model['provider']

            # 2. Legacy structure: llm.provider
            if 'llm' in config and 'provider' in config['llm']:
                return config['llm']['provider']

            # 3. New structure: api.providers.{provider_name} (least specific, fallback only)
            if 'api' in config and 'providers' in config['api']:
                providers = config['api']['providers']
                if providers:
                    # Return first provider
                    return list(providers.keys())[0]

            return None
        except Exception as e:
            print(f"Error extracting provider from config: {e}")
            return None

    def discover_models_from_provider(self, provider: str, config_path: str) -> List[str]:
        """
        Dynamically discover available models from the provider.

        This queries the provider's API to get a list of available models.
        Falls back to static list if discovery fails.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'argo')
            config_path: Path to project config for API credentials

        Returns:
            List of available model IDs
        """
        # Check cache first
        cache_key = f"{provider}:{config_path}"
        if cache_key in self._dynamic_model_cache:
            return self._dynamic_model_cache[cache_key]

        discovered_models = []

        try:
            # Try to discover models dynamically
            if provider.lower() in ['openai', 'azure']:
                discovered_models = self._discover_openai_models(config_path)
            elif provider.lower() == 'anthropic':
                discovered_models = self._discover_anthropic_models(config_path)
            elif provider.lower() == 'ollama':
                discovered_models = self._discover_ollama_models(config_path)
            elif provider.lower() in ['cborg', 'argo']:
                discovered_models = self._discover_openai_compatible_models(config_path, provider.lower())

            if discovered_models:
                # Cache the discovered models
                self._dynamic_model_cache[cache_key] = discovered_models
                return discovered_models

        except Exception as e:
            print(f"Failed to discover models from {provider}: {e}")

        # Fall back to static list (don't call get_available_models to avoid recursion)
        return self._available_models.get(provider.lower(), [])

    def _resolve_env_var(self, value: str) -> str:
        """
        Resolve environment variables in a string.

        Supports ${VAR_NAME} and $VAR_NAME syntax.

        Args:
            value: String that may contain environment variable references

        Returns:
            String with environment variables resolved
        """
        if not value or not isinstance(value, str):
            return value

        import os
        import re

        def replace_env_var(match):
            # Pattern matches: ${VAR_NAME} or $VAR_NAME
            if match.group(1):  # ${VAR_NAME}
                var_name = match.group(1)
            else:  # $VAR_NAME
                var_name = match.group(2)

            return os.environ.get(var_name, match.group(0))

        # Pattern matches ${VAR_NAME} or $VAR_NAME
        pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
        return re.sub(pattern, replace_env_var, value)

    def _discover_models_generic(self, config_path: str, provider_name: str, discovery_func) -> List[str]:
        """
        Generic model discovery method with provider-specific callback.

        Args:
            config_path: Path to project config file
            provider_name: Provider name (e.g., 'openai', 'ollama')
            discovery_func: Callable that takes (config, api_config) and returns list of model IDs

        Returns:
            List of discovered model IDs, or empty list on failure
        """
        try:
            # Load config
            config = load_config_safe(config_path)
            if not config:
                return []

            # Get provider-specific config
            api_config = config.get('api', {}).get('providers', {}).get(provider_name, {})
            if not api_config:
                return []

            # Call provider-specific discovery function
            model_ids = discovery_func(config, api_config)
            return sorted(model_ids) if model_ids else []

        except Exception as e:
            print(f"{provider_name.title()} model discovery failed: {e}")
            return []

    def _discover_openai_models(self, config_path: str) -> List[str]:
        """Discover models from OpenAI API."""
        def discover(config, api_config):
            import openai

            api_key = self._resolve_env_var(api_config.get('api_key', ''))
            if not api_key:
                return []

            client = openai.OpenAI(api_key=api_key)
            models = client.models.list()

            # Filter for chat models
            return [
                m.id for m in models.data
                if 'gpt' in m.id.lower() and not m.id.startswith('ft:')
            ]

        return self._discover_models_generic(config_path, 'openai', discover)

    def _discover_anthropic_models(self, config_path: str) -> List[str]:
        """Discover models from Anthropic API."""
        # Anthropic doesn't have a models list API, use static list
        return []

    def _discover_ollama_models(self, config_path: str) -> List[str]:
        """Discover models from Ollama API."""
        def discover(config, api_config):
            import requests

            base_url = self._resolve_env_var(api_config.get('base_url', 'http://localhost:11434'))
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()

            data = response.json()
            return [m['name'] for m in data.get('models', [])]

        return self._discover_models_generic(config_path, 'ollama', discover)

    def _discover_openai_compatible_models(self, config_path: str, provider_name: str) -> List[str]:
        """
        Discover models from OpenAI-compatible APIs (CBORG, Argo).

        Args:
            config_path: Path to project config file
            provider_name: Specific provider to query ('cborg' or 'argo')
        """
        def discover(config, api_config):
            import openai

            api_key = self._resolve_env_var(api_config.get('api_key', ''))
            base_url = self._resolve_env_var(api_config.get('base_url', ''))

            # Skip if credentials missing or look like placeholders
            if not api_key or not base_url:
                return []
            if 'your-' in api_key.lower() or 'placeholder' in api_key.lower():
                return []

            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                models = client.models.list()
                return [m.id for m in models.data]
            except Exception:
                # Silently fail on API errors
                return []

        return self._discover_models_generic(config_path, provider_name, discover)

    def get_available_models(self, provider: str, config_path: Optional[str] = None, use_dynamic: bool = True) -> List[str]:
        """
        Get list of available models for a provider.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            config_path: Optional path to config file for dynamic discovery
            use_dynamic: Whether to attempt dynamic model discovery

        Returns:
            List of model IDs available for this provider
        """
        # Try dynamic discovery if enabled and config path provided
        if use_dynamic and config_path:
            dynamic_models = self.discover_models_from_provider(provider, config_path)
            if dynamic_models:
                return dynamic_models

        # Fall back to static list
        return self._available_models.get(provider.lower(), [])

    def set_model_for_step(self, project_name: str, step: str, model_id: str):
        """
        Set the model preference for a specific step in a project.

        Args:
            project_name: Name of the project
            step: Infrastructure step name
            model_id: Model identifier to use for this step
        """
        if project_name not in self._preferences:
            self._preferences[project_name] = {}

        self._preferences[project_name][step] = model_id

    def get_model_for_step(self, project_name: str, step: str) -> Optional[str]:
        """
        Get the configured model for a specific step in a project.

        Args:
            project_name: Name of the project
            step: Infrastructure step name

        Returns:
            Model ID if configured, None otherwise
        """
        if project_name not in self._preferences:
            return None

        return self._preferences[project_name].get(step)

    def get_all_preferences(self, project_name: str) -> Dict[str, str]:
        """
        Get all model preferences for a project.

        Args:
            project_name: Name of the project

        Returns:
            Dictionary mapping step names to model IDs
        """
        return self._preferences.get(project_name, {}).copy()

    def set_all_preferences(self, project_name: str, preferences: Dict[str, str]):
        """
        Set all model preferences for a project at once.

        Args:
            project_name: Name of the project
            preferences: Dictionary mapping step names to model IDs
        """
        self._preferences[project_name] = preferences.copy()

    def clear_preferences(self, project_name: str):
        """
        Clear all model preferences for a project.

        Args:
            project_name: Name of the project
        """
        if project_name in self._preferences:
            del self._preferences[project_name]

    def has_preferences(self, project_name: str) -> bool:
        """
        Check if a project has any model preferences configured.

        Args:
            project_name: Name of the project

        Returns:
            True if project has preferences, False otherwise
        """
        return project_name in self._preferences and bool(self._preferences[project_name])

    def get_preference_count(self, project_name: str) -> int:
        """
        Get the number of configured steps for a project.

        Args:
            project_name: Name of the project

        Returns:
            Number of steps with configured models
        """
        if project_name not in self._preferences:
            return 0
        return len(self._preferences[project_name])
