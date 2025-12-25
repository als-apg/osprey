"""Lightweight LLM client for GUI routing - no singleton dependencies.

This module provides a simple, direct LLM client for routing decisions that
operates independently of the framework's global registry singleton. This
eliminates the "Registry not initialized" error that occurs when routing
happens before project selection.

The client reads the 'classifier' model configuration from gui_config.yml,
which the user has explicitly configured for their facility's LLM provider.

Key Features:
- Direct API calls to Anthropic, OpenAI, Ollama, Argo, CBORG, Stanford, Google
- No dependency on global registry singleton
- Uses user's explicitly configured provider from gui_config.yml
- Easy to test and maintain

Configuration:
The user must configure the 'classifier' model in gui_config.yml:

    models:
      classifier:
        provider: argo  # User's facility provider
        model_id: gpt5

    api:
      providers:
        argo:
          api_key: ${ARGO_API_KEY}
          base_url: https://argo-bridge.cels.anl.gov

Usage:
    # From GUI config (reads user's configured 'classifier' model)
    client = SimpleLLMClient.from_gui_config()

    # Make LLM call
    response = client.call("Your prompt here", max_tokens=500)
"""

import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleLLMClient:
    """
    Lightweight LLM client for GUI routing operations.

    This client provides direct LLM API access without framework singleton
    dependencies. It reads the user's explicitly configured provider from
    gui_config.yml, ensuring each facility uses their intended LLM service.

    Supported providers:
    - Anthropic (Claude models)
    - OpenAI (GPT models)
    - Ollama (local models)
    - Argo (ANL service)
    - CBORG (LBNL service)
    - Stanford (Stanford service)
    - Google (Gemini models)
    """

    def __init__(
        self,
        provider: str,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """Initialize LLM client with explicit configuration.

        Args:
            provider: Provider name ('anthropic', 'openai', 'ollama', 'argo', etc.)
            model_id: Model identifier (e.g., 'claude-3-sonnet-20240229', 'gpt5')
            api_key: API key (not needed for Ollama)
            base_url: Base URL (required for Ollama and some providers)

        Raises:
            ValueError: If required parameters are missing
        """
        self.provider = provider.lower()
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self._validate()

        logger.info(
            f"Initialized SimpleLLMClient with user-configured provider: "
            f"{self.provider}/{self.model_id}"
        )

    @classmethod
    def from_gui_config(cls, gui_config_path: Optional[str] = None) -> 'SimpleLLMClient':
        """Create client from GUI configuration file.

        This method reads the user's explicitly configured 'classifier' model
        from gui_config.yml. The user must have configured their facility's
        preferred LLM provider in this file.

        Args:
            gui_config_path: Path to gui_config.yml (default: auto-detect in pyqt dir)

        Returns:
            SimpleLLMClient instance configured with user's chosen provider

        Raises:
            FileNotFoundError: If gui_config.yml not found
            ValueError: If 'classifier' model not configured properly
        """
        import yaml

        # Auto-detect GUI config path if not provided
        if gui_config_path is None:
            # GUI config is in the pyqt directory
            pyqt_dir = Path(__file__).parent
            gui_config_path = pyqt_dir / 'gui_config.yml'

            if not gui_config_path.exists():
                raise FileNotFoundError(
                    f"GUI configuration file not found at {gui_config_path}.\n"
                    f"Please ensure gui_config.yml exists in the pyqt directory with "
                    f"a 'classifier' model configured for your facility's LLM provider."
                )

        logger.debug(f"Loading GUI config from: {gui_config_path}")

        # Load config file
        with open(gui_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get classifier model configuration (user's explicit choice)
        models = config.get('models', {})
        classifier_config = models.get('classifier', {})

        if not classifier_config:
            raise ValueError(
                f"'classifier' model not configured in {gui_config_path}.\n"
                f"Please add a 'classifier' model configuration with your facility's "
                f"LLM provider. Example:\n\n"
                f"models:\n"
                f"  classifier:\n"
                f"    provider: argo  # Your facility's provider\n"
                f"    model_id: gpt5\n"
            )

        provider = classifier_config.get('provider')
        model_id = classifier_config.get('model_id')

        if not provider or not model_id:
            raise ValueError(
                f"'classifier' model in {gui_config_path} is missing required fields.\n"
                f"Both 'provider' and 'model_id' must be specified."
            )

        # Get provider configuration
        providers = config.get('api', {}).get('providers', {})
        provider_config = providers.get(provider, {})

        if not provider_config:
            raise ValueError(
                f"Provider '{provider}' not configured in {gui_config_path}.\n"
                f"Please add provider configuration under api.providers.{provider}"
            )

        # Resolve environment variables in API key
        api_key = provider_config.get('api_key', '')
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]  # Extract VAR_NAME from ${VAR_NAME}
            api_key = os.getenv(env_var)
            if not api_key:
                logger.warning(
                    f"Environment variable {env_var} not set for {provider}. "
                    f"API calls may fail. Please set {env_var} in your environment."
                )

        base_url = provider_config.get('base_url')

        logger.info(
            f"Using user-configured provider from gui_config.yml: {provider}/{model_id}"
        )

        return cls(
            provider=provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url
        )

    def _validate(self):
        """Validate configuration.

        Raises:
            ValueError: If required parameters are missing
        """
        if not self.provider:
            raise ValueError("Provider is required")
        if not self.model_id:
            raise ValueError("Model ID is required")
        if self.provider not in ['ollama'] and not self.api_key:
            logger.warning(
                f"No API key provided for {self.provider}. "
                f"This may cause authentication errors."
            )
        if self.provider == 'ollama' and not self.base_url:
            raise ValueError("Base URL required for Ollama")

    def call(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0
    ) -> str:
        """Call LLM with prompt and return response.

        Args:
            prompt: Input prompt for the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            LLM response text

        Raises:
            ValueError: If provider is unsupported
            Exception: If API call fails
        """
        logger.debug(
            f"Calling {self.provider}/{self.model_id} "
            f"(max_tokens={max_tokens}, temperature={temperature})"
        )

        if self.provider == 'anthropic':
            return self._call_anthropic(prompt, max_tokens, temperature)
        elif self.provider in ['openai', 'argo', 'cborg', 'stanford']:
            # These all use OpenAI-compatible API
            return self._call_openai_compatible(prompt, max_tokens, temperature)
        elif self.provider == 'ollama':
            return self._call_ollama(prompt, max_tokens, temperature)
        elif self.provider == 'google':
            return self._call_google(prompt, max_tokens, temperature)
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: anthropic, openai, argo, cborg, stanford, ollama, google"
            )

    def _call_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Anthropic API directly."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            if message.content:
                response_text = message.content[0].text
                logger.debug(f"Anthropic response: {len(response_text)} chars")
                return response_text

            logger.warning("Anthropic returned empty content")
            return ""

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

    def _call_openai_compatible(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call OpenAI-compatible API (OpenAI, Argo, CBORG, Stanford).

        Note: Argo is the ANL (Argonne National Laboratory) service provider.
        """
        try:
            import openai

            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            logger.debug(f"{self.provider} response: {len(response_text)} chars")
            return response_text

        except Exception as e:
            logger.error(f"{self.provider} API call failed: {e}")
            raise

    def _call_ollama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Ollama API directly."""
        try:
            import ollama

            client = ollama.Client(host=self.base_url)
            response = client.chat(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )

            response_text = response['message']['content']
            logger.debug(f"Ollama response: {len(response_text)} chars")
            return response_text

        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise

    def _call_google(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Google Gemini API directly."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_id)

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )

            response_text = response.text
            logger.debug(f"Google response: {len(response_text)} chars")
            return response_text

        except Exception as e:
            logger.error(f"Google API call failed: {e}")
            raise
