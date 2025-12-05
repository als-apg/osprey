"""Stanford AI Playground Provider Adapter Implementation.

Stanford AI Playground is an OpenAI-compatible API proxy that provides access
to multiple LLM providers (Anthropic, OpenAI, Google, DeepSeek, etc.) through
a unified endpoint at https://aiapi-prod.stanford.edu/v1.

This adapter extends the framework's BaseProvider to integrate Stanford's API
with the Otter agent system, enabling model selection across various providers
while using a single API key and endpoint.
"""

import logging
from typing import Any, Optional, Union

import httpx
import openai
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from osprey.models.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class StanfordProviderAdapter(BaseProvider):
    """Stanford AI Playground provider adapter (OpenAI-compatible)."""

    # Metadata (single source of truth)
    name = "stanford"
    description = "Stanford AI Playground (multi-provider proxy)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = "https://aiapi-prod.stanford.edu/v1"
    default_model_id = "gpt-4o"
    health_check_model_id = "gpt-4.omini"  # Cheapest OpenAI model for health checks
    available_models = [
        # Anthropic Claude models
        "claude-3-7-sonnet",
        # OpenAI models
        "gpt-4o",
        "gpt-4.omini",
        "o3-mini",
        # Google models
        "gemini-2.0-flash-001",
        # DeepSeek models
        "deepseek-r1",
    ]

    # API key acquisition help
    api_key_url = "https://uit.stanford.edu/service/ai-api-gateway"
    api_key_instructions = [
        "Requires Stanford University affiliation",
        "Go to 'Get Started' -> 'Request the creation of a new API key'",
        "Log in with your Stanford credentials and complete the form",
        "Once approved, copy the API key from the notification email",
    ]
    api_key_note = "Access restricted to Stanford community"

    def create_model(
        self,
        model_id: str,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: Optional[float],
        http_client: Optional[httpx.AsyncClient],
    ) -> OpenAIModel:
        """Create Stanford AI model instance for PydanticAI.

        Args:
            model_id: Model identifier (e.g., 'claude-3-7-sonnet', 'gpt-4o')
            api_key: Stanford AI API key
            base_url: Stanford AI endpoint (default: https://aiapi-prod.stanford.edu/v1)
            timeout: Request timeout in seconds
            http_client: Optional async HTTP client for custom configurations

        Returns:
            OpenAIModel configured for Stanford AI Playground
        """
        if not base_url:
            base_url = self.default_base_url

        if http_client:
            client_args = {"api_key": api_key, "http_client": http_client}
            if base_url:
                client_args["base_url"] = base_url
            openai_client = openai.AsyncOpenAI(**client_args)
        else:
            effective_timeout = timeout if timeout is not None else 60.0
            client_args = {"api_key": api_key, "timeout": effective_timeout}
            if base_url:
                client_args["base_url"] = base_url
            openai_client = openai.AsyncOpenAI(**client_args)

        model = OpenAIModel(
            model_name=model_id,
            provider=PydanticOpenAIProvider(openai_client=openai_client),
        )
        model.model_id = model_id
        return model

    def execute_completion(
        self,
        message: str,
        model_id: str,
        api_key: Optional[str],
        base_url: Optional[str],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        output_format: Optional[Any] = None,
        **kwargs,
    ) -> Union[str, Any]:
        """Execute Stanford AI chat completion.

        Args:
            message: User message content
            model_id: Model to use (e.g., 'claude-3-7-sonnet')
            api_key: Stanford AI API key
            base_url: Stanford AI endpoint
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)
            thinking: Optional thinking parameters (not used for Stanford)
            system_prompt: Optional system prompt
            output_format: Optional Pydantic model for structured output
            **kwargs: Additional arguments

        Returns:
            String response or parsed object if output_format provided
        """
        # Check for thinking parameters (not supported by most models via Stanford)
        enable_thinking = kwargs.get("enable_thinking", False)
        budget_tokens = kwargs.get("budget_tokens")

        if enable_thinking or budget_tokens is not None:
            logger.warning("enable_thinking and budget_tokens are not used for Stanford provider.")

        if not base_url:
            base_url = self.default_base_url

        # Get http_client if provided
        http_client = kwargs.get("http_client")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        # Handle typed dict output flag
        is_typed_dict_output = kwargs.get("is_typed_dict_output", False)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        # Try new API first (max_completion_tokens for newer models)
        try:
            if output_format is not None:
                response = client.beta.chat.completions.parse(
                    model=model_id,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    response_format=output_format,
                )
            else:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
        except openai.BadRequestError as e:
            error_str = str(e).lower()
            # Fall back to old API (max_tokens for older/proxy models)
            # Handles: direct OpenAI errors, LiteLLM/Azure routing errors
            if (
                "max_tokens" in error_str
                or "unsupported parameter" in error_str
                or "unrecognized request argument" in error_str
            ):
                if output_format is not None:
                    response = client.beta.chat.completions.parse(
                        model=model_id,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format=output_format,
                    )
                else:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        max_tokens=max_tokens,
                    )
            else:
                raise

        if not response.choices:
            raise ValueError("Stanford AI API returned empty choices list")

        if output_format is not None:
            result = response.choices[0].message.parsed
            # Handle TypedDict conversion
            if is_typed_dict_output and hasattr(result, "model_dump"):
                return result.model_dump()
            return result
        else:
            return response.choices[0].message.content

    def check_health(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: float = 5.0,
        model_id: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Check Stanford AI API health.

        If model_id provided: Makes minimal chat completion
        If no model_id: Uses default health_check_model_id

        Args:
            api_key: Stanford AI API key
            base_url: Stanford AI endpoint
            timeout: Request timeout in seconds
            model_id: Optional specific model to test

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not api_key:
            return False, "API key not set"

        # Check for placeholder/template values
        if api_key.startswith("${") or "YOUR_API_KEY" in api_key.upper():
            return False, "API key not configured (placeholder value detected)"

        if not base_url:
            base_url = self.default_base_url

        # Use provided model or cheapest default from metadata
        test_model = model_id or self.health_check_model_id

        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)

            # Try new API first (max_completion_tokens for newer models)
            try:
                response = client.chat.completions.create(
                    model=test_model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_completion_tokens=50,
                    timeout=timeout,
                )
            except openai.BadRequestError as e:
                error_str = str(e).lower()
                # Fall back to old API (max_tokens for older/proxy models)
                # Handles: direct OpenAI errors, LiteLLM/Azure routing errors
                if (
                    "max_tokens" in error_str
                    or "unsupported parameter" in error_str
                    or "unrecognized request argument" in error_str
                ):
                    response = client.chat.completions.create(
                        model=test_model,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=50,
                        timeout=timeout,
                    )
                else:
                    raise

            if response.choices:
                return True, f"API accessible and model '{test_model}' working"
            else:
                return False, "API returned empty response"

        except openai.AuthenticationError:
            return False, "Authentication failed (invalid API key)"
        except openai.PermissionDeniedError:
            return False, "Permission denied (check API key permissions)"
        except openai.NotFoundError:
            return False, f"Model '{test_model}' not found"
        except openai.RateLimitError:
            return True, "API key valid (rate limited, but functional)"
        except openai.APITimeoutError:
            return False, "Request timeout"
        except openai.APIConnectionError as e:
            return False, f"Connection failed: {str(e)[:50]}"
        except openai.APIError as e:
            return False, f"API error: {str(e)[:50]}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)[:50]}"
