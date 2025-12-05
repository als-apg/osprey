"""OpenAI Provider Adapter Implementation."""

import logging
from typing import Any

import httpx
import openai
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from .base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProviderAdapter(BaseProvider):
    """OpenAI provider implementation."""

    # Metadata (single source of truth)
    name = "openai"
    description = "OpenAI (GPT models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = "https://api.openai.com/v1"
    default_model_id = "gpt-5"  # GPT-5 for general use
    health_check_model_id = "gpt-5-nano"  # Cheapest GPT-5 model for health checks
    available_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

    # API key acquisition information
    api_key_url = "https://platform.openai.com/api-keys"
    api_key_instructions = [
        "Sign up or log in to your OpenAI account",
        "Add billing information if not already set up",
        "Click '+ Create new secret key'",
        "Name your key and copy it (shown only once!)",
    ]
    api_key_note = None

    def create_model(
        self,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float | None,
        http_client: httpx.AsyncClient | None,
    ) -> OpenAIModel:
        """Create OpenAI model instance for PydanticAI."""
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
        api_key: str | None,
        base_url: str | None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking: dict | None = None,
        system_prompt: str | None = None,
        output_format: Any | None = None,
        **kwargs,
    ) -> str | Any:
        """Execute OpenAI chat completion."""
        # Check for thinking parameters (not supported by OpenAI)
        enable_thinking = kwargs.get("enable_thinking", False)
        budget_tokens = kwargs.get("budget_tokens")

        if enable_thinking or budget_tokens is not None:
            logger.warning("enable_thinking and budget_tokens are not used for OpenAI provider.")

        # Get http_client if provided
        http_client = kwargs.get("http_client")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        # Handle typed dict output flag
        is_typed_dict_output = kwargs.get("is_typed_dict_output", False)

        # Try new API (max_completion_tokens) first, fall back to old API (max_tokens)
        # This handles GPT-5, o1-series, and future models automatically
        try:
            if output_format is not None:
                # Use structured outputs with Pydantic model
                response = client.beta.chat.completions.parse(
                    model=model_id,
                    messages=[{"role": "user", "content": message}],
                    max_completion_tokens=max_tokens,
                    response_format=output_format,
                )
            else:
                # Regular text completion
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": message}],
                    max_completion_tokens=max_tokens,
                )
        except openai.BadRequestError as e:
            # Fall back to old API if max_completion_tokens not supported
            error_str = str(e).lower()
            if (
                "max_tokens" in error_str
                or "unsupported parameter" in error_str
                or "max_completion_tokens" in error_str
            ):
                if output_format is not None:
                    response = client.beta.chat.completions.parse(
                        model=model_id,
                        messages=[{"role": "user", "content": message}],
                        max_tokens=max_tokens,
                        response_format=output_format,
                    )
                else:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": message}],
                        max_tokens=max_tokens,
                    )
            else:
                raise

        if not response.choices:
            raise ValueError("OpenAI API returned empty choices list")

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
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check OpenAI API health.

        If model_id provided: Makes minimal chat completion (~$0.0001)
        If no model_id: Tests /v1/models endpoint (free)
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

        # If model_id provided, test with minimal completion call
        if test_model:
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)

                # Try new API first (max_completion_tokens for GPT-5, o1 models)
                try:
                    response = client.chat.completions.create(
                        model=test_model,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_completion_tokens=50,
                        timeout=timeout,
                    )
                except openai.BadRequestError as e:
                    error_str = str(e).lower()
                    # Fall back to old API (max_tokens for GPT-4 models)
                    if "max_tokens" in error_str or "unsupported parameter" in error_str:
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

        # No model_id: just test /v1/models endpoint (free)
        try:
            import requests

            test_url = base_url.rstrip("/") + "/models"
            headers = {"Authorization": f"Bearer {api_key}"}

            response = requests.get(test_url, headers=headers, timeout=timeout)

            if response.status_code == 200:
                return True, "API accessible and authenticated"
            elif response.status_code == 401:
                return False, "Authentication failed (invalid API key?)"
            else:
                return False, f"API returned status {response.status_code}"

        except requests.Timeout:
            return False, "Connection timeout"
        except requests.RequestException as e:
            return False, f"Connection failed: {str(e)[:50]}"
        except Exception as e:
            return False, f"Health check failed: {str(e)[:50]}"
