"""Anthropic Provider Adapter Implementation."""

from typing import Any

import anthropic
import httpx
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider as PydanticAnthropicProvider

from .base import BaseProvider


class AnthropicProviderAdapter(BaseProvider):
    """Anthropic AI provider implementation."""

    # Metadata (single source of truth)
    name = "anthropic"
    description = "Anthropic (Claude models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "claude-haiku-4-5-20251001"
    health_check_model_id = "claude-haiku-4-5-20251001"
    available_models = [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001"
    ]

    # API key acquisition information
    api_key_url = "https://console.anthropic.com/"
    api_key_instructions = [
        "Sign up or log in with your account",
        "Navigate to 'API Keys' in the settings",
        "Click 'Create Key' and name your key",
        "Copy the key (shown only once!)"
    ]
    api_key_note = None

    def create_model(
        self,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float | None,
        http_client: httpx.AsyncClient | None
    ) -> AnthropicModel:
        """Create Anthropic model instance for PydanticAI."""
        provider = PydanticAnthropicProvider(
            api_key=api_key,
            http_client=http_client
        )
        return AnthropicModel(
            model_name=model_id,
            provider=provider
        )

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
        **kwargs
    ) -> str | list | Any:
        """Execute Anthropic chat completion with extended thinking support."""
        # Get http_client if provided (for proxy support)
        http_client = kwargs.get("http_client")

        client = anthropic.Anthropic(
            api_key=api_key,
            http_client=http_client,
        )

        # Handle structured output if requested
        if output_format is not None:
            import json

            # Check if model supports native structured outputs (Sonnet 4.5, Opus 4.1)
            # Native structured outputs are in beta and only available for specific models
            supports_native_structured_outputs = (
                "claude-sonnet-4" in model_id or
                "claude-opus-4" in model_id
            )

            if supports_native_structured_outputs:
                # Use Anthropic's native structured outputs (beta)
                schema = output_format.model_json_schema()

                request_params = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": message}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": output_format.__name__,
                            "schema": schema
                        }
                    }
                }

                response = client.messages.create(**request_params)

                # Extract text content (should be valid JSON)
                text_parts = [
                    block.text for block in response.content
                    if isinstance(block, anthropic.types.TextBlock)
                ]
                response_text = "\n".join(text_parts).strip()

            else:
                # Fallback: Prompt-based structured output for models without native support (e.g., Haiku)
                schema = output_format.model_json_schema()
                structured_message = f"""{message}

You must respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with the JSON object, no additional text or markdown formatting."""

                request_params = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": structured_message}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

                response = client.messages.create(**request_params)

                # Extract text content
                text_parts = [
                    block.text for block in response.content
                    if isinstance(block, anthropic.types.TextBlock)
                ]
                response_text = "\n".join(text_parts).strip()

                # Clean up markdown code blocks if present (only for prompt-based approach)
                if response_text.startswith("```json"):
                    response_text = response_text[7:]  # Remove ```json
                elif response_text.startswith("```"):
                    response_text = response_text[3:]  # Remove ```

                if response_text.endswith("```"):
                    response_text = response_text[:-3]  # Remove trailing ```

                response_text = response_text.strip()

            # Parse and validate against the Pydantic model (common for both approaches)
            try:
                result = output_format.model_validate_json(response_text)

                # Handle TypedDict conversion
                is_typed_dict_output = kwargs.get("is_typed_dict_output", False)
                if is_typed_dict_output and hasattr(result, 'model_dump'):
                    return result.model_dump()
                return result
            except Exception as e:
                raise ValueError(f"Failed to parse structured output from Anthropic: {e}\nResponse: {response_text[:200]}") from e

        # Regular text completion (no structured output)
        request_params = {
            "model": model_id,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        # Add extended thinking if enabled
        enable_thinking = kwargs.get("enable_thinking", False)
        budget_tokens = kwargs.get("budget_tokens")

        if enable_thinking and budget_tokens is not None:
            if budget_tokens >= max_tokens:
                raise ValueError("budget_tokens must be less than max_tokens")
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens
            }

        response = client.messages.create(**request_params)

        if enable_thinking and "thinking" in request_params:
            return response.content  # Returns List[ContentBlock]
        else:
            # Concatenate text from all TextBlock instances
            text_parts = [
                block.text for block in response.content
                if isinstance(block, anthropic.types.TextBlock)
            ]
            return "\n".join(text_parts)

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None
    ) -> tuple[bool, str]:
        """Check Anthropic API health with minimal test call.

        Makes a minimal API call (~10 tokens, ~$0.0001) to verify the API key works.
        An API key that can't handle a penny's worth of tokens is a critical issue.
        """
        if not api_key:
            return False, "API key not set"

        # Check for placeholder/template values
        if api_key.startswith("${") or api_key.startswith("sk-ant-xxx"):
            return False, "API key not configured (placeholder value detected)"

        # Use provided model or cheapest default from metadata
        test_model = model_id or self.health_check_model_id

        try:
            client = anthropic.Anthropic(api_key=api_key)

            # Minimal test: 1 token in, 1 token out (~$0.0001 cost)
            _ = client.messages.create(
                model=test_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
                timeout=timeout
            )

            # If we got here, API key works
            return True, "API accessible and authenticated"

        except anthropic.AuthenticationError:
            return False, "Authentication failed (invalid API key)"
        except anthropic.PermissionDeniedError:
            return False, "Permission denied (check API key permissions)"
        except anthropic.RateLimitError:
            # Rate limited = API key works, just hitting limits
            return True, "API key valid (rate limited, but functional)"
        except anthropic.NotFoundError:
            # Model not found - API key works but model doesn't exist
            return False, f"Model '{test_model}' not found (check model ID)"
        except anthropic.BadRequestError as e:
            return False, f"Bad request: {str(e)[:50]}"
        except anthropic.APIConnectionError as e:
            return False, f"Connection failed: {str(e)[:50]}"
        except anthropic.APITimeoutError:
            return False, "Request timeout"
        except anthropic.InternalServerError as e:
            return False, f"Anthropic server error: {str(e)[:50]}"
        except anthropic.APIError as e:
            return False, f"API error: {str(e)[:50]}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)[:50]}"

