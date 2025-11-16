"""Google Provider Adapter Implementation."""

from typing import Any

import httpx
from google import genai
from google.genai import types as genai_types
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from .base import BaseProvider


class GoogleProviderAdapter(BaseProvider):
    """Google AI (Gemini) provider implementation."""

    # Metadata (single source of truth)
    name = "google"
    description = "Google (Gemini models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "gemini-2.5-flash"  # Latest Flash for general use
    health_check_model_id = "gemini-2.5-flash-lite"  # Cheapest/fastest for health checks
    available_models = [
        "gemini-2.5-pro",           # Most capable Gemini 2.5 model
        "gemini-2.5-flash",         # Fast and capable, good balance
        "gemini-2.5-flash-lite"     # Fastest, most cost-effective
    ]

    # API key acquisition information
    api_key_url = "https://aistudio.google.com/app/apikey"
    api_key_instructions = [
        "Sign in with your Google account",
        "Click 'Create API key'",
        "Select a Google Cloud project or create a new one",
        "Copy the generated API key"
    ]
    api_key_note = None

    def create_model(
        self,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float | None,
        http_client: httpx.AsyncClient | None
    ) -> GeminiModel:
        """Create Google Gemini model instance for PydanticAI."""
        provider = GoogleGLAProvider(
            api_key=api_key,
            http_client=http_client
        )
        return GeminiModel(model_name=model_id, provider=provider)

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
    ) -> str | Any:
        """Execute Google Gemini chat completion with thinking support."""
        # Suppress Google library's INFO logs (AFC messages, etc.)
        import logging
        google_logger = logging.getLogger('google.genai')
        original_level = google_logger.level
        google_logger.setLevel(logging.WARNING)

        try:
            client = genai.Client(api_key=api_key)
        finally:
            # Restore original log level
            google_logger.setLevel(original_level)

        # Handle thinking configuration
        enable_thinking = kwargs.get("enable_thinking", False)
        budget_tokens = kwargs.get("budget_tokens", 0)

        if not enable_thinking:
            budget_tokens = 0

        if budget_tokens >= max_tokens:
            raise ValueError("budget_tokens must be less than max_tokens.")

        # Handle structured output if requested
        if output_format is not None:
            import json
            # Google doesn't have native structured outputs like OpenAI/Anthropic
            # Use prompt-based approach
            schema = output_format.model_json_schema()
            structured_message = f"""{message}

You must respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with the JSON object, no additional text or markdown formatting."""

            response = client.models.generate_content(
                model=model_id,
                contents=[structured_message],
                config=genai_types.GenerateContentConfig(
                    **({"thinking_config": genai_types.ThinkingConfig(thinking_budget=budget_tokens)}),
                    max_output_tokens=max_tokens
                )
            )

            # Handle case where response.text is None
            if response.text is None:
                if response.candidates and response.candidates[0].finish_reason:
                    finish_reason = str(response.candidates[0].finish_reason)
                    raise ValueError(
                        f"Google model produced no output for structured response. "
                        f"Finish reason: {finish_reason}. Try increasing max_tokens (current: {max_tokens})"
                    )
                raise ValueError("Google model produced no output text for structured response")

            response_text = response.text.strip()

            # Clean up markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            elif response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```

            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```

            response_text = response_text.strip()

            # Parse and validate against the Pydantic model
            try:
                result = output_format.model_validate_json(response_text)

                # Handle TypedDict conversion
                is_typed_dict_output = kwargs.get("is_typed_dict_output", False)
                if is_typed_dict_output and hasattr(result, 'model_dump'):
                    return result.model_dump()
                return result
            except Exception as e:
                raise ValueError(f"Failed to parse structured output from Google: {e}\nResponse: {response_text[:200]}") from e

        # Regular text completion (no structured output)
        response = client.models.generate_content(
            model=model_id,
            contents=[message],
            config=genai_types.GenerateContentConfig(
                **({"thinking_config": genai_types.ThinkingConfig(thinking_budget=budget_tokens)}),
                max_output_tokens=max_tokens
            )
        )

        # Handle case where response.text is None (all tokens used for thinking)
        if response.text is None:
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason = str(response.candidates[0].finish_reason)
                raise ValueError(
                    f"Google model produced no output. Finish reason: {finish_reason}. "
                    f"Try increasing max_tokens (current: {max_tokens})"
                )
            raise ValueError("Google model produced no output text")

        return response.text

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None
    ) -> tuple[bool, str]:
        """Check Google API health with minimal test call.

        Makes a minimal API call (~10 tokens, ~$0.0001) to verify the API key works.
        An API key that can't handle a penny's worth of tokens is a critical issue.
        """
        if not api_key:
            return False, "API key not set"

        # Check for placeholder/template values
        if api_key.startswith("${") or "YOUR_API_KEY" in api_key.upper():
            return False, "API key not configured (placeholder value detected)"

        # Use provided model or cheapest default from metadata
        test_model = model_id or self.health_check_model_id

        try:
            # Suppress Google library's INFO logs (AFC messages, etc.)
            import logging
            google_logger = logging.getLogger('google.genai')
            original_level = google_logger.level
            google_logger.setLevel(logging.WARNING)

            try:
                client = genai.Client(api_key=api_key)
            finally:
                # Restore original log level
                google_logger.setLevel(original_level)

            # Minimal test with sufficient tokens for thinking + response
            # Note: Google models may use tokens for internal thinking, so we need
            # a reasonable buffer even for simple queries (100 tokens = ~$0.0001)
            response = client.models.generate_content(
                model=test_model,
                contents=["Hi"],
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=100
                )
            )

            # Check if we got a response (response.text may be None if all tokens used for thinking)
            if response.text or response.candidates:
                return True, "API accessible and authenticated"
            else:
                return False, "API responded but produced no output"

        except Exception as e:
            error_msg = str(e).lower()

            # Check for common error types
            if "authentication" in error_msg or "api key" in error_msg or "invalid" in error_msg:
                return False, "Authentication failed (invalid API key)"
            elif "permission" in error_msg or "denied" in error_msg:
                return False, "Permission denied (check API key permissions)"
            elif "quota" in error_msg or "rate" in error_msg:
                # Rate limited = API key works, just hitting limits
                return True, "API key valid (rate limited, but functional)"
            elif "timeout" in error_msg:
                return False, "Request timeout"
            else:
                return False, f"API error: {str(e)[:50]}"

