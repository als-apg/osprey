"""AskSage Provider Adapter Implementation."""

import json
from typing import Any

import openai

from osprey.utils.logger import get_logger

from .base import BaseProvider
from .litellm_adapter import _clean_json_response

logger = get_logger("asksage")


class AskSageProviderAdapter(BaseProvider):
    """
    AskSage provider implementation - both OpenAI and native.
    Native API is at: https://app.swaggerhub.com/apis-docs/asksageinc/ask-sage_server_api
    OpenAI API is not documented at all. It seems to also accept native parameters, so we use it for now.
    """

    name = "asksage"
    description = "AskSage proxy (supports multiple models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "google-claude-45-haiku"
    health_check_model_id = "google-claude-45-haiku"
    available_models = [
        "google-gemini-20-flash",
        "google-gemini-2.5-pro",
        "google-claude-45-haiku",
        "google-claude-45-sonnet",
        "google-claude-45-opus",
        "gpt-5-mini",
        "gpt-5.2",
    ]
    api_key_url = "https://api.civ.asksage.ai/server/v1"
    api_key_instructions = [
        "Follow the instructions on the AskSage website to get an API key.",
    ]
    api_key_note = None
    _models_cache: list[str] | None = None

    def get_available_models(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        force_refresh: bool = False,
        timeout: float = 5.0,
    ) -> list[str]:
        """
        Dynamically fetch available models.

        Falls back to static defaults if the request fails.
        """
        if self._models_cache is not None and not force_refresh:
            return self._models_cache

        import requests

        if not api_key:
            return False, "API key not set"

        if not base_url:
            return False, "Base URL not configured"

        try:
            test_url = base_url.rstrip("/") + "/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(test_url, headers=headers, timeout=timeout)

            if response.status_code == 200:
                js = response.json()
                models = [model["id"] for model in js["data"]]
                self._models_cache = models
                logger.debug(f"AskSage: models: {models}")
                return models
            elif response.status_code == 401:
                logger.debug(
                    f"Failed to refresh models from API ({response.status_code}); using static defaults"
                )
                return False, "Authentication failed (invalid API key?)"
            else:
                logger.debug(
                    f"Failed to refresh models from API ({response.status_code}); using static defaults"
                )
                return False, f"API returned status {response.status_code}"

        except requests.Timeout:
            logger.debug("Failed to refresh models from API (timeout); using static defaults")
            return False, "Connection timeout"
        except requests.RequestException as e:
            logger.debug(
                "Failed to refresh models from API (RequestException); using static defaults"
            )
            return False, f"Connection failed: {str(e)[:50]}"
        except Exception as e:
            logger.debug(f"Failed to refresh models from API: {str(e)[:50]}; using static defaults")
            return False, f"Health check failed: {str(e)[:50]}"

        self._models_cache = self.available_models
        return self.available_models

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
        """Execute AskSage chat completion."""
        # Ensure models list is populated for any UI callers that rely on metadata
        try:
            self.get_available_models(api_key=api_key, base_url=base_url)
        except Exception:
            pass  # Best-effort model list refresh; not required for completion

        # Check for thinking parameters (not supported by AskSage)
        enable_thinking = kwargs.get("enable_thinking", False)
        budget_tokens = kwargs.get("budget_tokens")

        if enable_thinking or budget_tokens is not None:
            logger.warning("enable_thinking and budget_tokens are not used for AskSage provider.")

        # Get http_client if provided
        http_client = kwargs.get("http_client")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        # Handle typed dict output flag
        is_typed_dict_output = kwargs.get("is_typed_dict_output", False)

        asksage_body_args = {
            "system_prompt": "-",
            "dataset": "none",
            "live": 0,
            "limit_references": 0,
        }

        if output_format is not None:
            schema = output_format.model_json_schema()
            structured_message = f"""{message}

You must respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with the JSON object, no additional text or markdown formatting."""
            messages = [{"role": "user", "content": structured_message}]
        else:
            messages = [{"role": "user", "content": message}]

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=asksage_body_args,
        )
        content = response.choices[0].message.content

        if output_format is not None:
            response_text = _clean_json_response(content)
            try:
                result = output_format.model_validate_json(response_text)

                if is_typed_dict_output and hasattr(result, "model_dump"):
                    return result.model_dump()
                return result
            except Exception as e:
                raise ValueError(
                    f"Failed to parse structured output from AskSage: {e}\n"
                    f"Response: {response_text[:200]}"
                ) from e
        else:
            return content

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check AskSage API health by testing models endpoint."""
        import requests

        if not api_key:
            return False, "API key not set"

        if not base_url:
            return False, "Base URL not configured"

        try:
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


# Ensure ABC doesn't block instantiation if metadata is fully defined
AskSageProviderAdapter.__abstractmethods__ = frozenset()
