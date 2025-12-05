"""ARGO Provider Adapter Implementation."""

import logging
import os
from typing import Any, List, Optional, Union

import httpx
import openai
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from .base import BaseProvider

logger = logging.getLogger(__name__)


class ArgoProviderAdapter(BaseProvider):
    """ARGO (ANL) provider implementation - OpenAI-compatible."""

    def __init_subclass__(cls, **kwargs):
        """
        Ensure ABC abstract flags are cleared once required methods are implemented.

        ABCMeta can mark the subclass abstract at creation time; resetting here avoids
        instantiation errors in environments that cache __abstractmethods__ aggressively.
        """
        super().__init_subclass__(**kwargs)
        cls.__abstractmethods__ = frozenset()

    # Metadata (single source of truth)
    name = "argo"
    description = "ANL Argo proxy (supports multiple models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = "https://argo-bridge.cels.anl.gov"
    default_model_id = "claudesonnet45"  # Claude 4.5 Sonnet via ARGO for general use
    health_check_model_id = "gpt5mini"  # Fast and cost-effective for health checks
    available_models = [
        "claudehaiku45",
        "claudeopus41",
        "claudesonnet45",
        "claudesonnet37",
        "gemini25flash",
        "gemini25pro",
        "gpt5",
        "gpt5mini",
    ]
    _models_cache: Optional[List[str]] = None

    # API key acquisition information
    api_key_url = None
    api_key_instructions = [
        "Argo uses the user login name which is obtained automatically from the $USER environment variable"
    ]
    api_key_note = None

    @classmethod
    def get_available_models(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        force_refresh: bool = False,
    ) -> List[str]:
        """
        Dynamically fetch available models from the Argo /models endpoint.

        Falls back to static defaults if the request fails or credentials are missing.
        """
        if cls._models_cache is not None and not force_refresh:
            return cls._models_cache

        api_key = api_key or os.environ.get("ARGO_API_KEY")
        base_url = base_url or os.environ.get("ARGO_BASE_URL") or cls.default_base_url

        if not api_key or not base_url:
            cls._models_cache = cls.available_models
            return cls.available_models

        try:
            url = base_url.rstrip("/") + "/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            models: List[str] = []
            if isinstance(data, dict):
                raw_models = data.get("data", [])
                if isinstance(raw_models, list):
                    for item in raw_models:
                        if isinstance(item, dict):
                            model_id = item.get("id") or item.get("model") or item.get("name")
                            if model_id:
                                models.append(model_id)

            if models:
                cls.available_models = models
                cls._models_cache = models
                return models

            logger.debug("ARGO: /models returned no entries; using static defaults")
        except Exception as exc:
            logger.debug(f"ARGO: failed to refresh models from API ({exc}); using static defaults")

        cls._models_cache = cls.available_models
        return cls.available_models

    def create_model(
        self,
        model_id: str,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: Optional[float],
        http_client: Optional[httpx.AsyncClient],
    ) -> OpenAIModel:
        """Create ARGO model instance for PydanticAI."""
        if http_client:
            client_args = {"api_key": api_key, "http_client": http_client, "base_url": base_url}
            openai_client = openai.AsyncOpenAI(**client_args)
        else:
            effective_timeout = timeout if timeout is not None else 60.0
            client_args = {"api_key": api_key, "timeout": effective_timeout, "base_url": base_url}
            openai_client = openai.AsyncOpenAI(**client_args)

        model = OpenAIModel(
            model_name=model_id,
            provider=PydanticOpenAIProvider(openai_client=openai_client),
        )
        model.model_id = model_id  # type: ignore[attr-defined]
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
        """Execute ARGO chat completion."""
        logger.debug(
            f"ARGO execute_completion called with output_format={output_format is not None}"
        )
        # Ensure models list is populated for any UI callers that rely on metadata
        try:
            self.get_available_models(api_key=api_key, base_url=base_url)
        except Exception:
            pass  # Don't block executions on model refresh errors
        # Check for thinking parameters (not supported by ARGO)
        enable_thinking = kwargs.get("enable_thinking", False)
        budget_tokens = kwargs.get("budget_tokens")

        if enable_thinking or budget_tokens is not None:
            logger.warning("enable_thinking and budget_tokens are not used for ARGO provider.")

        # Get http_client if provided
        http_client = kwargs.get("http_client")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        # Build messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        # Handle typed dict output flag
        is_typed_dict_output = kwargs.get("is_typed_dict_output", False)

        if output_format is not None:
            # ARGO has issues with structured output mode (returns Python booleans, etc.)
            # Use JSON mode with manual parsing directly - more reliable for ARGO
            logger.debug("ARGO: Using JSON mode with manual parsing (bypassing structured output)")

            # Build JSON instruction for the prompt
            import json

            schema = output_format.model_json_schema()

            # Create a simpler, example-based instruction
            # Extract field names and types
            fields = []
            if "properties" in schema:
                for field_name, field_info in schema["properties"].items():
                    field_type = field_info.get("type", "string")
                    field_desc = field_info.get("description", "")
                    fields.append(f'  "{field_name}": <{field_type}> // {field_desc}')

            fields_str = ",\n".join(fields)
            json_instruction = f"\n\nIMPORTANT: Respond with ONLY a valid JSON object containing the actual data (NOT the schema definition). Do not include markdown, code blocks, or explanations.\n\nYour response must be a JSON object with these fields:\n{{\n{fields_str}\n}}\n\nProvide the ACTUAL VALUES for each field based on the user's request. Start your response with {{ and end with }}"

            # Create new messages with JSON instruction
            json_messages = messages.copy()
            json_messages[-1] = {"role": "user", "content": message + json_instruction}

            # Use regular completion with JSON mode
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=json_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            except Exception as json_mode_error:
                logger.warning(
                    f"JSON mode request failed: {json_mode_error}, trying without JSON mode"
                )
                # Try without JSON mode as last resort
                response = client.chat.completions.create(
                    model=model_id,
                    messages=json_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            if not response.choices:
                raise ValueError("ARGO API returned empty choices list in JSON mode")

            raw_json = response.choices[0].message.content
            if not raw_json:
                logger.error("ARGO returned completely empty response")
                raise ValueError("ARGO returned empty response in JSON mode")

            logger.debug("ARGO: Processing JSON response")

            # Strip markdown code blocks FIRST (most common issue)
            cleaned_json = raw_json.strip()

            # Handle markdown code blocks with language identifier
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:].strip()  # Remove ```json
            elif cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:].strip()  # Remove generic ```

            # Remove trailing markdown
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3].strip()

            # NOW check if response starts with text before JSON (after markdown removal)
            if not cleaned_json.startswith("{") and not cleaned_json.startswith("["):
                logger.debug(
                    f"Response doesn't start with JSON. First 200 chars: {cleaned_json[:200]}"
                )
                # Try to find where JSON actually starts
                json_start = cleaned_json.find("{")
                if json_start > 0:
                    logger.debug(
                        f"Found JSON starting at position {json_start}, stripping prefix text"
                    )
                    cleaned_json = cleaned_json[json_start:].strip()
                else:
                    # No JSON found at all
                    logger.error(f"No JSON object found in response: {cleaned_json[:500]}")
                    raise ValueError("No JSON object found in model response")

            # Fix Python-style booleans to JSON-style booleans BEFORE parsing
            # This is a common issue where LLMs output Python syntax instead of JSON
            cleaned_json = cleaned_json.replace(": False", ": false")
            cleaned_json = cleaned_json.replace(": True", ": true")
            cleaned_json = cleaned_json.replace(": None", ": null")
            cleaned_json = cleaned_json.replace(",False", ",false")
            cleaned_json = cleaned_json.replace(",True", ",true")
            cleaned_json = cleaned_json.replace(",None", ",null")

            # Parse and validate with Pydantic
            try:
                result = output_format.model_validate_json(cleaned_json)
                if is_typed_dict_output and hasattr(result, "model_dump"):
                    return result.model_dump()
                return result
            except Exception as parse_error:
                logger.error(
                    f"Failed to parse JSON response. Cleaned JSON (first 500 chars): {cleaned_json[:500]}"
                )
                logger.error(f"Parse error: {parse_error}")
                # Try to parse as regular JSON to see what we got
                try:
                    import json

                    parsed = json.loads(cleaned_json)
                    logger.error(
                        f"JSON is valid but doesn't match schema. Keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'}"
                    )
                except:
                    logger.error("JSON is completely invalid")
                raise ValueError(f"Invalid JSON from model: {parse_error}")
        else:
            # Regular text completion
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if not response.choices:
                    raise ValueError("ARGO API returned empty choices list")
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"ARGO text completion error: {type(e).__name__}: {str(e)}")
                raise

    def check_health(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: float = 5.0,
        model_id: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Check ARGO API health.

        If model_id provided: Makes minimal chat completion (~$0.0001)
        If no model_id: Tests /v1/models endpoint (free)
        """
        if not api_key:
            return False, "API key not set"

        # Check for placeholder/template values
        if api_key.startswith("${") or "YOUR_API_KEY" in api_key.upper():
            return False, "API key not configured (placeholder value detected)"

        if not base_url:
            return False, "Base URL not configured"

        # Use provided model or cheapest default from metadata
        test_model = model_id or self.health_check_model_id

        # If model_id provided, test with minimal completion call
        if test_model:
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)

                response = client.chat.completions.create(
                    model=test_model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=50,
                    timeout=timeout,
                )

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


# Attempt to refresh available models at import time (best effort)
try:
    ArgoProviderAdapter.get_available_models(force_refresh=True)
except Exception:
    # Import-time failures should not break provider load
    pass

# Ensure ABC doesn't block instantiation if metadata is fully defined
ArgoProviderAdapter.__abstractmethods__ = frozenset()
