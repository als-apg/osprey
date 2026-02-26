"""ARGO Provider Adapter Implementation.

ARGO is ANL's (Argonne National Laboratory) OpenAI-compatible proxy service.
This provider uses LiteLLM as the backend while preserving ARGO-specific
functionality like dynamic model list refresh.

Structured output (JSON) is handled directly via httpx because the Argo API
does not support the response_format parameter. Instead, we use a system prompt
instructing the model to respond with JSON, then parse and validate the response.
"""

import json
import os
import re
from typing import Any

import httpx
from pydantic import BaseModel

from osprey.utils.logger import get_logger

from .base import BaseProvider
from .litellm_adapter import check_litellm_health, execute_litellm_completion

logger = get_logger("argo")


def _clean_json_response(text: str) -> str:
    """Clean markdown code blocks and fix common JSON issues from Argo response."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Fix Python-style booleans (True/False → true/false)
    text = re.sub(r":\s*True\b", ": true", text)
    text = re.sub(r":\s*False\b", ": false", text)
    text = re.sub(r",\s*True\b", ", true", text)
    text = re.sub(r",\s*False\b", ", false", text)

    # If response contains text before JSON, extract the JSON object
    if not text.startswith("{") and not text.startswith("["):
        # Try to find a JSON object in the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    return text


def _execute_argo_structured_output(
    model_id: str,
    message: str,
    output_format: type[BaseModel],
    api_key: str | None,
    base_url: str | None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    is_typed_dict_output: bool = False,
) -> BaseModel | dict:
    """Execute Argo structured output via direct httpx call.

    The Argo API does not support the response_format parameter, so we use
    a system prompt + schema instructions to get JSON output, then validate
    against the Pydantic model.
    """
    base_url = base_url or os.environ.get("ARGO_BASE_URL") or "https://argo-bridge.cels.anl.gov"
    api_key = api_key or os.environ.get("ARGO_API_KEY")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    schema = output_format.model_json_schema()

    system_message = (
        "You are a JSON-only response assistant. You MUST respond with valid JSON only. "
        "Never include explanations, markdown, or any text outside the JSON object."
    )

    structured_message = (
        f"Extract the requested information and respond with ONLY a JSON object.\n\n"
        f"REQUEST: {message}\n\n"
        f"REQUIRED JSON SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
        f"CRITICAL: Output ONLY the JSON object. No markdown, no explanations, "
        f"no code blocks. Just the raw JSON."
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": structured_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = httpx.post(url, json=payload, headers=headers, timeout=120.0)
    response.raise_for_status()

    data = response.json()

    # Extract content from OpenAI-compatible response
    response_text = ""
    if "choices" in data and data["choices"]:
        response_text = data["choices"][0].get("message", {}).get("content", "")

    if not response_text:
        raise ValueError(f"Empty response from Argo API for model {model_id}")

    response_text = _clean_json_response(response_text)

    # Parse and validate
    try:
        result = output_format.model_validate_json(response_text)
        if is_typed_dict_output and hasattr(result, "model_dump"):
            return result.model_dump()
        return result
    except Exception as e:
        raise ValueError(
            f"Failed to parse structured output from Argo ({model_id}): {e}\n"
            f"Response: {response_text[:500]}"
        ) from e


class ArgoProviderAdapter(BaseProvider):
    """ARGO (ANL) provider implementation using LiteLLM."""

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
    _models_cache: list[str] | None = None

    # API key acquisition information
    api_key_url = None
    api_key_instructions = [
        "Argo uses the user login name which is obtained automatically from the $USER environment variable"
    ]
    api_key_note = None

    # LiteLLM integration - ARGO is an OpenAI-compatible proxy
    is_openai_compatible = True

    @classmethod
    def get_available_models(
        cls,
        api_key: str | None = None,
        base_url: str | None = None,
        force_refresh: bool = False,
    ) -> list[str]:
        """Dynamically fetch available models from the Argo /models endpoint.

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

            models: list[str] = []
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
        """Execute ARGO chat completion.

        Structured output is handled directly via httpx (Argo API does not
        support response_format). Plain text completions go through LiteLLM.
        """
        # Ensure models list is populated for any UI callers that rely on metadata
        try:
            self.get_available_models(api_key=api_key, base_url=base_url)
        except Exception:
            pass  # Don't block executions on model refresh errors

        # Handle structured output directly (Argo doesn't support response_format)
        if output_format is not None:
            is_typed_dict_output = kwargs.get("is_typed_dict_output", False)
            return _execute_argo_structured_output(
                model_id=model_id,
                message=message,
                output_format=output_format,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                is_typed_dict_output=is_typed_dict_output,
            )

        # Plain text completion via LiteLLM
        return execute_litellm_completion(
            provider=self.name,
            message=message,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check ARGO API health via LiteLLM."""
        return check_litellm_health(
            provider=self.name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            model_id=model_id or self.health_check_model_id,
        )
