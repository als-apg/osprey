"""The output-token cap reaches each provider under the parameter its endpoint takes.

OpenAI's API takes ``max_completion_tokens`` on every chat model and refuses
``max_tokens`` on its reasoning models. LiteLLM rewrites ``max_tokens`` only for
the OpenAI families it recognises, so the adapter sends the parameter the
provider's adapter class declares instead of leaving it to that recognition.
Every other route keeps ``max_tokens``, which LiteLLM maps per provider.
The same declaration decides whether a request carries a temperature: OpenAI's
reasoning models refuse every temperature but their default.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from osprey.models.providers.litellm_adapter import (
    _max_tokens_param,
    check_litellm_health,
    execute_litellm_completion,
)

GATEWAY_URL = "https://gateway.example/v1"

# (provider, model_id, base_url, parameter the request carries)
ROUTES = [
    ("openai", "gpt-6-astra", "https://api.openai.com/v1", "max_completion_tokens"),
    ("openai", "gpt-5.6-luna", "https://api.openai.com/v1", "max_completion_tokens"),
    ("openai", "gpt-4o", "https://api.openai.com/v1", "max_completion_tokens"),
    ("anthropic", "claude-haiku-4-5-20251001", None, "max_tokens"),
    ("google", "gemini-2.5-flash", None, "max_tokens"),
    ("als-apg", "claude-haiku-4-5-20251001", GATEWAY_URL, "max_tokens"),
    ("cborg", "gpt-6-astra", GATEWAY_URL, "max_tokens"),
    ("amsc-i2", "claude-sonnet", GATEWAY_URL, "max_tokens"),
    ("stanford", "gpt-4o", GATEWAY_URL, "max_tokens"),
    ("vllm", "m", "http://localhost:8000/v1", "max_tokens"),
    ("ds4", "deepseek-v4-pro", "http://127.0.0.1:8000/v1", "max_tokens"),
]
_OTHER = {"max_tokens": "max_completion_tokens", "max_completion_tokens": "max_tokens"}


def _ids(rows):
    return [f"{r[0]}-{r[1]}" for r in rows]


class TestTheCompletionCarriesTheDeclaredParameter:
    @pytest.mark.parametrize(
        ("provider", "model_id", "base_url", "param"), ROUTES, ids=_ids(ROUTES)
    )
    def test_completion_request_shape(self, provider, model_id, base_url, param):
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(choices=[])
            execute_litellm_completion(
                provider=provider,
                message="hi",
                model_id=model_id,
                api_key="sk-test",
                base_url=base_url,
                max_tokens=64,
            )
        kwargs = mock_completion.call_args.kwargs
        assert kwargs[param] == 64
        assert _OTHER[param] not in kwargs

    @pytest.mark.parametrize(
        ("provider", "model_id", "base_url", "param"), ROUTES, ids=_ids(ROUTES)
    )
    def test_health_probe_request_shape(self, provider, model_id, base_url, param):
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock()
            ok, _ = check_litellm_health(
                provider=provider, api_key="sk-test", base_url=base_url, model_id=model_id
            )
        assert ok is True
        kwargs = mock_completion.call_args.kwargs
        assert kwargs[param] == 16
        assert _OTHER[param] not in kwargs

    def test_budget_check_still_reads_the_cap_on_openai(self):
        """The thinking-budget guard compares against the cap whatever it is sent as."""
        with pytest.raises(ValueError, match="budget_tokens must be less than max_tokens"):
            execute_litellm_completion(
                provider="openai",
                message="hi",
                model_id="gpt-6-astra",
                api_key="sk-test",
                base_url=None,
                max_tokens=100,
                enable_thinking=True,
                budget_tokens=100,
            )


class TestTheParameterIsDeclaredByTheAdapter:
    def test_openai_declares_max_completion_tokens(self):
        from osprey.models.providers.openai import OpenAIProviderAdapter

        assert OpenAIProviderAdapter.max_tokens_param == "max_completion_tokens"

    def test_base_default_is_max_tokens(self):
        from osprey.models.providers.base import BaseProvider

        assert BaseProvider.max_tokens_param == "max_tokens"

    def test_a_facility_adapter_declaration_is_honoured(self):
        """Not a name check: any adapter class declaring the attribute is read."""

        class _HouseOpenAI:
            max_tokens_param = "max_completion_tokens"

        class _Registry:
            def get_provider(self, name):
                return _HouseOpenAI if name == "house-openai" else None

        with patch(
            "osprey.models.provider_registry.get_provider_registry", return_value=_Registry()
        ):
            assert _max_tokens_param("house-openai") == "max_completion_tokens"
            assert _max_tokens_param("who-knows") == "max_tokens"


class TestTheWireCarriesTheDeclaredParameter:
    """What LiteLLM actually posts, with the HTTP send intercepted.

    Pins the pinned LiteLLM's side of the contract: it forwards
    ``max_completion_tokens`` for an OpenAI id it does not recognise, and still
    maps the adapter's ``max_tokens`` to Anthropic's own parameter.
    """

    @staticmethod
    def _posted_body(provider, model_id, base_url):
        bodies = []

        def fake_send(_client, request, **_kwargs):
            bodies.append(json.loads(request.content))
            raise httpx.ConnectError("intercepted", request=request)

        with patch.object(httpx.Client, "send", fake_send):
            ok, _ = check_litellm_health(
                provider=provider, api_key="sk-test", base_url=base_url, model_id=model_id
            )
        assert ok is False
        assert bodies, "no request reached the HTTP layer"
        return bodies[0]

    def test_an_openai_id_litellm_does_not_know_is_sent_max_completion_tokens(self):
        body = self._posted_body("openai", "gpt-6-astra", "https://api.openai.com/v1")
        assert body["max_completion_tokens"] == 16
        assert "max_tokens" not in body

    def test_anthropic_is_sent_its_own_max_tokens(self):
        body = self._posted_body("anthropic", "claude-haiku-4-5-20251001", None)
        assert body["max_tokens"] == 16
        assert "max_completion_tokens" not in body

    def test_an_openai_compatible_gateway_is_sent_max_tokens(self):
        body = self._posted_body("als-apg", "claude-haiku-4-5-20251001", GATEWAY_URL)
        assert body["max_tokens"] == 16
        assert "max_completion_tokens" not in body


# (provider, model_id, base_url, whether the request carries a temperature)
TEMPERATURE_ROUTES = [
    ("openai", "gpt-6-astra", "https://api.openai.com/v1", False),
    ("openai", "gpt-5.6-luna", "https://api.openai.com/v1", False),
    ("openai", "gpt-4o", "https://api.openai.com/v1", False),
    ("anthropic", "claude-haiku-4-5-20251001", None, True),
    ("google", "gemini-2.5-flash", None, True),
    ("als-apg", "claude-haiku-4-5-20251001", GATEWAY_URL, True),
    ("cborg", "gpt-6-astra", GATEWAY_URL, True),
    ("vllm", "m", "http://localhost:8000/v1", True),
]


class TestTheTemperatureIsSentOnlyWhereTheEndpointTakesIt:
    """OpenAI's reasoning models refuse every temperature but their default."""

    @pytest.mark.parametrize(
        ("provider", "model_id", "base_url", "sent"),
        TEMPERATURE_ROUTES,
        ids=_ids(TEMPERATURE_ROUTES),
    )
    def test_completion_request_shape(self, provider, model_id, base_url, sent):
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(choices=[])
            execute_litellm_completion(
                provider=provider,
                message="hi",
                model_id=model_id,
                api_key="sk-test",
                base_url=base_url,
                max_tokens=64,
                temperature=0.0,
            )
        kwargs = mock_completion.call_args.kwargs
        if sent:
            assert kwargs["temperature"] == 0.0
        else:
            assert "temperature" not in kwargs

    def test_openai_declares_no_temperature(self):
        from osprey.models.providers.base import BaseProvider
        from osprey.models.providers.openai import OpenAIProviderAdapter

        assert OpenAIProviderAdapter.accepts_temperature is False
        assert BaseProvider.accepts_temperature is True

    def test_a_gpt_5_completion_at_the_default_temperature_reaches_the_wire(self):
        """LiteLLM refuses temperature 0.0 on a GPT-5 id before sending anything."""
        bodies = []

        def fake_send(_client, request, **_kwargs):
            bodies.append(json.loads(request.content))
            raise httpx.ConnectError("intercepted", request=request)

        with patch.object(httpx.Client, "send", fake_send), pytest.raises(Exception) as exc:
            execute_litellm_completion(
                provider="openai",
                message="hi",
                model_id="gpt-5.6-luna",
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                max_tokens=64,
            )
        assert "UnsupportedParams" not in type(exc.value).__name__
        assert bodies, "no request reached the HTTP layer"
        assert "temperature" not in bodies[0]
        assert bodies[0]["max_completion_tokens"] == 64
