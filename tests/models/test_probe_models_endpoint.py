"""probe_models_endpoint: one listing probe for every route that declares one."""

from __future__ import annotations

import httpx
import pytest

from osprey.models.providers.als_apg import ALSAPGProviderAdapter
from osprey.models.providers.anthropic import AnthropicProviderAdapter
from osprey.models.providers.base import KEYLESS_API_KEY_PLACEHOLDER, BaseProvider
from osprey.models.providers.cborg import CBorgProviderAdapter
from osprey.models.providers.ds4 import DS4ProviderAdapter
from osprey.models.providers.google import GoogleProviderAdapter
from osprey.models.providers.health import HealthResult, probe_models_endpoint
from osprey.models.providers.openai import OpenAIProviderAdapter
from osprey.models.providers.vllm import VLLMProviderAdapter


class _Resp:
    def __init__(self, status_code: int = 200, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"data": []}

    def json(self):
        return self._payload


@pytest.fixture
def fake_get(monkeypatch):
    """Stand in for httpx.get; record each request and answer with ``state['resp']``."""
    state: dict = {"calls": [], "resp": _Resp(200, {"data": [{"id": "m"}]}), "raise": None}

    def _get(url, headers=None, timeout=None):
        state["calls"].append({"url": url, "headers": dict(headers or {}), "timeout": timeout})
        if state["raise"] is not None:
            raise state["raise"]
        return state["resp"]

    monkeypatch.setattr(httpx, "get", _get)
    for var in ("ALS_APG_BASE_URL",):
        monkeypatch.delenv(var, raising=False)
    return state


@pytest.fixture(autouse=True)
def _no_completion(monkeypatch):
    """The probe never makes a model call."""
    import litellm

    def _boom(*a, **kw):
        raise AssertionError("probe_models_endpoint must not call litellm.completion")

    monkeypatch.setattr(litellm, "completion", _boom)


def test_declared_probe_kinds():
    assert BaseProvider.models_probe is None
    assert BaseProvider.models_probe_base_url is None
    for cls in (
        OpenAIProviderAdapter,
        VLLMProviderAdapter,
        DS4ProviderAdapter,
        ALSAPGProviderAdapter,
        CBorgProviderAdapter,
    ):
        assert cls.models_probe == "bearer", cls.name
    assert AnthropicProviderAdapter.models_probe == "anthropic"
    assert GoogleProviderAdapter.models_probe is None
    assert AnthropicProviderAdapter.models_probe_base_url == "https://api.anthropic.com"
    assert OpenAIProviderAdapter.models_probe_base_url == "https://api.openai.com/v1"


def test_als_apg_lists_under_its_v1(fake_get):
    result = probe_models_endpoint(ALSAPGProviderAdapter, "https://h/v1", "k")
    assert fake_get["calls"][0]["url"] == "https://h/v1/models"
    assert fake_get["calls"][0]["headers"]["Authorization"] == "Bearer k"
    assert "x-api-key" not in fake_get["calls"][0]["headers"]
    assert result.reachable is True
    assert result.reason is None


def test_anthropic_direct_probes_the_vendor_listing(fake_get):
    result = probe_models_endpoint(AnthropicProviderAdapter, None, "sk-ant")
    call = fake_get["calls"][0]
    assert call["url"] == "https://api.anthropic.com/v1/models"
    assert call["headers"]["x-api-key"] == "sk-ant"
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert result == HealthResult(True, result.message, None)


def test_openai_direct_probes_the_vendor_listing(fake_get):
    result = probe_models_endpoint(OpenAIProviderAdapter, None, "sk")
    assert fake_get["calls"][0]["url"] == "https://api.openai.com/v1/models"
    assert result.reachable is True


def test_ds4_lists_under_v1(fake_get):
    probe_models_endpoint(DS4ProviderAdapter, "http://127.0.0.1:8000/v1", None)
    assert fake_get["calls"][0]["url"] == "http://127.0.0.1:8000/v1/models"


def test_base_without_v1_gets_it_appended(fake_get):
    probe_models_endpoint(VLLMProviderAdapter, "http://host:8001/", None)
    assert fake_get["calls"][0]["url"] == "http://host:8001/v1/models"


def test_keyless_route_never_sends_bearer_none(fake_get):
    probe_models_endpoint(VLLMProviderAdapter, "http://h:8000/v1", None)
    auth = fake_get["calls"][0]["headers"].get("Authorization")
    assert auth == f"Bearer {KEYLESS_API_KEY_PLACEHOLDER}"
    assert "None" not in auth


def test_missing_key_on_a_keyed_route_omits_the_header(fake_get):
    probe_models_endpoint(ALSAPGProviderAdapter, "https://h/v1", None)
    assert "Authorization" not in fake_get["calls"][0]["headers"]


def test_google_is_not_probed(fake_get):
    result = probe_models_endpoint(GoogleProviderAdapter, None, "k")
    assert result == HealthResult(None, "not probed", None)
    assert fake_get["calls"] == []


def test_no_resolvable_base_is_not_probed(fake_get):
    result = probe_models_endpoint(CBorgProviderAdapter, None, "k")
    assert result == HealthResult(None, "not probed", None)
    assert fake_get["calls"] == []


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (401, "auth"),
        (403, "auth"),
        (404, "unreachable"),
        (429, "unreachable"),
        (503, "unreachable"),
    ],
)
def test_status_codes_map_to_reasons(fake_get, status, reason):
    fake_get["resp"] = _Resp(status)
    result = probe_models_endpoint(ALSAPGProviderAdapter, "https://h/v1", "k", model_id="m")
    assert result.reachable is False
    assert result.reason == reason


def test_listing_without_the_model_reports_model(fake_get):
    fake_get["resp"] = _Resp(200, {"data": [{"id": "other"}]})
    result = probe_models_endpoint(ALSAPGProviderAdapter, "https://h/v1", "k", model_id="m")
    assert result.reachable is False
    assert result.reason == "model"


def test_listing_serving_the_model_is_healthy(fake_get):
    fake_get["resp"] = _Resp(200, {"data": [{"id": "other"}, {"id": "m"}]})
    result = probe_models_endpoint(ALSAPGProviderAdapter, "https://h/v1", "k", model_id="m")
    assert result.reachable is True
    assert result.reason is None


def test_connection_error_is_unreachable(fake_get):
    fake_get["raise"] = httpx.ConnectError("refused")
    result = probe_models_endpoint(DS4ProviderAdapter, "http://127.0.0.1:8000/v1", None)
    assert result.reachable is False
    assert result.reason == "unreachable"


def test_timeout_is_unreachable(fake_get):
    fake_get["raise"] = httpx.ReadTimeout("slow")
    result = probe_models_endpoint(DS4ProviderAdapter, "http://127.0.0.1:8000/v1", None)
    assert result.reachable is False
    assert result.reason == "unreachable"


def test_unexpected_error_never_raises(fake_get):
    fake_get["raise"] = RuntimeError("boom")
    result = probe_models_endpoint(DS4ProviderAdapter, "http://127.0.0.1:8000/v1", None)
    assert result.reachable is False
    assert result.reason == "unreachable"


def test_unparseable_listing_with_model_id_never_raises(fake_get):
    class _Bad(_Resp):
        def json(self):
            raise ValueError("not json")

    fake_get["resp"] = _Bad(200)
    result = probe_models_endpoint(ALSAPGProviderAdapter, "https://h/v1", "k", model_id="m")
    assert result.reachable is False
    assert result.reason == "unreachable"


def test_timeout_is_forwarded(fake_get):
    probe_models_endpoint(DS4ProviderAdapter, "http://127.0.0.1:8000/v1", None, timeout=2)
    assert fake_get["calls"][0]["timeout"] == 2
