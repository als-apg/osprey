"""Tests for the provider adapters that subclass ``LiteLLMDelegatingProvider``.

The adapters covered here add no method bodies: each is a block of class
attributes, and ``execute_completion`` / ``check_health`` are inherited
unchanged. They are therefore described by a table whose rows are the
attributes an adapter declares and whose columns are the behaviours those
declarations decide, and every shared behaviour is asserted once per row.

The patch targets are per adapter module because ``LiteLLMDelegatingProvider``
resolves ``execute_litellm_completion`` and ``check_litellm_health`` out of
``sys.modules[type(self).__module__]`` at call time, so a patch aimed anywhere
else would not be the one the call reads.

``anthropic`` subclasses the same base but keeps its own suite, which asserts
against ``litellm.completion`` rather than the module-level helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from tests.conftest import GATEWAY_BASE_URL

from osprey.models.providers.als_apg import ALSAPGProviderAdapter
from osprey.models.providers.amsc_i2 import AMSCI2ProviderAdapter
from osprey.models.providers.cborg import CBorgProviderAdapter
from osprey.models.providers.google import GoogleProviderAdapter
from osprey.models.providers.litellm_delegating import LiteLLMDelegatingProvider
from osprey.models.providers.openai import OpenAIProviderAdapter
from osprey.models.providers.stanford import StanfordProviderAdapter

#: Every adapter in the table resolves a supplied endpoint to itself, whatever
#: its policy for a missing one, so one endpoint serves every case that is not
#: about a missing one.
PROXY_URL = "https://proxy"
MODEL_ID = "m"
#: An explicit model id replaces the health-check default without being checked
#: against ``available_models``, so a value no adapter lists is what proves the
#: replacement is unconditional.
OTHER_MODEL_ID = "other-model"


class _Sample(BaseModel):
    result: str


@dataclass(frozen=True)
class _Adapter:
    """One delegating adapter and the behaviour its declarations decide.

    Every field holds an expected value written by hand, so the adapter is only
    ever the value under test.

    The last two fields are the two halves of one policy, and exactly one of
    them is meaningful per row: an adapter either refuses a missing endpoint,
    and ``missing_base_url_refusal`` is the message it refuses with, or it
    resolves one, and ``missing_base_url_resolves_to`` is what reaches the
    litellm layer.

    ``default_base_url`` and ``missing_base_url_resolves_to`` are separate
    columns and are not derived from one another. ``openai`` declares
    ``https://api.openai.com/v1`` and resolves a missing endpoint to ``None``,
    because a provider that requires no endpoint keeps resolving to nothing even
    with a default declared; ``stanford`` declares the same kind of value and
    does resolve to it. That the two columns disagree for one row and agree for
    another is the behaviour, not a redundancy.
    """

    adapter: type[LiteLLMDelegatingProvider]
    name: str
    description_keyword: str
    requires_base_url: bool
    default_base_url: str | None
    default_model_id: str
    health_check_model_id: str
    is_openai_compatible: bool
    supports_native_structured_output: bool | None
    litellm_prefix: str | None
    api_key_url: str | None
    declares_api_key_note: bool
    missing_base_url_refusal: str | None
    missing_base_url_resolves_to: str | None


LITELLM_ADAPTERS: tuple[_Adapter, ...] = (
    _Adapter(
        adapter=ALSAPGProviderAdapter,
        name="als-apg",
        description_keyword="ALS",
        requires_base_url=True,
        default_base_url=None,
        default_model_id="claude-haiku-4-5-20251001",
        health_check_model_id="claude-haiku-4-5-20251001",
        is_openai_compatible=True,
        supports_native_structured_output=None,
        litellm_prefix=None,
        api_key_url=None,
        declares_api_key_note=True,
        missing_base_url_refusal="Base URL required for als-apg",
        missing_base_url_resolves_to=None,
    ),
    _Adapter(
        adapter=AMSCI2ProviderAdapter,
        name="amsc-i2",
        description_keyword="American Science Cloud",
        requires_base_url=True,
        default_base_url=None,
        default_model_id="claude-haiku",
        health_check_model_id="claude-haiku",
        is_openai_compatible=True,
        supports_native_structured_output=True,
        litellm_prefix=None,
        api_key_url="https://api.i2-core.american-science-cloud.org/",
        declares_api_key_note=True,
        missing_base_url_refusal="Base URL required for amsc-i2",
        missing_base_url_resolves_to=None,
    ),
    _Adapter(
        adapter=CBorgProviderAdapter,
        name="cborg",
        description_keyword="CBorg",
        requires_base_url=True,
        default_base_url=None,
        default_model_id="anthropic/claude-haiku",
        health_check_model_id="anthropic/claude-haiku",
        is_openai_compatible=True,
        supports_native_structured_output=True,
        litellm_prefix=None,
        api_key_url="https://cborg.lbl.gov",
        declares_api_key_note=True,
        missing_base_url_refusal="Base URL required for cborg",
        missing_base_url_resolves_to=None,
    ),
    _Adapter(
        adapter=GoogleProviderAdapter,
        name="google",
        description_keyword="Gemini",
        requires_base_url=False,
        default_base_url=None,
        default_model_id="gemini-3.8-flash",
        health_check_model_id="gemini-3.5-flash-lite",
        is_openai_compatible=False,
        supports_native_structured_output=None,
        litellm_prefix="gemini",
        api_key_url="https://aistudio.google.com/app/apikey",
        declares_api_key_note=False,
        missing_base_url_refusal=None,
        missing_base_url_resolves_to=None,
    ),
    _Adapter(
        adapter=OpenAIProviderAdapter,
        name="openai",
        description_keyword="GPT",
        requires_base_url=False,
        default_base_url="https://api.openai.com/v1",
        default_model_id="gpt-5.6-sol",
        health_check_model_id="gpt-5.6-luna",
        is_openai_compatible=False,
        supports_native_structured_output=None,
        litellm_prefix="",
        api_key_url="https://platform.openai.com/api-keys",
        declares_api_key_note=False,
        missing_base_url_refusal=None,
        missing_base_url_resolves_to=None,
    ),
    _Adapter(
        adapter=StanfordProviderAdapter,
        name="stanford",
        description_keyword="Stanford",
        requires_base_url=True,
        default_base_url="https://aiapi-prod.stanford.edu/v1",
        default_model_id="gpt-4o",
        health_check_model_id="gpt-4o-mini",
        is_openai_compatible=True,
        supports_native_structured_output=True,
        litellm_prefix=None,
        api_key_url="https://uit.stanford.edu/service/ai-api-gateway",
        declares_api_key_note=True,
        missing_base_url_refusal=None,
        missing_base_url_resolves_to="https://aiapi-prod.stanford.edu/v1",
    ),
)

_REFUSING = tuple(row for row in LITELLM_ADAPTERS if row.missing_base_url_refusal)
_RESOLVING = tuple(row for row in LITELLM_ADAPTERS if not row.missing_base_url_refusal)

_ALS_APG = next(row for row in LITELLM_ADAPTERS if row.name == "als-apg")
_STANFORD = next(row for row in LITELLM_ADAPTERS if row.name == "stanford")


def _ids(rows: tuple[_Adapter, ...]) -> list[str]:
    return [row.name for row in rows]


# The module an adapter is defined in is the namespace its inherited body looks
# the helpers up in, so deriving the target from ``adapter.__module__`` is the
# same lookup rather than a second spelling of it.
def _completion_target(row: _Adapter) -> str:
    return f"{row.adapter.__module__}.execute_litellm_completion"


def _health_target(row: _Adapter) -> str:
    return f"{row.adapter.__module__}.check_litellm_health"


@pytest.fixture(autouse=True)
def _no_ambient_base_url_override(monkeypatch):
    """Clear every endpoint override the table's adapters declare.

    An adapter that declares one reads it at request time by design, so an
    exported value would rewrite the endpoint every assertion here is about. A
    case that exercises an override sets the variable itself, which still wins
    because the fixture runs first.
    """
    for row in LITELLM_ADAPTERS:
        if row.adapter.base_url_env_var:
            monkeypatch.delenv(row.adapter.base_url_env_var, raising=False)


class TestTheDeclaredMetadata:
    """Registry- and routing-facing metadata is the single source of truth.

    Each case pins what one declaration is.
    """

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_registry_name_is_the_declared_one(self, row):
        assert row.adapter.name == row.name

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_description_names_the_provider(self, row):
        assert row.description_keyword in row.adapter.description

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_requirement_flags_are_declared(self, row):
        """An adapter that needs no endpoint still needs a key and a model id."""
        assert row.adapter.requires_api_key is True
        assert row.adapter.requires_base_url is row.requires_base_url
        assert row.adapter.requires_model_id is True
        assert row.adapter.supports_proxy is True

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_declared_default_endpoint(self, row):
        assert row.adapter.default_base_url == row.default_base_url

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_default_and_health_models(self, row):
        assert row.adapter.default_model_id == row.default_model_id
        assert row.adapter.health_check_model_id == row.health_check_model_id

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_available_models_include_the_default_and_health_models(self, row):
        models = row.adapter.available_models
        assert len(models) > 0
        assert row.default_model_id in models
        assert row.health_check_model_id in models

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_litellm_routing_metadata(self, row):
        """The three together decide routing and native response schemas.

        They decide how a model id is routed and whether a response schema is
        sent natively; ``None`` for the structured-output flag defers to
        litellm's own detection rather than asserting support either way.
        """
        assert row.adapter.is_openai_compatible is row.is_openai_compatible
        assert row.adapter.supports_native_structured_output is (
            row.supports_native_structured_output
        )
        assert row.adapter.litellm_prefix == row.litellm_prefix

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_api_key_help_is_present(self, row):
        assert row.adapter.api_key_url == row.api_key_url
        assert len(row.adapter.api_key_instructions) > 0
        assert (row.adapter.api_key_note is not None) is row.declares_api_key_note

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_adapter_is_instantiable(self, row):
        assert isinstance(row.adapter(), row.adapter)


class TestForwardingACompletion:
    """The inherited body passes the caller's arguments to the litellm helper.

    It calls under the adapter's own name, adding only the resolved endpoint and
    the declared defaults.
    """

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_litellm_result_is_returned_unchanged(self, row):
        with patch(_completion_target(row), return_value="hello") as mock_exec:
            result = row.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url=PROXY_URL
            )
        assert result == "hello"
        mock_exec.assert_called_once()

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_core_arguments_and_defaults_reach_the_litellm_layer(self, row):
        """The ``base_url`` assertion is the one every row shares.

        A supplied endpoint is used verbatim by every adapter in the table,
        whatever it does with a missing one.
        """
        with patch(_completion_target(row), return_value="ok") as mock_exec:
            row.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url=PROXY_URL
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["provider"] == row.name
        assert kwargs["message"] == "hi"
        assert kwargs["model_id"] == MODEL_ID
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == PROXY_URL
        assert kwargs["max_tokens"] == 1024
        assert kwargs["temperature"] == 0.0
        assert kwargs["output_format"] is None

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_overrides_replace_the_defaults(self, row):
        with patch(_completion_target(row), return_value="ok") as mock_exec:
            row.adapter().execute_completion(
                message="hi",
                model_id=MODEL_ID,
                api_key="key",
                base_url=PROXY_URL,
                max_tokens=64,
                temperature=0.7,
                output_format=_Sample,
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["max_tokens"] == 64
        assert kwargs["temperature"] == 0.7
        assert kwargs["output_format"] is _Sample

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_extra_keyword_arguments_pass_through(self, row):
        """The body forwards keyword arguments it does not name.

        A model-specific option therefore needs no adapter change to reach
        litellm.
        """
        with patch(_completion_target(row), return_value="ok") as mock_exec:
            row.adapter().execute_completion(
                message="hi",
                model_id=MODEL_ID,
                api_key="key",
                base_url=PROXY_URL,
                enable_thinking=True,
                budget_tokens=256,
                reasoning_effort="high",
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["enable_thinking"] is True
        assert kwargs["budget_tokens"] == 256
        assert kwargs["reasoning_effort"] == "high"


class TestForwardingAHealthProbe:
    """The health probe forwards the same resolved endpoint completions use.

    It falls back to the declared health-check model.
    """

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_health_result_is_returned_unchanged(self, row):
        with patch(_health_target(row), return_value=(True, "ok")) as mock_health:
            result = row.adapter().check_health(api_key="key", base_url=PROXY_URL)
        assert result == (True, "ok")
        mock_health.assert_called_once()

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_health_arguments_and_the_default_timeout_reach_the_litellm_layer(self, row):
        with patch(_health_target(row), return_value=(True, "ok")) as mock_health:
            row.adapter().check_health(api_key="key", base_url=PROXY_URL)
        kwargs = mock_health.call_args.kwargs
        assert kwargs["provider"] == row.name
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == PROXY_URL
        assert kwargs["timeout"] == 5.0

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_the_model_id_defaults_to_the_health_check_model(self, row):
        with patch(_health_target(row), return_value=(True, "ok")) as mock_health:
            row.adapter().check_health(api_key="key", base_url=PROXY_URL)
        assert mock_health.call_args.kwargs["model_id"] == row.health_check_model_id

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_an_explicit_model_id_replaces_the_default(self, row):
        with patch(_health_target(row), return_value=(True, "ok")) as mock_health:
            row.adapter().check_health(api_key="key", base_url=PROXY_URL, model_id=OTHER_MODEL_ID)
        assert mock_health.call_args.kwargs["model_id"] == OTHER_MODEL_ID

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_a_custom_timeout_is_forwarded(self, row):
        with patch(_health_target(row), return_value=(True, "ok")) as mock_health:
            row.adapter().check_health(api_key="key", base_url=PROXY_URL, timeout=12.0)
        assert mock_health.call_args.kwargs["timeout"] == 12.0

    @pytest.mark.parametrize("row", LITELLM_ADAPTERS, ids=_ids(LITELLM_ADAPTERS))
    def test_a_health_failure_propagates(self, row):
        with patch(_health_target(row), return_value=(False, "down")):
            result = row.adapter().check_health(api_key="key", base_url=PROXY_URL)
        assert result == (False, "down")


class TestAMissingEndpoint:
    """An adapter either has a source for an endpoint or it does not.

    What it does when it does not is the one behaviour the table's rows
    genuinely disagree on.
    """

    @pytest.mark.parametrize("row", _REFUSING, ids=_ids(_REFUSING))
    def test_a_missing_endpoint_is_refused(self, row):
        """No endpoint is invented for an adapter that has no source for one.

        The refusal is raised in the adapter rather than left to the model
        client, so a direct call fails for a reason a deployer can act on
        instead of as an authentication failure against a host nobody
        configured.
        """
        with pytest.raises(ValueError, match=row.missing_base_url_refusal):
            row.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url=None
            )

    @pytest.mark.parametrize("row", _REFUSING, ids=_ids(_REFUSING))
    def test_a_missing_endpoint_is_refused_by_the_health_probe(self, row):
        """The health probe refuses the same missing endpoint completions do."""
        with pytest.raises(ValueError, match=row.missing_base_url_refusal):
            row.adapter().check_health(api_key="key", base_url=None)

    @pytest.mark.parametrize("row", _RESOLVING, ids=_ids(_RESOLVING))
    def test_a_missing_endpoint_resolves_to_the_declared_policy(self, row):
        """An adapter that requires an endpoint and declares a default uses it.

        An adapter that requires none forwards nothing, because litellm derives
        the endpoint from the model prefix there and forwarding a declared
        default would pin a route the client chooses.
        """
        with patch(_completion_target(row), return_value="ok") as mock_exec:
            row.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url=None
            )
        assert mock_exec.call_args.kwargs["base_url"] == row.missing_base_url_resolves_to

    @pytest.mark.parametrize("row", _RESOLVING, ids=_ids(_RESOLVING))
    def test_a_missing_endpoint_resolves_to_the_declared_policy_for_the_health_probe(self, row):
        """The health probe resolves a missing endpoint the same way."""
        with patch(_health_target(row), return_value=(True, "ok")) as mock_health:
            row.adapter().check_health(api_key="key", base_url=None)
        assert mock_health.call_args.kwargs["base_url"] == row.missing_base_url_resolves_to

    def test_some_adapter_refuses_a_missing_endpoint(self):
        """The table must keep an example of the refusing policy."""
        assert _REFUSING, "no row in the table refuses a missing endpoint"

    def test_some_adapter_resolves_a_missing_endpoint(self):
        """The table must keep an example of the resolving policy."""
        assert _RESOLVING, "no row in the table resolves a missing endpoint"


class TestTheALSAPGEndpointOverride:
    """The declared environment variable beats every other source of an endpoint.

    It is both how a deployment names its gateway and the break-glass lever for
    pointing an already-deployed system at a fallback without rebuilding an
    image, so it beats an explicit argument too.
    """

    def test_declares_the_env_var_name(self):
        assert _ALS_APG.adapter.base_url_env_var == "ALS_APG_BASE_URL"

    def test_no_default_endpoint_resolves_to_nothing(self):
        """A default here would be one organisation's host reached silently.

        The adapter requires an endpoint, so the requirement gate would accept
        the default instead of asking for the deployment's own.
        """
        assert _ALS_APG.adapter.effective_base_url(None) is None

    def test_env_var_wins_over_explicit_base_url(self, monkeypatch):
        monkeypatch.setenv("ALS_APG_BASE_URL", "https://fallback.example.org/v1")
        with patch(_completion_target(_ALS_APG), return_value="ok") as mock_exec:
            _ALS_APG.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url=PROXY_URL
            )
        assert mock_exec.call_args.kwargs["base_url"] == "https://fallback.example.org/v1"

    def test_env_var_fills_missing_base_url(self, monkeypatch):
        monkeypatch.setenv("ALS_APG_BASE_URL", "https://fallback.example.org/v1")
        with patch(_completion_target(_ALS_APG), return_value="ok") as mock_exec:
            _ALS_APG.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url=None
            )
        assert mock_exec.call_args.kwargs["base_url"] == "https://fallback.example.org/v1"

    def test_empty_env_var_is_ignored(self, monkeypatch):
        """An empty export must not blank a configured URL."""
        monkeypatch.setenv("ALS_APG_BASE_URL", "")
        with patch(_completion_target(_ALS_APG), return_value="ok") as mock_exec:
            _ALS_APG.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url=GATEWAY_BASE_URL
            )
        assert mock_exec.call_args.kwargs["base_url"] == GATEWAY_BASE_URL

    def test_an_unresolved_placeholder_is_not_a_url(self, monkeypatch):
        """The config resolver keeps a reference verbatim when the variable is unset.

        Resolving to nothing is the refusal, rather than handing the literal
        string to litellm as a hostname.
        """
        monkeypatch.delenv("ALS_APG_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="Base URL required for als-apg"):
            _ALS_APG.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url="${ALS_APG_BASE_URL}"
            )

    def test_env_var_redirects_check_health_too(self, monkeypatch):
        """The health probe must reach the endpoint completions actually use."""
        monkeypatch.setenv("ALS_APG_BASE_URL", "https://fallback.example.org/v1")
        with patch(_health_target(_ALS_APG), return_value=(True, "ok")) as mock_health:
            _ALS_APG.adapter().check_health(api_key="key", base_url=PROXY_URL)
        assert mock_health.call_args.kwargs["base_url"] == "https://fallback.example.org/v1"

    def test_unset_env_var_preserves_existing_behavior(self, monkeypatch):
        monkeypatch.delenv("ALS_APG_BASE_URL", raising=False)
        with patch(_completion_target(_ALS_APG), return_value="ok") as mock_exec:
            _ALS_APG.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url=PROXY_URL
            )
        assert mock_exec.call_args.kwargs["base_url"] == PROXY_URL


class TestTheStanfordEndpointFallback:
    """A falsy endpoint resolves the way an absent one does.

    This stays outside the table because stanford is the only row with both a
    required endpoint and a declared default, which is the pair that makes a
    falsy value distinguishable from a supplied one.
    """

    def test_an_empty_endpoint_falls_back_to_the_default(self):
        """An empty string is no endpoint."""
        with patch(_completion_target(_STANFORD), return_value="ok") as mock_exec:
            _STANFORD.adapter().execute_completion(
                message="hi", model_id=MODEL_ID, api_key="key", base_url=""
            )
        assert mock_exec.call_args.kwargs["base_url"] == _STANFORD.missing_base_url_resolves_to
