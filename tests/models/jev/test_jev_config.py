"""The ``api.jev`` settings model: defaults, validation, the endpoint rule and route/key resolution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from osprey.models.jev.config import (
    OPENROUTER_KEY_ENV,
    REFUSED_MODEL_ALIASES,
    TYPESAFE_KEY_ENV,
    Budget,
    JevSettings,
    endpoint,
    resolve_base_url,
    resolve_key,
    resolve_route,
)
from osprey.models.jev.errors import DecisionAuthError

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_settings_defaults_match_the_measured_values() -> None:
    s = JevSettings()
    assert s.route == "typesafe"
    assert s.max_choice_options == 255
    assert s.max_concurrency == 4
    assert s.input_cost_per_million_tokens == 0.042

    ts = s.routes.typesafe
    assert ts.model == "jev-1.13.0"
    assert ts.base_url == "https://api.typesafe.ai"
    assert ts.api_key is None
    assert ts.max_request_tokens == 60_000
    assert ts.max_state_plus_question_tokens == 30_000

    orr = s.routes.openrouter
    assert orr.model == "typesafe/jev-1.13"
    assert orr.max_request_tokens == 30_000
    assert orr.max_state_plus_question_tokens == 30_000


def test_budget_defaults_interactive_and_batch() -> None:
    s = JevSettings()
    assert s.budgets.interactive == Budget(timeout_s=2.0, max_retries=1, total_s=3.0)
    assert s.budgets.batch == Budget(timeout_s=10.0, max_retries=2, total_s=30.0)


def test_settings_route_selects_the_active_route_block() -> None:
    s = JevSettings(route="openrouter")
    assert s.active_route is s.routes.openrouter
    assert JevSettings().active_route is JevSettings().routes.typesafe


def test_settings_openrouter_route_has_no_key_or_base_fields() -> None:
    # The openrouter route borrows base and key from api.providers.openrouter.
    with pytest.raises(ValidationError):
        JevSettings.model_validate({"routes": {"openrouter": {"api_key": "x"}}})
    with pytest.raises(ValidationError):
        JevSettings.model_validate({"routes": {"openrouter": {"base_url": "https://x"}}})


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_settings_from_config_without_block_gives_defaults() -> None:
    assert JevSettings.from_config({}) == JevSettings()
    assert JevSettings.from_config({"api": {}}) == JevSettings()
    assert JevSettings.from_config({"api": {"jev": None}}) == JevSettings()


def test_settings_from_config_reads_the_api_jev_block() -> None:
    config = {
        "api": {
            "providers": {"openrouter": {"api_key": "${OPENROUTER_API_KEY}"}},
            "jev": {
                "route": "openrouter",
                "max_concurrency": 2,
                "budgets": {"interactive": {"timeout_s": 1.5}},
                "routes": {
                    "typesafe": {"base_url": "https://proxy.example/v1"},
                    "openrouter": {"max_request_tokens": 32_000},
                },
            },
        }
    }
    s = JevSettings.from_config(config)
    assert s.route == "openrouter"
    assert s.max_concurrency == 2
    assert s.budgets.interactive == Budget(timeout_s=1.5, max_retries=1, total_s=3.0)
    assert s.budgets.batch == Budget(timeout_s=10.0, max_retries=2, total_s=30.0)
    assert s.routes.typesafe.base_url == "https://proxy.example/v1"
    assert s.routes.typesafe.max_request_tokens == 60_000
    assert s.routes.openrouter.max_request_tokens == 32_000
    assert s.routes.openrouter.model == "typesafe/jev-1.13"


def test_settings_from_config_refuses_a_non_mapping_block() -> None:
    with pytest.raises(ValidationError):
        JevSettings.from_config({"api": {"jev": "typesafe"}})


def test_settings_typesafe_is_never_an_api_provider() -> None:
    # A typesafe entry under api.providers is not how the route is configured;
    # the settings model reads api.jev only and ignores the provider table.
    config = {"api": {"providers": {"typesafe": {"base_url": "https://elsewhere"}}}}
    s = JevSettings.from_config(config)
    assert s.routes.typesafe.base_url == "https://api.typesafe.ai"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_settings_unknown_route_refused() -> None:
    with pytest.raises(ValidationError):
        JevSettings(route="anthropic")  # type: ignore[arg-type]


def test_settings_unknown_key_refused() -> None:
    with pytest.raises(ValidationError):
        JevSettings.model_validate({"max_concurency": 3})
    with pytest.raises(ValidationError):
        JevSettings.model_validate({"routes": {"typesafe": {"max_tokens": 1}}})


@pytest.mark.parametrize("alias", ["jev-latest", "jev-preview", "~typesafe/jev-latest"])
@pytest.mark.parametrize("route", ["typesafe", "openrouter"])
def test_alias_model_ids_refused_on_every_route(alias: str, route: str) -> None:
    assert alias in REFUSED_MODEL_ALIASES
    with pytest.raises(ValidationError, match="alias"):
        JevSettings.model_validate({"routes": {route: {"model": alias}}})


def test_alias_refusal_is_exact_so_versioned_ids_pass() -> None:
    s = JevSettings.model_validate(
        {"routes": {"openrouter": {"model": "typesafe/jev-1.13-20260917"}}}
    )
    assert s.routes.openrouter.model == "typesafe/jev-1.13-20260917"


def test_settings_empty_model_refused() -> None:
    with pytest.raises(ValidationError):
        JevSettings.model_validate({"routes": {"typesafe": {"model": "  "}}})


@pytest.mark.parametrize("route", ["typesafe", "openrouter"])
def test_settings_state_plus_question_must_fit_one_request(route: str) -> None:
    with pytest.raises(ValidationError, match="max_state_plus_question_tokens"):
        JevSettings.model_validate(
            {
                "routes": {
                    route: {"max_request_tokens": 20_000, "max_state_plus_question_tokens": 20_001}
                }
            }
        )
    ok = JevSettings.model_validate(
        {
            "routes": {
                route: {"max_request_tokens": 20_000, "max_state_plus_question_tokens": 20_000}
            }
        }
    )
    assert getattr(ok.routes, route).max_state_plus_question_tokens == 20_000


@pytest.mark.parametrize("value", [0, 256, 1000])
def test_settings_max_choice_options_bounded(value: int) -> None:
    with pytest.raises(ValidationError):
        JevSettings(max_choice_options=value)


def test_settings_max_choice_options_accepts_the_bound() -> None:
    assert JevSettings(max_choice_options=255).max_choice_options == 255
    assert JevSettings(max_choice_options=2).max_choice_options == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_concurrency", 0),
        ("input_cost_per_million_tokens", -0.01),
    ],
)
def test_settings_scalar_bounds(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        JevSettings.model_validate({field: value})


@pytest.mark.parametrize(
    "budget",
    [
        {"timeout_s": 0},
        {"timeout_s": -1},
        {"max_retries": -1},
        {"total_s": 0},
        {"timeout_s": 5, "total_s": 4},
    ],
)
def test_budget_invalid_values_refused(budget: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        JevSettings.model_validate({"budgets": {"batch": budget}})


def test_budget_zero_retries_allowed() -> None:
    s = JevSettings.model_validate({"budgets": {"interactive": {"max_retries": 0}}})
    assert s.budgets.interactive.max_retries == 0


@pytest.mark.parametrize("route", ["typesafe", "openrouter"])
def test_settings_token_limits_positive(route: str) -> None:
    with pytest.raises(ValidationError):
        JevSettings.model_validate({"routes": {route: {"max_request_tokens": 0}}})
    with pytest.raises(ValidationError):
        JevSettings.model_validate({"routes": {route: {"max_state_plus_question_tokens": 0}}})


def test_settings_are_frozen() -> None:
    s = JevSettings()
    with pytest.raises(ValidationError):
        s.max_concurrency = 8  # type: ignore[misc]


def test_settings_api_key_not_in_repr() -> None:
    s = JevSettings.model_validate({"routes": {"typesafe": {"api_key": "sk-SECRET-value"}}})
    assert s.routes.typesafe.api_key == "sk-SECRET-value"
    assert "sk-SECRET-value" not in repr(s)
    assert "sk-SECRET-value" not in str(s)


# ---------------------------------------------------------------------------
# Endpoint rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base,expected",
    [
        ("https://api.typesafe.ai", "https://api.typesafe.ai/v1/systemone"),
        ("https://api.typesafe.ai/", "https://api.typesafe.ai/v1/systemone"),
        ("https://api.typesafe.ai/v1", "https://api.typesafe.ai/v1/systemone"),
        ("https://api.typesafe.ai/v1/", "https://api.typesafe.ai/v1/systemone"),
        ("https://openrouter.ai/api", "https://openrouter.ai/api/v1/systemone"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/systemone"),
        ("https://openrouter.ai/api/v1/", "https://openrouter.ai/api/v1/systemone"),
    ],
)
def test_endpoint_accepts_both_base_spellings(base: str, expected: str) -> None:
    assert endpoint(base) == expected


def test_endpoint_only_strips_a_trailing_v1_segment() -> None:
    assert endpoint("https://host/v1beta") == "https://host/v1beta/v1/systemone"
    assert endpoint("https://host/v1/proxy") == "https://host/v1/proxy/v1/systemone"


def test_endpoint_of_the_default_typesafe_base() -> None:
    assert (
        endpoint(JevSettings().routes.typesafe.base_url) == "https://api.typesafe.ai/v1/systemone"
    )


# ---------------------------------------------------------------------------
# Route resolution: base URL and key (environment is monkeypatched only)
# ---------------------------------------------------------------------------

_FAKE_KEY = "synthetic-test-key"
_OPENROUTER = JevSettings(route="openrouter")
_TYPESAFE = JevSettings()
_OPENROUTER_DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/systemone"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in ("OPENROUTER_BASE_URL", OPENROUTER_KEY_ENV, TYPESAFE_KEY_ENV, "JEV_TEST_KEY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _providers(block: dict[str, object]) -> dict[str, object]:
    return {"api": {"providers": {"openrouter": block}}}


@pytest.mark.usefixtures("clean_env")
@pytest.mark.parametrize("base", ["https://gw.example/api", "https://gw.example/api/v1"])
def test_route_openrouter_both_base_spellings_give_one_endpoint(base: str) -> None:
    config = _providers({"base_url": base, "api_key": _FAKE_KEY})
    resolved = resolve_route(_OPENROUTER, config)
    assert resolved.endpoint == "https://gw.example/api/v1/systemone"
    assert resolved.route == "openrouter"
    assert resolved.model == "typesafe/jev-1.13"


def test_route_openrouter_without_provider_block_uses_default_endpoint(clean_env) -> None:
    clean_env.setenv(OPENROUTER_KEY_ENV, _FAKE_KEY)
    for config in ({}, {"api": {}}, {"api": {"providers": {}}}):
        assert resolve_route(_OPENROUTER, config).endpoint == _OPENROUTER_DEFAULT_ENDPOINT


@pytest.mark.usefixtures("clean_env")
def test_route_openrouter_unresolved_base_placeholder_uses_default() -> None:
    config = _providers({"base_url": "${JEV_TEST_UNSET_BASE}", "api_key": _FAKE_KEY})
    assert resolve_route(_OPENROUTER, config).endpoint == _OPENROUTER_DEFAULT_ENDPOINT


def test_route_env_override_moves_chat_and_jev_together(clean_env) -> None:
    from osprey.models.providers.openrouter import OpenRouterProviderAdapter

    clean_env.setenv("OPENROUTER_BASE_URL", "https://moved.example/v1")
    config = _providers({"base_url": "https://gw.example/api", "api_key": _FAKE_KEY})
    base = resolve_base_url(_OPENROUTER, config)
    assert base == OpenRouterProviderAdapter.effective_base_url("https://gw.example/api")
    assert resolve_route(_OPENROUTER, config).endpoint == "https://moved.example/v1/systemone"


def test_route_typesafe_uses_its_own_base(clean_env) -> None:
    clean_env.setenv(TYPESAFE_KEY_ENV, _FAKE_KEY)
    clean_env.setenv("OPENROUTER_BASE_URL", "https://moved.example/v1")
    resolved = resolve_route(_TYPESAFE, {})
    assert resolved.endpoint == "https://api.typesafe.ai/v1/systemone"
    assert resolved.model == "jev-1.13.0"
    assert resolved.route == "typesafe"


def test_key_openrouter_from_provider_block_after_env_expansion(clean_env) -> None:
    clean_env.setenv("JEV_TEST_KEY", _FAKE_KEY)
    clean_env.setenv(OPENROUTER_KEY_ENV, "other-synthetic-key")
    config = _providers({"api_key": "${JEV_TEST_KEY}"})
    assert resolve_key(_OPENROUTER, config) == _FAKE_KEY


def test_key_openrouter_falls_back_to_env(clean_env) -> None:
    clean_env.setenv(OPENROUTER_KEY_ENV, _FAKE_KEY)
    assert resolve_key(_OPENROUTER, _providers({"api_key": "${JEV_TEST_KEY}"})) == _FAKE_KEY
    assert resolve_key(_OPENROUTER, _providers({})) == _FAKE_KEY


def test_key_typesafe_resolves_env_in_code(clean_env) -> None:
    clean_env.setenv(TYPESAFE_KEY_ENV, _FAKE_KEY)
    assert resolve_key(_TYPESAFE, {}) == _FAKE_KEY


def test_key_typesafe_config_key_wins(clean_env) -> None:
    clean_env.setenv(TYPESAFE_KEY_ENV, "other-synthetic-key")
    settings = JevSettings.model_validate({"routes": {"typesafe": {"api_key": _FAKE_KEY}}})
    assert resolve_key(settings, {}) == _FAKE_KEY


@pytest.mark.parametrize("case", ["missing", "empty", "placeholder"])
@pytest.mark.parametrize("route", ["typesafe", "openrouter"])
def test_key_missing_empty_or_placeholder_raises_no_key(clean_env, case: str, route: str) -> None:
    env_var = OPENROUTER_KEY_ENV if route == "openrouter" else TYPESAFE_KEY_ENV
    configured: str | None = None
    if case == "empty":
        clean_env.setenv(env_var, "")
        configured = "   "
    elif case == "placeholder":
        clean_env.setenv(env_var, "${JEV_TEST_UNSET_KEY}")
        configured = "${JEV_TEST_UNSET_KEY}"
    if route == "openrouter":
        settings = _OPENROUTER
        config = _providers({} if configured is None else {"api_key": configured})
    else:
        settings = JevSettings.model_validate({"routes": {"typesafe": {"api_key": configured}}})
        config = {}
    with pytest.raises(DecisionAuthError) as info:
        resolve_route(settings, config)
    assert info.value.error_type == "no_key"
    assert info.value.status is None
    assert env_var in info.value.message


def test_key_not_in_resolved_route_repr(clean_env) -> None:
    clean_env.setenv(TYPESAFE_KEY_ENV, _FAKE_KEY)
    assert _FAKE_KEY not in repr(resolve_route(_TYPESAFE, {}))
