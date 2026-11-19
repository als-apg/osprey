"""The one list of features that make their own model call, and the "no key" rule.

A personal Claude login covers the agent. The features listed by
:mod:`osprey.models.inner_calls` send a completion themselves and need the
provider's API key. These tests pin which features a project config turns on,
that a keyless provider is never reported, that an unset ``${VAR}`` is no key,
and that :func:`get_chat_completion` refuses such a key before it reaches the
provider.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.models.api_keys import api_key_env_var, api_key_missing
from osprey.models.inner_calls import (
    FEATURES,
    SESSION_NAMING,
    inner_model_calls,
    missing_key_warnings,
)

CATALOG: dict[str, Any] = {
    "anthropic": {"api_key": "${ANTHROPIC_API_KEY}"},
    "cborg": {"api_key": "${CBORG_API_KEY}"},
    "ollama": {"api_key": "ollama"},
}


def _config(**blocks: Any) -> dict[str, Any]:
    """A project config on ``anthropic`` with the catalog above, *blocks* laid over."""
    return {"claude_code": {"provider": "anthropic"}, "api": {"providers": CATALOG}, **blocks}


def _features(config: dict[str, Any], **environ: str) -> dict[str, tuple[str, str, bool]]:
    return {
        call.feature: (call.provider, call.key_env_var, call.key_present)
        for call in inner_model_calls(config, environ=environ)
    }


ALWAYS_ON = {"session naming", "logbook compose"}


class TestWhichFeaturesAConfigTurnsOn:
    def test_a_bare_project_names_the_two_features_that_are_always_on(self):
        assert set(_features(_config())) == ALWAYS_ON

    def test_no_provider_means_no_feature(self):
        assert inner_model_calls({"api": {"providers": CATALOG}}, environ={}) == []

    @pytest.mark.parametrize("mode", ["hierarchical", "middle_layer", "graph"])
    def test_only_the_in_context_channel_finder_asks_a_model(self, mode):
        found = _features(_config(channel_finder={"pipeline_mode": mode}))
        assert "in-context channel finder" not in found

    def test_the_in_context_channel_finder_is_on_the_agents_provider(self):
        found = _features(_config(channel_finder={"pipeline_mode": "in_context"}))
        assert found["in-context channel finder"] == ("anthropic", "ANTHROPIC_API_KEY", False)

    def test_its_own_subagent_provider_wins(self):
        finder = {
            "pipeline_mode": "in_context",
            "pipelines": {"in_context": {"subagent_provider": "cborg"}},
        }
        found = _features(_config(channel_finder=finder))
        assert found["in-context channel finder"] == ("cborg", "CBORG_API_KEY", False)

    def test_logbook_compose_takes_its_own_provider_over_the_agents(self):
        found = _features(_config(logbook={"composition": {"provider": "cborg"}}))
        assert found["logbook compose"][0] == "cborg"
        assert found["session naming"][0] == "anthropic"

    @pytest.mark.parametrize(
        ("module", "feature"),
        [
            ("semantic_processor", "ARIEL semantic processor"),
            ("image_caption", "ARIEL image captions"),
        ],
    )
    def test_an_ariel_module_is_listed_only_when_enabled(self, module, feature):
        def ariel(enabled: bool) -> dict[str, Any]:
            block = {"enabled": enabled, "provider": "cborg"}
            return _config(ariel={"enhancement_modules": {module: block}})

        assert feature not in _features(ariel(False))
        assert _features(ariel(True))[feature] == ("cborg", "CBORG_API_KEY", False)

    def test_health_model_chat_is_listed_per_provider_under_models(self):
        models = {
            "a": {"provider": "cborg", "model_id": "x"},
            "b": {"provider": "cborg", "model_id": "y"},
            "c": {"provider": "anthropic", "model_id": "z"},
        }
        calls = inner_model_calls(_config(models=models), environ={})
        chat = [call.provider for call in calls if call.feature == "health model chat"]
        assert chat == ["anthropic", "cborg"]

    def test_the_session_namer_can_find_its_own_row(self):
        assert SESSION_NAMING in FEATURES
        assert SESSION_NAMING.name in _features(_config())


class TestAKeylessProviderIsNeverReported:
    def test_a_feature_on_a_keyless_provider_is_left_out(self):
        ariel = {
            "enhancement_modules": {"semantic_processor": {"enabled": True, "provider": "ollama"}}
        }
        assert "ARIEL semantic processor" not in _features(_config(ariel=ariel))

    def test_a_whole_project_on_a_keyless_provider_reports_nothing(self):
        config = _config(channel_finder={"pipeline_mode": "in_context"})
        config["claude_code"]["provider"] = "ollama"
        assert inner_model_calls(config, environ={}) == []

    def test_a_provider_the_registry_does_not_hold_is_left_to_the_call(self):
        config = _config()
        config["claude_code"]["provider"] = "no-such-provider"
        assert inner_model_calls(config, environ={}) == []


class TestWhetherTheKeyIsSet:
    def test_a_reference_the_environment_resolves_is_a_key(self):
        found = _features(_config(), ANTHROPIC_API_KEY="sk-ant")
        assert found["session naming"] == ("anthropic", "ANTHROPIC_API_KEY", True)
        assert missing_key_warnings(_config(), environ={"ANTHROPIC_API_KEY": "sk-ant"}) == []

    def test_an_unset_reference_is_no_key(self):
        assert _features(_config())["session naming"][2] is False

    def test_an_entry_with_no_api_key_is_no_key_even_with_the_variable_set(self):
        # The call reads api.providers.<name>.api_key and nothing else.
        config = _config()
        config["api"] = {"providers": {"anthropic": {"base_url": "https://api.anthropic.com"}}}
        found = _features(config, ANTHROPIC_API_KEY="sk-ant")
        assert found["session naming"] == ("anthropic", "ANTHROPIC_API_KEY", False)

    def test_the_variable_named_is_the_one_the_entry_references(self):
        config = _config()
        config["api"] = {"providers": {"anthropic": {"api_key": "${SITE_CLAUDE_KEY}"}}}
        assert _features(config)["session naming"][1] == "SITE_CLAUDE_KEY"

    def test_a_literal_key_is_a_key(self):
        config = _config()
        config["api"] = {"providers": {"anthropic": {"api_key": "sk-ant-literal"}}}
        assert _features(config)["session naming"][2] is True

    def test_the_warning_names_the_feature_the_variable_and_the_loss(self):
        assert missing_key_warnings(_config(), environ={}) == [
            "session naming needs ANTHROPIC_API_KEY (not set): "
            "sessions keep their first-prompt title",
            "logbook compose needs ANTHROPIC_API_KEY (not set): "
            "the compose panel cannot draft an entry",
        ]

    @pytest.mark.parametrize("value", [None, "", "${MISSING}", "$MISSING"])
    def test_what_counts_as_no_key(self, value):
        assert api_key_missing(value) is True

    @pytest.mark.parametrize("value", ["sk-ant", "EMPTY", "prefix-${MISSING}"])
    def test_what_counts_as_a_key(self, value):
        assert api_key_missing(value) is False

    def test_the_env_var_of_a_provider(self):
        assert api_key_env_var("anthropic") == "ANTHROPIC_API_KEY"
        assert api_key_env_var("anthropic", "${SITE_CLAUDE_KEY}") == "SITE_CLAUDE_KEY"
        assert api_key_env_var("ollama") is None


class TestCompletionRefusesAPlaceholderKey:
    """``get_chat_completion`` fails locally on an unset ``${VAR}`` key and sends nothing."""

    @pytest.fixture
    def sent(self, monkeypatch) -> list[dict[str, Any]]:
        from osprey.models.provider_registry import get_provider_registry

        calls: list[dict[str, Any]] = []

        def fake_execute(self, **kwargs):  # noqa: ARG001 - stands in for the adapter's execute
            calls.append(kwargs)
            return "ok"

        provider_class = get_provider_registry().get_provider("anthropic")
        monkeypatch.setattr(provider_class, "execute_completion", fake_execute)
        return calls

    def _entry(self, monkeypatch, api_key: object) -> None:
        from osprey.models import completion as completion_module

        monkeypatch.setattr(
            completion_module,
            "get_provider_config",
            lambda provider: {"api_key": api_key, "default_model_id": "a-model"},
        )

    def test_an_unset_placeholder_is_refused_by_name_and_nothing_is_sent(self, monkeypatch, sent):
        from osprey.models.completion import get_chat_completion

        self._entry(monkeypatch, "${ANTHROPIC_API_KEY}")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set") as refused:
            get_chat_completion(message="ping", provider="anthropic", max_tokens=4)
        assert "API key required for anthropic" in str(refused.value)
        assert sent == []

    def test_a_missing_key_names_the_providers_own_variable(self, monkeypatch, sent):
        from osprey.models.completion import get_chat_completion

        self._entry(monkeypatch, None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            get_chat_completion(message="ping", provider="anthropic", max_tokens=4)
        assert sent == []

    def test_a_real_key_still_reaches_the_provider(self, monkeypatch, sent):
        from osprey.models.completion import get_chat_completion

        self._entry(monkeypatch, "sk-ant")
        assert get_chat_completion(message="ping", provider="anthropic", max_tokens=4) == "ok"
        assert [call["api_key"] for call in sent] == ["sk-ant"]
