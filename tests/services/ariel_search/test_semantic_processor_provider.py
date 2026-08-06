"""The semantic processor resolves its LLM from one provider key.

``ariel.enhancement_modules.semantic_processor`` used to carry two ``provider``
keys: the module-level one fed tier resolution, the nested ``model.provider``
selected the LLM and sourced its credentials.  They could diverge silently, and
omitting the module-level one substituted ``ariel.embedding.provider`` — an
*embedding* endpoint — for an LLM module.

There is now one key.  The module-level ``provider`` drives both the tier lookup
and the completion call, it has no default, and a missing value is an error that
names the config key rather than a quiet fall-through to Ollama.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.enhancement.semantic_processor.processor import (
    SemanticProcessorModule,
)


@pytest.fixture
def provider_models(monkeypatch) -> None:
    """Give ``cborg`` a tier map so tier aliases resolve to a concrete model ID."""
    import osprey.models.config as models_config

    def fake_provider_config(provider: str, config_path: str | None = None) -> dict[str, Any]:
        if provider == "cborg":
            return {"models": {"haiku": "anthropic/claude-haiku"}}
        return {}

    monkeypatch.setattr(models_config, "get_provider_config", fake_provider_config)


def _config(**overrides: Any) -> dict[str, Any]:
    """Build a semantic_processor module config dict."""
    config: dict[str, Any] = {"enabled": True, "provider": "cborg"}
    config.update(overrides)
    return config


class TestModuleProviderDrivesBothUses:
    """The single module-level key feeds tier resolution and the completion call."""

    def test_tier_alias_resolves_from_module_provider(self, provider_models) -> None:
        """A tier alias is resolved against the module-level provider."""
        module = SemanticProcessorModule()
        module.configure(_config(model={"model_id": "haiku", "max_tokens": 256}))

        assert module._model_config["model_id"] == "anthropic/claude-haiku"

    def test_module_provider_reaches_model_config(self, provider_models) -> None:
        """The completion call is given the module-level provider."""
        module = SemanticProcessorModule()
        module.configure(_config(model={"model_id": "haiku"}))

        assert module._model_config["provider"] == "cborg"

    def test_nested_model_provider_is_no_longer_required(self, provider_models) -> None:
        """Omitting the retired ``model.provider`` still yields a usable model config."""
        module = SemanticProcessorModule()
        module.configure(_config(model={"model_id": "anthropic/claude-haiku", "max_tokens": 256}))

        assert module._model_config == {
            "provider": "cborg",
            "model_id": "anthropic/claude-haiku",
            "max_tokens": 256,
        }

    def test_legacy_nested_provider_does_not_win(self, provider_models) -> None:
        """A leftover ``model.provider`` from an old config cannot override the module key."""
        module = SemanticProcessorModule()
        module.configure(
            _config(model={"provider": "ollama", "model_id": "anthropic/claude-haiku"})
        )

        assert module._model_config["provider"] == "cborg"

    @pytest.mark.asyncio
    async def test_provider_reaches_the_completion_call(self, provider_models) -> None:
        """The provider the module resolved is the one the LLM call is made with."""
        import osprey.models.completion as completion

        captured: dict[str, Any] = {}

        def fake_completion(message: str, model_config: dict[str, Any] | None = None, **kwargs):
            captured["model_config"] = model_config
            return '{"keywords": ["vacuum"], "summary": "Pump swapped."}'

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(completion, "get_chat_completion", fake_completion)
            module = SemanticProcessorModule()
            module.configure(_config(model={"model_id": "haiku"}))
            result = await module._process_text("VP-103 replaced")

        assert result is not None
        assert captured["model_config"]["provider"] == "cborg"
        assert captured["model_config"]["model_id"] == "anthropic/claude-haiku"


class TestMissingProviderIsAnError:
    """No provider means an actionable error, never a substituted default."""

    def test_missing_provider_raises_naming_the_key(self) -> None:
        """The message names the dotted config key the operator has to set."""
        module = SemanticProcessorModule()

        with pytest.raises(ValueError) as excinfo:
            module.configure({"enabled": True, "model": {"model_id": "claude-haiku"}})

        assert "ariel.enhancement_modules.semantic_processor.provider" in str(excinfo.value)

    def test_missing_provider_does_not_substitute_ollama(self) -> None:
        """The embedding default is never silently used for the LLM module."""
        module = SemanticProcessorModule()

        with pytest.raises(ValueError) as excinfo:
            module.configure({"enabled": True})

        assert "ollama" not in str(excinfo.value)
        assert module._model_config == {}

    def test_nested_provider_alone_is_not_enough(self) -> None:
        """The retired nested key does not satisfy the requirement."""
        module = SemanticProcessorModule()

        with pytest.raises(ValueError):
            module.configure({"enabled": True, "model": {"provider": "cborg"}})


class TestEmbeddingProviderIsNotAFallback:
    """``ariel.embedding.provider`` stops leaking into the LLM module's config."""

    def test_semantic_processor_gets_no_embedding_fallback(self) -> None:
        """A provider-less semantic_processor is handed ``None``, not ``ollama``."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {"semantic_processor": {"enabled": True}},
                "embedding": {"provider": "ollama"},
            }
        )

        module_config = config.get_enhancement_module_config("semantic_processor")

        assert module_config is not None
        assert module_config["provider"] is None

    def test_explicit_provider_is_passed_through(self) -> None:
        """An explicit module provider survives the config round-trip."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "semantic_processor": {"enabled": True, "provider": "cborg"}
                },
                "embedding": {"provider": "ollama"},
            }
        )

        module_config = config.get_enhancement_module_config("semantic_processor")

        assert module_config is not None
        assert module_config["provider"] == "cborg"

    def test_embedding_modules_keep_the_fallback(self) -> None:
        """Negative control: the fallback still serves the module it was written for."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "text_embedding": {
                        "enabled": True,
                        "models": [{"name": "nomic-embed-text", "dimension": 768}],
                    }
                },
                "embedding": {"provider": "ollama"},
            }
        )

        module_config = config.get_enhancement_module_config("text_embedding")

        assert module_config is not None
        assert module_config["provider"] == "ollama"


class TestEndToEndThroughConfig:
    """The whole path from ``config.yml`` shape to a configured module."""

    def test_configured_provider_survives_the_full_path(self, provider_models) -> None:
        """One key in the config dict lands on the module's model config."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "semantic_processor": {
                        "enabled": True,
                        "provider": "cborg",
                        "model": {"model_id": "haiku", "max_tokens": 256},
                    }
                },
                "embedding": {"provider": "ollama"},
            }
        )
        module_config = config.get_enhancement_module_config("semantic_processor")
        assert module_config is not None

        module = SemanticProcessorModule()
        module.configure(module_config)

        assert module._model_config["provider"] == "cborg"
        assert module._model_config["model_id"] == "anthropic/claude-haiku"

    def test_provider_less_config_fails_loudly(self) -> None:
        """An enabled module with no provider errors instead of running on Ollama."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "semantic_processor": {"enabled": True, "model": {"model_id": "haiku"}}
                },
                "embedding": {"provider": "ollama"},
            }
        )
        module_config = config.get_enhancement_module_config("semantic_processor")
        assert module_config is not None

        with pytest.raises(ValueError, match="semantic_processor.provider"):
            SemanticProcessorModule().configure(module_config)
