"""Tests for the LLM-based channel name generator.

The LLM/provider boundary is mocked completely so these tests run keyless and
offline. The single seam is ``get_chat_completion`` imported into the module.
"""

from __future__ import annotations

import pytest

from osprey.services.channel_finder.tools import llm_channel_namer as mod
from osprey.services.channel_finder.tools.llm_channel_namer import (
    ChannelNames,
    LLMChannelNamer,
    create_namer_from_config,
)


@pytest.fixture()
def namer() -> LLMChannelNamer:
    return LLMChannelNamer(provider="als-apg", model_id="test-model", batch_size=2)


def _patch_completion(monkeypatch, result, calls: list | None = None):
    """Patch the module-level get_chat_completion to return ``result``.

    ``result`` may be a value (returned for every call) or a callable invoked
    with the keyword arguments of each call.
    """

    def fake(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        if callable(result):
            return result(**kwargs)
        return result

    monkeypatch.setattr(mod, "get_chat_completion", fake)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptForBatch:
    def test_includes_exact_channel_count_and_entries(self, namer: LLMChannelNamer):
        channels = [
            {"short_name": "SX3Set", "description": "steering coil set value"},
            {"short_name": "IP149A", "description": "ion pump pressure"},
        ]
        prompt = namer._create_prompt_for_batch(channels)
        assert "generate EXACTLY 2 names" in prompt
        assert '1. Short: "SX3Set"' in prompt
        assert '2. Short: "IP149A"' in prompt
        assert "steering coil set value" in prompt
        assert "ion pump pressure" in prompt


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


class TestIsValidChannelName:
    @pytest.mark.parametrize(
        "name",
        ["Abc", "Valid123Name", "Ab_cd", "SteeringCoilXSetPoint"],
    )
    def test_accepts_valid_names(self, namer: LLMChannelNamer, name: str):
        assert namer._is_valid_channel_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "",  # empty
            "Ab",  # too short
            "abc",  # lowercase first char
            "Ab-cd",  # non-alnum char
            "Ab cd",  # space
            "A" * 81,  # too long
        ],
    )
    def test_rejects_invalid_names(self, namer: LLMChannelNamer, name: str):
        assert namer._is_valid_channel_name(name) is False


# ---------------------------------------------------------------------------
# generate_names_batch
# ---------------------------------------------------------------------------


class TestGenerateNamesBatch:
    def test_empty_batch_returns_empty_without_calling_llm(
        self, namer: LLMChannelNamer, monkeypatch
    ):
        calls: list = []
        _patch_completion(monkeypatch, ChannelNames(names=[]), calls)
        assert namer.generate_names_batch([]) == []
        assert calls == []

    def test_success_returns_model_names(self, namer: LLMChannelNamer, monkeypatch):
        channels = [
            {"short_name": "a", "description": "d1"},
            {"short_name": "b", "description": "d2"},
        ]
        _patch_completion(monkeypatch, ChannelNames(names=["NameOne", "NameTwo"]))
        assert namer.generate_names_batch(channels) == ["NameOne", "NameTwo"]

    def test_accepts_dict_shaped_result(self, namer: LLMChannelNamer, monkeypatch):
        channels = [{"short_name": "a", "description": "d1"}]
        _patch_completion(monkeypatch, {"names": ["NameOne"]})
        assert namer.generate_names_batch(channels) == ["NameOne"]

    def test_invalid_generated_name_falls_back_to_short_name(
        self, namer: LLMChannelNamer, monkeypatch
    ):
        channels = [
            {"short_name": "keepme", "description": "d1"},
            {"short_name": "fallback_original", "description": "d2"},
        ]
        # Second name is invalid (lowercase) so the original short_name is used.
        _patch_completion(monkeypatch, ChannelNames(names=["ValidName", "bad"]))
        result = namer.generate_names_batch(channels)
        assert result == ["ValidName", "fallback_original"]

    def test_wrong_count_falls_back_to_short_names(self, namer: LLMChannelNamer, monkeypatch):
        channels = [
            {"short_name": "s1", "description": "d1"},
            {"short_name": "s2", "description": "d2"},
        ]
        # Only one name returned for two channels -> ValueError -> fallback path.
        _patch_completion(monkeypatch, ChannelNames(names=["OnlyOne"]))
        assert namer.generate_names_batch(channels) == ["s1", "s2"]

    def test_llm_exception_falls_back_to_short_names(self, namer: LLMChannelNamer, monkeypatch):
        channels = [{"short_name": "s1", "description": "d1"}]

        def boom(**kwargs):
            raise RuntimeError("provider down")

        _patch_completion(monkeypatch, boom)
        assert namer.generate_names_batch(channels) == ["s1"]

    def test_provider_config_passthrough_when_credentials_set(self, monkeypatch):
        namer = LLMChannelNamer(
            provider="als-apg",
            model_id="m",
            base_url="https://example/api",
            api_key="secret",
        )
        calls: list = []
        _patch_completion(monkeypatch, ChannelNames(names=["NameOne"]), calls)
        namer.generate_names_batch([{"short_name": "a", "description": "d"}])
        assert len(calls) == 1
        cfg = calls[0]["provider_config"]
        assert cfg == {"base_url": "https://example/api", "api_key": "secret"}
        assert calls[0]["base_url"] == "https://example/api"

    def test_provider_config_none_when_no_credentials(self, namer: LLMChannelNamer, monkeypatch):
        calls: list = []
        _patch_completion(monkeypatch, ChannelNames(names=["NameOne"]), calls)
        namer.generate_names_batch([{"short_name": "a", "description": "d"}])
        assert calls[0]["provider_config"] is None


# ---------------------------------------------------------------------------
# resolve_duplicates
# ---------------------------------------------------------------------------


class TestResolveDuplicates:
    def test_no_duplicates_returns_names_unchanged_without_llm(
        self, namer: LLMChannelNamer, monkeypatch
    ):
        calls: list = []
        _patch_completion(monkeypatch, ChannelNames(names=[]), calls)
        channels = [
            {"short_name": "a", "description": "d1"},
            {"short_name": "b", "description": "d2"},
        ]
        names = ["Alpha", "Beta"]
        assert namer.resolve_duplicates(channels, names) == ["Alpha", "Beta"]
        assert calls == []

    def test_duplicates_replaced_with_regenerated_names(self, namer: LLMChannelNamer, monkeypatch):
        channels = [
            {"short_name": "a", "description": "d1"},
            {"short_name": "b", "description": "d2"},
            {"short_name": "c", "description": "d3"},
        ]
        names = ["Dup", "Dup", "Unique"]
        _patch_completion(monkeypatch, ChannelNames(names=["DupSpecificA", "DupSpecificB"]))
        result = namer.resolve_duplicates(channels, names)
        assert result == ["DupSpecificA", "DupSpecificB", "Unique"]

    def test_duplicate_resolution_llm_failure_keeps_original(
        self, namer: LLMChannelNamer, monkeypatch
    ):
        channels = [
            {"short_name": "a", "description": "d1"},
            {"short_name": "b", "description": "d2"},
        ]
        names = ["Dup", "Dup"]

        def boom(**kwargs):
            raise RuntimeError("provider down")

        _patch_completion(monkeypatch, boom)
        assert namer.resolve_duplicates(channels, names) == ["Dup", "Dup"]

    def test_resolution_prompt_lists_groups_and_counts(self, namer: LLMChannelNamer):
        groups = {
            "Dup": [
                {"short_name": "a", "description": "d1", "original_name": "Dup"},
                {"short_name": "b", "description": "d2", "original_name": "Dup"},
            ]
        }
        prompt = namer._create_duplicate_resolution_prompt(groups)
        assert "GENERATE EXACTLY 2 UNIQUE NAMES" in prompt
        assert "DUPLICATE GROUP: 'Dup'" in prompt
        assert "needs 2 distinct names" in prompt
        assert '"Dup"' in prompt  # previous (too generic) name shown


# ---------------------------------------------------------------------------
# generate_names (batching orchestration)
# ---------------------------------------------------------------------------


class TestGenerateNames:
    def test_batches_cover_all_channels_and_resolves(self, namer: LLMChannelNamer, monkeypatch):
        # batch_size=2, five channels -> three batches.
        channels = [{"short_name": f"s{i}", "description": f"d{i}"} for i in range(5)]

        seen_batch_sizes: list[int] = []

        def fake_batch(batch):
            seen_batch_sizes.append(len(batch))
            return [f"Name{c['short_name']}" for c in batch]

        monkeypatch.setattr(namer, "generate_names_batch", fake_batch)
        # No duplicates in the produced names -> resolve returns unchanged.
        result = namer.generate_names(channels)

        assert result == [f"Names{i}" for i in range(5)]
        assert seen_batch_sizes == [2, 2, 1]


# ---------------------------------------------------------------------------
# create_namer_from_config
# ---------------------------------------------------------------------------


class TestCreateNamerFromConfig:
    def test_missing_provider_raises(self, monkeypatch):
        import osprey.utils.config as config_mod

        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda *a, **k: {"channel_finder": {"channel_name_generation": {"llm_model": {}}}},
        )
        with pytest.raises(ValueError, match="provider is"):
            create_namer_from_config()

    def test_builds_namer_from_config_values(self, monkeypatch):
        import osprey.models.tiers as tiers_mod
        import osprey.utils.config as config_mod

        config = {
            "channel_finder": {
                "channel_name_generation": {
                    "llm_batch_size": 7,
                    "llm_model": {
                        "provider": "als-apg",
                        "model_id": "haiku",
                        "max_tokens": 555,
                    },
                }
            },
            "api": {
                "providers": {
                    "als-apg": {
                        "base_url": "https://als/api",
                        "api_key": "k",
                    }
                }
            },
        }
        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: config)
        monkeypatch.setattr(
            tiers_mod, "resolve_model_id", lambda provider, mid: f"{provider}/{mid}"
        )

        namer = create_namer_from_config()
        assert namer.provider == "als-apg"
        assert namer.model_id == "als-apg/haiku"
        assert namer.max_tokens == 555
        assert namer.batch_size == 7
        assert namer.base_url == "https://als/api"
        assert namer.api_key == "k"

    def test_uses_explicit_config_path(self, monkeypatch, tmp_path):
        import osprey.models.tiers as tiers_mod
        import osprey.utils.config as config_mod

        seen: list = []

        def fake_load(path=None):
            seen.append(path)
            return {
                "channel_finder": {"channel_name_generation": {"llm_model": {"provider": "cborg"}}}
            }

        monkeypatch.setattr(config_mod, "load_config", fake_load)
        monkeypatch.setattr(tiers_mod, "resolve_model_id", lambda p, m: m)
        cfg_path = tmp_path / "config.yml"
        create_namer_from_config(str(cfg_path))
        assert seen == [str(cfg_path)]
