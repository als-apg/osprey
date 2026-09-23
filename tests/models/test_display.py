"""Human-facing model names and Claude Code alias candidates."""

from __future__ import annotations

import pytest

from osprey.models.display import claude_code_alias_candidates, display_model_name


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("claude-sonnet-5", "Sonnet 5"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5"),
        ("claude-haiku-4-5", "Haiku 4.5"),
        ("claude-fable-5-1", "Fable 5.1"),
        ("claude-opus-5", "Opus 5"),
        ("claude-opus-5-5", "Opus 5.5"),
        ("claudeopus41", "Opus 4.1"),
        ("claudehaiku45", "Haiku 4.5"),
        ("us.anthropic.claude-opus-4-8", "Opus 4.8"),
        ("us.anthropic.claude-opus-4-6-v1", "Opus 4.6"),
        ("claude-opus-4.6-high", "Opus 4.6 high"),
        ("anthropic/claude-sonnet-5", "Sonnet 5"),
        ("claude-opus-5-5[1m]", "Opus 5.5"),
        ("claude-sonnet", "Sonnet"),
        ("gpt-6-sol", "gpt-6-sol"),
        ("ollama/gpt-oss:20b", "gpt-oss:20b"),
        ("gemini-3.8-flash", "gemini-3.8-flash"),
        ("deepseek-v4-pro", "deepseek-v4-pro"),
    ],
)
def test_display_model_name(model_id, expected):
    assert display_model_name(model_id) == expected


def test_display_name_never_carries_the_vendor():
    for model_id in ("claude-sonnet-5", "anthropic/claude-opus-5", "us.anthropic.claude-opus-4-8"):
        name = display_model_name(model_id)
        assert "Claude" not in name and "claude" not in name
        assert "anthropic" not in name


class TestAliasCandidates:
    def test_newest_version_of_a_family_wins(self):
        assert claude_code_alias_candidates(["claude-sonnet-5", "claude-sonnet-4-6"]) == {
            "sonnet": "claude-sonnet-5"
        }

    def test_order_of_the_served_list_does_not_decide(self):
        assert claude_code_alias_candidates(["claude-sonnet-4-6", "claude-sonnet-5"]) == {
            "sonnet": "claude-sonnet-5"
        }

    def test_minor_version_ranks_above_its_major(self):
        served = ["claude-opus-5", "claude-opus-5-5"]
        assert claude_code_alias_candidates(served) == {"opus": "claude-opus-5-5"}

    def test_argo_compact_spelling_fills_all_three(self):
        assert claude_code_alias_candidates(
            ["claudehaiku45", "claudesonnet45", "claudeopus41"]
        ) == {
            "haiku": "claudehaiku45",
            "sonnet": "claudesonnet45",
            "opus": "claudeopus41",
        }

    def test_the_als_apg_list_fills_all_three_and_never_fable(self):
        served = [
            "claude-fable-5-1",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-4-5-20251001",
        ]
        assert claude_code_alias_candidates(served) == {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-5",
            "opus": "claude-opus-5",
        }

    def test_an_unversioned_id_still_qualifies(self):
        served = ["claude-haiku", "claude-sonnet", "claude-opus"]
        assert claude_code_alias_candidates(served) == {
            "haiku": "claude-haiku",
            "sonnet": "claude-sonnet",
            "opus": "claude-opus",
        }

    def test_a_versioned_id_ranks_above_an_unversioned_one(self):
        served = ["claude-sonnet", "claude-sonnet-4-6"]
        assert claude_code_alias_candidates(served) == {"sonnet": "claude-sonnet-4-6"}

    def test_a_tie_is_broken_by_id(self):
        served = ["claude-haiku-4-5-20251001", "claude-haiku-4-5"]
        assert claude_code_alias_candidates(served) == {"haiku": "claude-haiku-4-5"}

    def test_a_non_claude_gateway_yields_nothing(self):
        assert claude_code_alias_candidates(["gpt-6-sol"]) == {}
        assert claude_code_alias_candidates(["gpt-oss:20b", "mistral:7b"]) == {}
