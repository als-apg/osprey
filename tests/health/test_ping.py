"""A ping is one one-token completion with the model the deployment uses."""

from __future__ import annotations

from typing import Any

import pytest

from osprey.health.ping import model_for, ping_provider


def test_the_model_next_to_the_role_wins_over_the_catalogue_default():
    config = {
        "claude_code": {"provider": "gw", "default_model": "big"},
        "api": {"providers": {"gw": {"default_model": "small"}}},
    }
    assert model_for(config, "gw") == "big"
    assert model_for({"api": {"providers": {"gw": {"default_model": "small"}}}}, "gw") == "small"
    assert model_for({}, "gw") is None


def test_an_embedding_role_is_not_pinged_with_a_completion():
    config = {
        "ariel": {
            "text_embedding": {"provider": "local", "model": "embedder"},
            "image_caption": {"provider": "local", "model": "captioner"},
        }
    }
    assert model_for(config, "local") == "captioner"


async def test_a_ping_sends_one_token_and_reports_latency(monkeypatch: pytest.MonkeyPatch):
    sent: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> str:
        sent.update(kwargs)
        return "OK"

    monkeypatch.setattr("osprey.models.completion.get_chat_completion", fake_completion)
    config = {
        "models": {"m": {"provider": "anthropic", "model_id": "claude-x"}},
        "api": {"providers": {"anthropic": {}}},
    }
    result = await ping_provider(config, "anthropic")

    assert result["status"] == "ok"
    assert result["model"] == "claude-x"
    assert result["latency_ms"] >= 0
    assert sent["max_tokens"] == 1


async def test_a_failing_completion_is_the_answer(monkeypatch: pytest.MonkeyPatch):
    def boom(**_: Any) -> str:
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr("osprey.models.completion.get_chat_completion", boom)
    config = {"api": {"providers": {"anthropic": {"default_model": "m"}}}}
    result = await ping_provider(config, "anthropic")

    assert result["status"] == "error"
    assert "401" in result["message"]
