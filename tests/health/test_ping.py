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
    assert sent["max_tokens"] >= 1


async def test_a_failing_completion_is_the_answer(monkeypatch: pytest.MonkeyPatch):
    def boom(**_: Any) -> str:
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr("osprey.models.completion.get_chat_completion", boom)
    config = {"api": {"providers": {"anthropic": {"default_model": "m"}}}}
    result = await ping_provider(config, "anthropic")

    assert result["status"] == "error"
    assert "401" in result["message"]


async def test_concurrent_pings_of_one_provider_send_one_request(
    monkeypatch: pytest.MonkeyPatch,
):
    import asyncio
    import threading

    calls: list[int] = []
    release = threading.Event()

    def slow_completion(**kwargs: Any) -> str:
        calls.append(kwargs["max_tokens"])
        release.wait(5)
        return "OK"

    monkeypatch.setattr("osprey.models.completion.get_chat_completion", slow_completion)
    config = {"api": {"providers": {"anthropic": {"default_model": "m"}}}}

    first = asyncio.ensure_future(ping_provider(config, "anthropic"))
    second = asyncio.ensure_future(ping_provider(config, "anthropic"))
    await asyncio.sleep(0.2)
    release.set()
    results = await asyncio.gather(first, second)

    assert len(calls) == 1
    assert [r["status"] for r in results] == ["ok", "ok"]


async def test_a_ping_leaves_room_for_a_reasoning_model(monkeypatch: pytest.MonkeyPatch):
    from osprey.health import ping

    sent: dict[str, Any] = {}
    monkeypatch.setattr(
        "osprey.models.completion.get_chat_completion", lambda **kw: sent.update(kw) or "OK"
    )
    await ping_provider({"api": {"providers": {"p": {"default_model": "m"}}}}, "p")

    assert sent["max_tokens"] == ping.MAX_TOKENS >= 32
