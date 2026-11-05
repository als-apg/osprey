"""Tests for utils/redis_client.py — get_redis_url and get_redis_client."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest
import redis as redis_lib

import osprey_optimizer.utils.redis_client as rc_module
from osprey_optimizer.utils.redis_state_store import RedisStateStore


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-level cached clients between tests."""
    rc_module.reset_redis_clients()
    yield
    rc_module.reset_redis_clients()


class TestGetRedisUrl:
    @pytest.mark.parametrize(
        "env_value,expected",
        [
            (None, "redis://localhost:6379/0"),
            ("redis://custom:1234/1", "redis://custom:1234/1"),
        ],
        ids=["default", "from_env"],
    )
    def test_get_redis_url(
        self, monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: str
    ) -> None:
        if env_value is None:
            monkeypatch.delenv("REDIS_URL", raising=False)
        else:
            monkeypatch.setenv("REDIS_URL", env_value)
        assert rc_module.get_redis_url() == expected


class TestGetRedisClient:
    def test_caching_returns_same_instance(self) -> None:
        mock_client = MagicMock()
        with patch(
            "osprey_optimizer.utils.redis_client.Redis.from_url", return_value=mock_client
        ) as mock_from_url:
            first = rc_module.get_redis_client(binary=False)
            second = rc_module.get_redis_client(binary=False)
        assert first is second
        mock_from_url.assert_called_once()


class TestPingUrl:
    @pytest.mark.parametrize("ping_fails", [False, True], ids=["ping_ok", "ping_fails"])
    def test_pings_and_always_closes_throwaway_client(self, ping_fails: bool) -> None:
        """The throwaway client is closed on both the success and failure paths."""
        mock_client = MagicMock()
        if ping_fails:
            mock_client.ping.side_effect = redis_lib.ConnectionError("down")
        ctx = pytest.raises(redis_lib.ConnectionError) if ping_fails else contextlib.nullcontext()
        with (
            patch(
                "osprey_optimizer.utils.redis_client.Redis.from_url",
                return_value=mock_client,
            ) as mock_from_url,
            ctx,
        ):
            rc_module.ping_url("redis://localhost:6379")
        mock_from_url.assert_called_once_with("redis://localhost:6379")
        mock_client.ping.assert_called_once()
        mock_client.close.assert_called_once()


def test_require_redis_client_uninitialized_raises() -> None:
    """_require_redis_client with no client -> raise RuntimeError."""
    store = RedisStateStore()
    with pytest.raises(RuntimeError, match="Redis client not initialized"):
        store._require_redis_client()
