"""The optimizer ServerDefinition: opt-in, no launch token, nothing gated, error guidance on every tool."""

import asyncio
import importlib
import pkgutil

from osprey.registry.mcp import resolve_servers
from tests.registry._ctx import base_ctx


def _resolve(cfg=None):
    (optimizer,) = [s for s in resolve_servers(cfg or {}, base_ctx()) if s["name"] == "optimizer"]
    return optimizer


def _registered_tools() -> list[str]:
    """The tools the server's ``@mcp.tool()`` decorators register."""
    from osprey.mcp_server.optimizer import server, tools

    for found in pkgutil.iter_modules(tools.__path__):
        importlib.import_module(f"{tools.__name__}.{found.name}")
    return [tool.name for tool in asyncio.run(server.mcp.list_tools())]


def test_module_and_env():
    optimizer = _resolve()
    assert optimizer["args"] == ["-m", "osprey.mcp_server.optimizer"]
    assert optimizer["env"]["OSPREY_CONFIG"] == "/tmp/test-project/build/config.yml"
    assert optimizer["env"]["CONFIG_FILE"] == "/tmp/test-project/build/config.yml"
    assert optimizer["env"]["OPTIMIZER_URL"] == "${OPTIMIZER_URL:-}"


def test_no_launch_token_reaches_the_server():
    assert not any("TOKEN" in name for name in _resolve()["env"])


def test_off_by_default_and_on_by_override():
    assert _resolve()["enabled"] is False
    assert _resolve({"servers": {"optimizer": {"enabled": True}}})["enabled"] is True


def test_every_registered_tool_is_allowed_and_nothing_asks_or_gates():
    """Drafts move nothing, and pause/cancel are stops, which must never wait on a prompt."""
    optimizer = _resolve()
    allowed = optimizer["permissions_allow"]
    assert len(allowed) == len(set(allowed))
    assert sorted(allowed) == sorted(_registered_tools())
    assert optimizer["permissions_ask"] == []
    assert optimizer["hooks_pre"] == []


def test_error_guidance_runs_after_every_tool():
    (rule,) = _resolve()["hooks_post"]
    assert rule["matcher"] == "mcp__optimizer__.*"
    assert "osprey_error_guidance.py" in rule["hooks"][0]["command"]
