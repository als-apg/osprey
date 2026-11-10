"""The optimizer ServerDefinition: opt-in, no launch token, stops ask, error guidance on every tool."""

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


def test_every_registered_tool_is_allowed_or_asked_and_only_the_stops_ask():
    optimizer = _resolve()
    allowed, asked = optimizer["permissions_allow"], optimizer["permissions_ask"]
    assert len(allowed) == len(set(allowed))
    assert not set(allowed) & set(asked)
    assert sorted([*allowed, *asked]) == sorted(_registered_tools())
    assert set(asked) == {"pause_job", "cancel_job"}


def test_the_stops_carry_approval_only_never_the_kill_switch():
    rules = {rule["matcher"]: rule for rule in _resolve()["hooks_pre"]}
    assert set(rules) == {"mcp__optimizer__pause_job", "mcp__optimizer__cancel_job"}
    for rule in rules.values():
        (hook,) = rule["hooks"]
        assert "osprey_approval.py" in hook["command"]


def test_error_guidance_runs_after_every_tool():
    (rule,) = _resolve()["hooks_post"]
    assert rule["matcher"] == "mcp__optimizer__.*"
    assert "osprey_error_guidance.py" in rule["hooks"][0]["command"]
