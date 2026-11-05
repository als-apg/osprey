"""The optimizer ServerDefinition: opt-in, read-only, error guidance on every tool."""

from osprey.registry.mcp import resolve_servers
from tests.registry._ctx import base_ctx

_READ_TOOLS = [
    "list_environments",
    "get_environment",
    "environment_status",
    "read_variables",
    "read_observables",
    "list_presets",
    "initial_bounds",
    "resolve_bounds",
    "list_jobs",
    "get_job",
    "current_job",
    "get_job_config",
    "list_results",
    "get_result",
    "list_models",
]


def _resolve(cfg=None):
    (optimizer,) = [s for s in resolve_servers(cfg or {}, base_ctx()) if s["name"] == "optimizer"]
    return optimizer


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


def test_every_tool_is_allowed_and_nothing_asks_or_gates():
    optimizer = _resolve()
    assert optimizer["permissions_allow"] == _READ_TOOLS
    assert optimizer["permissions_ask"] == []
    assert optimizer["hooks_pre"] == []


def test_error_guidance_runs_after_every_tool():
    (rule,) = _resolve()["hooks_post"]
    assert rule["matcher"] == "mcp__optimizer__.*"
    assert "osprey_error_guidance.py" in rule["hooks"][0]["command"]
