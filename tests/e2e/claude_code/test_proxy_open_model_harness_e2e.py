"""L5 — full e2e: an OPEN model drives the real Claude Code harness + an MCP tool
through OSPREY's translation proxy.

Real `claude` CLI (via claude_agent_sdk) -> ANTHROPIC_BASE_URL = OSPREY proxy ->
open model. In-process MCP server exposes read_pv; we assert the agent actually
calls it and relays the result. This is the cborg-coder scenario; locally an
Ollama open model stands in (swap proxy upstream + model = the CBORG case).

VERY SLOW and OPT-IN. A local 32B at ~24K context over 2 turns is ~20 min, so these
are gated behind explicit env flags and never run in the default suite:

  * Ollama variant: OSPREY_E2E_OLLAMA=1  + a large-context tool model present.
        Default 4096 num_ctx TRUNCATES Claude Code's ~24K-token 32-tool prompt and
        the model returns empty — so a large-num_ctx variant is required, e.g.:
            printf 'FROM qwen2.5:32b\\nPARAMETER num_ctx 32768\\n' | ollama create qwen2.5:32b-32k -f -
  * CBORG: OSPREY_E2E_CBORG=1 + on LBLnet/VPN. cborg-coder has a 114K context, so no
        num_ctx workaround is needed there.

Experiment branch: experiment/cborg-claude-code (issue #259).
"""

from __future__ import annotations

import os
import subprocess

import httpx
import pytest

from osprey.infrastructure.proxy.lifecycle import start_proxy, stop_proxy

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

OLLAMA_BASE = "http://localhost:11434/v1"
OLLAMA_MODEL = os.environ.get("OSPREY_E2E_OLLAMA_MODEL", "qwen2.5:32b-32k")
CBORG_BASE = "https://api.cborg.lbl.gov/v1"
CBORG_MODEL = os.environ.get("OSPREY_E2E_CBORG_MODEL", "cborg-coder")


def _claude_cli() -> bool:
    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        return (
            subprocess.run(
                ["claude", "--version"], capture_output=True, timeout=10, env=env
            ).returncode
            == 0
        )
    except Exception:
        return False


def _ollama_has(model: str) -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        return r.status_code == 200 and any(m["name"] == model for m in r.json().get("models", []))
    except Exception:
        return False


@tool("read_pv", "Read the current value of an EPICS process variable (PV).", {"name": str})
async def _read_pv(args):
    return {"content": [{"type": "text", "text": f"{args['name']} = 3.14159 mm"}]}


async def _drive_harness(upstream_base: str, upstream_key: str, model: str):
    """Run the real claude CLI against `model` via the proxy; return (tool_calls, final_text)."""
    server = create_sdk_mcp_server(name="controls", version="1.0.0", tools=[_read_pv])
    port = start_proxy(upstream_base, upstream_key)
    options = ClaudeAgentOptions(
        model=model,
        permission_mode="bypassPermissions",
        max_turns=6,
        mcp_servers={"controls": server},
        allowed_tools=["mcp__controls__read_pv"],
        setting_sources=[],
        env={
            "CLAUDECODE": "",
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            "ANTHROPIC_AUTH_TOKEN": upstream_key,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_SMALL_FAST_MODEL": model,
        },
    )
    tool_calls: list[str] = []
    texts: list[str] = []
    try:
        async for msg in query(
            prompt="Use the read_pv tool to read the value of the PV named BPM:01:X, then report the value.",
            options=options,
        ):
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, ToolUseBlock):
                        tool_calls.append(b.name)
                    elif isinstance(b, TextBlock):
                        texts.append(b.text)
            elif isinstance(msg, ResultMessage):
                pass
    finally:
        stop_proxy()
    return tool_calls, " ".join(texts)


@pytest.mark.skipif(
    os.environ.get("OSPREY_E2E_OLLAMA") != "1"
    or not _claude_cli()
    or not _ollama_has(OLLAMA_MODEL),
    reason="opt-in: set OSPREY_E2E_OLLAMA=1, install claude CLI, and create a large-num_ctx Ollama model",
)
async def test_open_model_drives_claude_code_harness_via_proxy_ollama():
    tool_calls, final_text = await _drive_harness(OLLAMA_BASE, "ollama", OLLAMA_MODEL)
    assert any("read_pv" in t for t in tool_calls), (
        f"no MCP tool call: {tool_calls} / {final_text!r}"
    )
    assert "3.14159" in final_text, f"tool result not relayed: {final_text!r}"


def _cborg_reachable() -> bool:
    key = os.environ.get("CBORG_API_KEY")
    if not key:
        return False
    try:
        r = httpx.post(
            CBORG_BASE + "/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": CBORG_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
            timeout=30.0,
        )
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    os.environ.get("OSPREY_E2E_CBORG") != "1" or not _claude_cli() or not _cborg_reachable(),
    reason="opt-in: set OSPREY_E2E_CBORG=1, install claude CLI, and be on LBLnet/VPN",
)
async def test_cborg_coder_drives_claude_code_harness_via_proxy():
    tool_calls, final_text = await _drive_harness(
        CBORG_BASE, os.environ["CBORG_API_KEY"], CBORG_MODEL
    )
    assert any("read_pv" in t for t in tool_calls), (
        f"no MCP tool call: {tool_calls} / {final_text!r}"
    )
    assert "3.14159" in final_text, f"tool result not relayed: {final_text!r}"
