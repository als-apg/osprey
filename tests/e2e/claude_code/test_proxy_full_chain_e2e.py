"""L3→L5 bridge — the full runtime chain claude_cmd performs, end to end, locally.

Reproduces exactly what `osprey claude` does for a custom OpenAI-compatible provider:

    resolve()  ->  inject_provider_env()  ->  start_proxy()  ->  ANTHROPIC_BASE_URL rewrite

then sends a real Anthropic Messages request (text + tool) to the *running* proxy and
checks the translated open-model response. Upstream is local Ollama, standing in for
CBORG's self-hosted models (swap base_url + key + model = the CBORG case).

Covers lifecycle.start_proxy/stop_proxy (the real daemon thread), which the in-process
TestClient tests in test_proxy_live_roundtrip.py do not.

Experiment branch: experiment/cborg-claude-code (issue #259).
"""

from __future__ import annotations

import httpx
import pytest

from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver, inject_provider_env
from osprey.infrastructure.proxy.lifecycle import start_proxy, stop_proxy

OLLAMA_BASE = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:32b"


def _ollama_up() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=5.0).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(), reason="local Ollama not running")


@pytest.fixture
def running_proxy():
    """Run the resolve->inject->start_proxy chain; yield the rewritten base URL."""
    api_providers = {
        "local-oss": {
            "api_key": "${LOCAL_OSS_API_KEY}",
            "base_url": OLLAMA_BASE,
            "models": {"haiku": OLLAMA_MODEL, "sonnet": OLLAMA_MODEL, "opus": OLLAMA_MODEL},
        }
    }
    cc_config = {"provider": "local-oss", "default_model": "sonnet"}

    spec = ClaudeCodeModelResolver.resolve(cc_config, api_providers)
    assert spec.needs_proxy and spec.upstream_base_url == OLLAMA_BASE

    # Mimic the shell env claude_cmd starts from (derived secret env var name).
    environ: dict[str, str] = {spec.auth_secret_env: "ollama"}
    inject_provider_env(environ, spec)
    assert environ["ANTHROPIC_AUTH_TOKEN"] == "ollama"

    port = start_proxy(spec.upstream_base_url, environ.get(spec.auth_env_var))
    base = f"http://127.0.0.1:{port}"  # what claude_cmd writes to ANTHROPIC_BASE_URL
    try:
        yield base
    finally:
        stop_proxy()


def test_full_chain_text(running_proxy):
    resp = httpx.post(
        f"{running_proxy}/v1/messages",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly the word: PONG"}],
            "max_tokens": 16,
        },
        timeout=120.0,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["type"] == "message"
    text = "".join(b.get("text", "") for b in body["content"] if b["type"] == "text")
    assert text.strip(), body


def test_full_chain_tool_call(running_proxy):
    resp = httpx.post(
        f"{running_proxy}/v1/messages",
        json={
            "model": OLLAMA_MODEL,
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "Use read_pv to read PV BPM:01:X. You must call the tool."}
            ],
            "tools": [
                {
                    "name": "read_pv",
                    "description": "Read the current value of an EPICS process variable.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                }
            ],
            "tool_choice": {"type": "any"},
        },
        timeout=120.0,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
    assert tool_blocks, f"open model emitted no tool call through full chain: {body}"
    assert tool_blocks[0]["name"] == "read_pv"
    assert isinstance(tool_blocks[0]["input"], dict)
