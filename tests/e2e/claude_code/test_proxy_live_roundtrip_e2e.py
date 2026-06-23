"""L2c/L2d — LIVE round-trip through the proxy against real OpenAI-compatible upstreams.

Drives the actual proxy route (real httpx, real model) so translation is validated
against a real server emitting real tool calls — not a mock. Each upstream is probed
for usability and SKIPPED if unavailable, so the suite is green wherever it runs:

  * ollama (local)  — real OPEN model (qwen2.5 / gpt-oss) over OpenAI format. This is
                      the closest local analog to CBORG's self-hosted `cborg-coder`:
                      it proves an open model can drive Claude Code's tool loop through
                      the proxy. Runs whenever Ollama is up.
  * cborg  (VPN)    — the actual target: same proxy, upstream=api.cborg.lbl.gov/v1,
                      model=cborg-coder. Runs only on LBLnet/VPN (CBORG IP-allowlists).
  * openai          — frontier sanity check. Runs only with a funded OPENAI_API_KEY.

The ONLY difference between upstreams is base_url + model — the whole point of #259.

Experiment branch: experiment/cborg-claude-code.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.infrastructure.proxy.app import create_proxy_app

UPSTREAMS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key": "ollama",  # Ollama ignores the key
        "model": "qwen2.5:32b",
    },
    "cborg": {
        "base_url": "https://api.cborg.lbl.gov/v1",
        "key": os.environ.get("CBORG_API_KEY", ""),
        "model": "cborg-coder",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key": os.environ.get("OPENAI_API_KEY", ""),
        "model": "gpt-4o-mini",
    },
}


@lru_cache(maxsize=None)
def _usable(name: str) -> bool:
    """Real minimal probe: True only if the upstream returns a 200 completion.

    Catches CBORG's 403 IP block, OpenAI's 429 quota error, and Ollama being down.
    """
    u = UPSTREAMS[name]
    if not u["key"]:
        return False
    try:
        r = httpx.post(
            u["base_url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {u['key']}", "Content-Type": "application/json"},
            json={"model": u["model"], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            timeout=120.0,  # first Ollama call may load a large model
        )
        return r.status_code == 200
    except Exception:
        return False


def _client(name: str) -> TestClient:
    u = UPSTREAMS[name]
    return TestClient(create_proxy_app(u["base_url"], upstream_api_key=u["key"]))


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _skip_if_unusable(name: str) -> None:
    if not _usable(name):
        pytest.skip(f"upstream '{name}' not usable (down / off-VPN / no-quota / key unset)")


@pytest.mark.parametrize("name", list(UPSTREAMS))
def test_text_through_proxy(name):
    _skip_if_unusable(name)
    u = UPSTREAMS[name]
    resp = _client(name).post(
        "/v1/messages",
        json={
            "model": u["model"],
            "messages": [{"role": "user", "content": "Reply with exactly the word: PONG"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["type"] == "message"
    text = "".join(b.get("text", "") for b in body["content"] if b["type"] == "text")
    assert text.strip(), body
    assert body["usage"]["output_tokens"] > 0


@pytest.mark.parametrize("name", list(UPSTREAMS))
def test_tool_call_through_proxy(name):
    """The reliability question for open models: a well-formed tool call that
    survives Anthropic->OpenAI->Anthropic translation."""
    _skip_if_unusable(name)
    u = UPSTREAMS[name]
    resp = _client(name).post(
        "/v1/messages",
        json={
            "model": u["model"],
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": "Use the read_pv tool to read the PV named BPM:01:X. You must call the tool.",
                }
            ],
            "tools": [
                {
                    "name": "read_pv",
                    "description": "Read the current value of an EPICS process variable.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string", "description": "PV name"}},
                        "required": ["name"],
                    },
                }
            ],
            "tool_choice": {"type": "any"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
    assert tool_blocks, f"[{name}] emitted no tool call: {body}"
    assert tool_blocks[0]["name"] == "read_pv"
    assert isinstance(tool_blocks[0]["input"], dict)


@pytest.mark.parametrize("name", list(UPSTREAMS))
def test_streaming_through_proxy(name):
    """Claude Code streams; verify the Anthropic SSE envelope the SDK expects."""
    _skip_if_unusable(name)
    u = UPSTREAMS[name]
    resp = _client(name).post(
        "/v1/messages",
        json={
            "model": u["model"],
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "Count: one two three"}],
        },
    )
    assert resp.status_code == 200, resp.text
    types = [e.get("type") for e in _parse_sse(resp.text)]
    assert types and types[0] == "message_start", types
    assert "content_block_delta" in types, types
    assert types[-1] == "message_stop", types
