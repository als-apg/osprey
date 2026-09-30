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
                      Its image cases skip until a model that takes images is pinned.
  * openai          — frontier sanity check. Runs only with a funded OPENAI_API_KEY.

Each upstream pins `model` for the text, tool and streaming cases, and `image_model` for
the image cases. `image_model` is a model id that takes images, or None, in which case
that upstream's image cases skip by name.

Upstreams differ only in base_url, key and the models they pin.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import zlib
from functools import cache

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.infrastructure.proxy.app import create_proxy_app
from osprey.infrastructure.proxy.lifecycle import _request_shape

UPSTREAMS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key": "ollama",  # Ollama ignores the key
        "model": "qwen2.5:32b",
        # the route carries no images unless the catalog opts in
        "image_model": None,
    },
    "cborg": {
        "base_url": "https://api.cborg.lbl.gov/v1",
        "key": os.environ.get("CBORG_API_KEY", ""),
        "model": "cborg-coder",
        # no model that takes images is pinned; the image cases skip by name
        "image_model": None,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key": os.environ.get("OPENAI_API_KEY", ""),
        "model": "gpt-4o-mini",
        "image_model": "gpt-4o-mini",
    },
}


@cache
def _usable(name: str, model: str) -> bool:
    """Real minimal probe: True only if the upstream returns a 200 completion.

    Catches CBORG's 403 IP block, OpenAI's 429 quota error, and Ollama being down.
    The probe is per model, so an upstream's image model is probed on its own and not
    vouched for by its text model.
    """
    u = UPSTREAMS[name]
    if not u["key"]:
        return False
    try:
        r = httpx.post(
            u["base_url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {u['key']}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
            timeout=120.0,  # first Ollama call may load a large model
        )
        return r.status_code == 200
    except Exception:
        return False


def _client(name: str) -> TestClient:
    u = UPSTREAMS[name]
    return TestClient(
        create_proxy_app(u["base_url"], upstream_api_key=u["key"], **_request_shape(name))
    )


def _solid_png(rgb: tuple[int, int, int], size: int = 16) -> str:
    """A base64 PNG of one solid colour, built in place so the module keeps no fixture."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    row = b"\x00" + bytes(rgb) * size  # filter type 0, then the pixels
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _red_square() -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _solid_png((255, 0, 0))},
    }


_COLOUR_QUESTION = "What colour is this square? Answer with one word."


def _skip_if_unusable(name: str, model: str) -> None:
    if not _usable(name, model):
        pytest.skip(
            f"upstream '{name}' model '{model}' not usable (down / off-VPN / no-quota / key unset)"
        )


def _skip_unless_images_reach(name: str) -> str:
    """Return the model id the image cases send, or skip by name when the route carries
    no images, the upstream pins no image model, or that model does not answer."""
    if not _request_shape(name)["supports_images"]:
        pytest.skip(f"upstream '{name}': route declares no images")
    model = UPSTREAMS[name]["image_model"]
    if model is None:
        pytest.skip(f"upstream '{name}': no image model pinned")
    _skip_if_unusable(name, model)
    return model


def _answer_text(resp) -> str:
    assert resp.status_code == 200, resp.text
    return "".join(b.get("text", "") for b in resp.json()["content"] if b["type"] == "text")


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


@pytest.mark.parametrize("name", list(UPSTREAMS))
def test_text_through_proxy(name):
    _skip_if_unusable(name, UPSTREAMS[name]["model"])
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
    _skip_if_unusable(name, UPSTREAMS[name]["model"])
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
    _skip_if_unusable(name, UPSTREAMS[name]["model"])
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


@pytest.mark.parametrize("name", list(UPSTREAMS))
def test_an_image_reaches_a_vision_model_through_proxy(name):
    model = _skip_unless_images_reach(name)
    resp = _client(name).post(
        "/v1/messages",
        json={
            "model": model,
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [_red_square(), {"type": "text", "text": _COLOUR_QUESTION}],
                }
            ],
        },
    )
    assert "red" in _answer_text(resp).casefold()


@pytest.mark.parametrize("name", list(UPSTREAMS))
def test_a_tool_result_image_reaches_a_vision_model_through_proxy(name):
    model = _skip_unless_images_reach(name)
    resp = _client(name).post(
        "/v1/messages",
        json={
            "model": model,
            "max_tokens": 16,
            "tools": [
                {
                    "name": "read_screen",
                    "description": "Capture the operator screen.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "messages": [
                {"role": "user", "content": "Read the screen."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "read_screen", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [_red_square()],
                        },
                        {"type": "text", "text": _COLOUR_QUESTION},
                    ],
                },
            ],
        },
    )
    assert "red" in _answer_text(resp).casefold()
