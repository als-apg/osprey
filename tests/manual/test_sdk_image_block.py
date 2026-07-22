"""Manual spike: prove the Agent SDK streams a base64 image block to the CLI.

This harness empirically verifies a single load-bearing assumption: that
``claude_agent_sdk``'s *stream-json input* path accepts a user message whose
``content`` is a **block list** containing a base64-encoded image block, and
that the block reaches the model end-to-end through the ``claude`` CLI
subprocess. Static research only confirmed the assumption up to the SDK
boundary (``_internal/client.py`` writes each streamed dict verbatim to the
CLI's stdin via ``json.dumps(dict) + "\n"``); this run closes the gap by
sending a real image and checking the model names its color.

The exact message dict shape the CLI accepted (the deliverable for the
downstream prompt-assembly task):

    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "<base64 PNG bytes, no data: URI prefix>",
                    },
                },
                {"type": "text", "text": "What color is this square? ..."},
            ],
        },
    }

Gotchas discovered (for the prompt-assembly task):

  * The top-level envelope is ``{"type": "user", "message": {...}}`` — the
    image/text blocks live under ``message.content``, NOT at the top level.
    ``message`` mirrors the Anthropic Messages API user-turn shape exactly.
  * The image block is the Anthropic native shape
    ``{"type": "image", "source": {"type": "base64", "media_type": ...,
    "data": ...}}``. ``data`` is the **raw base64 string** — no ``data:``
    URI prefix, no whitespace/newlines.
  * Block ordering is free: image-then-text works (verified here). The model
    resolves "this square" against the image regardless.
  * The SDK writes the dict verbatim — it does no validation of the block
    schema. A malformed block is only rejected downstream by the CLI/API, so
    the caller owns getting ``media_type`` and base64 right.
  * Provider env (base_url / auth token / model) must be delivered via
    ``ClaudeAgentOptions.env`` (or the ambient process env). ``options.env``
    overrides the inherited environment for the spawned CLI subprocess.

Skipped by default. To run it once against a real provider::

    OSPREY_RUN_SDK_IMAGE_SPIKE=1 \
      PYTHONPATH=$WORKTREE/src \
      $VENV/bin/python -m pytest tests/manual/test_sdk_image_block.py -s

It also runs standalone: ``python tests/manual/test_sdk_image_block.py``.
"""

from __future__ import annotations

import asyncio
import base64
import os
import struct
import tempfile
import zlib
from collections.abc import AsyncIterator

import pytest

# --------------------------------------------------------------------------
# Provider wiring — als-apg is Anthropic-native (no translation proxy) and
# IP-unrestricted (works off-VPN / from CI), so it is the default here. CBORG
# is LBLnet-gated and would 403 off-VPN. Values mirror osprey's own resolver
# (CLAUDE_CODE_PROVIDERS in osprey.cli.claude_code_resolver).
# --------------------------------------------------------------------------
_PROVIDERS = [
    ("ALS_APG_API_KEY", "https://llm.gianlucamartino.com", "claude-haiku-4-5-20251001"),
    ("CBORG_API_KEY", "https://api.cborg.lbl.gov", "claude-haiku-4-5"),
    ("ANTHROPIC_API_KEY", None, "claude-haiku-4-5-20251001"),
]


def _select_provider() -> tuple[str, str | None, str] | None:
    """Return (secret, base_url, model) for the first provider with a key."""
    for env_var, base_url, model in _PROVIDERS:
        secret = os.environ.get(env_var)
        if secret and "${" not in secret:
            return secret, base_url, model
    return None


def _provider_env(secret: str, base_url: str | None, model: str) -> dict[str, str]:
    """Build the CLI subprocess env for the chosen provider.

    Proxy providers authenticate with ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL
    (bare origin, no /v1). Direct Anthropic uses ANTHROPIC_API_KEY only. The
    three tier-model vars plus ANTHROPIC_MODEL pin every model the CLI might
    reach to the one cheap haiku-class model.
    """
    env = {
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url.rstrip("/").removesuffix("/v1")
        env["ANTHROPIC_AUTH_TOKEN"] = secret
    else:
        env["ANTHROPIC_API_KEY"] = secret
    return env


# --------------------------------------------------------------------------
# Test image — a 32x32 solid-red PNG built in-process (no binary fixture,
# no PIL dependency): raw RGB scanlines, zlib-compressed, wrapped in the
# minimal IHDR/IDAT/IEND chunk structure.
# --------------------------------------------------------------------------
def _red_png_b64(size: int = 32) -> str:
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # RGB, 8-bit
    row = b"\x00" + b"\xff\x00\x00" * size  # filter byte 0 + red pixels
    idat = zlib.compress(row * size, 9)
    png = (
        b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _image_message() -> dict:
    """The exact stream-json user message the CLI is expected to accept."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _red_png_b64(),
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "What color is this square? Answer with a single "
                        "lowercase color word and nothing else."
                    ),
                },
            ],
        },
    }


async def _stream_once(message: dict) -> AsyncIterator[dict]:
    """Yield exactly one streamed message, then close the input stream.

    Returning after the single yield exhausts the async iterable, which the
    SDK turns into an EOF on the CLI's stdin — the CLI processes the one turn
    and emits its result.
    """
    yield message


async def _run_spike() -> tuple[str, object]:
    """Send the image message through the SDK/CLI; return (reply_text, result)."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    selected = _select_provider()
    if selected is None:
        raise RuntimeError("no provider key available")
    env = _provider_env(*selected)

    reply_parts: list[str] = []
    result: object = None

    # Clean cwd + empty setting_sources = SDK isolation: no project/user
    # settings files can override the provider env we inject.
    with tempfile.TemporaryDirectory() as cwd:
        options = ClaudeAgentOptions(
            env=env,
            model=env["ANTHROPIC_MODEL"],
            max_turns=1,
            allowed_tools=[],
            setting_sources=[],
            permission_mode="bypassPermissions",
            cwd=cwd,
        )
        async for msg in query(prompt=_stream_once(_image_message()), options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        reply_parts.append(block.text)
            elif isinstance(msg, ResultMessage):
                result = msg

    return " ".join(reply_parts).strip(), result


# Skip-by-default gates:
#   * requires_als_apg / provider key — auto-skips in CI (no key present).
#   * OSPREY_RUN_SDK_IMAGE_SPIKE — explicit opt-in so a local `pytest tests/`
#     with keys present still skips this real-API, cost-incurring run.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("OSPREY_RUN_SDK_IMAGE_SPIKE"),
        reason="manual spike; set OSPREY_RUN_SDK_IMAGE_SPIKE=1 to run",
    ),
    pytest.mark.skipif(
        _select_provider() is None,
        reason="no provider API key (ALS_APG_API_KEY / CBORG_API_KEY / ANTHROPIC_API_KEY)",
    ),
]


def test_sdk_accepts_base64_image_block():
    """The CLI accepts a streamed base64 image block and the model reads it."""
    reply, result = asyncio.run(_run_spike())
    print(f"\n[spike] model reply: {reply!r}")
    if result is not None:
        print(
            f"[spike] result: subtype={getattr(result, 'subtype', None)!r} "
            f"is_error={getattr(result, 'is_error', None)!r} "
            f"cost_usd={getattr(result, 'total_cost_usd', None)!r}"
        )
    assert result is None or not getattr(result, "is_error", False), (
        f"CLI returned an error result: {result!r}"
    )
    assert "red" in reply.lower(), f"expected the model to name red, got: {reply!r}"


if __name__ == "__main__":
    os.environ.setdefault("OSPREY_RUN_SDK_IMAGE_SPIKE", "1")
    _reply, _result = asyncio.run(_run_spike())
    print(f"model reply: {_reply!r}")
    print(f"result: {_result!r}")
    assert "red" in _reply.lower(), f"expected red, got {_reply!r}"
    print("PASS: CLI accepted the base64 image block and the model named the color.")
