#!/usr/bin/env python
"""Single-model harness runner for the CBORG model matrix (issue #259).

Drives the REAL `claude` CLI -> OSPREY Anthropic<->OpenAI translation proxy ->
one CBORG-hosted model -> an in-process MCP tool (`read_pv`), and reports whether
the model actually called the tool and relayed its result.

This is the standalone, parameterized form of
tests/experimental/test_e2e_open_model_harness.py: one model per process so the
proxy's per-process singleton gets its own OS-allocated port (parallel-safe).

Usage:
    .venv/bin/python scripts/benchmark/matrix_run.py --model cborg-coder
    .venv/bin/python scripts/benchmark/matrix_run.py --model lbl/gpt-oss-120b --timeout 300

Key resolution (first hit wins): --key-file, $CBORG_API_KEY, ~/.cborg_key
Emits exactly one JSON object as the LAST line of stdout (machine-parseable);
all human/diagnostic logging goes to stderr.

Result schema (one JSON object):
    {
      "model": "cborg-coder",
      "ok": true,                 # tool was called AND result relayed
      "tool_called": true,        # any mcp__controls__read_pv call seen
      "result_relayed": true,     # "3.14159" present in final assistant text
      "tool_calls": ["mcp__controls__read_pv"],
      "num_turns": 2,             # from ResultMessage (null if absent)
      "final_text": "...",        # joined assistant text blocks (truncated)
      "latency_s": 12.4,
      "error": null,              # error string, or "timeout", or null
      "base_url": "https://api.cborg.lbl.gov/v1"
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CBORG_BASE = "https://api.cborg.lbl.gov/v1"
SENTINEL = "3.14159"  # value the mock PV returns; presence in final text == relayed


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def _resolve_key(key_file: str | None) -> str:
    if key_file:
        return Path(key_file).expanduser().read_text().strip()
    if os.environ.get("CBORG_API_KEY"):
        return os.environ["CBORG_API_KEY"].strip()
    p = Path.home() / ".cborg_key"
    if p.exists():
        return p.read_text().strip()
    raise SystemExit("no CBORG key: pass --key-file, set CBORG_API_KEY, or create ~/.cborg_key")


def _claude_cli_ok() -> bool:
    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        return subprocess.run(["claude", "--version"], capture_output=True, timeout=10, env=env).returncode == 0
    except Exception:
        return False


async def _drive(base_url: str, key: str, model: str):
    """Reproduces test_e2e_open_model_harness._drive_harness, capturing num_turns too."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
        create_sdk_mcp_server,
        query,
        tool,
    )

    from osprey.infrastructure.proxy.lifecycle import start_proxy, stop_proxy

    @tool("read_pv", "Read the current value of an EPICS process variable (PV).", {"name": str})
    async def _read_pv(args):
        return {"content": [{"type": "text", "text": f"{args['name']} = {SENTINEL} mm"}]}

    server = create_sdk_mcp_server(name="controls", version="1.0.0", tools=[_read_pv])
    port = start_proxy(base_url, key)
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
            "ANTHROPIC_AUTH_TOKEN": key,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_SMALL_FAST_MODEL": model,
        },
    )
    tool_calls: list[str] = []
    texts: list[str] = []
    num_turns = None
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
                num_turns = getattr(msg, "num_turns", None)
    finally:
        stop_proxy()
    return tool_calls, " ".join(texts), num_turns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=CBORG_BASE)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--timeout", type=float, default=420.0, help="per-run wall-clock seconds")
    args = ap.parse_args()

    result = {
        "model": args.model,
        "ok": False,
        "tool_called": False,
        "result_relayed": False,
        "tool_calls": [],
        "num_turns": None,
        "final_text": "",
        "latency_s": None,
        "error": None,
        "base_url": args.base_url,
    }

    if not _claude_cli_ok():
        result["error"] = "claude CLI not available"
        print(json.dumps(result))
        return 1

    key = _resolve_key(args.key_file)
    _log(f"[{args.model}] driving harness via proxy -> {args.base_url}")
    t0 = time.monotonic()
    try:
        tool_calls, final_text, num_turns = asyncio.run(
            asyncio.wait_for(_drive(args.base_url, key, args.model), timeout=args.timeout)
        )
        result["tool_calls"] = tool_calls
        result["final_text"] = final_text[:2000]
        result["num_turns"] = num_turns
        result["tool_called"] = any("read_pv" in t for t in tool_calls)
        result["result_relayed"] = SENTINEL in final_text
        result["ok"] = result["tool_called"] and result["result_relayed"]
    except asyncio.TimeoutError:
        result["error"] = "timeout"
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
    result["latency_s"] = round(time.monotonic() - t0, 1)

    _log(f"[{args.model}] ok={result['ok']} tool_called={result['tool_called']} "
         f"relayed={result['result_relayed']} turns={result['num_turns']} "
         f"latency={result['latency_s']}s err={result['error']}")
    print(json.dumps(result))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
