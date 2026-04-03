#!/usr/bin/env python3
"""
---
name: Channel Finder Feedback Capture
description: Silently captures channel finder search results into pending review store
summary: Captures CF tool results for operator review
event: PostToolUse
tools: mcp__channel-finder__build_channels, mcp__channel-finder__get_channels
---

## Flow

```
stdin --> Parse JSON
              |
              v
         Is CF tool?  --NO--> EXIT (silent)
              |
             YES
              v
         Parse tool_response
              |
              v
         total > 0?  --NO--> EXIT (silent)
              |
             YES
              v
         Resolve store path
              |
              v
         Try import PendingReviewStore
              |
         +----+----+
         |         |
     installed  not installed
         |         |
         v         v
     Use store   Inline JSON append
         |         |
         v         v
         EXIT(0)
```

COMPLETELY SILENT: never writes to stdout. Always exits 0.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import get_hook_input, get_project_dir, log_hook

# Top-level guard: never crash the agent
hook_input = None
try:
    # ----------------------------------------------------------------
    # 1. Read hook input from stdin
    # ----------------------------------------------------------------
    hook_input = get_hook_input()
    if not hook_input:
        log_hook("cf-feedback-capture", {}, status="empty-input")
        sys.exit(0)

    # ----------------------------------------------------------------
    # 2. Gate on tool name
    # ----------------------------------------------------------------
    tool_name = hook_input.get("tool_name", "")
    CAPTURE_TOOLS = {
        "mcp__channel-finder__build_channels",
        "mcp__channel-finder__get_channels",
    }
    if tool_name not in CAPTURE_TOOLS:
        log_hook("cf-feedback-capture", hook_input, status="skip-tool", detail=tool_name)
        sys.exit(0)

    # ----------------------------------------------------------------
    # 3. Parse tool_response, skip if total <= 0
    # ----------------------------------------------------------------
    tool_response_raw = hook_input.get("tool_response", "")

    # Handle content block array format: [{"type": "text", "text": "..."}]
    if isinstance(tool_response_raw, list):
        texts = [b.get("text", "") for b in tool_response_raw if isinstance(b, dict)]
        tool_response_raw = texts[0] if texts else ""

    try:
        tool_response = (
            json.loads(tool_response_raw)
            if isinstance(tool_response_raw, str)
            else tool_response_raw
        )
    except (json.JSONDecodeError, ValueError):
        log_hook(
            "cf-feedback-capture",
            hook_input,
            status="parse-error",
            detail=f"type={type(tool_response_raw).__name__}",
        )
        sys.exit(0)

    # Unwrap {"result": "..."} wrapping from Claude Code's PostToolUse format
    if isinstance(tool_response, dict) and "result" in tool_response and len(tool_response) == 1:
        inner = tool_response["result"]
        if isinstance(inner, str):
            try:
                tool_response = json.loads(inner)
            except (json.JSONDecodeError, ValueError):
                tool_response = inner  # keep as-is if not JSON
        else:
            tool_response = inner  # already a dict/list

    total = 0
    if isinstance(tool_response, dict):
        total = tool_response.get("total", 0)
    if not total or total <= 0:
        resp_type = type(tool_response).__name__
        resp_keys = list(tool_response.keys()) if isinstance(tool_response, dict) else None
        log_hook(
            "cf-feedback-capture",
            hook_input,
            status="no-total",
            detail=f"type={resp_type} keys={resp_keys}",
        )
        sys.exit(0)

    # ----------------------------------------------------------------
    # 4. Resolve store path
    # ----------------------------------------------------------------
    project_dir = get_project_dir(hook_input)
    if not project_dir:
        log_hook("cf-feedback-capture", hook_input, status="no-cwd")
        sys.exit(0)

    store_path = os.path.join(project_dir, "data", "feedback", "pending_reviews.json")

    # ----------------------------------------------------------------
    # 5. Extract fields from hook input
    # ----------------------------------------------------------------
    tool_input = hook_input.get("tool_input", {})
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, ValueError):
            tool_input = {}

    item = {
        "query": tool_input.get("query", ""),
        "facility": tool_input.get("facility", ""),
        "tool_name": tool_name,
        "tool_response": tool_response_raw
        if isinstance(tool_response_raw, str)
        else json.dumps(tool_response_raw),
        "channel_count": total,
        "selections": tool_input.get("selections", {}),
        "session_id": hook_input.get("session_id", ""),
        "transcript_path": hook_input.get("transcript_path", ""),
    }

    # ----------------------------------------------------------------
    # 5b. Extract agent task from sub-agent transcript (delegation prompt)
    #
    # transcript_path points to the MAIN session transcript.
    # The channel-finder sub-agent's transcript lives at:
    #   <session_stem>/subagents/<agent>.jsonl
    # We find the most recently modified .jsonl there (the active
    # sub-agent) and read its first user message — that's the
    # delegation prompt the main agent sent to channel-finder.
    # ----------------------------------------------------------------
    agent_task = ""
    transcript = item.get("transcript_path", "")
    if transcript and os.path.isfile(transcript):
        try:
            stem = os.path.splitext(transcript)[0]  # strip .jsonl
            subagent_dir = os.path.join(stem, "subagents")
            subagent_transcript = ""
            if os.path.isdir(subagent_dir):
                # Find the most recently modified .jsonl (the active sub-agent)
                candidates = [
                    os.path.join(subagent_dir, f)
                    for f in os.listdir(subagent_dir)
                    if f.endswith(".jsonl")
                ]
                if candidates:
                    subagent_transcript = max(candidates, key=os.path.getmtime)

            if subagent_transcript and os.path.isfile(subagent_transcript):
                with open(subagent_transcript) as tf:
                    for line in tf:
                        try:
                            entry = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if entry.get("type") == "user":
                            msg = entry.get("message", {})
                            content = msg.get("content", "")
                            if isinstance(content, str) and content.strip():
                                agent_task = content.strip()
                                break
        except OSError:
            pass
    item["agent_task"] = agent_task

    # ----------------------------------------------------------------
    # 6. Try importing PendingReviewStore; fallback to inline append
    # ----------------------------------------------------------------
    try:
        from osprey.services.channel_finder.feedback.pending_store import PendingReviewStore

        store = PendingReviewStore(store_path)
        store.capture(item)
    except ImportError:
        # Inline fallback when osprey is not installed
        import fcntl
        import uuid
        from datetime import UTC, datetime

        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        lock_path = store_path + ".lock"

        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                # Load existing data
                data = {"version": 1, "items": {}}
                if os.path.exists(store_path):
                    try:
                        with open(store_path) as f:
                            data = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        data = {"version": 1, "items": {}}

                # Add new item
                item_id = str(uuid.uuid4())
                item["id"] = item_id
                item["captured_at"] = datetime.now(UTC).isoformat()
                data.setdefault("items", {})[item_id] = item

                # Evict if over 500
                items = data["items"]
                if len(items) > 500:
                    sorted_ids = sorted(
                        items.keys(),
                        key=lambda k: items[k].get("captured_at", ""),
                    )
                    for evict_id in sorted_ids[: len(items) - 500]:
                        del items[evict_id]

                # Atomic write via tmp + replace
                fd_num, tmp_path = tempfile.mkstemp(dir=os.path.dirname(store_path), suffix=".tmp")
                try:
                    with os.fdopen(fd_num, "w") as tmp_f:
                        json.dump(data, tmp_f, indent=2)
                    os.replace(tmp_path, store_path)
                except BaseException:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    log_hook("cf-feedback-capture", hook_input, status="captured", detail=f"channels={total}")

except SystemExit:
    raise
except BaseException as exc:
    # Never crash the agent, but log the exception for diagnostics
    log_hook("cf-feedback-capture", hook_input or {}, status="exception", detail=str(exc))

sys.exit(0)
