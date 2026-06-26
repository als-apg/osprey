"""Shared SDK-context primitives for operator and dispatch call sites.

Two narrow helpers:

* ``build_system_prompt`` — builds a ``SystemPromptPreset`` dict that
  appends facility-local time (and optional extras) to the Claude Code
  default system prompt. The SDK emits this as ``--append-system-prompt``
  only, so the CLI's default prompt is preserved.

* ``make_tool_allowlist`` — builds a ``CanUseTool`` callback that denies
  anything not in a given allowlist. Dispatch uses this as the real
  enforcement layer, since ``permission_mode="bypassPermissions"`` makes
  the SDK's ``allowed_tools`` advisory.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

try:
    from claude_agent_sdk import (
        PermissionResultAllow,
        PermissionResultDeny,
    )
    from claude_agent_sdk.types import CanUseTool, SystemPromptPreset

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    PermissionResultAllow = object  # type: ignore[assignment,misc]
    PermissionResultDeny = object  # type: ignore[assignment,misc]
    CanUseTool = Any  # type: ignore[assignment,misc]
    SystemPromptPreset = dict  # type: ignore[assignment,misc]


def build_system_prompt(
    tz: ZoneInfo,
    extra: str | None = None,
) -> SystemPromptPreset:
    """Build a SystemPromptPreset that appends facility-local time.

    The returned dict is suitable for ``ClaudeAgentOptions.system_prompt``.
    The SDK sends a preset-with-append as ``--append-system-prompt`` only,
    so Claude Code's built-in prompt remains loaded.

    Args:
        tz: Facility timezone (use ``get_facility_timezone()`` at call sites).
        extra: Optional second line appended after the timestamp.
    """
    stamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")
    append = f"Facility local time: {stamp}"
    if extra:
        append = f"{append}\n{extra}"
    return {"type": "preset", "preset": "claude_code", "append": append}


def make_tool_allowlist(allowed: Iterable[str]) -> CanUseTool:
    """Build a ``CanUseTool`` callback enforcing a closed tool allowlist.

    Any tool whose name is not in ``allowed`` is denied with a message
    naming the blocked tool. Dispatch runs use this to enforce per-trigger
    allowlists at the SDK permission layer, because
    ``permission_mode="bypassPermissions"`` makes ``allowed_tools`` advisory.
    """
    allowed_set = frozenset(allowed)

    async def can_use_tool(tool_name, tool_input, context):  # type: ignore[no-untyped-def]
        if tool_name in allowed_set:
            return PermissionResultAllow()
        return PermissionResultDeny(
            message=f"Tool {tool_name!r} is not in this trigger's allowed_tools list",
        )

    return can_use_tool
