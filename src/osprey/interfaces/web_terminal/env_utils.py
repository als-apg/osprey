"""Shared environment helpers for the OSPREY Web Terminal.

Centralises the ``CLAUDE_CODE_*`` stripping logic so the operator (SDK) path
and the interactive PTY path cannot drift apart.
"""

from __future__ import annotations

# The telemetry master switch must survive the strip so that OpenTelemetry
# stays enabled on both the web-terminal chat path and the interactive PTY
# path. It starts with the ``CLAUDE_CODE_`` prefix but is a user-facing
# configuration toggle, not an internal session marker.
_TELEMETRY_MASTER_SWITCH = "CLAUDE_CODE_ENABLE_TELEMETRY"

_STRIP_PREFIXES = ("CLAUDECODE", "CLAUDE_CODE_")


def strip_claude_code_env(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``env`` without Claude Code internal session variables.

    Strips every key beginning with ``CLAUDECODE`` or ``CLAUDE_CODE_`` (nesting
    detection, entrypoint tracking, beta flags) so a nested launch does not
    inherit the parent session's markers. The telemetry master switch
    ``CLAUDE_CODE_ENABLE_TELEMETRY`` is preserved so telemetry survives on both
    the operator and PTY paths. ``OTEL_*`` keys do not carry the stripped
    prefix and therefore pass through untouched.

    Args:
        env: Source environment mapping (typically a copy of ``os.environ``).

    Returns:
        A new dict containing only the retained keys.
    """
    return {
        k: v
        for k, v in env.items()
        if k == _TELEMETRY_MASTER_SWITCH or not k.startswith(_STRIP_PREFIXES)
    }
