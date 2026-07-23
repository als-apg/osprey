"""Clean child-environment helpers for launching Claude Code agents.

Centralises the ``CLAUDE_CODE_*`` stripping and base-env preparation shared by
every path that spawns a Claude Code child process: the Web Terminal's SDK
operator path and interactive PTY path (``interfaces.web_terminal``) and the
dispatch worker (``mcp_server.dispatch_worker``). Living in ``agent_runner``
keeps this single source of truth below both consumers so neither layer has to
import the other.
"""

from __future__ import annotations

import os
from pathlib import Path

from osprey.utils.shell_resolver import user_bin_dirs

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


def build_base_child_env() -> dict[str, str]:
    """Build the base environment shared by every Claude Code child launcher.

    The interactive PTY path (:func:`pty_manager.build_pty_env`), the SDK
    operator subprocess (:func:`build_clean_env`), and the dispatch worker all
    need the same three preparation steps, in the same order, before each
    overlays its own path-specific keys:

    1. Strip Claude Code internal session variables via
       :func:`strip_claude_code_env` (preserving the telemetry master switch).
    2. Resolve the auth-token conflict: when token-based auth is configured
       (e.g. the CBORG proxy at LBNL), drop a stale ``ANTHROPIC_API_KEY`` that
       Claude Code would otherwise auto-load from the project ``.env`` and warn
       about.
    3. Augment ``PATH`` with user-local bin dirs (e.g. ``~/.local/bin``) so child
       processes resolve their dependencies even from a non-login context
       (lifecycle hooks, ``nohup``, etc.).

    Centralising all three here — not just the strip step — is what keeps the
    launch paths from drifting apart, the reason this helper exists.

    Returns:
        A fresh env dict, ready for path-specific keys to be overlaid.
    """
    env = strip_claude_code_env(dict(os.environ))

    if env.get("ANTHROPIC_AUTH_TOKEN"):
        env.pop("ANTHROPIC_API_KEY", None)

    extra_dirs = user_bin_dirs()
    if extra_dirs:
        env["PATH"] = os.pathsep.join(extra_dirs) + os.pathsep + env.get("PATH", "")

    return env


def build_clean_env(project_cwd: str | None = None) -> dict[str, str]:
    """Build a clean environment dict for the SDK subprocess.

    Layers the SDK-specific keys on top of :func:`build_base_child_env` (which
    strips ``CLAUDECODE``/``CLAUDE_CODE_*`` variables while preserving the
    telemetry master switch, resolves the auth-token conflict, and augments
    ``PATH``): auto-sets ``OSPREY_CONFIG`` from the project directory.

    Args:
        project_cwd: Optional project directory. When ``OSPREY_CONFIG`` is not
            already set and this directory contains ``config.yml``, the variable
            is set automatically so hooks can locate the configuration.
    """
    env = build_base_child_env()

    # Auto-set OSPREY_CONFIG when a config.yml exists in the project directory
    if "OSPREY_CONFIG" not in env and project_cwd:
        config_path = Path(project_cwd) / "config.yml"
        if config_path.exists():
            env["OSPREY_CONFIG"] = str(config_path)

    # Note: OSPREY_HOOK_DEBUG is intentionally NOT propagated here.
    # Hooks read config.yml directly for hot-reloadable debug toggle.

    return env
