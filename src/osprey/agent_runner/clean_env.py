"""Clean child-environment helpers for launching Claude Code agents.

Centralises the ``CLAUDE_CODE_*`` stripping and base-env preparation shared by
every path that spawns a Claude Code child process: the Web Terminal's SDK
operator path and interactive PTY path (``interfaces.web_terminal``) and the
dispatch worker (``mcp_server.dispatch_worker``). Living in ``agent_runner``
keeps this single source of truth below both consumers so neither layer has to
import the other.

The base env also drops the credentials named by
:mod:`osprey.utils.sensitive_env` — the web-terminal operator secret, the panel
token, the event-dispatcher and dispatch-worker tokens, and any
``*_LAUNCH_TOKEN``. How much that deny step actually buys depends on the path it
lands on, and the two cases are genuinely different:

* **Real removal — the PTY child and ``osprey chat``.**
  :func:`pty_manager.build_pty_env` and :func:`osprey.cli.chat_cmd.chat` hand
  the result to ``subprocess.Popen(..., env=env)`` / ``subprocess.run(...,
  env=env)`` as the child's *complete* environment, so a name absent from this
  dict is absent from the child and from every grandchild it spawns (the MCP
  servers the agent session starts). These are the consumers where the strip is
  what closes the variable.
* **Inert deletion — every SDK path.** ``ClaudeAgentOptions.env`` is not a
  replacement environment: claude-agent-sdk builds the CLI's environment as
  ``{**os.environ, **options.env}``, so ``options.env`` is an *overlay*. A key
  this helper declined to copy is simply not overlaid, and the child inherits
  the parent's value from ``os.environ`` regardless. Both SDK consumers work
  this way — the web-terminal operator session and the dispatch worker, which
  passes :func:`build_clean_env` straight into ``ClaudeAgentOptions.env``.

What compensates on the SDK paths differs by consumer, and only one of them is
covered. In the web server it is :func:`osprey.interfaces.web_auth.close_env_carriers`,
called from every interface app's construction, that removes the operator
secret and the panel token from ``os.environ`` — that removal, not this strip,
is what truly closes those two names there. It has to be construction-time and
unconditional rather than a one-off pop inside ``_populate``: a launcher
re-publishes the operator secret on purpose so a ``--reload`` worker or
``--detach`` child inherits it, and on the direct-serve path (plain
``osprey web``, and the per-user container's ``CMD``) that "child" is the same,
already-populated process, where ``_populate`` never runs a second time. In the
dispatch worker nothing removes anything — the worker container holds the
event-dispatcher and dispatch-worker tokens in its own ``os.environ`` for the
lifetime of the process — so on that path the deny step provides no containment
whatsoever, and the honest reading is that it is inert rather than
defence-in-depth.

The dispatcher-side tokens and the ``*_LAUNCH_TOKEN`` family cannot be removed
the way the operator secret is: the web server must keep them in its own
``os.environ`` for upstream proxy injection into the services it fronts. The
safety authority for a write-armed endpoint therefore remains that endpoint's
own in-tool re-checks, never this dict.

**The panel token is deliberately put back on the web-terminal child paths.**
It is the weak, panel-tier-only credential
(:data:`osprey.interfaces.web_auth.PANEL_TIER_ROUTES`), and the agent's MCP
panel tools and its SessionStart/UserPromptSubmit/approval hooks cannot do
their job without it. The three seams that re-add it are
:func:`osprey.interfaces.web_terminal.routes.websocket._build_extra_env` (PTY),
:func:`osprey.interfaces.web_terminal.operator_session.build_operator_child_env`
(the SDK operator and chat sessions) and :func:`osprey.cli.chat_cmd.chat`
(``osprey chat``). Each takes the value from the credential holder, never from
``os.environ``. The operator secret is re-added by nothing, anywhere.

One consequence of the ``*_LAUNCH_TOKEN`` strip is deliberate and worth stating
plainly rather than discovering as a bug report: **on the two real-child paths
— the PTY child and ``osprey chat`` — it disarms the Bluesky MCP server's
queue-launch tool.** That server is a grandchild of the spawned child, so it
inherits this environment;
:func:`osprey.bluesky_bridge_connection.resolve_launch_token` then finds no
``BLUESKY_LAUNCH_TOKEN`` and (absent a local dev ``bluesky.launch_token`` in
``config.yml``) refuses the launch client-side, before the bridge is contacted.
Severing the agent-initiated arm is the point: an agent that can reach the
token can start a queue without a human in the loop. Under the Web Terminal the
human panel-click path is unaffected, because
:mod:`osprey.interfaces.bluesky_web.queue_relay` resolves the token
server-side, inside the web server whose ``os.environ`` still holds it. In the
``osprey chat`` shape that relay is not running at all — chat hosts only the
companion panel servers — so there the disarm is total: no path in that session
launches a queue.
"""

from __future__ import annotations

import os
from pathlib import Path

from osprey.utils.sensitive_env import strip_sensitive
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
    2. Drop the sensitive credentials named by
       :mod:`osprey.utils.sensitive_env` via :func:`strip_sensitive`, so a
       spawned child never inherits a value that would let agent-run code
       authenticate as the server that launched it. Operational keys such as
       ``PATH`` and ``OSPREY_WEB_PORT`` are untouched — only the credential
       names match. Read this module's docstring before relying on this step:
       it genuinely closes those names on the real child-process paths, but on
       the SDK overlay path an absent key is inherited from ``os.environ``
       anyway, which makes it defence-in-depth there rather than a boundary.
    3. Resolve the auth-token conflict: when token-based auth is configured
       (e.g. the CBORG proxy at LBNL), drop a stale ``ANTHROPIC_API_KEY`` that
       Claude Code would otherwise auto-load from the project ``.env`` and warn
       about.
    4. Augment ``PATH`` with user-local bin dirs (e.g. ``~/.local/bin``) so child
       processes resolve their dependencies even from a non-login context
       (lifecycle hooks, ``nohup``, etc.).

    Centralising all four here — not just the strip step — is what keeps the
    launch paths from drifting apart, the reason this helper exists.

    Returns:
        A fresh env dict, ready for path-specific keys to be overlaid. The
        caller's own ``os.environ`` is never mutated.
    """
    env = strip_sensitive(strip_claude_code_env(dict(os.environ)))

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
    telemetry master switch, drops the sensitive credentials, resolves the
    auth-token conflict, and augments ``PATH``): auto-sets ``OSPREY_CONFIG``
    from the project directory.

    Both consumers hand the result to the SDK as ``ClaudeAgentOptions.env``,
    which the SDK overlays onto ``os.environ`` rather than substituting for it,
    so omitting a credential here does not by itself keep it away from the
    child. In the web server
    :func:`osprey.interfaces.web_auth.close_env_carriers` removes the operator
    secret and the panel token from ``os.environ`` at app construction, which is
    what actually closes them; in the dispatch worker nothing compensates, and
    the credential deny step buys no containment at all. See this module's
    docstring.

    Web-terminal callers should use
    :func:`osprey.interfaces.web_terminal.operator_session.build_operator_child_env`
    rather than this function directly: it layers the panel token the agent's
    panel tools and hooks need on top of this result. This function stays
    panel-token-free because the dispatch worker also calls it and has no panel
    host to talk to.

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
