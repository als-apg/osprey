"""Sensitive-environment scrub shared by every agent-code execution sandbox.

Three call sites spawn a local subprocess to run agent-generated Python:
``python_executor.executor`` (the general-purpose Python execution sandbox),
``workspace.execution.sandbox_executor`` (the lighter visualization-only
sandbox), and ``services.bluesky_bridge.plan_validation`` (plan import
checking). None of them has any reason to reach the surfaces those
credentials unlock: they never call panel routes, never authenticate a
web-terminal session, and never post to the event-dispatcher API. Handing
them the tokens anyway would let agent-run code call a write-gated endpoint
directly, from outside the tool layer whose in-tool ``writes_enabled``
re-check is the actual write-safety authority.

The set of names to drop is not defined here. It lives in
:mod:`osprey.utils.sensitive_env`, the dependency-free leaf module that
``agent_runner`` also uses, so that the PTY child and the execution
sandboxes cannot drift apart. This module only re-exports that set and wraps
:func:`~osprey.utils.sensitive_env.strip_sensitive` under the name the
sandboxes already import.
"""

from collections.abc import Mapping

from osprey.utils.sensitive_env import (
    SENSITIVE_ENV_EXACT,
    SENSITIVE_ENV_SUFFIXES,
    strip_sensitive,
)

__all__ = ["SENSITIVE_ENV_EXACT", "SENSITIVE_ENV_SUFFIXES", "scrub_sensitive_env"]


def scrub_sensitive_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of *env* with agent-forbidden credentials removed.

    Drops any key in :data:`SENSITIVE_ENV_EXACT` and any key ending in one of
    :data:`SENSITIVE_ENV_SUFFIXES`, delegating the match to
    :func:`osprey.utils.sensitive_env.strip_sensitive`. Used to build the
    environment passed to an agent-code execution subprocess, so the
    sandboxed code cannot read these secrets even though the parent process
    needs them for its own MCP/server plumbing. *env* is not mutated, so
    ``os.environ`` may be passed directly.
    """
    return strip_sensitive(env)
