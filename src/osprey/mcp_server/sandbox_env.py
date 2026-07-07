"""Sensitive-environment scrub shared by every agent-code execution sandbox.

Both ``python_executor.executor`` (the general-purpose Python execution
sandbox) and ``workspace.execution.sandbox_executor`` (the lighter
visualization-only sandbox) spawn a local subprocess to run agent-generated
Python code. Both must exclude write-arming secrets from that subprocess's
environment: agent-run code must never be able to read a token that gates a
write-capable action from outside the sandbox (e.g. the Bluesky bridge's
``/runs/{id}/promote`` endpoint, event-dispatch webhooks) and call the gated
endpoint directly. The in-tool ``writes_enabled`` re-check inside
``launch_scan`` is the actual write-safety authority — not possession of a
token — so this scrub closes the read side of that attack surface.

Centralized here (rather than duplicated per sandbox) so the two deny-lists
cannot drift.
"""

# Exact names for known secrets that don't fit the suffix pattern below.
_SENSITIVE_ENV_EXACT: tuple[str, ...] = ("EVENT_DISPATCHER_TOKEN",)
# Suffix patterns so future write-arming tokens (e.g. a second bridge's
# *_PROMOTE_TOKEN) are excluded without needing a code change here.
_SENSITIVE_ENV_SUFFIXES: tuple[str, ...] = ("_PROMOTE_TOKEN",)


def _scrub_sensitive_env(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of *env* with write-arming secrets removed.

    Drops any key in :data:`_SENSITIVE_ENV_EXACT` and any key ending in one of
    :data:`_SENSITIVE_ENV_SUFFIXES`. Used to build the environment passed to
    an agent-code execution subprocess so the sandboxed code cannot read
    these secrets, even though the parent process needs them for its own
    MCP/server plumbing.
    """
    return {
        key: value
        for key, value in env.items()
        if key not in _SENSITIVE_ENV_EXACT and not key.endswith(_SENSITIVE_ENV_SUFFIXES)
    }
