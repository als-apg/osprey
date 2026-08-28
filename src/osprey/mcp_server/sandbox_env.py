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

The set of CREDENTIAL names to drop is not defined here. It lives in
:mod:`osprey.utils.sensitive_env`, the dependency-free leaf module that
``agent_runner`` also uses, so that the PTY child and the execution
sandboxes cannot drift apart. This module only re-exports that set and wraps
:func:`~osprey.utils.sensitive_env.strip_sensitive` under the name the
sandboxes already import.

What IS defined here is the second, sandbox-only narrowing: the web-terminal
address book and the navigation-only perimeter stamp. Those are not credential
policy - the PTY child that shares ``osprey.utils.sensitive_env`` *is* the web
terminal and must keep them - but no execution sandbox has any use for them,
and both spawn paths must drop exactly the same set. Hence
:func:`scrub_sandbox_child_env`: one definition, used by every sandbox that
spawns a child, so a name added for one path cannot go missing on the other.
"""

from collections.abc import Mapping

from osprey.utils.sensitive_env import (
    SENSITIVE_ENV_EXACT,
    SENSITIVE_ENV_SUFFIXES,
    strip_sensitive,
)

__all__ = [
    "PERIMETER_DENY_PORTS_ENV",
    "PERIMETER_MARKER_ENV",
    "SANDBOX_CHILD_ENV_DROP_NAMES",
    "SANDBOX_CHILD_ENV_DROP_PREFIXES",
    "SENSITIVE_ENV_EXACT",
    "SENSITIVE_ENV_SUFFIXES",
    "WEB_TERMINAL_ENV_NAMES_TO_DROP",
    "scrub_sandbox_child_env",
    "scrub_sensitive_env",
]

# The web-terminal address family, dropped from every sandbox child on top of
# the shared credential scrub. A sandbox's only callback surface is the
# `save_artifact` helper its execution wrapper injects, which writes to the
# filesystem: nothing in a child resolves a terminal URL or calls a web-terminal
# route, so these names buy it nothing and only tell agent code where a surface
# it must not reach is listening. OSPREY_TERMINAL_SECRET is already gone via
# scrub_sensitive_env; dropping the whole family is still right, because the
# rest of it (bind host, landing URL, external origin, the per-user
# OSPREY_TERMINAL_SECRET_<USER> names) is the same address book.
#
# Deliberately NOT added to the shared deny-list in osprey.utils.sensitive_env:
# that set is shared with the PTY child, and the PTY child *is* the web terminal
# - it must keep these (see the module docstring). This is a per-sandbox
# narrowing, not a credential policy.
WEB_TERMINAL_ENV_NAMES_TO_DROP: tuple[str, ...] = ("OSPREY_WEB_PORT",)

#: The navigation-only perimeter stamp, rendered onto every per-user web
#: terminal container by the deployment's compose overlay when
#: ``modules.web_terminals.auth.method`` is ``none``. The marker names the
#: posture; the deny-list names the deployment's own web ports (nginx, the TLS
#: listener when TLS is on, and every roster user's terminal), which under that
#: posture are reachable from inside such a container as whoever owns them -
#: nginx injects each user's operator secret, and these containers share the
#: host network namespace.
#:
#: Read in the PARENT process and handed to the sandbox as a wrapper argument.
#: They are NOT in the ``OSPREY_TERMINAL_`` family on purpose: that prefix is
#: dropped from the child too, and a stamp the parent could not read back would
#: be inert. The child never sees them either (they are in
#: :data:`SANDBOX_CHILD_ENV_DROP_NAMES`) - executed code is told what it may not
#: reach by the process that spawned it, and a sandbox that re-derived the list
#: could equally derive an empty one.
PERIMETER_MARKER_ENV = "OSPREY_WEB_PERIMETER"
PERIMETER_DENY_PORTS_ENV = "OSPREY_WEB_PERIMETER_DENY_PORTS"

#: Every exact name dropped from a sandbox child's environment: the web-terminal
#: address book plus the perimeter stamp the parent has already consumed by the
#: time the child is spawned.
SANDBOX_CHILD_ENV_DROP_NAMES: tuple[str, ...] = (
    *WEB_TERMINAL_ENV_NAMES_TO_DROP,
    PERIMETER_MARKER_ENV,
    PERIMETER_DENY_PORTS_ENV,
)

#: Matched by prefix so a terminal variable added later is covered without a
#: code change here - the same reasoning as SENSITIVE_ENV_SUFFIXES.
SANDBOX_CHILD_ENV_DROP_PREFIXES: tuple[str, ...] = ("OSPREY_TERMINAL_",)


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


def scrub_sandbox_child_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return the environment an agent-code execution child may be spawned with.

    :func:`scrub_sensitive_env` first (the credential policy shared with the PTY
    child), then the sandbox-only narrowing: every name in
    :data:`SANDBOX_CHILD_ENV_DROP_NAMES` and every name starting with one of
    :data:`SANDBOX_CHILD_ENV_DROP_PREFIXES`.

    Both spawn paths - the general-purpose python executor and the lighter
    visualization sandbox - call THIS function rather than each applying the
    drop themselves, so a name added for one path cannot go missing on the
    other. *env* is not mutated, so ``os.environ`` may be passed directly.
    """
    scrubbed = scrub_sensitive_env(env)
    for name in tuple(scrubbed):
        if name in SANDBOX_CHILD_ENV_DROP_NAMES or name.startswith(SANDBOX_CHILD_ENV_DROP_PREFIXES):
            scrubbed.pop(name, None)
    return scrubbed
