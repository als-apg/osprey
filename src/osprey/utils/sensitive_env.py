"""Canonical list of environment variables agent-run code must never see.

Several places in Osprey hand a copy of the parent process's environment to
something an agent can influence: the PTY child that runs the agent session
(``agent_runner.clean_env``), and the subprocesses that execute
agent-generated Python (``mcp_server.python_executor.executor``,
``mcp_server.workspace.execution.sandbox_executor``,
``services.bluesky_bridge.plan_validation``). Each of those hand-offs must
drop the same set of credentials, for the same reason: possession of one of
these values lets code call a write-armed or session-authenticating endpoint
directly, from outside the tool layer whose in-tool re-checks are the actual
safety authority.

This module is the single place that set is written down. It deliberately
imports nothing from ``osprey`` — only the standard library — so that both
``agent_runner`` and ``mcp_server`` can depend on it without creating an
import cycle between those two packages.

Two match shapes are supported:

* :data:`SENSITIVE_ENV_EXACT` — full variable names, matched exactly.
* :data:`SENSITIVE_ENV_SUFFIXES` — name suffixes, so a future credential that
  follows an established naming convention (a second bridge's
  ``*_LAUNCH_TOKEN``, say) is covered the day it is introduced rather than
  the day someone remembers to add it here.

Both forms are case-sensitive, matching the case-sensitive environment
semantics of the POSIX platforms Osprey runs its sandboxes on.
"""

from collections.abc import Mapping

#: Exact variable names to remove. ``OSPREY_TERMINAL_SECRET`` authenticates a
#: web-terminal session, ``OSPREY_PANEL_TOKEN`` authorizes panel-backend calls,
#: ``EVENT_DISPATCHER_TOKEN`` authenticates callers to the event-dispatcher API,
#: and ``DISPATCH_WORKER_TOKEN`` is that token's peer on the other leg — it
#: authenticates the dispatcher to a worker's ``/dispatch`` API, which is what
#: actually starts an agent run. The two are issued, rendered and deployed
#: together everywhere (``deployment.container_lifecycle``,
#: ``cli.templates.scaffolding``, every service compose template), so a set that
#: held one and not the other would be an oversight rather than a policy. Each
#: is a credential that would let agent code act as the server that launched it.
SENSITIVE_ENV_EXACT: tuple[str, ...] = (
    "OSPREY_TERMINAL_SECRET",
    "OSPREY_PANEL_TOKEN",
    "EVENT_DISPATCHER_TOKEN",
    "DISPATCH_WORKER_TOKEN",
)

#: Name suffixes to remove. ``*_LAUNCH_TOKEN`` arms a bridge's plan-launch
#: endpoint (e.g. ``BLUESKY_LAUNCH_TOKEN``); matching on the suffix means a
#: second bridge adopting the same convention is covered without a code change.
SENSITIVE_ENV_SUFFIXES: tuple[str, ...] = ("_LAUNCH_TOKEN",)


def is_sensitive(name: str) -> bool:
    """Return whether *name* is a credential agent-run code must not see.

    True when *name* appears in :data:`SENSITIVE_ENV_EXACT` or ends with one
    of :data:`SENSITIVE_ENV_SUFFIXES`. Exposed alongside
    :func:`strip_sensitive` for callers that decide one name at a time — for
    instance popping a single variable out of ``os.environ`` — so the match
    rule is not re-implemented per call site.
    """
    return name in SENSITIVE_ENV_EXACT or name.endswith(SENSITIVE_ENV_SUFFIXES)


def strip_sensitive(env: Mapping[str, str]) -> dict[str, str]:
    """Return a new dict holding every entry of *env* except the sensitive ones.

    *env* is never mutated, so a caller may pass ``os.environ`` itself and use
    the result as a child process's ``env=`` without disturbing its own
    process environment. Entries are dropped when :func:`is_sensitive` accepts
    their name; every other key keeps its original value.
    """
    return {key: value for key, value in env.items() if not is_sensitive(key)}
