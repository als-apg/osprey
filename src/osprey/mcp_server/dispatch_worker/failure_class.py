"""Failure-class taxonomy and error-result stamping for dispatch worker runs.

Every error result a dispatch run can produce is stamped with exactly one
``failure_class`` string so downstream consumers (retry logic, dashboards,
conversational bridges) can decide uniformly whether a failure is worth
retrying and where the fault lies.

Taxonomy — three mutually exclusive classes:

* ``provider`` — the model provider / upstream API is at fault: authentication
  or API errors, HTTP 429 rate limits, and the inactivity watchdog firing (no
  bytes from the provider within the window). Generally retryable.
* ``infrastructure`` — the dispatch worker's own machinery failed before or
  around the run: the outer worker timeout, the stale-run sweep, and the SDK
  not being installed. Retryable once the infra recovers.
* ``run`` — the agent run itself reached a terminal state that is neither the
  provider's nor the worker's fault: an agent-level exception, a user
  cancellation, or a resource cap (``max_turns`` / ``max_budget_usd``). Not
  usefully retryable as-is.

Public API — consumed by the ``sdk_runner`` and ``dispatch_api`` stamp sites
(Tasks 1.3 / 1.4):

* :data:`FAILURE_PROVIDER` / :data:`FAILURE_INFRASTRUCTURE` / :data:`FAILURE_RUN`
  — the three class strings.
* :data:`FAILURE_CLASSES` — frozenset of the three valid values.
* :func:`classify_exception` — table-driven exception→class mapping for use at
  the *generic* ``except Exception`` sites, where the cause is not known from
  the call site. Known-cause sites (the inactivity watchdog, the worker
  timeout, the stale sweep, user cancel, SDK-missing) must stamp their literal
  class directly rather than route a synthetic exception through here.
* :func:`is_budget_subtype` — ``True`` for a ``ResultMessage`` subtype that
  signals a resource cap; reuses ``agent_runner.verdict._BUDGET_SUBTYPES`` (the
  single source of truth) rather than re-listing the subtypes.
* :func:`register_counter_hook` — install the process-lifetime per-class
  counter callback. The counters module is owned by a separate task; until a
  hook is registered, :func:`_stamp` only stamps and does not count.
* :func:`_stamp` — write the class and tool-call count onto an error result and
  bump the per-class counter via the registered hook.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

# Single source of truth for the resource-cap subtypes. Reused, not duplicated,
# so a change to the verdict taxonomy propagates here automatically.
from osprey.agent_runner.verdict import _BUDGET_SUBTYPES

logger = logging.getLogger("osprey.mcp_server.dispatch_worker.failure_class")

# ---------------------------------------------------------------------------
# Class constants
# ---------------------------------------------------------------------------

FAILURE_PROVIDER: str = "provider"
"""Upstream provider / API fault (auth, API errors, 429, inactivity stall)."""

FAILURE_INFRASTRUCTURE: str = "infrastructure"
"""Worker-side fault (outer timeout, stale sweep, SDK not installed)."""

FAILURE_RUN: str = "run"
"""Agent-level terminal state (agent error, user cancel, resource cap)."""

FAILURE_CLASSES: frozenset[str] = frozenset({FAILURE_PROVIDER, FAILURE_INFRASTRUCTURE, FAILURE_RUN})
"""The complete set of valid ``failure_class`` values."""

# ---------------------------------------------------------------------------
# Exception → class mapping (table-driven)
# ---------------------------------------------------------------------------

# Exception *type names* that unambiguously indicate a provider / upstream-API
# fault. Matched by ``type(exc).__name__`` so we do not need to import the
# provider SDK's exception classes (they are only present inside the worker
# container, and matching by name keeps this module import-light and testable).
_PROVIDER_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "APIError",
        "APIStatusError",
        "APIConnectionError",
        "APITimeoutError",
        "APIResponseValidationError",
        "InternalServerError",
        "OverloadedError",
    }
)

# Case-insensitive substrings in the exception message that indicate a provider
# fault. Kept broad because generic exceptions surfaced by the CLI often carry
# only a formatted message, not a typed class.
_PROVIDER_MESSAGE_SUBSTRINGS: tuple[str, ...] = (
    "rate limit",
    "429",
    "overloaded",
    "529",
    "authentication",
    "unauthorized",
    "401 ",
    "403 ",
    "invalid api key",
    "invalid x-api-key",
    "api key",
    "credential",
    "permission denied",
    "insufficient_quota",
)

# Maximum depth to walk the ``__cause__`` / ``__context__`` chain so a provider
# error wrapped in a generic exception is still classified correctly, without
# risking an unbounded walk on a pathological cycle.
_MAX_CAUSE_DEPTH = 8


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return ``exc`` followed by its cause/context chain (bounded, cycle-safe)."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and len(chain) < _MAX_CAUSE_DEPTH and id(cur) not in seen:
        chain.append(cur)
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return chain


def classify_exception(exc: BaseException) -> str:
    """Map an exception raised at a *generic* error site to a failure class.

    Resolution order:

    1. If any exception in the cause chain has a type name in
       :data:`_PROVIDER_TYPE_NAMES`, the class is :data:`FAILURE_PROVIDER`.
    2. Otherwise, if the message contains a provider substring, likewise
       :data:`FAILURE_PROVIDER`.
    3. Otherwise the fault is attributed to the agent run itself:
       :data:`FAILURE_RUN`.

    This function is deliberately conservative about ``infrastructure``: worker
    machinery faults (timeout, stale sweep, SDK-missing) are known at their
    call sites and must be stamped with the literal class there, so nothing
    routed through this generic mapping is classified as infrastructure.

    Args:
        exc: The caught exception.

    Returns:
        One of :data:`FAILURE_PROVIDER` or :data:`FAILURE_RUN`.
    """
    chain = _exception_chain(exc)
    for e in chain:
        if type(e).__name__ in _PROVIDER_TYPE_NAMES:
            return FAILURE_PROVIDER
    for e in chain:
        msg = str(e).lower()
        if any(sub in msg for sub in _PROVIDER_MESSAGE_SUBSTRINGS):
            return FAILURE_PROVIDER
    return FAILURE_RUN


def is_budget_subtype(subtype: str | None) -> bool:
    """Return ``True`` when ``subtype`` marks a resource cap (turns / budget).

    A completed ``ResultMessage`` carrying one of these subtypes is an
    agent-level terminal state and should be stamped :data:`FAILURE_RUN`.
    """
    return subtype is not None and subtype in _BUDGET_SUBTYPES


# ---------------------------------------------------------------------------
# Per-class counter seam
# ---------------------------------------------------------------------------

# Decoupling seam for the process-lifetime per-class counters, whose module is
# owned by a later task. That module calls :func:`register_counter_hook` once at
# import/boot to install a callback; :func:`_stamp` invokes it (if present) with
# the failure class it just stamped. Keeping the dependency inverted this way
# lets the two modules ship and be tested independently.
_counter_hook: Callable[[str], None] | None = None


def register_counter_hook(hook: Callable[[str], None] | None) -> None:
    """Install (or clear, with ``None``) the per-class counter callback.

    The callback receives the stamped ``failure_class`` string and is expected
    to increment a process-lifetime counter for it. Passing ``None`` detaches
    the hook (useful for test isolation).
    """
    global _counter_hook
    _counter_hook = hook


def _bump_counter(failure_class: str) -> None:
    """Invoke the registered counter hook, swallowing any error it raises.

    Counting is observability, never load-bearing: a broken or absent hook must
    not turn a stamped error result into a second failure.
    """
    hook = _counter_hook
    if hook is None:
        return
    try:
        hook(failure_class)
    except Exception:  # pragma: no cover - defensive
        logger.debug("failure-class counter hook raised for %r", failure_class, exc_info=True)


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------


def _stamp(
    result: dict[str, Any], failure_class: str, num_tool_calls: int | None
) -> dict[str, Any]:
    """Stamp a failure class and tool-call count onto an error ``result``.

    Mutates ``result`` in place (and returns it, for convenient chaining):

    * ``result["failure_class"]`` is set to ``failure_class``.
    * ``result["num_tool_calls"]`` is set to ``num_tool_calls`` (``None`` is a
      valid value — the count is unknown at some sites).

    The process-lifetime per-class counter for ``failure_class`` is bumped via
    the registered hook (a no-op until one is registered).

    Args:
        result: The error-result dict to annotate.
        failure_class: One of :data:`FAILURE_CLASSES`.
        num_tool_calls: Number of tool calls the run made before failing, or
            ``None`` when unavailable.

    Returns:
        The same ``result`` dict, annotated.

    Raises:
        ValueError: If ``failure_class`` is not a recognised class.
    """
    if failure_class not in FAILURE_CLASSES:
        raise ValueError(
            f"unknown failure_class {failure_class!r}; expected one of {sorted(FAILURE_CLASSES)}"
        )
    result["failure_class"] = failure_class
    result["num_tool_calls"] = num_tool_calls
    _bump_counter(failure_class)
    return result
