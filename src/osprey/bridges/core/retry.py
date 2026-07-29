"""Pure retry-policy decisions for the drain loop.

The drain loop periodically re-examines entries the bridge parked in its retry
queue (``status == "queued"`` in the :class:`~osprey.bridges.core.dedup.DedupStore`)
and, for each, must decide: is this failure the kind worth retrying at all? is it
*safe* to fire a fresh dispatch? has it aged enough to try again? should we give
up and file a GitLab issue instead? which queued entries collapse into one?

Every function here is a **pure** decision — no I/O, no clock, no config lookup
of its own: the caller passes ``now`` (epoch seconds, ``time.time()``) and the
``cfg`` tunables. That keeps them exhaustively table-testable and free of channel
and dispatch-internal imports (stdlib only).

Contracts consumed:

  * ``result`` — a dispatch result dict
    (:func:`osprey.bridges.core.dispatch_client` shape) carrying ``failure_class``
    (one of ``provider`` / ``infrastructure`` / an agent-or-user class / ``None``),
    ``num_tool_calls`` (``int`` or ``None`` when the worker never reported one),
    and ``error_code`` (a dispatcher pool-error code, e.g. an input_files
    rejection, or ``None``) which can force a non-retryable verdict.
  * ``entry`` — a :class:`~osprey.bridges.core.dedup.DedupStore` value dict. Beyond
    the core ``run_id`` / ``status`` it may carry retry bookkeeping the drain stamps
    into claim meta: ``queued_at`` (epoch seconds the entry entered the queue),
    ``attempts`` (count of redispatch attempts so far), ``num_tool_calls`` (the
    worst tool-call count observed across prior attempts), and ``coalesce_key``
    (a channel-supplied grouping key).
  * ``cfg`` — anything exposing ``retry_min_age`` / ``retry_give_up`` /
    ``retry_lifetime_cap`` (a :class:`~osprey.bridges.core.config.CoreConfig`),
    duck-typed so tests can pass a stub.
"""

from __future__ import annotations

from typing import Any

# A failed run is worth retrying only when the failure is external to the agent's
# reasoning — a provider outage or infrastructure fault that a later attempt may
# clear. An agent/user-level failure (bad question, tool refusal) would fail
# identically on redispatch, so it is NOT retryable.
RETRYABLE_FAILURE_CLASSES = frozenset({"provider", "infrastructure"})

# A dispatcher pool result may reject a capability-carrying dispatch outright with
# a structured error_code. Such a rejection is deterministic — an identical
# redispatch would be rejected identically — so it is NON-retryable regardless of
# the failure_class the (bridge-local, infrastructure-defaulted) error result
# otherwise carries. Today these are the input_files rejections.
NON_RETRYABLE_ERROR_CODES = frozenset({"input_files_cap_exceeded", "input_files_invalid"})


def _is_int(value: Any) -> bool:
    """True for a real integer count (``bool`` is excluded — ``True``/``False``
    are ints in Python but never a valid tool-call count)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _age(entry: dict[str, Any], now: float) -> float:
    """Seconds since the entry entered the retry queue, clamped to ``>= 0``.

    A missing/invalid ``queued_at`` is treated as *just now* (age 0) so a
    malformed entry is neither retried early nor abandoned early. A ``queued_at``
    in the future (clock skew) also clamps to 0.
    """
    q = entry.get("queued_at")
    if not isinstance(q, (int, float)) or isinstance(q, bool):
        q = now
    return max(0.0, now - q)


def ceiling_age(entry: dict[str, Any], now: float) -> float:
    """Seconds since the entry FIRST entered the retry queue, for the give-up
    ceilings.

    A re-park refreshes ``queued_at`` (the per-attempt pacing clock read by
    :func:`is_eligible`) but preserves ``first_queued_at`` — anchoring the
    ceilings there is what stops a cycling entry (park -> retry -> fail ->
    re-park) from re-aging forever and never giving up. Entries persisted
    before ``first_queued_at`` existed fall back to ``queued_at``.
    """
    q = entry.get("first_queued_at")
    if not isinstance(q, (int, float)) or isinstance(q, bool):
        return _age(entry, now)
    return max(0.0, now - q)


def _retried(entry: dict[str, Any]) -> bool:
    """Whether the drain has already made a redispatch attempt for this entry.

    Honors either signal the drain may stamp: a boolean ``retried`` flag (set by
    the ``queued -> in_flight`` transition) or a positive ``attempts`` count.
    """
    if entry.get("retried"):
        return True
    a = entry.get("attempts", 0)
    return _is_int(a) and a > 0


def is_retryable(result: dict[str, Any]) -> bool:
    """Whether ``result``'s failure class warrants any retry at all.

    A structured ``error_code`` in :data:`NON_RETRYABLE_ERROR_CODES` (an
    input_files rejection) forces ``False`` first — such a rejection is
    deterministic even though the bridge-local error result defaults
    ``failure_class`` to the retryable ``"infrastructure"``. Otherwise ``True``
    iff ``failure_class`` is one of :data:`RETRYABLE_FAILURE_CLASSES`; a missing
    or ``None`` class (and a missing/unrelated ``error_code``) -> ``False``
    (unknown is not retried).
    """
    if result.get("error_code") in NON_RETRYABLE_ERROR_CODES:
        return False
    return result.get("failure_class") in RETRYABLE_FAILURE_CLASSES


def may_redispatch(entry: dict[str, Any], result: dict[str, Any]) -> bool:
    """Whether it is *safe* to fire a brand-new dispatch for this entry.

    A redispatch re-runs the agent from scratch, so it is only safe when the
    failed run took **no** side-effecting action — i.e. a *known* tool-call count
    of exactly 0. The count is read from the freshly-observed ``result`` and from
    any worst-case count the drain persisted on the ``entry`` across earlier
    attempts; redispatch is permitted only when at least one of those is known
    AND every known count is 0. If a tool count is unknown everywhere, we cannot
    prove safety, so we refuse (fail-closed).

    Orthogonal to :func:`is_retryable` — the caller checks both: retryable *and*
    safe to redispatch.
    """
    counts = [result.get("num_tool_calls"), entry.get("num_tool_calls")]
    known = [c for c in counts if _is_int(c)]
    return bool(known) and all(c == 0 for c in known)


def is_eligible(entry: dict[str, Any], now: float, cfg: Any) -> bool:
    """Whether a queued entry has aged past ``retry_min_age`` and may be retried
    on this drain pass.

    Holds a just-failed dispatch back long enough for a transient outage to
    clear. ``True`` iff ``age > cfg.retry_min_age`` (strict; a clamped-to-0 age
    can never exceed a positive threshold).
    """
    return bool(_age(entry, now) > cfg.retry_min_age)


def give_up_due(entry: dict[str, Any], now: float, gate_ever_open: bool, cfg: Any) -> bool:
    """Whether a queued entry should be abandoned (a GitLab give-up issue filed)
    instead of retried further.

    Two independent ceilings:

      * the hard **lifetime cap** (``retry_lifetime_cap``, ~7d): once reached the
        entry is reaped no matter what — an unconditional backstop against an
        entry living forever.
      * the softer **give-up age** (``retry_give_up``, ~48h): applied ONLY if the
        entry has actually been retried (a ``retried`` flag or ``attempts > 0``)
        or the health gate was observed open at some point since it was enqueued
        (``gate_ever_open``).
        If neither holds, the outage has been continuous and the entry never got
        a fair attempt, so the give-up age is *extended* rather than triggered —
        we keep it queued until the lifetime cap.

    Both ceilings measure :func:`ceiling_age` — anchored at the entry's FIRST
    park, not its latest re-park — so a cycling entry still ages out.
    """
    age = ceiling_age(entry, now)
    if age >= cfg.retry_lifetime_cap:
        return True
    if age >= cfg.retry_give_up and (_retried(entry) or gate_ever_open):
        return True
    return False


def coalesce_key(entry: dict[str, Any]) -> tuple:
    """Grouping key for collapsing redundant queued entries into one.

    When several messages for the same conversation pile up while dispatch is
    down, the drain redispatches only the newest per key and supersedes the rest.
    The key is channel-supplied and persisted in claim meta as ``coalesce_key``
    (e.g. Chat ``space``/``thread`` or an email thread id); it is normalized to a
    hashable tuple: a list/tuple becomes ``tuple(key)``, any scalar becomes a
    1-tuple, and a missing/empty key becomes the empty tuple ``()``.

    ``()`` is the *no-group* sentinel: an entry without a coalesce key must NOT
    be collapsed with other keyless entries, so the drain treats an empty tuple
    as "stands alone" rather than as a shared group.
    """
    key = entry.get("coalesce_key")
    if key is None or key == "" or key == [] or key == ():
        return ()
    if isinstance(key, (list, tuple)):
        return tuple(key)
    return (key,)
