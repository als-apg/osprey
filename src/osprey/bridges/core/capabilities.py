"""Per-dispatch capability probe for the OSPREY dispatch pair.

Some dispatches carry a payload that only newer dispatcher/worker builds accept —
the prime case is ``input_files`` (attachment bytes handed to the agent). Before
a bridge fires such a dispatch it must confirm BOTH halves of the pair advertise
the capability: a dispatch whose files reach an old worker would fail (or worse,
silently drop them) *after* the user's message was already accepted.

:func:`pair_supports` is that go/no-go check. It GETs ``/health`` on both the
dispatcher and the worker and returns ``True`` only when the named capability
appears in the ``"capabilities"`` list of BOTH bodies. It is **fail-closed**:
any non-200, timeout, malformed body, missing/non-list ``capabilities`` key, or
transport error on EITHER endpoint yields ``False``.

There is deliberately **no caching** — capability is re-probed immediately before
every capability-carrying dispatch, so a mid-session worker downgrade (or a
just-completed upgrade) is observed on the very next dispatch rather than sticking
to a stale answer.

Consistent with :mod:`osprey.bridges.core.health_gate` (which GETs the same two
``/health`` endpoints for the drain gate): a short per-request timeout, and the
default client honors ``cfg.trust_env`` — ``False`` by default, so it ignores
``HTTP(S)_PROXY`` and reaches the localhost endpoints directly. Tests inject their
own :class:`httpx.MockTransport` client.

Like every ``osprey.bridges.core`` module this carries ZERO osprey imports and
ZERO channel (chat/email) imports — pure stdlib + httpx.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# Short per-request timeout: a hung dispatcher/worker must fail the probe closed
# quickly rather than stall the dispatch it gates. Matches health_gate._TIMEOUT.
_TIMEOUT = 10.0


def _health_capabilities(http: httpx.Client, base_url: str) -> frozenset[str] | None:
    """The capability set advertised by ``{base_url}/health``, or ``None`` on any
    failure.

    Fail-closed: a non-200, non-JSON body, or a missing/non-list ``capabilities``
    key -> ``None``. ``None`` is distinct from an empty ``frozenset`` (a healthy
    endpoint that advertises no capabilities) — the caller treats both as "does
    not support X", but only the empty set means the probe actually succeeded.
    Non-string entries in the list are ignored.
    """
    try:
        resp = http.get(f"{base_url}/health")
        if resp.status_code != 200:
            logger.warning("capability probe %s/health returned %d", base_url, resp.status_code)
            return None
        caps = resp.json().get("capabilities")
        if not isinstance(caps, list):
            logger.warning("capability probe %s/health missing capabilities list", base_url)
            return None
        return frozenset(c for c in caps if isinstance(c, str))
    except Exception as exc:
        logger.warning("capability probe %s/health failed: %s", base_url, exc)
        return None


def pair_supports(cfg, capability: str, http: httpx.Client | None = None) -> bool:
    """Return ``True`` only if BOTH the dispatcher and the worker advertise
    ``capability`` in their ``/health`` body.

    Probes ``cfg.dispatcher_url`` then ``cfg.worker_url`` (the real
    :class:`~osprey.bridges.core.config.CoreConfig` field names, shared with
    :func:`osprey.bridges.core.health_gate.gate_open`). **Fail-closed**: if either
    probe fails for any reason, or either side omits the capability, returns
    ``False``. The worker is not probed when the dispatcher already lacks it
    (short-circuit).

    No caching — call this immediately before every dispatch that carries the
    capability's payload. ``http`` is injectable for tests; otherwise a client is
    built and closed here, honoring ``cfg.trust_env`` for proxy handling.
    """
    own_client = http is None
    client = http or httpx.Client(timeout=_TIMEOUT, trust_env=cfg.trust_env)
    try:
        disp = _health_capabilities(client, cfg.dispatcher_url)
        if disp is None or capability not in disp:
            return False
        work = _health_capabilities(client, cfg.worker_url)
        return work is not None and capability in work
    finally:
        if own_client:
            client.close()
