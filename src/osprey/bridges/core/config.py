"""Channel-neutral configuration shared by every dispatch bridge.

:class:`CoreConfig` carries ONLY the fields a bridge needs to talk to the
``osprey.dispatch`` pair and persist its local crash-safety stores — the
dispatcher/worker URLs + tokens, the trigger name, the poll budget, and the
dedup/history store paths. Nothing here is chat-, email-, or otherwise
channel-specific, and (like every ``osprey.bridges.core`` module) it carries ZERO
osprey dispatch imports.

Each channel's own config *composes* a ``CoreConfig`` for these neutral fields
and adds its channel-specific ones on top.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from osprey.port_layout import default_port

# Statuses that mean an entry needs no further work. "completed"/"error" are
# the worker's terminal run states; "post_failed" is bridge-local — the run
# finished but delivering the reply failed permanently (do NOT retry on
# restart). "superseded" is bridge-local too — a queued entry dropped because a
# newer question in the same conversation coalesced over it; it MUST be
# terminal, or DedupStore.reconcile() would re-surface it on restart and re-run
# the deliberately dropped question. None of these is channel-specific, so they
# live in the core.
TERMINAL_STATUSES = frozenset({"completed", "error", "post_failed", "superseded"})

# Spellings accepted as "on" for boolean env vars, matching the rest of the
# repo (``osprey.interfaces.vendor``, ``osprey.services.bluesky_bridge``).
_TRUTHY = {"1", "true", "yes", "on"}

#: Where a bridge reaches the dispatch pair when nothing tells it otherwise: the
#: layout's ``dispatcher`` slot and worker 1 of its ``worker`` band, both at the
#: layout's DEFAULT base — the one place that base is legitimate, because a
#: bridge has no project config to resolve one from. Neither is what a deployed
#: bridge uses: every bridge compose service sets ``DISPATCHER_URL``/``WORKER_URL``
#: from its own deployment's block, so these are reached only by a bridge started
#: by hand against a default-base project.
_DEFAULT_DISPATCHER_URL = f"http://localhost:{default_port('dispatcher')}"
_DEFAULT_WORKER_URL = f"http://localhost:{default_port('worker', 1)}"


@dataclass
class CoreConfig:
    """Channel-neutral dispatch + local-store configuration.

    Field names are the contract every ``osprey.bridges.core`` collaborator reads
    (:class:`~osprey.bridges.core.dispatch_client.DispatchClient`,
    :func:`~osprey.bridges.core.artifacts.fetch_artifact`, the stores). A channel
    config that exposes these same attributes is a structural superset and works
    anywhere a ``CoreConfig`` is expected.
    """

    # --- Dispatch pipeline endpoints ---------------------------------------
    dispatcher_url: str = _DEFAULT_DISPATCHER_URL
    worker_url: str = _DEFAULT_WORKER_URL
    event_dispatcher_token: str = ""
    """Bearer token for the dispatcher webhook (inbound to the dispatcher slot)."""

    dispatch_worker_token: str = ""
    """Bearer token for the worker run-status/artifact endpoints (the worker slot)."""

    trigger: str = ""
    """Dispatcher trigger to fire (``POST /webhook/{trigger}``). Channel-set —
    the ONLY per-channel difference in the dispatched agent."""

    # --- Polling controls ---------------------------------------------------
    poll_interval: float = 2.0
    """Seconds between worker status polls."""

    poll_budget: float = 330.0
    """Max seconds to poll before giving up. Must be at least ``worker_timeout``."""

    worker_timeout: float = 300.0
    """The worker's own per-run wall-clock cap, in seconds — the deployment's
    ``DISPATCH_TIMEOUT_SEC``. Not used to time anything here; it is the floor
    ``poll_budget`` is validated against, so a facility that raises the worker's
    cap cannot leave the bridge abandoning runs that are still alive."""

    # --- HTTP client behavior -----------------------------------------------
    trust_env: bool = False
    """Whether outbound HTTP clients honor ambient ``HTTP(S)_PROXY``/``NO_PROXY``.

    Default ``False``: the dispatcher, the worker, and (where configured) the
    alert host are reached directly, so a proxy inherited from a dev shell or CI
    runner cannot silently mount itself in front of them. A site whose bridge
    genuinely sits behind an egress proxy sets this true."""

    # --- Retry queue / drain controls --------------------------------------
    drain_interval: float = 60.0
    """Seconds between drain-loop passes over the retry queue."""

    retry_min_age: float = 1200.0
    """Minimum age (s) a queued entry must reach before the drain retries it —
    holds a failed dispatch back long enough for a transient outage to clear."""

    retry_give_up: float = 172800.0
    """Age (s) past which a queued entry that still cannot dispatch is abandoned
    (an alert issue is filed instead of retrying forever). Default 48h."""

    retry_lifetime_cap: float = 604800.0
    """Hard ceiling (s) on how long any entry may live in the store before the
    drain reaps it regardless of status. Default 7d."""

    # --- GitLab alerts (give-up issue filing; opt-in) -----------------------
    gitlab_url: str = ""
    """Base URL of the GitLab instance used to file give-up issues. Empty means
    the site has no alert host: no issues are filed and the health gate skips
    its alert-issue check entirely."""

    gitlab_project: str = ""
    """Project path (namespace/name) that give-up issues are filed against."""

    gitlab_issues_token: str = ""
    """Token for the GitLab issues API. Unprefixed shared secret (like
    ``event_dispatcher_token``) — read from ``GITLAB_ISSUES_TOKEN``."""

    # --- Local state --------------------------------------------------------
    dedup_path: str = "/data/dedup.json"
    """Path to the persisted message_id -> run store (bridge-owned volume)."""

    history_path: str = "/data/history.json"
    """Path to the persisted per-conversation transcript store (same volume)."""

    def __post_init__(self) -> None:
        # The worker caps runs at `worker_timeout`; polling must outlast that to
        # observe the terminal (timeout) result instead of racing it.
        if self.poll_budget < self.worker_timeout:
            raise ValueError(
                f"poll_budget must be >= {self.worker_timeout}s (worker cap); "
                f"got {self.poll_budget}"
            )
        # Drain/retry tunables are durations — a non-positive value would make the
        # drain loop spin or reap everything immediately.
        for name in (
            "worker_timeout",
            "drain_interval",
            "retry_min_age",
            "retry_give_up",
            "retry_lifetime_cap",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0; got {value}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CoreConfig:
        """Build the neutral config from environment variables.

        Reads channel-neutral names (no channel prefix); a channel config maps
        its own prefixed vars onto these fields when it composes a
        ``CoreConfig``.
        """
        e = os.environ if env is None else env

        def _f(name: str, default: float) -> float:
            raw = e.get(name)
            if raw is None or raw == "":
                return default
            return float(raw)

        def _b(name: str, default: bool) -> bool:
            raw = e.get(name)
            if raw is None or raw == "":
                return default
            return raw.strip().lower() in _TRUTHY

        return cls(
            dispatcher_url=e.get("DISPATCHER_URL", _DEFAULT_DISPATCHER_URL).rstrip("/"),
            worker_url=e.get("WORKER_URL", _DEFAULT_WORKER_URL).rstrip("/"),
            event_dispatcher_token=e.get("EVENT_DISPATCHER_TOKEN", ""),
            dispatch_worker_token=e.get("DISPATCH_WORKER_TOKEN", ""),
            trigger=e.get("DISPATCH_TRIGGER", ""),
            poll_interval=_f("POLL_INTERVAL", 2.0),
            poll_budget=_f("POLL_BUDGET", 330.0),
            # Same var the worker itself reads, so a deployment that raises the
            # cap raises it for both halves from one setting.
            worker_timeout=_f("DISPATCH_TIMEOUT_SEC", 300.0),
            trust_env=_b("BRIDGE_TRUST_ENV", False),
            drain_interval=_f("DRAIN_INTERVAL", 60.0),
            retry_min_age=_f("RETRY_MIN_AGE", 1200.0),
            retry_give_up=_f("RETRY_GIVE_UP", 172800.0),
            retry_lifetime_cap=_f("RETRY_LIFETIME_CAP", 604800.0),
            gitlab_url=e.get("GITLAB_URL", "").rstrip("/"),
            gitlab_project=e.get("GITLAB_PROJECT", ""),
            gitlab_issues_token=e.get("GITLAB_ISSUES_TOKEN", ""),
            dedup_path=e.get("DEDUP_PATH", "/data/dedup.json"),
            history_path=e.get("HISTORY_PATH", "/data/history.json"),
        )

    def require(self, *names: str) -> None:
        """Raise if any of the named fields are unset. Used by an entrypoint to
        fail fast before entering its poll loop."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise ValueError(f"missing required config: {', '.join(missing)}")
