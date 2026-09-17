"""Configuration for the Microsoft Teams bridge.

:class:`TeamsBridgeConfig` carries the Teams-specific settings — the bot's app
registration (id, secret, tenant), which Azure cloud it lives in, and the Service
Bus queue the relay enqueues activities on — and *composes* a
:class:`~osprey.bridges.core.CoreConfig` for everything channel-neutral (the
dispatcher/worker endpoints, their tokens, the trigger, and the poll/retry
budgets). Composition, not subclassing: the engine's collaborators are handed
``cfg.core``, so nothing Teams-specific can leak into them.

``from_env`` reads the ``TEAMS_*`` names itself and delegates every neutral name
to :meth:`CoreConfig.from_env` **unprefixed** — a deployment sets
``POLL_BUDGET``/``DEDUP_PATH``/``DISPATCH_TRIGGER``, not ``TEAMS_``-prefixed
spellings of them, so every adapter reads one set of names for the settings they
genuinely share.

Two values carry a default here and two deliberately do not:

*  ``TEAMS_CLOUD`` defaults to ``commercial`` — the cloud almost every tenant is
   in, and the one an operator who has never heard of GCC High should not have to
   name. An unrecognised value is rejected outright in :meth:`__post_init__`
   rather than silently treated as commercial: the wrong login host would surface
   only as an opaque authentication failure from inside a worker thread.
*  ``version_tag`` falls back to the installed distribution's version, resolved
   **once**, here — see :meth:`from_env`.
*  The trigger has no code default; its ``teams-question`` default lives at the
   deployment surface only (the profile block renders it as ``DISPATCH_TRIGGER``
   in the compose template).
*  The app registration and the queue coordinates have no defaults at all: every
   one of them names a specific tenant's resources, so a shipped literal would be
   one facility's secret compiled into every deployment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata

from osprey.bridges.core import CoreConfig

DISTRIBUTION = "osprey-framework"
"""Distribution whose installed version stands in for an unset
``APP_VERSION_DISPLAY``. Named once so :meth:`TeamsBridgeConfig.from_env` and its
tests cannot disagree about which package is being asked."""


@dataclass(frozen=True)
class CloudEndpoints:
    """The two hosts that differ between the Azure clouds a bot can live in.

    Nothing else about the bridge is cloud-dependent: replies always go to the
    ``serviceUrl`` the activity itself carried, so the outbound host is never
    chosen from a table.
    """

    login_host: str
    """Host of the AAD token endpoint, ``POST https://{host}/{tenant}/oauth2/v2.0/token``."""

    token_scope: str
    """Scope requested for that token — the Bot Connector audience in this cloud."""


CLOUDS: Mapping[str, CloudEndpoints] = {
    "commercial": CloudEndpoints(
        login_host="login.microsoftonline.com",
        token_scope="https://api.botframework.com/.default",
    ),
    "gcchigh": CloudEndpoints(
        login_host="login.microsoftonline.us",
        token_scope="https://api.botframework.us/.default",
    ),
}
"""Every cloud the bridge supports, keyed by the value an operator sets as
``TEAMS_CLOUD``. Closed on purpose: a value outside this table is rejected at
construction, so adding a cloud is a deliberate edit here rather than a string
an operator can guess at."""

DEFAULT_CLOUD = "commercial"
"""Cloud assumed when ``TEAMS_CLOUD`` is unset — or renders empty under compose,
where the variable is optional and therefore bare."""

CORE_URL_ENV = {"dispatcher_url": "DISPATCHER_URL", "worker_url": "WORKER_URL"}
"""The two :class:`CoreConfig` URL fields :meth:`TeamsBridgeConfig.require_startup`
does not cover, mapped to the environment variables an operator actually sets. One
mapping, so :func:`require_boot` and its error message cannot name different things."""


@dataclass(frozen=True)
class TeamsBridgeConfig:
    """Microsoft Teams settings plus a :class:`CoreConfig` projection.

    Immutable: every collaborator (the Service Bus receive loop, the shared
    ``ChannelOps`` instance, the drain thread) reads the same instance
    concurrently, so it is built once at startup and never mutated. Build a
    variant with :func:`dataclasses.replace` — which re-runs
    :meth:`__post_init__`, so a variant cannot slip past the cloud check.
    """

    app_id: str = ""
    """The bot's app-registration (client) id. Also the audience the relay
    validates inbound Bot Framework tokens against, and — as ``28:{app_id}`` —
    the mention target the channel filter matches, so the bridge answers messages
    that actually @mention this bot rather than every message in a channel."""

    app_secret: str = ""
    """Client secret for that app registration, exchanged for a Bot Connector
    token. A credential: no default, and never logged."""

    tenant_id: str = ""
    """Directory (tenant) id the app registration lives in — the ``{tenant}``
    segment of the token endpoint. Single-tenant bots have no usable default."""

    cloud: str = DEFAULT_CLOUD
    """Which Azure cloud the bot is registered in, a key of :data:`CLOUDS`.
    Selects the login host and the token scope, and nothing else."""

    servicebus_connection_string: str = ""
    """Connection string for the Service Bus namespace the relay enqueues
    activities on, scoped to a listen-only policy. A credential: no default."""

    servicebus_queue: str = ""
    """Queue within that namespace. Exactly one bridge consumes a given queue —
    two consumers would split a conversation's activities between them."""

    version_tag: str = ""
    """OPTIONAL deployed release tag (e.g. ``v2026.7.1+abc1234``) shown in the ack
    so every conversation says which release answered. Resolved once in
    :meth:`from_env`; empty renders the ack without the parenthetical, and is
    never a startup requirement."""

    core: CoreConfig = field(default_factory=CoreConfig)
    """The channel-neutral half, handed to the engine's collaborators as-is."""

    def __post_init__(self) -> None:
        # Reject an unknown cloud here rather than falling back to commercial: a
        # GCC High tenant misspelled as "gcc-high" would otherwise be sent to the
        # commercial login host, and the only symptom would be an authentication
        # failure from inside the token refresh that says nothing about the
        # misspelling. This is the one Teams setting with a closed vocabulary, so
        # it is the one that can be checked at all.
        if self.cloud not in CLOUDS:
            raise ValueError(
                f"TEAMS_CLOUD must be one of {', '.join(sorted(CLOUDS))}; got {self.cloud!r}"
            )

    @property
    def login_host(self) -> str:
        """Host of the AAD token endpoint for :attr:`cloud`."""
        return CLOUDS[self.cloud].login_host

    @property
    def token_scope(self) -> str:
        """Bot Connector scope requested for :attr:`cloud`."""
        return CLOUDS[self.cloud].token_scope

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TeamsBridgeConfig:
        """Build the config from environment variables.

        Reads the ``TEAMS_*`` names itself and delegates every neutral name to
        :meth:`CoreConfig.from_env`. No trigger default is applied here.

        ``version_tag`` is resolved once, at this point: ``APP_VERSION_DISPLAY``
        if the image build baked one in, else the installed distribution's
        version, else empty. The distribution lookup is a metadata read and the
        answer cannot change while the process runs, so doing it here rather than
        in a property keeps it off the path of every ack the bridge posts. A
        source checkout has no installed distribution at all — that is a plainer
        ack, not a failure, so :class:`~importlib.metadata.PackageNotFoundError`
        is swallowed.

        Both reads fall back on an *empty* value as well as an absent one:
        compose renders an unset bare ``${VAR}`` as ``""``, so a check for the
        absent key alone would leave every composed deployment with a blank tag
        and, for the cloud, a value that no table row matches.

        Args:
            env: Mapping to read instead of :data:`os.environ` (tests).

        Returns:
            The parsed config. Completeness is *not* checked — call
            :meth:`require_startup` (or :func:`require_boot`) before starting any
            thread.

        Raises:
            ValueError: If ``TEAMS_CLOUD`` names a cloud outside :data:`CLOUDS`.
        """
        e = os.environ if env is None else env
        return cls(
            app_id=e.get("TEAMS_APP_ID", ""),
            app_secret=e.get("TEAMS_APP_SECRET", ""),
            tenant_id=e.get("TEAMS_TENANT_ID", ""),
            cloud=e.get("TEAMS_CLOUD", "") or DEFAULT_CLOUD,
            servicebus_connection_string=e.get("TEAMS_SERVICEBUS_CONNECTION_STRING", ""),
            servicebus_queue=e.get("TEAMS_SERVICEBUS_QUEUE", ""),
            version_tag=e.get("APP_VERSION_DISPLAY", "") or _installed_version(),
            core=CoreConfig.from_env(e),
        )

    def require_startup(self) -> None:
        """Raise unless everything the bridge cannot run without is set.

        Covers the app registration, the queue coordinates, and the neutral
        trigger plus both dispatch tokens. ``cloud`` and ``version_tag`` are
        deliberately excluded: the cloud has a checked default, and without a
        version tag the ack simply omits it.

        The error names the missing **environment variables** (not field names),
        since that is what a deployment sets, and names all of them in one raise
        so a half-configured deployment is fixed in one pass. Call this before
        starting the receive or drain thread.

        Raises:
            ValueError: If any required value is unset, listing the missing names.
        """
        checks: tuple[tuple[str, str], ...] = (
            ("TEAMS_APP_ID", self.app_id),
            ("TEAMS_APP_SECRET", self.app_secret),
            ("TEAMS_TENANT_ID", self.tenant_id),
            ("TEAMS_SERVICEBUS_CONNECTION_STRING", self.servicebus_connection_string),
            ("TEAMS_SERVICEBUS_QUEUE", self.servicebus_queue),
            ("DISPATCH_TRIGGER", self.core.trigger),
            ("EVENT_DISPATCHER_TOKEN", self.core.event_dispatcher_token),
            ("DISPATCH_WORKER_TOKEN", self.core.dispatch_worker_token),
        )
        missing = [name for name, value in checks if not value]
        if missing:
            raise ValueError(f"missing required config: {', '.join(missing)}")


def _installed_version() -> str:
    """The installed distribution's version, or ``""`` when it is not installed."""
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return ""


def require_boot(cfg: TeamsBridgeConfig) -> None:
    """Raise unless the environment carries everything the bridge cannot run without.

    Two checks. The second is not redundant with the first:
    :meth:`TeamsBridgeConfig.require_startup` deliberately does not cover the
    dispatcher/worker URLs, which have code defaults — and under compose those
    defaults are a trap. An unset **bare** ``${VAR}`` renders as an *empty string*,
    not an absent key, so ``CoreConfig.from_env``'s ``localhost`` fallbacks never
    fire and the bridge boots with ``dispatcher_url == ""``. It would then POST
    every question to a protocol-less URL and fail — not loudly, which is the
    problem: the dispatch client turns the transport error into an error result
    and the receive loop settles the message either way, so nothing redelivers and
    nothing crashes. The bridge would sit there answering no one, one
    parked-or-errored question at a time, with the cause visible only to someone
    correlating per-event logs. A startup abort naming the variable is strictly
    better than that, which is what the second check restores. (The rendered
    compose template guards the same two with ``${VAR:?...}``; this check is what
    covers a hand-written or otherwise un-rendered deployment.)

    Args:
        cfg: The config to check.

    Raises:
        ValueError: If anything required is unset, naming the missing **environment
            variables** — the abort has to be diagnosable from the container log alone.
    """
    cfg.require_startup()
    try:
        cfg.core.require(*CORE_URL_ENV)
    except ValueError as exc:
        # The engine's check reports FIELD names; an operator sets env vars. Re-raise
        # in the spelling a deployment uses, keeping the original as the cause.
        missing = [
            env for field_name, env in CORE_URL_ENV.items() if not getattr(cfg.core, field_name)
        ]
        raise ValueError(f"missing required config: {', '.join(missing)}") from exc
