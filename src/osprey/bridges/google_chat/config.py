"""Configuration for the Google Chat bridge.

:class:`GoogleChatBridgeConfig` carries the Google-specific settings — the
service-account key, the Pub/Sub subscription events are pulled from, the Chat
app's own id, and the optional GCS bucket artifacts are published through — and
*composes* a :class:`~osprey.bridges.core.CoreConfig` for everything
channel-neutral (the dispatcher/worker endpoints, their tokens, the trigger, and
the poll/retry budgets). Composition, not subclassing: the engine's collaborators
are handed ``cfg.core``, so nothing Google-specific can leak into them.

``from_env`` reads the ``GCHAT_*``/``GCS_*`` names itself and delegates every
neutral name to :meth:`CoreConfig.from_env` **unprefixed** — a deployment sets
``POLL_BUDGET``/``DEDUP_PATH``/``DISPATCH_TRIGGER``, not ``GCHAT_``-prefixed
spellings of them, so the two adapters read one set of names for the settings
they genuinely share.

Two values deliberately have **no code default**:

*  the trigger, whose ``gchat-question`` default lives at the deployment surface
   only (the profile block renders it as ``DISPATCH_TRIGGER`` in the compose
   template);
*  the subscription, which is a fully-qualified, project-scoped Pub/Sub resource
   name and therefore cannot have a facility-neutral default at all.

Both are what keeps :meth:`GoogleChatBridgeConfig.require_startup` a meaningful
check for a deployment that was not rendered by ``osprey build`` — a default here
would silently point a facility's bridge at somebody else's project.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from osprey.bridges.core import CoreConfig

logger = logging.getLogger(__name__)

SUBSCRIPTION_PREFIX = "projects/"
"""Leading segment of a fully-qualified pull-subscription name. The Pub/Sub pull
API takes ``projects/{project}/subscriptions/{sub}`` and nothing shorter; a bare
subscription id is the plausible mistake, so it is worth warning about."""

SUBSCRIPTION_INFIX = "/subscriptions/"
"""The second half of that shape, checked with the prefix above."""

CORE_URL_ENV = {"dispatcher_url": "DISPATCHER_URL", "worker_url": "WORKER_URL"}
"""The two :class:`CoreConfig` URL fields :meth:`GoogleChatBridgeConfig.require_startup`
does not cover, mapped to the environment variables an operator actually sets. One
mapping, so :func:`require_boot` and its error message cannot name different things."""


@dataclass(frozen=True)
class GoogleChatBridgeConfig:
    """Google Chat settings plus a :class:`CoreConfig` projection.

    Immutable: every collaborator (the subscriber loop, the shared ``ChannelOps``
    instance, the drain thread) reads the same instance concurrently, so it is
    built once at startup and never mutated. Build a variant with
    :func:`dataclasses.replace`.
    """

    sa_key: str = ""
    """Path to the service-account JSON key the bridge authenticates as (scope
    ``chat.bot``). The same identity posts replies and pulls the subscription."""

    subscription: str = ""
    """Fully-qualified Pub/Sub pull subscription carrying the Chat events, as
    ``projects/{project}/subscriptions/{sub}``. No default: it names a specific
    facility's GCP project, so a shipped literal would point every deployment
    that forgot to set it at one facility's queue."""

    app_id: str = ""
    """The Chat app's own user id, e.g. ``users/1234567890``. Also the mention
    target the space filter matches, so the bridge answers messages that actually
    @mention this bot rather than every message in a space."""

    gcs_bucket: str = ""
    """GCS bucket artifacts (plots, files) are uploaded to for delivery. OPTIONAL:
    unset disables image delivery and the bridge still posts text-only answers,
    which is why it is absent from :meth:`require_startup`."""

    gcs_project: str = ""
    """GCP project owning ``gcs_bucket``. OPTIONAL, same rationale as above."""

    version_tag: str = ""
    """OPTIONAL deployed release tag (e.g. ``v2026.7.1+abc1234``), baked into the
    image as ``APP_VERSION_DISPLAY`` and appended to the ack so every conversation
    shows which release answered. Unset renders the ack without it — never a
    startup requirement."""

    core: CoreConfig = field(default_factory=CoreConfig)
    """The channel-neutral half, handed to the engine's collaborators as-is."""

    def __post_init__(self) -> None:
        # A bare subscription id ("gchat-events-sub") is the plausible mistake and
        # would fail only on the first pull, inside the subscriber thread. Warn
        # loudly at boot instead — but never raise: the resource name is Google's
        # to validate, and a shape check that rejects a name Google would accept
        # would be a bridge that refuses to start for no reason.
        if self.subscription and not (
            self.subscription.startswith(SUBSCRIPTION_PREFIX)
            and SUBSCRIPTION_INFIX in self.subscription
        ):
            logger.warning(
                "GCHAT_SUBSCRIPTION does not look fully-qualified (%s): the Pub/Sub "
                "pull API expects projects/{project}/subscriptions/{subscription}.",
                self.subscription,
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GoogleChatBridgeConfig:
        """Build the config from environment variables.

        Reads the ``GCHAT_*``/``GCS_*`` names itself and delegates every neutral
        name to :meth:`CoreConfig.from_env`. No trigger and no subscription
        default is applied here.

        Args:
            env: Mapping to read instead of :data:`os.environ` (tests).

        Returns:
            The parsed config. Completeness is *not* checked — call
            :meth:`require_startup` (or :func:`require_boot`) before starting any
            thread.
        """
        e = os.environ if env is None else env
        return cls(
            sa_key=e.get("GCHAT_SA_KEY", ""),
            subscription=e.get("GCHAT_SUBSCRIPTION", ""),
            app_id=e.get("GCHAT_APP_ID", ""),
            gcs_bucket=e.get("GCS_BUCKET", ""),
            gcs_project=e.get("GCS_PROJECT", ""),
            version_tag=e.get("APP_VERSION_DISPLAY", ""),
            core=CoreConfig.from_env(e),
        )

    def require_startup(self) -> None:
        """Raise unless everything the bridge cannot run without is set.

        Covers the Google credentials, the subscription and the app id, plus the
        neutral trigger and both dispatch tokens. The optional
        ``gcs_bucket``/``gcs_project`` and ``version_tag`` are deliberately
        excluded: without a bucket the bridge still answers, text-only, and
        without a version tag the ack simply omits it.

        The error names the missing **environment variables** (not field names),
        since that is what a deployment sets, and names all of them in one raise
        so a half-configured deployment is fixed in one pass. Call this before
        starting the subscriber or drain thread.

        Raises:
            ValueError: If any required value is unset, listing the missing names.
        """
        checks: tuple[tuple[str, str], ...] = (
            ("GCHAT_SA_KEY", self.sa_key),
            ("GCHAT_SUBSCRIPTION", self.subscription),
            ("GCHAT_APP_ID", self.app_id),
            ("DISPATCH_TRIGGER", self.core.trigger),
            ("EVENT_DISPATCHER_TOKEN", self.core.event_dispatcher_token),
            ("DISPATCH_WORKER_TOKEN", self.core.dispatch_worker_token),
        )
        missing = [name for name, value in checks if not value]
        if missing:
            raise ValueError(f"missing required config: {', '.join(missing)}")


def require_boot(cfg: GoogleChatBridgeConfig) -> None:
    """Raise unless the environment carries everything the bridge cannot run without.

    Two checks. The second is not redundant with the first:
    :meth:`GoogleChatBridgeConfig.require_startup` deliberately does not cover the
    dispatcher/worker URLs, which have code defaults — and under compose those
    defaults are a trap. An unset **bare** ``${VAR}`` renders as an *empty string*,
    not an absent key, so ``CoreConfig.from_env``'s ``localhost`` fallbacks never
    fire and the bridge boots with ``dispatcher_url == ""``. It would then POST
    every question to a protocol-less URL and fail — not loudly, which is the
    problem: ``DispatchClient.run`` turns an ``httpx.HTTPError`` into an error
    result, and the subscriber acks every message from a ``finally`` regardless, so
    nothing redelivers and nothing crashes. The bridge would sit there answering no
    one, one parked-or-errored question at a time, with the cause visible only to
    someone correlating per-event logs. A startup abort naming the variable is
    strictly better than that, which is what the second check restores. (The rendered compose template
    guards the same two with ``${VAR:?...}``; this check is what covers a
    hand-written or otherwise un-rendered deployment.)

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
