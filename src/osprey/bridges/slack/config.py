"""Configuration for the Slack bridge.

:class:`SlackBridgeConfig` carries the Slack-specific settings — the bot token
(``xoxb-``) used for the Web API, the app-level token (``xapp-``) that opens the
Socket Mode connection, whether agent mentions become real Slack mentions, and
the release tag shown in the ack — and *composes* a
:class:`~osprey.bridges.core.CoreConfig` for everything channel-neutral (the
dispatcher/worker endpoints, their tokens, the trigger, and the poll/retry
budgets). Composition, not subclassing: the engine's collaborators are handed
``cfg.core``, so nothing Slack-specific can leak into them.

``from_env`` reads the ``SLACK_*`` names itself and delegates every neutral name
to :meth:`CoreConfig.from_env` **unprefixed** — a deployment sets
``POLL_BUDGET``/``DEDUP_PATH``/``DISPATCH_TRIGGER``, not ``SLACK_``-prefixed
spellings of them.

Defaults:

*  Both tokens have no default: each names one workspace's app installation, so a
   shipped literal would be one facility's secret compiled into every deployment.
   Neither ever appears in ``repr`` — a config logged at startup, or carried in a
   traceback, must not leak a credential.
*  ``version_tag`` falls back to the installed distribution's version, resolved
   **once**, in :meth:`SlackBridgeConfig.from_env`.
*  The trigger has no code default; its ``slack-question`` default lives at the
   deployment surface only (the profile block renders it as ``DISPATCH_TRIGGER``).
*  ``SLACK_MENTIONS`` is optional and defaults on.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata

from osprey.bridges.core import CoreConfig, env_flag

DISTRIBUTION = "osprey-framework"
"""Distribution whose installed version stands in for an unset
``APP_VERSION_DISPLAY``. Named once so :meth:`SlackBridgeConfig.from_env` and its
tests cannot disagree about which package is being asked."""

CORE_URL_ENV = {"dispatcher_url": "DISPATCHER_URL", "worker_url": "WORKER_URL"}
"""The two :class:`CoreConfig` URL fields :meth:`SlackBridgeConfig.require_startup`
does not cover, mapped to the environment variables an operator actually sets. One
mapping, so :func:`require_boot` and its error message cannot name different things."""


@dataclass(frozen=True)
class SlackBridgeConfig:
    """Slack settings plus a :class:`CoreConfig` projection.

    Immutable: every collaborator (the socket loop, the shared ``ChannelOps``
    instance, the drain thread) reads the same instance concurrently, so it is
    built once at startup and never mutated. Build a variant with
    :func:`dataclasses.replace`.
    """

    bot_token: str = field(default="", repr=False)
    """Bot user OAuth token (``xoxb-…``) authorising every Web API call. A
    credential: no default, and excluded from ``repr``."""

    app_token: str = field(default="", repr=False)
    """App-level token (``xapp-…``, ``connections:write``) that opens the Socket
    Mode connection. A credential: no default, and excluded from ``repr``."""

    version_tag: str = ""
    """OPTIONAL deployed release tag shown in the ack so every conversation says
    which release answered. Resolved once in :meth:`from_env`; empty renders the
    ack without it, and is never a startup requirement."""

    mentions: bool = True
    """Whether an agent's ``<@ID>`` becomes a real Slack mention. On by default.
    Off, every mention is posted as plain text and the agent is told to name
    people instead. Set from the build profile, rendered as ``SLACK_MENTIONS``."""

    core: CoreConfig = field(default_factory=CoreConfig)
    """The channel-neutral half, handed to the engine's collaborators as-is."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SlackBridgeConfig:
        """Build the config from environment variables.

        Reads the ``SLACK_*`` names itself and delegates every neutral name to
        :meth:`CoreConfig.from_env`. No trigger default is applied here.

        ``version_tag`` is ``APP_VERSION_DISPLAY`` if set and non-empty, else the
        installed distribution's version, else empty (a source checkout). An empty
        value falls back like an absent one, because compose renders an unset bare
        ``${VAR}`` as ``""``.

        Args:
            env: Mapping to read instead of :data:`os.environ` (tests).

        Returns:
            The parsed config. Completeness is *not* checked — call
            :meth:`require_startup` (or :func:`require_boot`) before starting any
            thread.
        """
        e = os.environ if env is None else env
        return cls(
            bot_token=e.get("SLACK_BOT_TOKEN", ""),
            app_token=e.get("SLACK_APP_TOKEN", ""),
            version_tag=e.get("APP_VERSION_DISPLAY", "") or _installed_version(),
            mentions=env_flag(e.get("SLACK_MENTIONS"), True),
            core=CoreConfig.from_env(e),
        )

    def require_startup(self) -> None:
        """Raise unless everything the bridge cannot run without is set.

        Covers both Slack tokens plus the neutral trigger and both dispatch
        tokens. ``version_tag`` and ``mentions`` are excluded: both have a usable
        default.

        The error names the missing **environment variables** (not field names),
        all of them in one raise, so a half-configured deployment is fixed in one
        pass.

        Raises:
            ValueError: If any required value is unset, listing the missing names.
        """
        checks: tuple[tuple[str, str], ...] = (
            ("SLACK_BOT_TOKEN", self.bot_token),
            ("SLACK_APP_TOKEN", self.app_token),
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


def require_boot(cfg: SlackBridgeConfig) -> None:
    """Raise unless the environment carries everything the bridge cannot run without.

    Two checks. The second is not redundant with the first:
    :meth:`SlackBridgeConfig.require_startup` does not cover the dispatcher/worker
    URLs, which have code defaults — and under compose an unset bare ``${VAR}``
    renders as an *empty string*, so ``CoreConfig.from_env``'s ``localhost``
    fallbacks never fire and the bridge would boot posting every question to a
    protocol-less URL, failing quietly one question at a time. A startup abort
    naming the variable is strictly better.

    Args:
        cfg: The config to check.

    Raises:
        ValueError: If anything required is unset, naming the missing **environment
            variables**.
    """
    cfg.require_startup()
    try:
        cfg.core.require(*CORE_URL_ENV)
    except ValueError as exc:
        # The engine's check reports FIELD names; an operator sets env vars.
        missing = [
            env for field_name, env in CORE_URL_ENV.items() if not getattr(cfg.core, field_name)
        ]
        raise ValueError(f"missing required config: {', '.join(missing)}") from exc
