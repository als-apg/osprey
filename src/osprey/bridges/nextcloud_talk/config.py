"""Configuration for the Nextcloud Talk bridge.

:class:`NextcloudBridgeConfig` carries the Nextcloud-specific settings — the
instance URL, the bot's credentials, the Talk rooms to watch, and the path of the
adapter-owned poll-offset store — and *composes* a
:class:`~osprey.bridges.core.CoreConfig` for everything channel-neutral (the
dispatcher/worker endpoints, their tokens, the trigger, and the poll/retry
budgets). Composition, not subclassing: the engine's collaborators are handed
``cfg.core``, so nothing channel-specific can leak into them.

``from_env`` deliberately applies **no trigger default**. The
``nextcloud-question`` default lives at the deployment surface only (the profile
block renders it as ``DISPATCH_TRIGGER`` in the compose template), which is what
keeps :meth:`NextcloudBridgeConfig.require_startup` a meaningful check for a
deployment that was not rendered by ``osprey build``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from osprey.bridges.core import CoreConfig

logger = logging.getLogger(__name__)


def _split_rooms(raw: str) -> tuple[str, ...]:
    """Split a comma-separated ``NEXTCLOUD_ROOMS`` value into room tokens.

    Each entry is whitespace-stripped and empty entries are dropped, so a value
    written across a wrapped compose line (or with a trailing comma) yields the
    tokens an operator meant.

    Args:
        raw: The raw env value, possibly empty.

    Returns:
        The room tokens in the order they were written.
    """
    return tuple(token for token in (part.strip() for part in raw.split(",")) if token)


@dataclass(frozen=True)
class NextcloudBridgeConfig:
    """Nextcloud Talk settings plus a :class:`CoreConfig` projection.

    Immutable: every collaborator (the room threads, the shared ``ChannelOps``
    instance, the drain thread) reads the same instance concurrently, so it is
    built once at startup and never mutated. Build a variant with
    :func:`dataclasses.replace`.
    """

    base_url: str = ""
    """Base URL of the Nextcloud instance, without a trailing ``/``. HTTPS is the
    production expectation; anything else logs a warning (never raises)."""

    bot_account: str = ""
    """Nextcloud user id the bridge authenticates as. Also the mention target the
    group-room filter matches, and the owner of the DAV tree artifacts go into."""

    app_password: str = ""
    """App password for ``bot_account`` (Basic auth on every OCS/DAV call)."""

    rooms: tuple[str, ...] = ()
    """Talk room tokens to poll — one long-poll thread per entry."""

    offsets_path: str = "/data/offsets.json"
    """Path to the persisted per-room ``lastKnownMessageId`` store. Adapter-owned,
    and on the same bridge volume as ``core.dedup_path``/``core.history_path`` —
    losing it alone would replay or skip history."""

    core: CoreConfig = field(default_factory=CoreConfig)
    """The channel-neutral half, handed to the engine's collaborators as-is."""

    def __post_init__(self) -> None:
        # Plaintext HTTP puts the bot's app password on the wire, but a facility
        # may legitimately run the bridge against an internal host, so this is a
        # loud warning and never a hard error.
        if self.base_url and not self.base_url.lower().startswith("https://"):
            logger.warning(
                "NEXTCLOUD_BASE_URL is not https (%s): bot credentials and message "
                "content cross the network unencrypted. HTTPS is the production "
                "expectation.",
                self.base_url,
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> NextcloudBridgeConfig:
        """Build the config from environment variables.

        Reads the ``NEXTCLOUD_*`` names itself and delegates every neutral name to
        :meth:`CoreConfig.from_env`. No trigger default is applied here.

        Args:
            env: Mapping to read instead of :data:`os.environ` (tests).

        Returns:
            The parsed config. Completeness is *not* checked — call
            :meth:`require_startup` before starting any thread.
        """
        e = os.environ if env is None else env
        return cls(
            base_url=e.get("NEXTCLOUD_BASE_URL", "").rstrip("/"),
            bot_account=e.get("NEXTCLOUD_BOT_ACCOUNT", ""),
            app_password=e.get("NEXTCLOUD_APP_PASSWORD", ""),
            rooms=_split_rooms(e.get("NEXTCLOUD_ROOMS", "")),
            offsets_path=e.get("OFFSETS_PATH", "/data/offsets.json"),
            core=CoreConfig.from_env(e),
        )

    def require_startup(self) -> None:
        """Raise unless everything the bridge cannot run without is set.

        Covers the Nextcloud credentials and rooms plus the neutral trigger and
        both dispatch tokens. The error names the missing **environment
        variables** (not field names), since that is what a deployment sets, and
        names all of them in one raise so a half-configured deployment is fixed in
        one pass. Call this before starting any room or drain thread.

        Raises:
            ValueError: If any required value is unset, listing the missing names.
        """
        checks: tuple[tuple[str, str | tuple[str, ...]], ...] = (
            ("NEXTCLOUD_BASE_URL", self.base_url),
            ("NEXTCLOUD_BOT_ACCOUNT", self.bot_account),
            ("NEXTCLOUD_APP_PASSWORD", self.app_password),
            ("NEXTCLOUD_ROOMS", self.rooms),
            ("DISPATCH_TRIGGER", self.core.trigger),
            ("EVENT_DISPATCHER_TOKEN", self.core.event_dispatcher_token),
            ("DISPATCH_WORKER_TOKEN", self.core.dispatch_worker_token),
        )
        missing = [name for name, value in checks if not value]
        if missing:
            raise ValueError(f"missing required config: {', '.join(missing)}")
