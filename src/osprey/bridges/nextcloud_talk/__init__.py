"""Nextcloud Talk adapter for the OSPREY dispatch bridge engine.

Contributes only Talk's wire format and platform I/O: OCS long-poll ingestion with
persisted per-room offsets, OCS chat replies, and WebDAV+share artifact delivery.
All ordering, dedup, retry, and crash recovery stay in
:mod:`osprey.bridges.core`, which this package imports from its root only.
"""

from .client import DavTooLargeError, RoomInfo, TalkApiError, TalkClient, upload_dir
from .config import NextcloudBridgeConfig
from .events import parse_event, resolve_reply_context
from .offsets import OffsetStore
from .ops import NextcloudTalkOps
from .poller import RoomPoller
from .rooms import RoomDirectory, RoomTypeUnresolved

__all__ = [
    "DavTooLargeError",
    "NextcloudBridgeConfig",
    "NextcloudTalkOps",
    "OffsetStore",
    "RoomDirectory",
    "RoomPoller",
    "RoomInfo",
    "RoomTypeUnresolved",
    "TalkApiError",
    "TalkClient",
    "parse_event",
    "resolve_reply_context",
    "upload_dir",
]
