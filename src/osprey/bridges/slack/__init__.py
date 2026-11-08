"""Slack adapter for the OSPREY dispatch bridge engine.

Contributes only Slack's wire format and platform I/O: Socket Mode ingestion of
Events API envelopes and Web API replies. All ordering, dedup, retry, and crash
recovery stay in :mod:`osprey.bridges.core`, which this package imports from its
root only.

Importing this package root pulls in no heavy dependency, so it is safe in a dev
checkout, during test collection, or while ``osprey build`` renders the compose
template.

``__main__`` is deliberately absent from the re-exports: importing it from here
would run the process entrypoint's module body a second time under
``python -m osprey.bridges.slack``.
"""

from .client import SlackApiError, SlackAuthError, SlackWebClient
from .config import SlackBridgeConfig, require_boot
from .events import (
    SL_CHANNEL,
    SL_CHANNEL_TYPE,
    SL_FILES,
    SL_PARENT_FILES,
    SL_TEAM,
    SL_TEXT_HEAD,
    SL_THREAD_TS,
    SL_TS,
    BotIdentity,
    parse_event,
)
from .formatting import markdown_to_mrkdwn
from .inbox import Inbox
from .ops import (
    ACK_TEXT,
    ANSWER_CHUNK_CHARS,
    EMPTY_ANSWER_TEXT,
    ERROR_TEXT,
    GIVEUP_TEXT,
    QUEUED_TEXT,
    RESUMED_TEXT,
    SKIPPED_FILES_NOTE,
    SUPERSEDED_TEXT,
    VERSION_TAG_SUFFIX,
    SlackOps,
    ack_text,
    quote_prefix,
    skipped_files_note,
)
from .receiver import Ingest, serve
from .socket import SocketModeConnection

__all__ = [
    "ACK_TEXT",
    "ANSWER_CHUNK_CHARS",
    "BotIdentity",
    "EMPTY_ANSWER_TEXT",
    "ERROR_TEXT",
    "GIVEUP_TEXT",
    "Inbox",
    "Ingest",
    "QUEUED_TEXT",
    "RESUMED_TEXT",
    "SKIPPED_FILES_NOTE",
    "SL_CHANNEL",
    "SL_CHANNEL_TYPE",
    "SL_FILES",
    "SL_PARENT_FILES",
    "SL_TEAM",
    "SL_TEXT_HEAD",
    "SL_THREAD_TS",
    "SL_TS",
    "SUPERSEDED_TEXT",
    "SlackApiError",
    "SlackAuthError",
    "SlackBridgeConfig",
    "SlackOps",
    "SlackWebClient",
    "SocketModeConnection",
    "VERSION_TAG_SUFFIX",
    "ack_text",
    "markdown_to_mrkdwn",
    "parse_event",
    "quote_prefix",
    "require_boot",
    "serve",
    "skipped_files_note",
]
