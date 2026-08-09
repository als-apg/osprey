"""Google Chat adapter for the OSPREY dispatch bridge engine.

Contributes only Google Chat's wire format and platform I/O: Pub/Sub event
ingestion, Chat API replies, and GCS-backed artifact delivery. All ordering,
dedup, retry, and crash recovery stay in :mod:`osprey.bridges.core`, which this
package imports from its root only.

**Importing this package root requires no Google library.** Every ``google.*`` and
``googleapiclient`` import in this package is function-local, inside the default
factory that needs it, so the re-exports below are safe to import in an environment
without the ``gchat`` extra installed — a dev checkout, test collection, a
``osprey build`` run rendering the compose template. The extra is needed at the
moment the bridge actually builds a Chat service, a storage client or a Pub/Sub
subscriber, which is to say in the container and nowhere else. Keep it that way:
a module-level Google import here would make the whole package unimportable off a
deployment host.

``__main__`` is deliberately absent from the re-exports: importing it from here
would run the process entrypoint's module body a second time under
``python -m osprey.bridges.google_chat``.
"""

from .client import ChatClient, build_chat_service, chunk_text
from .config import GoogleChatBridgeConfig, require_boot
from .events import parse_event, resolve_reply_context
from .formatting import markdown_to_chat
from .gcs import publish_artifacts, publish_documents
from .inbound import download_attachments
from .ops import GoogleChatOps
from .subscriber import decode_pubsub_data, serve

__all__ = [
    "ChatClient",
    "GoogleChatBridgeConfig",
    "GoogleChatOps",
    "build_chat_service",
    "chunk_text",
    "decode_pubsub_data",
    "download_attachments",
    "markdown_to_chat",
    "parse_event",
    "publish_artifacts",
    "publish_documents",
    "require_boot",
    "resolve_reply_context",
    "serve",
]
