"""Microsoft Teams adapter for the OSPREY dispatch bridge engine.

Contributes only Teams' wire format and platform I/O: Service Bus ingestion of
the activities an Azure Function relay enqueued, Bot Connector replies, and
inline image delivery. All ordering, dedup, retry, and crash recovery stay in
:mod:`osprey.bridges.core`, which this package imports from its root only.

**Importing this package root requires neither** ``azure-servicebus`` **nor**
``Pillow``. Every import of either is function-local, inside the one factory that
needs it, so the re-exports below are safe to import in an environment without
the ``teams`` extra installed — a dev checkout, test collection, an
``osprey build`` run rendering the compose template. The extra is needed at the
moment the bridge actually opens a queue receiver or downsizes a plot, which is
to say in the container and nowhere else. Keep it that way: a module-level
``azure.servicebus`` import here would make the whole package unimportable off a
deployment host.

The posted wording is re-exported alongside the types — the ack, the quote prefix,
the skipped-images note and the five terminal texts — because the end-to-end tier
asserts what a conversation received *by equality against these names*. A test that
re-spelled the wording would prove only that someone typed it twice; composing the
expectation from the same constants the posting path used is what pins the thing that
matters.

``__main__`` is deliberately absent from the re-exports: importing it from here
would run the process entrypoint's module body a second time under
``python -m osprey.bridges.teams``.
"""

from .client import ConnectorClient, TokenSource
from .config import TeamsBridgeConfig, require_boot
from .events import (
    MS_ACTIVITY_ID,
    MS_CONVERSATION_ID,
    MS_CONVERSATION_TYPE,
    MS_SERVICE_URL,
    MS_TENANT_ID,
    parse_event,
    resolve_reply_context,
)
from .formatting import markdown_to_teams
from .ops import (
    ANSWER_CHUNK_CHARS,
    EMPTY_ANSWER_TEXT,
    ERROR_TEXT,
    GIVEUP_TEXT,
    QUEUED_TEXT,
    SUPERSEDED_TEXT,
    TeamsOps,
    ack_text,
    quote_prefix,
    skipped_images_note,
)
from .receiver import (
    QueueReceiver,
    ReceiverFactory,
    ServiceBusQueueReceiver,
    make_receiver,
    serve,
)

__all__ = [
    "ANSWER_CHUNK_CHARS",
    "ConnectorClient",
    "EMPTY_ANSWER_TEXT",
    "ERROR_TEXT",
    "GIVEUP_TEXT",
    "MS_ACTIVITY_ID",
    "MS_CONVERSATION_ID",
    "MS_CONVERSATION_TYPE",
    "MS_SERVICE_URL",
    "MS_TENANT_ID",
    "QUEUED_TEXT",
    "QueueReceiver",
    "ReceiverFactory",
    "SUPERSEDED_TEXT",
    "ServiceBusQueueReceiver",
    "TeamsBridgeConfig",
    "TeamsOps",
    "TokenSource",
    "ack_text",
    "make_receiver",
    "markdown_to_teams",
    "parse_event",
    "quote_prefix",
    "require_boot",
    "resolve_reply_context",
    "serve",
    "skipped_images_note",
]
