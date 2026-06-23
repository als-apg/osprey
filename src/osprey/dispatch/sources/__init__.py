"""Built-in trigger sources for the OSPREY event dispatcher.

Public surface: the :class:`TriggerSource` protocol, the :data:`FireCallback`
type alias, and the built-in :class:`WebhookSource` and :class:`CronSource`
implementations.
"""

from osprey.dispatch.sources.base import FireCallback, TriggerSource
from osprey.dispatch.sources.cron import CronSource
from osprey.dispatch.sources.webhook import WebhookSource

__all__ = [
    "CronSource",
    "FireCallback",
    "TriggerSource",
    "WebhookSource",
]
