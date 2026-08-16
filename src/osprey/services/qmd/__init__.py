"""Client for the qmd search sidecar.

The sidecar indexes a deployment's markdown corpora and answers semantic
queries over HTTP. This package holds the Python side of that conversation and
nothing else — it never starts, stops, or supervises the daemon.
"""

from osprey.services.qmd.client import (
    QMDClient,
    QMDClientError,
    QMDResponse,
    QMDSearchResult,
    QMDTransport,
    QMDUnavailableError,
)

__all__ = [
    "QMDClient",
    "QMDClientError",
    "QMDResponse",
    "QMDSearchResult",
    "QMDTransport",
    "QMDUnavailableError",
]
