"""Client for the qmd search sidecar.

Each sidecar indexes one of a deployment's markdown corpora and answers
semantic queries over HTTP. This package holds the Python side of that conversation and
nothing else — it never starts, stops, or supervises the daemon.
"""

from osprey.services.qmd.client import (
    QMDClient,
    QMDClientError,
    QMDIndexStatus,
    QMDResponse,
    QMDSearchResult,
    QMDTransport,
    QMDUnavailableError,
)

__all__ = [
    "QMDClient",
    "QMDClientError",
    "QMDIndexStatus",
    "QMDResponse",
    "QMDSearchResult",
    "QMDTransport",
    "QMDUnavailableError",
]
