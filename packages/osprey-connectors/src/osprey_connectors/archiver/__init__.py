"""The archiver connector contract.

``ArchiverConnector`` and its metadata model, the processing modes and the
shared time-range helpers every in-tree archiver builds on, the field name a
channel is stored under, and the client arguments for the MongoDB archive
store, which the archive recorder and the deploy-time seed open without the
connector. The concrete archivers are the
``_*_archiver_connector`` modules beside this one.
"""

from osprey_connectors.archiver._base import (
    ArchiverConnector,
    ArchiverMetadata,
)
from osprey_connectors.archiver._field_names import (
    channel_address,
    field_name,
)
from osprey_connectors.archiver._mongo_client import (
    CA_FILE_ERRORS,
    address_overrides,
    mongo_client_kwargs,
)
from osprey_connectors.archiver._timerange import (
    PROCESSING_MODES,
    Processing,
    aggregate_series,
    decimate_raw,
    long_frame,
    reject_non_numeric,
    require_datetime,
    resolve_processing,
    to_utc,
)

__all__ = [
    "ArchiverConnector",
    "ArchiverMetadata",
    "CA_FILE_ERRORS",
    "PROCESSING_MODES",
    "Processing",
    "address_overrides",
    "channel_address",
    "field_name",
    "aggregate_series",
    "decimate_raw",
    "long_frame",
    "mongo_client_kwargs",
    "reject_non_numeric",
    "require_datetime",
    "resolve_processing",
    "to_utc",
]
