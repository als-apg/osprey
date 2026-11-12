"""The archiver connector contract.

``ArchiverConnector``, its metadata model and store extent, the processing modes, the
field name a channel is stored under, and the client arguments for the MongoDB
archive store, which the archive recorder and the deploy-time seed open
without the connector. The concrete archivers are the
``_*_archiver_connector`` modules beside this one.
"""

from osprey_connectors.archiver._base import (
    ArchiveExtent,
    ArchiverConnector,
    ArchiverMetadata,
)
from osprey_connectors.archiver._field_names import field_name
from osprey_connectors.archiver._mongo_client import (
    CA_FILE_ERRORS,
    address_overrides,
    mongo_client_kwargs,
)
from osprey_connectors.archiver._timerange import PROCESSING_MODES

__all__ = [
    "ArchiveExtent",
    "ArchiverConnector",
    "ArchiverMetadata",
    "CA_FILE_ERRORS",
    "PROCESSING_MODES",
    "address_overrides",
    "field_name",
    "mongo_client_kwargs",
]
