"""The archive service contract.

``ChannelArchive``, the fifth service kind, with the coverage and store extent
it answers, the processing modes, the field name a channel is stored under, and
the client arguments for the MongoDB archive store, which the archive recorder
and the deploy-time seed open without the archive. The concrete archives are
the ``_*_archiver_connector`` modules beside this one, which the factory reaches
from their type names.
"""

from osprey_connectors.archiver._base import (
    ArchiveCoverage,
    ArchiveExtent,
    ChannelArchive,
)
from osprey_connectors.archiver._field_names import field_name
from osprey_connectors.archiver._mongo_client import (
    CA_FILE_ERRORS,
    address_overrides,
    mongo_client_kwargs,
)
from osprey_connectors.archiver._timerange import PROCESSING_MODES, ProcessingMode

__all__ = [
    "CA_FILE_ERRORS",
    "PROCESSING_MODES",
    "ArchiveCoverage",
    "ArchiveExtent",
    "ChannelArchive",
    "ProcessingMode",
    "address_overrides",
    "field_name",
    "mongo_client_kwargs",
]
