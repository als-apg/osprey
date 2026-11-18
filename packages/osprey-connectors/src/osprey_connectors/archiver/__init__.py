"""The archive service contract.

``ChannelArchive``, the fifth service kind, with the coverage and store extent
it answers and the processing modes. The concrete archives are the
``_*_archiver_connector`` modules beside this one, which the factory reaches
from their type names. How the MongoDB store names a channel and how a process
reaches it is :mod:`osprey_connectors.archive_store`.
"""

from osprey_connectors.archiver._base import (
    ArchiveCoverage,
    ArchiveExtent,
    ChannelArchive,
)
from osprey_connectors.archiver._timerange import PROCESSING_MODES, ProcessingMode

__all__ = [
    "PROCESSING_MODES",
    "ArchiveCoverage",
    "ArchiveExtent",
    "ChannelArchive",
    "ProcessingMode",
]
