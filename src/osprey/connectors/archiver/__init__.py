"""Archiver connector implementations."""

from osprey.connectors.archiver._timerange import PROCESSING_MODES
from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata

# PROCESSING_MODES is the one piece of _timerange that belongs to the package's
# public surface: callers outside the connectors (the archiver_read tool, and
# anything validating a user-supplied mode) need the valid set without reaching
# into a private module. Everything else in _timerange stays connector-internal.
__all__ = ["PROCESSING_MODES", "ArchiverConnector", "ArchiverMetadata"]
