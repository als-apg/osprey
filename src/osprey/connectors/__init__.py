"""
Connector abstraction for control systems and archivers.

This package provides pluggable connectors for different control systems
(EPICS, LabVIEW, Tango, Mock, etc.) and archiver systems. Connectors implement
standard interfaces that allow capabilities to work independently of the
underlying control system.

"""

from osprey.connectors.factory import ConnectorFactory
from osprey.connectors.types import EPICS, EPICS_ARCHIVER, MOCK, MOCK_ARCHIVER

__all__ = ["ConnectorFactory", "EPICS", "EPICS_ARCHIVER", "MOCK", "MOCK_ARCHIVER"]
