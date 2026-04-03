"""Connector type constants.

Single source of truth for built-in connector type name strings.
Custom connectors use dotted module paths (e.g., 'mypackage.TangoConnector')
and don't need constants here.
"""

# -- Control system connector types (have implementations) --
MOCK = "mock"
EPICS = "epics"

# -- Archiver connector types --
MOCK_ARCHIVER = "mock_archiver"
EPICS_ARCHIVER = "epics_archiver"

# -- CLI choice lists (only types with implementations) --
CLI_CONTROL_SYSTEM_TYPES = [MOCK, EPICS]
CLI_ARCHIVER_TYPES = [MOCK_ARCHIVER, EPICS_ARCHIVER]
