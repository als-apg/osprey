"""Compatibility shim: this module now lives in osprey_connectors."""

import sys

from osprey_connectors.archiver import epics_archiver_connector as _mod

sys.modules[__name__] = _mod
