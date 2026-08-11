"""Compatibility shim: this module now lives in osprey_connectors."""

import sys

from osprey_connectors import pv_taxonomy as _mod

sys.modules[__name__] = _mod
