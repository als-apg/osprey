"""Compatibility shim: this module now lives in osprey_connectors."""

import sys

from osprey_connectors import relative_time as _mod

sys.modules[__name__] = _mod
