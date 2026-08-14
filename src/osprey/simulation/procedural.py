"""Compatibility shim: this module now lives in osprey_connectors."""

import sys

from osprey_connectors.simulation import procedural as _mod

sys.modules[__name__] = _mod
