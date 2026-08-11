"""Compatibility shim: this module now lives in osprey_connectors."""

import sys

from osprey_connectors import honesty as _mod

sys.modules[__name__] = _mod
