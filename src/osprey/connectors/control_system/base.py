"""Compatibility shim: this module now lives in osprey_connectors."""

import sys

from osprey_connectors.control_system import base as _mod

sys.modules[__name__] = _mod
