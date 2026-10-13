"""Compatibility shim: this module now lives in osprey_connectors.

Both spellings resolve to one module object, and the star import is what a type
checker reads in place of that substitution.
"""

import sys

from osprey_connectors.control_system import _limits_validator as _mod
from osprey_connectors.control_system._limits_validator import *  # noqa: F403

sys.modules[__name__] = _mod
