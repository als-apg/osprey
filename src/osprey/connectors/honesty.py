"""Compatibility shim: this module now lives in osprey_connectors.

Both spellings resolve to one module object, and the star import is what a type
checker reads in place of that substitution.
"""

import sys

from osprey_connectors import honesty as _mod
from osprey_connectors.honesty import *  # noqa: F403

sys.modules[__name__] = _mod
