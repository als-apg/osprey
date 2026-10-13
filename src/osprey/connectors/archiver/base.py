"""Compatibility shim: this module now lives in osprey_connectors.

Both spellings resolve to one module object, and the star import is what a type
checker reads in place of that substitution.
"""

import sys

from osprey_connectors.archiver import _base as _mod
from osprey_connectors.archiver._base import *  # noqa: F403

sys.modules[__name__] = _mod
