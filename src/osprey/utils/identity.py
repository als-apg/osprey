"""Compatibility shim: this module now lives in osprey_connectors.

Both spellings resolve to one module object, and the star import is what a type
checker reads in place of that substitution.

The star import carries public names only, so the one private name a caller
takes through this path, the ladder's path-component rule, is named for the
type checker by hand.
"""

import sys

from osprey_connectors import identity as _mod
from osprey_connectors.identity import *  # noqa: F403
from osprey_connectors.identity import _usable as _usable

sys.modules[__name__] = _mod
