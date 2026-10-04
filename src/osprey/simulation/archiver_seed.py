"""The osprey spelling of osprey_connectors.simulation.archive.

Both spellings resolve to one module object, and the star import is what a type
checker reads in place of that substitution.
"""

import sys

from osprey_connectors.simulation import archive as _mod
from osprey_connectors.simulation.archive import *  # noqa: F403

sys.modules[__name__] = _mod
