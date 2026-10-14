"""Compatibility shim: this module now lives in osprey_connectors.

It is ``osprey_connectors.archiver._timerange``, private to its package now, and this
path is that module under another name, so code outside the framework that
imports it here gets exactly what it got before. The framework itself imports
the public ``osprey_connectors.archiver`` namespace.

Both spellings resolve to one module object, and the star import is what a type
checker reads in place of that substitution.
"""

import sys

from osprey_connectors.archiver import _timerange as _mod
from osprey_connectors.archiver._timerange import *  # noqa: F403

sys.modules[__name__] = _mod
