"""``scripts/check.py``, loaded once for every test module that reads it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: The name the module is registered under; no import uses it.
NAME = "_check_by_path"

_spec = importlib.util.spec_from_file_location(
    NAME, Path(__file__).resolve().parents[2] / "scripts" / "check.py"
)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
# import-time required because scripts/ is not a package: check.py is loaded
# by path and registered in sys.modules before exec so @dataclass can resolve
# annotations through cls.__module__. Doing it here, once, gives every reader
# the same module.
sys.modules[NAME] = check
_spec.loader.exec_module(check)
