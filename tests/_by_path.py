"""Importing a standalone script or template module by its file path.

``scripts/`` and the shipped template trees are not packages, so their suites
load the file under test directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module(name: str, path: Path | str, *, register: bool = False) -> ModuleType:
    """Execute the file at ``path`` as a module called ``name`` and return it.

    Args:
        name: The module's ``__name__``.
        path: The source file.
        register: Put the module in :data:`sys.modules` before executing it, as
            a module defining dataclasses needs (``dataclasses`` resolves
            annotations through ``cls.__module__``). A call at a test module's
            import time must leave this off: an import-time write to
            ``sys.modules`` is process state no fixture can undo, and
            ``tests/infrastructure/test_import_time_audit.py`` reads such
            writes only where they are spelled out.

    Returns:
        The executed module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
