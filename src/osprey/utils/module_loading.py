"""Import a facility-supplied ``.py`` file by its path, not by a dotted name.

Bluesky plan files, health plugin files and sequence modules all arrive as files
on disk that are not on ``sys.path``. Each is imported under a synthetic module
name derived from its resolved absolute path, so two files that share a stem
(a shipped and a facility-tier ``orm.py``) never collide in ``sys.modules``, and
re-loading the same file always lands on the same name.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_DIGEST_HEX_CHARS = 16


def module_name_for(path: Path, prefix: str) -> str:
    """The ``sys.modules`` name `load_by_path` gives the file at *path*.

    The digest only keeps the name short and filesystem-agnostic; nothing here is
    a security boundary, hence ``usedforsecurity=False`` (a FIPS-strict host
    otherwise refuses sha1 outright).
    """
    resolved = str(Path(path).resolve())
    digest = hashlib.sha1(resolved.encode(), usedforsecurity=False).hexdigest()
    return f"{prefix}{digest[:_DIGEST_HEX_CHARS]}"


def load_by_path(path: Path, prefix: str) -> ModuleType:
    """Execute the ``.py`` file at *path* and return it as a module.

    The module is registered in ``sys.modules`` before it executes, because
    dataclasses, pickling and postponed annotations resolve a class's own module
    through ``sys.modules`` while the file is still running. If execution raises
    anything at all, the entry is removed again before the exception propagates,
    so a half-initialised module never shadows a later, fixed load of the same
    file.

    Raises:
        ImportError: no import machinery handles *path*.
        BaseException: whatever the file raises while it executes, unchanged.
    """
    resolved = Path(path).resolve()
    name = module_name_for(resolved, prefix)
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"no Python import machinery handles {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
