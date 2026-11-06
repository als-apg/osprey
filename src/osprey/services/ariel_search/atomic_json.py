"""Write a JSON document whole: the ARIEL draft store's one writer.

Both draft writers (the web API's ``write_draft`` and the MCP ``entry_create``
draft branch) call :func:`atomic_write_json`, each with a path under its own
resolved drafts directory. The bytes go through
:func:`osprey_connectors.safety_file.atomic_replace`, so a reader sees the old
file or the new one, never half a file. The temporary file is named
``.<name>.<random>.tmp``, so it never matches the store's ``draft-*.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osprey_connectors.safety_file import atomic_replace

__all__ = ["atomic_write_json"]

#: The mode drafts had under the usual umask before writes became atomic.
_FILE_MODE = 0o644


def atomic_write_json(path: Path, obj: Any) -> None:
    """Replace ``path`` with ``obj`` as indented JSON, whole or not at all.

    The parent directory must exist. Both callers surface a failure to someone
    who can retry, so the flush is required.
    """
    data = json.dumps(obj, indent=2).encode("utf-8")
    atomic_replace(path, data, mode=_FILE_MODE, sync="required")
