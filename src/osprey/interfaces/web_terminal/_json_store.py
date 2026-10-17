"""Atomic JSON documents on disk, shared by the web terminal's small stores.

A store here is a directory of independent JSON documents that a browser writes
and something else — a CLI reading a container volume, the next page load, an
operator with ``cat`` — reads back. Two properties make that safe, and both
live in this module so every store gets them the same way:

**A write is all-or-nothing.** :func:`write_json_atomic` replaces the document
whole through ``osprey_connectors.safety_file.atomic_replace``, so a concurrent
reader sees the complete new document or the complete previous one — never a
truncation, and never an empty file where a document used to be. Temporary
names are dot-prefixed and ``.tmp``-suffixed so the globs a store reads itself
with cannot match a document that is still being written.
Atomicity is per file only: a store that spreads one logical record across
several documents still has to order its writes so an interruption leaves a
recoverable state.

**A read never raises.** :func:`read_json_object` answers ``None`` for a
document that is absent, unreadable, unparseable, or not a JSON object, so a
caller falls back to its default instead of failing. A corrupt preferences blob
must degrade to the deployment default; a damaged record must not hide the
undamaged ones beside it.

The module is deliberately free of application imports (the standard library
and the standard-library-only ``osprey_connectors.safety_file``):
:mod:`~osprey.interfaces.web_terminal.feedback_store` holds that contract
because the CLI reads feedback stores out of container volumes with no
web-terminal app in scope, and a helper it imports has to hold it too.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from osprey_connectors.safety_file import atomic_replace

logger = logging.getLogger(__name__)

__all__ = ["read_json_object", "write_json_atomic"]


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Serialize *data* to *path* as JSON, replaced whole at mode 0600.

    The document goes through ``atomic_replace``, which flushes it when the
    filesystem allows. On any failure the error is re-raised and a broken write
    leaves neither debris nor a damaged target — whatever was at *path* before
    is still there, whole.

    Values JSON cannot encode are stringified rather than raising, matching
    what a diagnostic document (a timestamp, a :class:`~pathlib.Path`) needs.
    Keys still have to be JSON keys: a non-string key raises :class:`TypeError`
    like any other serialization failure.

    ``path.parent`` must already exist. Creating it belongs to the calling
    store, which knows whether an absent directory is a first write or a
    missing mount.
    """
    text = json.dumps(data, indent=2, default=str)
    atomic_replace(path, text.encode("utf-8"), mode=0o600, sync="best_effort")


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Return the JSON object at *path*, or ``None`` if there isn't one.

    ``None`` covers every degraded outcome — the file is absent, cannot be
    read, is not valid JSON, or parses to something other than an object — so
    the caller has one branch to write and never a ``try``. An absent document
    is silent (an unused store is a normal state); everything else is logged at
    debug level, because a damaged document is worth finding but is never worth
    failing the request over.
    """
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.debug("json store: unreadable document %s", path, exc_info=True)
        return None
    if not isinstance(document, dict):
        logger.debug("json store: document %s is not a JSON object", path)
        return None
    return document
