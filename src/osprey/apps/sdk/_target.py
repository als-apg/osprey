"""Which machine an owner's session is on, for an application backend: :func:`owner_target`.

An application backend that holds the read-only control-context tree answers a
request from the machine the request's owner is on. That machine is stated in one
place, the owner's record in the tree; nobody tells the backend.

The read fails closed the way the write verdict's tree reader does: nothing here
follows a symbolic link (the tree is group-writable, so a link planted under
another owner's name would decide which machine is read), and a record that is
present and cannot be read names no machine rather than a guess.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Any

from osprey_connectors.control_context import RECORD_FILENAME, parse_record
from osprey_connectors.posture_store import CONTROL_CONTEXT_TREE_ENV_VAR, CONTROL_TREE_MARKER_NAME
from osprey_connectors.types import baseline_target

logger = logging.getLogger(__name__)

#: A record is a few hundred bytes; anything far larger is not one.
_MAX_RECORD_BYTES = 64 * 1024


def _tree_bind() -> str:
    """The tree bind as set, empty when this backend holds no tree."""
    return (os.environ.get(CONTROL_CONTEXT_TREE_ENV_VAR) or "").strip()


def _is_segment(owner: str) -> bool:
    return owner not in {"", ".", ".."} and not any(c in owner for c in "/\\\x00")


def _read_record(tree: Path, owner: str) -> tuple[str | None, bool]:
    """``(target, read)``: the owner's record target, ``None`` with *read* when there is none."""
    marker = os.lstat(tree / CONTROL_TREE_MARKER_NAME)
    if not stat.S_ISREG(marker.st_mode):
        return None, False
    try:
        owner_dir = os.lstat(tree / owner)
    except FileNotFoundError:
        return None, True
    if not stat.S_ISDIR(owner_dir.st_mode):
        return None, False
    try:
        fd = os.open(tree / owner / RECORD_FILENAME, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None, True
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            return None, False
        raw = handle.read(_MAX_RECORD_BYTES + 1)
    record = None if len(raw) > _MAX_RECORD_BYTES else parse_record(raw.decode("utf-8"))
    return (record.target, True) if record is not None else (None, False)


def owner_target(section: Any, owner: str | None) -> str | None:
    """The control target *owner*'s session acts on, for a backend holding the tree.

    Stable: part of the application SDK's public surface. An owner with no
    record has not switched, and a backend with no tree bound holds no
    session's record, so both are the deployment's baseline, as is a call that
    names no owner.

    Args:
        section: The ``control_system`` section of the resolved config.
        owner: The name the request's gate stamped (:func:`owner_from_header`),
            or ``None``.

    Returns:
        The target's name, or ``None`` when the owner's record may exist and
        cannot be read: a machine must not be answered for by a guess, so the
        caller refuses. Never raises.
    """
    bind = _tree_bind()
    if owner is None or not bind:
        return baseline_target(section)
    tree = os.path.expanduser(bind)
    if not os.path.isabs(tree) or not _is_segment(owner):
        logger.warning("No control target for %r: the tree bind or the name is unusable", owner)
        return None
    try:
        target, read = _read_record(Path(tree), owner)
    except (OSError, UnicodeDecodeError, ValueError):
        logger.warning("No control target for %r: its record is unreadable", owner, exc_info=True)
        return None
    if not read:
        logger.warning("No control target for %r: its record is not a record", owner)
        return None
    return target if target is not None else baseline_target(section)
