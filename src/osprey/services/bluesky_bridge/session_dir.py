"""Resolves the bridge-owned, writable session-plan directory.

An authored (session-tier) plan file is the one thing the bridge itself
writes, rather than merely discovers — every directory layer
``plan_loader.py`` otherwise scans (``shipped``/``preset``/``facility``) is
operator-supplied and read-only from the bridge's point of view. This module
is the single place that resolves *where* an authored plan file lives, so
both the write path (task 2.3's ``POST /plans/session``) and the discovery
path (task 2.4's session-tier directory layer) agree on the same directory
without either hardcoding it.

Ephemeral by design, like the run registry (``runs.py``) and the validation
record store (``validation_record.py``): the default directory has no deploy
volume behind it, so a container rebuild/restart loses every session plan
along with the (equally in-memory) validation records that referenced them —
consistent with the bridge's existing "stateless except explicit, documented
exceptions" posture. A deploy that wants session plans to survive a restart
sets ``BLUESKY_SESSION_PLAN_DIR`` to a mounted, writable path.
"""

from __future__ import annotations

import os
from pathlib import Path

# Set per bridge instance to override the default below with a persistent
# (e.g. volume-mounted) writable directory.
_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"

# In-image default: a sibling of the shipped plan directory
# (``plan_loader.py``'s ``_SHIPPED_PLANS_DIR``), created on first use. Writable
# in every deploy today (it lives inside the installed package tree, which is
# not mounted read-only), but not persisted across a container rebuild —
# see the module docstring.
_DEFAULT_SESSION_PLAN_DIR = Path(__file__).parent / "plans_session"


def resolve_session_plan_dir() -> Path:
    """The bridge's writable session-plan directory, creating it if needed.

    Resolution order:

    1. ``BLUESKY_SESSION_PLAN_DIR`` env var — set per bridge instance to
       point at a persistent, writable directory.
    2. The in-image default (see :data:`_DEFAULT_SESSION_PLAN_DIR`).

    Always ``mkdir(parents=True, exist_ok=True)``s the resolved directory
    before returning it, so every caller (this task's write/validate routes,
    task 2.4's session-tier directory layer) can rely on the directory
    existing without each re-implementing the same check.
    """
    raw = os.environ.get(_SESSION_PLAN_DIR_ENV)
    directory = Path(raw) if raw else _DEFAULT_SESSION_PLAN_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory
