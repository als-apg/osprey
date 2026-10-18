"""A run owns its temporary root; nothing else on the host numbers or prunes it.

Left to itself, pytest numbers every run's base temporary directory under the
shared ``{temproot}/pytest-of-<user>/`` and, when a run exits, deletes all but
the newest three entries there, along with any lock-less entry named
``pytest-*`` that is not a number. Two runs on one host would delete each
other's live trees. So the run gives itself a fresh directory as
``--basetemp`` before pytest's own temporary-path setup reads the option,
outside ``pytest-of-<user>`` and not named ``pytest-*``, in
``PYTEST_DEBUG_TEMPROOT`` or the system temp dir. An xdist run's workers each
get ``<root>/popen-gwN`` from their controller, so the whole run lives under
one root.

pytest never removes a given base temporary directory, so the run removes the
root it claimed itself, as ``tmp_path_retention_policy`` says: ``none`` always,
``failed`` after a green run, ``all`` never. A kept root is named once in the
terminal summary. A ``--basetemp`` the caller gives is used as is and never
removed.

The leading underscore keeps the module out of pytest collection.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

#: Prefix of every root a run claims.
ROOT_PREFIX = "osprey-tests-"

#: Line naming a root the run kept, followed by its path.
KEPT_LINE = "temporary files kept at"

_OWNED_ROOT = pytest.StashKey[Path]()
_KEPT_ROOT = pytest.StashKey[Path]()


def claim(config: pytest.Config) -> None:
    """Give the run a fresh temporary root unless it was given one.

    An xdist worker is given ``<controller root>/popen-gwN`` by its controller
    and claims nothing.
    """
    if config.option.basetemp or hasattr(config, "workerinput"):
        return
    temproot = os.environ.get("PYTEST_DEBUG_TEMPROOT") or tempfile.gettempdir()
    root = Path(tempfile.mkdtemp(prefix=ROOT_PREFIX, dir=temproot)).resolve()
    config.option.basetemp = str(root)
    config.stash[_OWNED_ROOT] = root


def release(config: pytest.Config, exitstatus: int) -> None:
    """Remove the root this run claimed, as the retention policy says."""
    root = config.stash.get(_OWNED_ROOT, None)
    if root is None:
        return
    policy = config.getini("tmp_path_retention_policy")
    if policy == "none" or (policy == "failed" and exitstatus == 0):
        shutil.rmtree(root, ignore_errors=True)
    if root.exists():
        config.stash[_KEPT_ROOT] = root


def kept_root(config: pytest.Config) -> Path | None:
    """The root this run claimed and kept, if any."""
    return config.stash.get(_KEPT_ROOT, None)
