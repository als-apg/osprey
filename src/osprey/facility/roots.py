"""The two facility roots of a deployment and the predicates that walk them.

A deployment's facility facts live in one tree, ``<data>/facility``, unless its
profile names a **machine root** (the ``machine:`` key). Then the machine facts
(identity, classes, records, imported layers, models, decks, seeds, limits,
scenarios, measurement, fixes) are read from the machine root, and the
deployment's own content is read from the **content root**, ``<data>/facility``.
Without the key the one tree plays both roles: :meth:`FacilityRoots.single`.

Everything here is path arithmetic on relative paths; nothing touches the disk.
Every copy, hash and fold of the machine root filters its entries through
:func:`is_machine_entry`, so the digest, the profile-input fold, the render copy
and image staging agree on what a machine fact is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath

__all__ = [
    "DEPLOYMENT_CONTENT_DIRS",
    "FacilityRoots",
    "is_machine_entry",
    "is_vcs_metadata",
]

DEPLOYMENT_CONTENT_DIRS: tuple[str, ...] = ("knowledge", "ariel", "benchmarks")
"""The top-level directories of a facility tree that belong to the deployment.

Exactly these three, because they are what a deployment writes about its own
use of the machine rather than what the machine is (ruling 2026-11-07):
``knowledge/`` is the operators' notes the agent reads, ``ariel/`` the logbook
vocabulary of the deployment's ARIEL service, and ``benchmarks/`` the query
datasets the deployment scores its agent against. Two deployments mounting the
same machine differ in all three and agree on everything else, so everything
else is a machine fact. While mounted they are read from the content root only,
and the same names inside the machine root are skipped.
"""

_VCS_METADATA = ".git"


@dataclass(frozen=True)
class FacilityRoots:
    """Where a deployment reads machine facts and where it reads its own content.

    ``machine`` holds the machine facts; ``content`` holds the deployment's
    :data:`DEPLOYMENT_CONTENT_DIRS`. They are the same directory for a
    single-tree deployment.
    """

    machine: Path
    content: Path

    @property
    def mounted(self) -> bool:
        """True when the machine root is a separate tree from the content root."""
        return self.machine != self.content

    @classmethod
    def single(cls, path: Path) -> FacilityRoots:
        """One facility tree playing both roles, as every deployment without ``machine:``."""
        return cls(machine=path, content=path)


def is_vcs_metadata(rel: PurePath) -> bool:
    """True when ``rel`` is, or lies inside, a ``.git`` entry at any depth.

    ``.git`` may be a directory (a clone) or a file (a submodule or worktree
    pointer); both are version control, never facts. ``.gitignore``,
    ``.gitattributes`` and ``.github`` are ordinary files and are not matched.
    """
    return _VCS_METADATA in rel.parts


def is_machine_entry(roots: FacilityRoots, rel: PurePath) -> bool:
    """True when ``rel`` (relative to ``roots.machine``) is a machine fact.

    VCS metadata never is. When mounted, a top-level content directory is not
    either, because the deployment's own copy in the content root is the one
    read; a content name deeper down (``imported/x/knowledge/a.yaml``) stays a
    machine entry. A single tree keeps its content directories, as today.
    """
    if is_vcs_metadata(rel):
        return False
    return not (roots.mounted and rel.parts and rel.parts[0] in DEPLOYMENT_CONTENT_DIRS)
