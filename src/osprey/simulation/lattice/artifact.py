"""Canonical ``.mat`` build-artifact I/O for the shared ALS-U AR ring.

The one ring is serialized with ``at.save_mat(ring, path, use='RING')`` and
committed as tracked binary data under the packaged ``osprey.templates`` data
root, so a clean-checkout hatchling wheel ships it (hatchling packages only
VCS-tracked files). The committed file is located at runtime via the
``osprey.templates`` package path — never ``__file__`` parent-climbing.
"""

from __future__ import annotations

from pathlib import Path

import at
from at import Lattice

import osprey.templates

# Location of the committed artifact, relative to the ``osprey.templates`` root.
_MAT_RELPATH = "apps/control_assistant/data/lattice/als_u_ar.mat"


def canonical_mat_path() -> Path:
    """Return the runtime path of the committed canonical ``.mat``.

    Resolved from the packaged :mod:`osprey.templates` root so it works in an
    editable checkout, an installed wheel, and a wheel-drop container image
    alike.
    """
    return Path(osprey.templates.__file__).parent / _MAT_RELPATH


def save_canonical_mat(ring: Lattice, path: str | Path | None = None) -> Path:
    """Serialize ``ring`` to a canonical ``.mat`` (``use='RING'``).

    Args:
        ring: The lattice to serialize.
        path: Destination path; defaults to :func:`canonical_mat_path`. Parent
            directories are created if missing.

    Returns:
        The path written.
    """
    dest = Path(path) if path is not None else canonical_mat_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    at.save_mat(ring, str(dest), use="RING")
    return dest


def load_canonical_ring(path: str | Path | None = None) -> Lattice:
    """Load a canonical ``.mat`` back into an :class:`at.Lattice`.

    Args:
        path: Source path; defaults to the committed
            :func:`canonical_mat_path`.

    Returns:
        The round-tripped ring.
    """
    src = Path(path) if path is not None else canonical_mat_path()
    return at.load_mat(str(src), use="RING")
