"""The served lattice: a saved pyAT ring, checked against its bindings.

``build_ring`` is the virtual accelerator's one lattice acquisition path. It
takes a served tree -- a :class:`~...manifest.paths.ManifestPaths` over a
facility's ``data/`` directory -- and returns the ring the model drives. The
lattice is data the facility exported, not code this package carries: one
facility's ring is another's, and nothing here names a family, an element or
an energy.

Two files in that tree describe one accelerator, and three checks make sure
they still describe the same one:

1. **The lattice is the one the bindings were derived against.** Every element
   name, polynomial index and nominal in ``va_bindings.json`` was read off one
   particular ring at export time, so the document stamps that ring's
   ``lattice_sha256`` and this refuses any other file. The digest also settles
   the energy: ``energy_gev`` was read off the same bytes, so a ring whose
   energy disagrees cannot get past the digest, and there is no second check
   for it.
2. **Every bound element name belongs to exactly one element.** lume-pyat
   addresses elements by ``FamName``, and a real lattice repeats names freely
   -- a deck that calls every beam monitor in the ring ``BPM`` is ordinary --
   so a name shared by two elements does not say which one a channel writes. A missing name is the
   same refusal from the other side: the document describes an element this
   ring does not have.
3. **Longitudinal motion is on for the cavity alone.** ``enable_6d`` is given
   :class:`at.RFCavity` and nothing else, so the ring solves a 6D closed orbit
   through the RF bucket while the dipoles and quadrupoles stay
   non-radiative -- the way MML reads its own simulator model, and the
   condition lume-pyat 0.2.0's orbit guard is written against. A ring with no
   cavity stays 4D, and lume-pyat dispatches on ``ring.is_6d`` accordingly.

A refusal is a :class:`~...bindings.BindingsError`: what breaks is always the
agreement between the document and the tree it is served from, so the error
points at the key in the document that no longer holds and names the lattice
file it was checked against.

No caching: each consumer builds its own ring, so a consumer never inherits
another's writes.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import at

from osprey.services.virtual_accelerator.bindings import (
    BindingsDocument,
    BindingsError,
    load_bindings,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths

#: Read size for the lattice digest; the file is JSON of a whole ring.
_CHUNK = 1 << 20


def build_ring(paths: ManifestPaths) -> at.Lattice:
    """Build the ring a served tree describes.

    Args:
        paths: The tree to serve, holding the saved lattice and the bindings
            derived against it.

    Returns:
        The ring, with longitudinal motion enabled for its RF cavities and
        radiation left off.

    Raises:
        FileNotFoundError: The tree carries no lattice or no bindings file;
            the message names the one that is missing.
        BindingsError: The bindings document is refused by its own schema, the
            lattice is not the file the bindings were derived against, or a
            bound element name is not carried by exactly one element.
        OSError: A file could not be read.
    """
    document = load_bindings(paths.va_bindings)
    ring = _load_lattice(paths, document)
    _check_bound_elements(ring, document, paths)
    ring.enable_6d(at.RFCavity)
    return ring


def _load_lattice(paths: ManifestPaths, document: BindingsDocument) -> at.Lattice:
    """Load the lattice file, refusing one the bindings do not describe."""
    path = paths.lattice_json
    if not path.is_file():
        raise FileNotFoundError(f"VA lattice file not found: {path}")
    digest = _sha256(path)
    if digest != document.lattice_sha256:
        raise BindingsError(
            "lattice_sha256",
            f"the bindings were derived against the lattice {document.lattice_sha256}, "
            f"but {path} hashes to {digest}: serve the lattice these bindings describe, "
            "or re-emit the bindings for this one",
            source=str(paths.va_bindings),
        )
    return at.load_lattice(path)


def _check_bound_elements(
    ring: at.Lattice, document: BindingsDocument, paths: ManifestPaths
) -> None:
    """Refuse a bound element name the ring does not carry exactly once.

    Raises:
        BindingsError: naming the first slice, in document order, whose
            element the lattice has none or several of.
    """
    census = Counter(element.FamName for element in ring)
    for position, binding in enumerate(document.bindings):
        for slot, slice_ in enumerate(binding.slices):
            found = census[slice_.element]
            if found == 1:
                continue
            carries = "does not carry" if found == 0 else f"has {found} of"
            raise BindingsError(
                f"bindings[{position}].slices[{slot}].element",
                f"family {binding.family!r} binds element {slice_.element!r}, which "
                f"{paths.lattice_json} {carries}: an element is addressed by its FamName, "
                "so a bound name must belong to exactly one element",
                source=str(paths.va_bindings),
            )


def _sha256(path: Path) -> str:
    """Return the lowercase hex sha256 of a file's bytes.

    The same digest the emit lane stamps into the document (see
    ``services/mml/canonical.py::sha256_of``); it is spelt again here rather
    than imported so the served virtual accelerator does not depend on the
    exporter that fed it.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
