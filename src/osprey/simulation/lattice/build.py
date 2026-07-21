"""Regenerate the committed canonical ALS-U AR ``.mat`` artifact.

Run::

    python -m osprey.simulation.lattice.build

This rebuilds the ring from source (:func:`osprey.simulation.lattice.build_ring`)
and writes it to the committed artifact path
(``templates/apps/control_assistant/data/lattice/als_u_ar.mat``). Commit the
resulting ``.mat`` as tracked binary data — the hatchling wheel ships only
VCS-tracked files, so an uncommitted artifact would be absent from a
clean-checkout install. The round-trip / regen-equality test guards staleness.
"""

from __future__ import annotations

from osprey.simulation.lattice.artifact import save_canonical_mat
from osprey.simulation.lattice.ring import build_ring


def main() -> None:
    """Build the ring and write the canonical ``.mat`` to the committed path."""
    ring = build_ring()
    dest = save_canonical_mat(ring)
    print(f"Wrote {len(ring)}-element ALS-U AR ring to {dest}")


if __name__ == "__main__":
    main()
