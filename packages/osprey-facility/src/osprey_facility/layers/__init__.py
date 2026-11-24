"""Layers: the importers that turn an outside export into facility sources.

Each layer reads one outside format and writes record sources under
``data/facility/imported/<layer>/``; the build merges them with the authored
sources field by field. A seed-once file an import creates opens with
:data:`HEADER`, whichever format wrote it, and is a person's from then on.
"""

from __future__ import annotations

from osprey_facility import source_tree

__all__ = ["FORMER_HEADER", "HEADER", "LAYER_ROOT"]

#: The directory, under ``data/facility``, that holds one directory per imported layer.
LAYER_ROOT = source_tree.layer_root()

#: The first line of every file an import seeds, in every format.
HEADER = "# Seeded once by the facility import; edit this file by hand."

#: The header the facility import seeded files under before HEADER; a file that
#: opens with it reads as seeded.
FORMER_HEADER = "# Seeded once by the mml import; edit this file by hand."
