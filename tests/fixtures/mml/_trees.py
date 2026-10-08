"""The MML trees OSPREY supports, and the proof each one is held to.

One row per fixture tree under this directory that carries a whole export (a
deck, its sampled facts and a reviewed ``imported/mml/mapping.yaml``). A row
states what is claimed of the tree, and every lane that proves a claim
parametrises from the rows instead of naming trees of its own:

* ``imports`` -- ``osprey facility import mml`` writes the tree's records
  under its committed mapping (``tests/facility/_mml_built.py`` and the layer
  tests that read it).
* ``builds`` -- ``osprey build`` accepts the imported tree once the limits
  records its stops name are widened (``tests/cli/test_mml_build_recipes.py``).
* ``golden`` -- the in-memory build of the imported tree is held to a
  committed fingerprint (``tests/facility/test_fixture_fingerprints.py``).
* ``check`` -- the parity checks the tree takes. ``A``: pyAT replays the
  Middle Layer's own model recipe on the built deck and meets the tree's
  ``<stem>.model.json``, which only a tree a real Middle Layer exported has.
  ``B``: a measurement on the served model meets pyAT's exact physics on the
  same lattice. ``C`` names a capability list and is proven by no tree.
* ``boots`` -- the install recipe ends in a container serving the tree's own
  channels (``tests/va/e2e/test_mml_trees_boot.py``).

``lines`` names the exports of a tree that are single-pass transport lines;
every other export is a periodic lattice.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CHECKS", "SUPPORTED", "Tree", "by_name", "exports_taking", "names"]

#: The parity checks a tree may take.
CHECKS: tuple[str, ...] = ("A", "B", "C")


@dataclass(frozen=True)
class Tree:
    """One supported MML tree and what is claimed of it.

    Attributes:
        name: The fixture directory.
        exports: The stem every file of one export is named after, in import
            order.
        lines: The exports that are single-pass transport lines.
        imports: The tree imports under its committed mapping.
        builds: ``osprey build`` accepts the imported tree.
        golden: The built facility file is held to a committed fingerprint.
        check: The parity checks the tree takes, from :data:`CHECKS`.
        boots: The built tree is served by a container.
    """

    name: str
    exports: tuple[str, ...]
    lines: tuple[str, ...] = ()
    imports: bool = True
    builds: bool = True
    golden: bool = False
    check: tuple[str, ...] = ()
    boots: bool = True


#: Every supported tree, in the order the parity checks read them.
SUPPORTED: tuple[Tree, ...] = (
    Tree(
        name="spear3",
        exports=("spear3.storagering",),
        golden=True,
        check=("A",),
    ),
    Tree(
        name="nsls2",
        exports=("nsls2.storagering", "nsls2.ltb"),
        lines=("nsls2.ltb",),
        golden=True,
        check=("A",),
    ),
    Tree(
        name="synthetic",
        exports=("quokka.sr",),
        check=("B",),
    ),
)


def by_name() -> dict[str, Tree]:
    """Every supported tree, keyed by its name."""
    return {tree.name: tree for tree in SUPPORTED}


def names(claim: str) -> tuple[str, ...]:
    """The sorted names of the trees a boolean claim holds for.

    Args:
        claim: ``imports``, ``builds``, ``golden`` or ``boots``.
    """
    return tuple(sorted(tree.name for tree in SUPPORTED if getattr(tree, claim) is True))


def exports_taking(check: str, *, lines: bool) -> tuple[tuple[str, str], ...]:
    """The ``(tree, stem)`` of every export that takes one parity check.

    Args:
        check: One of :data:`CHECKS`.
        lines: ``True`` for the transport lines, ``False`` for the periodic
            lattices.
    """
    return tuple(
        (tree.name, stem)
        for tree in SUPPORTED
        if check in tree.check
        for stem in tree.exports
        if (stem in tree.lines) is lines
    )
