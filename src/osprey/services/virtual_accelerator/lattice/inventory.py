"""Derives the SR pyat-coupled device inventory from the namespace-union manifest.

The manifest (osprey.services.virtual_accelerator.manifest) is the single
source of truth for which SR magnet/BPM devices exist and how many of each --
this module never hardcodes device counts. If the paradigm channel DBs
change, the manifest's partition changes, and this inventory (and the
lattice built from it) changes with it: the DB fixes the lattice, not vice
versa.
"""

from __future__ import annotations

from osprey.services.virtual_accelerator.manifest import PARTITION_PYAT_COUPLED, build_manifest

# The MAG families plus the DIAG BPM family that make up partition (a).
PYAT_COUPLED_FAMILIES = frozenset({"DIPOLE", "QF", "QD", "HCM", "VCM", "SF", "SD", "BPM"})


def pyat_coupled_device_ids() -> dict[str, list[str]]:
    """Return {family: sorted device-id list} for every family in partition (a).

    Device ids are the manifest's zero-padded device strings (e.g. "01".."24"),
    collected from whichever pyat-coupled channel exists for that device
    (CURRENT:SP/RB for magnets, POSITION:X/Y for BPMs).

    Returns:
        Dict mapping family name (e.g. "DIPOLE", "BPM") to a sorted list of
        device-id strings.

    Raises:
        ValueError: if the manifest's pyat-coupled partition contains a family
            outside PYAT_COUPLED_FAMILIES (a sign the manifest's classification
            changed underneath this module).
    """
    manifest = build_manifest()
    devices: dict[str, set[str]] = {}
    for ch in manifest["channels"]:
        if ch["partition"] != PARTITION_PYAT_COUPLED:
            continue
        family = ch["family"]
        if family not in PYAT_COUPLED_FAMILIES:
            raise ValueError(
                f"unexpected pyat-coupled family '{family}' not in {sorted(PYAT_COUPLED_FAMILIES)}; "
                "the manifest's partition (a) classification changed -- update this module to match"
            )
        devices.setdefault(family, set()).add(ch["device"])

    return {family: sorted(ids) for family, ids in devices.items()}
