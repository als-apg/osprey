"""The frozen export contract: the keys a reader may find, and the words it branches on.

The exporter test holds ``mml_export.m`` to this contract and the fixture tests
hold a committed export to it, so both read it from here. A key or a word is
added by changing this module, the file header of ``mml_export.m`` and the
reader together, never by one of the three alone.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The version token the shipped exporter writes. A committed export carries
#: the token of the run that wrote it, which :func:`exporter_of` reads; a test
#: about a committed tree compares against that, never against this.
EXPORTER_VERSION = "mml_export 2.1.0"

#: The exporter versions whose run writes the sixth file, ``<stem>.model.json``:
#: the Middle Layer's own model answers, kept for verification only. An export
#: at any other version has no model file, and a reader holds no tree to one.
MODEL_FILE_EXPORTERS = frozenset({"mml_export 2.1.0"})

#: The sections of ``model.json``, in the order the exporter writes them. Each
#: holds what the Middle Layer's model answered, or ``{"refused": <message>}``.
MODEL_SECTION_KEYS = (
    "state",
    "tune",
    "chromaticity",
    "dispersion",
    "orbit_response",
    "tune_response",
    "chromaticity_response",
)

#: The unit systems a response section of ``model.json`` is stated in, one
#: block under each.
MODEL_RESPONSE_UNIT_KEYS = ("physics", "hardware")


def exporter_of(tree_dir: Path) -> str:
    """Return the exporter token of the committed export in ``tree_dir``.

    The token is the ``_export.exporter`` of the tree's ``*.ao.json``. A tree
    holding several sub-machines is one export run per sub-machine, and every
    one of them has to name the same token: a tree whose files disagree states
    no version at all.

    Args:
        tree_dir: A fixture directory holding one or more ``*.ao.json`` files.

    Returns:
        The exporter token the tree was written with.

    Raises:
        ValueError: The tree holds no ``*.ao.json``, or its files name
            different tokens.
    """
    exports = sorted(Path(tree_dir).glob("*.ao.json"))
    if not exports:
        raise ValueError(f"{tree_dir} holds no *.ao.json")
    tokens = {
        path.name: json.loads(path.read_text(encoding="utf-8"))["_export"]["exporter"]
        for path in exports
    }
    if len(set(tokens.values())) != 1:
        raise ValueError(f"{tree_dir} mixes exporter versions: {tokens}")
    return next(iter(tokens.values()))


#: The ``va.json`` key set, frozen: what a reader of a 2.0 export is entitled
#: to find, and all it is entitled to find.
VA_LATTICE_KEYS = ("elements", "famname_sha256", "energy_gev", "ringparam_indices")
VA_FAMILY_KEYS = (
    "device_list",
    "fields",
    "nominals",
    "Setpoint",
    "Monitor",
    "energy_candidate",
    "energy_table",
    "refused",
)
VA_SETPOINT_KEYS = ("calibration", "energy_scaling", "energy_deviation")
VA_MONITOR_KEYS = ("calibration", "monitor_inverse", "readout")
VA_NOMINAL_KEYS = ("values", "units", "at_type", "at_index", "synthetic")
VA_ENERGY_TABLE_KEYS = (
    "device_row",
    "grid",
    "values",
    "finite_span",
    "I_nom",
    "energy_at_nominal",
)
VA_CALIBRATION_KEYS = (
    "kind",
    "gain",
    "offset",
    "grid",
    "values",
    "finite_span",
    "grid_source",
    "anchor",
    "fcn",
)

#: The closed vocabularies beside those keys: the words a reader branches on,
#: and all the words it has to branch on.
VA_VOCABULARIES = {
    "kind": ("linear", "table"),
    "grid_source": ("range", "setpoint", "fallback"),
    "anchor": ("nominal", "range_midpoint", "zero"),
    "energy_scaling": ("brho", "none"),
}

#: What the Middle Layer corrects a monitor's own readings by, one value per
#: device under each key the family states. ``gain`` and ``offset`` are already
#: inside the conversion beside them; ``roll`` and ``crunch`` carry a beam
#: monitor's two planes into the model's and are in no conversion at all.
VA_READOUT_KEYS = ("gain", "offset", "roll", "crunch")

#: What a line carries where a table carries its sampled points: a calibration
#: of one kind carries one of these pairs and never the other.
VA_LINEAR_KEYS = ("gain", "offset")
VA_TABLE_KEYS = ("grid", "values", "finite_span")

#: One block of the response document, and one side of that block.
VA_RESPONSE_BLOCK_KEYS = (
    "monitor",
    "actuator",
    "origin",
    "timestamp",
    "gev",
    "units",
    "units_string",
    "modulation_method",
    "actuator_delta",
    "data",
)
VA_RESPONSE_SIDE_KEYS = ("family", "device_list", "mode", "status", "data")

#: The provenance block every file of an export carries.
EXPORT_BLOCK_KEYS = ("exporter", "machine", "submachine", "matlab", "timestamp")
