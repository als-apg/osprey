"""The frozen 2.0 export contract: the keys a reader may find, and the words it branches on.

The exporter test holds ``mml_export.m`` to this contract and the fixture tests
hold a committed export to it, so both read it from here. A key or a word is
added by changing this module, the file header of ``mml_export.m`` and the
reader together, never by one of the three alone.
"""

from __future__ import annotations

#: The version token an export carries, and the one the importer reads a full
#: virtual-accelerator input set from.
EXPORTER_VERSION = "mml_export 2.0.0"

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
