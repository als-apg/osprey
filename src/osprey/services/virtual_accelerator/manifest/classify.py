"""Address partitioning and EPICS record-type derivation.

Partitions every namespace address into exactly one of three physics-
fidelity tiers the future IOC needs to treat differently:

  pyat-coupled -- backed by the AT lattice model: the SR magnet currents that
                  actually steer the beam, plus the SR BPM readbacks that
                  observe it.
  sp-echo      -- writable but physics-free: a write to the setpoint just
                  echoes onto the readback, with no lattice model behind it.
                  Covers the BR/BTS transport-line magnets (upstream of/
                  outside the storage-ring lattice) and the SR RF/vacuum
                  setpoint+readback pairs (RF and vacuum are not part of the
                  AT lattice model either).
  static-noisy -- everything else: golden references, status/fault flags,
                  and slow telemetry (temperatures, pressures, radiation
                  monitors) that just needs a plausible noisy constant.

An address is assigned to `pyat-coupled` or `sp-echo` only when it clears an
explicit rule below; everything else falls through to `static-noisy`.
"""

from __future__ import annotations

from osprey.simulation.facility_spec import ALS_U_AR

# Spec-derived: every magnet/corrector family declared by the facility spec
# (excludes the BPM monitor family, which is gated separately below).
MAG_FAMILIES = frozenset(f.name for f in ALS_U_AR.families if f.kind in ("magnet", "corrector"))

PARTITION_PYAT_COUPLED = "pyat-coupled"
PARTITION_SP_ECHO = "sp-echo"
PARTITION_STATIC_NOISY = "static-noisy"

# SR RF/VAC fields that carry a real writable-setpoint + readback pair.
# Pure telemetry fields in the same systems (POWER, TEMPERATURE, PRESSURE,
# ION-PUMP CURRENT) have no setpoint counterpart and stay static-noisy.
_SR_RF_VAC_SP_ECHO_FIELDS = frozenset({"VOLTAGE", "FREQUENCY", "TUNER"})


def classify_partition(path: dict[str, str]) -> str:
    """Classify one expanded channel's hierarchy path into a manifest partition.

    Args:
        path: Hierarchy path as produced by HierarchicalChannelDatabase,
            mapping "ring"/"system"/"family"/"device"/"field"/"subfield" to
            the selected value for this channel.

    Returns:
        One of PARTITION_PYAT_COUPLED, PARTITION_SP_ECHO, PARTITION_STATIC_NOISY.
    """
    ring, system, family, field, subfield = (
        path["ring"],
        path["system"],
        path["family"],
        path["field"],
        path["subfield"],
    )

    if (
        ring == "SR"
        and system == "MAG"
        and family in MAG_FAMILIES
        and field == "CURRENT"
        and subfield in ("SP", "RB")
    ):
        return PARTITION_PYAT_COUPLED

    if (
        ring == "SR"
        and system == "DIAG"
        and family == "BPM"
        and field == "POSITION"
        and subfield in ("X", "Y")
    ):
        return PARTITION_PYAT_COUPLED

    if ring in ("BR", "BTS") and system == "MAG":
        return PARTITION_SP_ECHO

    if (
        ring == "SR"
        and system in ("RF", "VAC")
        and field in _SR_RF_VAC_SP_ECHO_FIELDS
        and subfield in ("SP", "RB")
    ):
        return PARTITION_SP_ECHO

    return PARTITION_STATIC_NOISY


# --- EPICS record type ---------------------------------------------------

RECORD_TYPE_BINARY = "bi"
RECORD_TYPE_ANALOG = "ai"
RECORD_TYPE_STRING = "stringin"
# The two remaining gateway channel shapes: a 512-byte char waveform ("long
# string", e.g. a status/message channel wider than stringin's 40 bytes) and
# a multi-bit binary (discrete enum state). `derive_record_type` never emits
# either -- the tutorial namespace has no such channel -- but a file-backed
# manifest (see loaders.load_manifest_file) may declare them, and
# ioc/records.py dispatches on them like any other record type.
RECORD_TYPE_LONG_STRING = "longstringin"
RECORD_TYPE_MBB = "mbbi"

# Field/subfield tokens that indicate a two-state boolean signal rather than
# a continuous measurement.
_BOOLEAN_SUBFIELDS = frozenset(
    {
        "VALID",
        "FAULT",
        "READY",
        "ON",
        "INTERLOCK",
        "ALARM",
        "CONNECTED",
        "OPEN",
        "CLOSE",
        "CLOSED",
    }
)
_BOOLEAN_FIELDS = frozenset({"STATUS", "CONTROL"})


def derive_record_type(path: dict[str, str]) -> tuple[str, bool]:
    """Derive an EPICS record type and noise flag from a channel's hierarchy path.

    Args:
        path: Hierarchy path as produced by HierarchicalChannelDatabase.

    Returns:
        (record_type, noise) where noise indicates whether the future IOC
        should apply simulated measurement noise to this address:
          - booleans -> "bi", noise=False (a status flag doesn't jitter)
          - floats   -> "ai", noise=True (a measurement or setpoint readback does)
          - strings  -> "stringin", noise=False (reserved: the current
            namespace has no genuinely string-valued channel; kept for
            forward compatibility with future non-numeric DB additions)
    """
    field, subfield = path["field"], path["subfield"]

    if field in _BOOLEAN_FIELDS or subfield in _BOOLEAN_SUBFIELDS:
        return RECORD_TYPE_BINARY, False

    return RECORD_TYPE_ANALOG, True
