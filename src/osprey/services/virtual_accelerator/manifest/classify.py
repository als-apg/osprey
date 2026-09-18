"""The partition vocabulary, the setpoint vocabulary, and EPICS record types.

Every namespace address sits in exactly one of three physics-fidelity tiers,
which the IOC serves differently:

  pyat-coupled -- backed by the lattice model: the setpoints a write actually
                  steers the beam with, plus the monitors that observe it.
                  These are the addresses the bindings document claims, and
                  the bindings are the only thing that puts an address here.
  sp-echo      -- writable but physics-free: a write to the setpoint just
                  echoes onto the readback, with no lattice model behind it.
  static-noisy -- everything else: golden references, status/fault flags,
                  and slow telemetry (temperatures, pressures, radiation
                  monitors) that just needs a plausible noisy constant.

Which partition an address lands in is read off the facility's own bindings
(see :mod:`~osprey.services.virtual_accelerator.bindings`) by
:mod:`~osprey.services.virtual_accelerator.manifest.build`, never decided from
the address text here: a rule keyed on a ring, system and family name would be
one facility's naming convention masquerading as a physics fact. What this
module still owns is the vocabulary those partitions are spelled in, the
setpoint/readback subfields, and the record type an address is served as --
all of which are the framework's own and the same for every facility.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# The hierarchy level names a classified channel is described by, and the
# identity keys the manifest carries per channel. A hierarchical database
# declares its own level names, and a facility whose tree is not levelled this
# way carries no path in these terms -- the caller compares the declared names
# with these and records no path rather than guessing one.
CLASSIFIER_LEVELS = ("ring", "system", "family", "device", "field", "subfield")

PARTITION_PYAT_COUPLED = "pyat-coupled"
PARTITION_SP_ECHO = "sp-echo"
PARTITION_STATIC_NOISY = "static-noisy"

# The manifest's setpoint/readback vocabulary, and the one place it is spelled.
# A channel's ADDRESS text is free -- any facility's namespace loads through
# `loaders.load_manifest_file` -- but the `subfield` VALUE is reserved: "SP"
# marks the writable channel, "RB" marks its readback, and a channel carrying
# any other token is neither written nor paired with one. Deliberately not
# facility-configurable: a typo in a per-facility spelling would silently
# disable every write on the machine rather than fail loudly.
SETPOINT_SUBFIELD = "SP"
READBACK_SUBFIELD = "RB"


def setpoint_addresses(channels: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """The addresses a manifest declares writable, read off its own subfields.

    The one answer to "which of these channels is a setpoint", for every layer
    that needs it: the drive-limit and value-range readers, which have to pick
    the writable half out of a limits file holding an entry per address. Asking
    the address text instead ties those layers to one facility's spelling.
    """
    return frozenset(
        channel["address"] for channel in channels if channel["subfield"] == SETPOINT_SUBFIELD
    )


def pyat_coupled_setpoint_addresses(channels: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """The addresses a manifest declares writable AND backed by the lattice model.

    The setpoint half of the pyat-coupled partition -- the magnet currents a
    write actually steers the beam with, as opposed to the sp-echo setpoints
    that only echo onto their readback. Read off the manifest's own
    ``partition`` and ``subfield``, because that is where the answer is: the
    address text is a facility's own spelling and says nothing about whether a
    lattice element is behind it.
    """
    return frozenset(
        channel["address"]
        for channel in channels
        if channel["partition"] == PARTITION_PYAT_COUPLED
        and channel["subfield"] == SETPOINT_SUBFIELD
    )


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
