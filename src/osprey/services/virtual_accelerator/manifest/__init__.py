"""Namespace-union manifest generator for the PyAT virtual accelerator.

The Control Assistant Tutorial's channel-finder databases already define the
channel namespace the virtual accelerator must serve: the tutorial ships
three interchangeable file "paradigm" formats (in_context, hierarchical,
middle_layer) that describe the same set of PV addresses. The ``graph``
paradigm is deliberately not among them -- it answers from a seeded store
rather than a tier file, so it contributes no manifest source.

What this package fixes, and what it does not. A channel's ADDRESS text is
free: it is carried through verbatim from whatever source declared it, and the
bundled demo tree's six colon-separated tokens
(``{ring}:{system}:{family}:{device}:{field}:{subfield}``) are that tree's
spelling rather than a shape anything here requires -- a facility whose
addresses are three parts, or slashed, loads through the same call (see
``loaders.load_manifest_file``). What IS reserved is the ``subfield`` VALUE:
``SETPOINT_SUBFIELD`` marks the writable channel and ``READBACK_SUBFIELD``
marks its readback, and a channel carrying any other token is neither written
nor paired with one.

This package expands all three file formats at their build-resolved tier,
verifies they agree, unions in the scenario-seed ``machine.json`` channels,
reconciles the machine-state template against the result, and classifies
every address into a physics-fidelity partition (pyat-coupled / sp-echo /
static-noisy) plus an EPICS record type. The served channel set is derived
from these sources -- never hand-listed.

See :func:`build.build_manifest` for the entry point.
"""

from .build import build_manifest
from .classify import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    READBACK_SUBFIELD,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
    RECORD_TYPE_LONG_STRING,
    RECORD_TYPE_MBB,
    RECORD_TYPE_STRING,
    SETPOINT_SUBFIELD,
    classify_partition,
    derive_record_type,
    pyat_coupled_setpoint_addresses,
    setpoint_addresses,
)
from .loaders import MANIFEST_CHANNEL_KEYS, ManifestFileError, load_manifest_file

__all__ = [
    "build_manifest",
    "classify_partition",
    "derive_record_type",
    "load_manifest_file",
    "ManifestFileError",
    "MANIFEST_CHANNEL_KEYS",
    "PARTITION_PYAT_COUPLED",
    "PARTITION_SP_ECHO",
    "PARTITION_STATIC_NOISY",
    "READBACK_SUBFIELD",
    "RECORD_TYPE_ANALOG",
    "RECORD_TYPE_BINARY",
    "RECORD_TYPE_LONG_STRING",
    "RECORD_TYPE_MBB",
    "RECORD_TYPE_STRING",
    "SETPOINT_SUBFIELD",
    "pyat_coupled_setpoint_addresses",
    "setpoint_addresses",
]
