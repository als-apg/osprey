"""Namespace-union manifest generator for the PyAT virtual accelerator.

The Control Assistant Tutorial's channel-finder databases already define the
channel namespace the virtual accelerator must serve: the tutorial ships
three interchangeable "paradigm" formats (in_context, hierarchical,
middle_layer) that describe the same set of PV addresses under the grammar
``{ring}:{system}:{family}:{device}:{field}:{subfield}``.

This package expands all three at their build-resolved tier, verifies they
agree, unions in the scenario-seed ``machine.json`` channels, reconciles the
machine-state template against the result, and classifies every address into
a physics-fidelity partition (pyat-coupled / sp-echo / static-noisy) plus an
EPICS record type. The served channel set is derived from these sources --
never hand-listed.

See :func:`build.build_manifest` for the entry point.
"""

from .build import build_manifest
from .classify import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
    RECORD_TYPE_LONG_STRING,
    RECORD_TYPE_MBB,
    RECORD_TYPE_STRING,
    classify_partition,
    derive_record_type,
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
    "RECORD_TYPE_ANALOG",
    "RECORD_TYPE_BINARY",
    "RECORD_TYPE_LONG_STRING",
    "RECORD_TYPE_MBB",
    "RECORD_TYPE_STRING",
]
