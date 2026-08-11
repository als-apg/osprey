"""Connector type constants, and the fallbacks a factory applies to them.

Single source of truth for built-in connector type name strings.
Custom connectors use dotted module paths (e.g., 'mypackage.TangoConnector')
and don't need constants here.

The two resolvers at the bottom are here rather than in
:mod:`osprey_connectors.factory` so that a guard deciding what a deployment
*will build* can share them with the factory that builds it. A guard which
re-implements "what does this config select" is a guard that can disagree with
the answer, and the disagreement is a bypass rather than a discrepancy — see
:mod:`osprey_connectors.honesty`.
"""

from typing import Any

# -- Control system connector types (have implementations) --
MOCK = "mock"
EPICS = "epics"
VIRTUAL_ACCELERATOR = "virtual_accelerator"
DOOCS = "doocs"

# -- Archiver connector types --
MOCK_ARCHIVER = "mock_archiver"
EPICS_ARCHIVER = "epics_archiver"
MONGODB_ARCHIVER = "mongodb_archiver"
DOOCS_ARCHIVER = "doocs_archiver"

# -- CLI choice lists (only types with implementations) --
CLI_CONTROL_SYSTEM_TYPES = [MOCK, EPICS, VIRTUAL_ACCELERATOR, DOOCS]
CLI_ARCHIVER_TYPES = [MOCK_ARCHIVER, EPICS_ARCHIVER, MONGODB_ARCHIVER, DOOCS_ARCHIVER]


def _resolve_type(section: Any, fallback: str) -> str:
    """The connector type a factory builds from a config section.

    A section that is missing, is not a mapping, or carries no usable ``type``
    resolves to *fallback* — the factory's documented fail-closed default, which
    it announces with a ``… is not set; defaulting to …`` warning. Empty and
    ``None`` count as absent (YAML gives ``None`` for a bare ``type:``); any
    other value is returned as written, so a typo reaches the factory's
    "Unknown … type" error rather than being quietly rounded to something.
    """
    value = section.get("type") if isinstance(section, dict) else None
    return str(value) if value else fallback


def resolve_archiver_type(section: Any) -> str:
    """The archiver an ``archiver:`` config section actually selects.

    Absent means the mock: a config that says nothing about its archiver is a
    config that gets the synthesizing one, which is the fact the honesty rule is
    really about.
    """
    return _resolve_type(section, MOCK_ARCHIVER)


def resolve_control_system_type(section: Any) -> str:
    """The control system a ``control_system:`` config section actually selects."""
    return _resolve_type(section, MOCK)
