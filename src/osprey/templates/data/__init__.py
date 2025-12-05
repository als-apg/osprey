"""Shared template configuration data.

This module contains configuration data that can be used across multiple
templates, such as facility presets for EPICS gateway configurations.
"""

from .facility_presets import (
    FACILITY_PRESETS,
    get_facility_choices,
    get_facility_config,
    list_facilities,
)

__all__ = [
    "FACILITY_PRESETS",
    "get_facility_choices",
    "get_facility_config",
    "list_facilities",
]
