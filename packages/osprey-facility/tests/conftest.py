"""Pytest configuration for the ``osprey-facility`` suite.

The guards are framework-free and imported by name from ``_harness._guards``
(see that module for why not as a plugin). This file sits above the
``osprey_facility_tests`` package so its session hook loads before collection
and so pytest puts this directory on ``sys.path``.
"""

from osprey_facility_tests._harness._guards import (  # noqa: F401 - fixtures and hooks, used by name
    entry_points_read_afresh,
    pytest_configure,
    restore_cwd,
    restore_environ,
)
