"""Shared collection-time setup for the Virtual Accelerator test suite.

``manifest``, ``lattice``, ``ioc`` and ``serving`` are importable as
``osprey.services.virtual_accelerator.{manifest,lattice,ioc,serving}`` --
ordinary packaged imports, no sys.path setup needed.

pytest loads this conftest before collecting any test module in this
directory tree (including tests/va/e2e/).
"""

from __future__ import annotations
