"""Shared collection-time setup for the Virtual Accelerator test suite.

docker/virtual-accelerator/{manifest,lattice,ioc} are not importable dotted
packages (their parent directory name contains a hyphen), so every test
module under this directory imports them by putting docker/virtual-accelerator
directly on sys.path -- "manifest", "lattice", and "ioc" are then valid
top-level module names.

pytest loads this conftest before collecting any test module in this
directory tree (including tests/va/e2e/), so the path is set up exactly once
for the whole suite instead of being repeated per test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VA_PARENT = REPO_ROOT / "docker" / "virtual-accelerator"
if str(VA_PARENT) not in sys.path:
    sys.path.insert(0, str(VA_PARENT))
