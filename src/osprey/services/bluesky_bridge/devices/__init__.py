"""ophyd-async device factories for the Bluesky bridge.

Both submodules (``mock.py``, ``epics.py``) import ophyd-async and live behind
the optional ``osprey-framework[bluesky-bridge]`` extra — this package (and
its submodules) must never be imported from the bridge lifecycle core
(``app.py``, ``runs.py``, ``scanner.py``, ``security.py``), which stays
import-clean of bluesky/ophyd so it can be built and unit-tested without the
extra installed. This ``__init__.py`` itself imports nothing from either
submodule, so ``from osprey.services.bluesky_bridge import devices`` alone
stays cheap; callers import ``devices.mock`` or ``devices.epics`` explicitly.
"""

from __future__ import annotations
