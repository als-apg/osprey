"""ophyd-async device factories for the Bluesky bridge.

Both submodules (``mock.py``, ``connector.py``) import ophyd-async and live
behind the optional ``osprey-framework[bluesky-bridge]`` extra — this package
(and its submodules) must never be imported from the bridge lifecycle core
(``app.py``, ``runs.py``, ``plan_runner.py``, ``security.py``), which stays
import-clean of bluesky/ophyd so it can be built and unit-tested without the
extra installed. This ``__init__.py`` itself imports nothing from either
submodule, so ``from osprey.services.bluesky_bridge import devices`` alone
stays cheap; callers import ``devices.mock`` or ``devices.connector``
explicitly. There is no raw Channel Access device factory in this package —
every device that talks to a real IOC is mediated by the OSPREY connector
(``devices/connector.py``); the earlier direct-CA ``devices/epics.py`` layer
has been removed.
"""

from __future__ import annotations
