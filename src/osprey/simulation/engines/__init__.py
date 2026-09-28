"""Simulation engine plug-ins, discovered through the ``osprey.simulation.engines`` group.

Each entry point names a module that implements the plug-in contract for one
engine: ``locate``, ``prepare``, ``start_values`` and ``plane`` read a deck and
a model's wiring without building a model. A facility build reaches an engine
through its entry point, never by importing the module by name.
"""

from __future__ import annotations

__all__ = ["ENTRY_POINT_GROUP"]

#: The entry-point group every engine plug-in registers under.
ENTRY_POINT_GROUP = "osprey.simulation.engines"
