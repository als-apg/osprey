"""Execution subsystem — channel write safety limits.

Provides :class:`LimitsValidator` for enforcing configured min/max/step
constraints on control system channel writes.

Note: LimitsValidator's public home is osprey_connectors.control_system.
"""

from osprey_connectors.control_system import LimitsValidator

__all__ = ["LimitsValidator"]
