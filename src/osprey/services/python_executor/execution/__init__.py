"""Execution subsystem — channel write safety limits.

Provides :class:`LimitsValidator` for enforcing configured min/max/step
constraints on control system channel writes.
"""

from .limits_validator import LimitsValidator

__all__ = ["LimitsValidator"]
