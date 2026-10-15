"""Custom optimization generators.

Generators in this sub-package are resolved by name in
``optimization.xopt_backend._CUSTOM_GENERATORS`` before falling back to Xopt's
built-in ``get_generator`` registry.  To add a new custom generator, define
a ``Generator`` subclass here and register it in that dict.
"""

from .baxus import BAxUSGenerator

__all__ = ["BAxUSGenerator"]
