"""Channel Access serving layer for the virtual accelerator.

:mod:`serving.pvdb` turns the namespace-union channel manifest into the
process-variable database the CA server is handed, plus one shim per channel
carrying the ``.set()``/``.get()`` surface the existing value sources
(``ioc.engine_source``, ``ioc.physics_bridge``) already drive.

**Importing this package must stay cheap.** It is on the no-lattice boot
path, whose whole purpose is coming up without the accelerator-physics
stack, and the serving package it hosts (``lume_pva_apg``) drags the
compiled CA server extension in with it. So:

* nothing here imports ``lume_pva_apg`` (or the CA server library) at module
  import time -- those imports live inside functions, or inside a module
  reached lazily through the ``__getattr__`` hook below;
* modules of this package that *do* need them at module level (the runner
  subclasses a class it must import to define itself) are re-exported
  lazily, PEP 562 style, never with an eager ``from .runner import ...``
  here.

Attribute access is unchanged for callers: ``serving.runner`` works, it just
pays the import when it is first touched.
"""

from typing import Any

from .pvdb import (
    ManifestContractError,
    PVRecord,
    ServingRecords,
    build_serving_pvdb,
)

__all__ = [
    "ManifestContractError",
    "PVRecord",
    "ServingRecords",
    "build_serving_pvdb",
]

# Submodules resolved on first attribute access rather than at package
# import. Naming one here is not a promise that it exists yet -- an absent
# module still raises ModuleNotFoundError on access, exactly as a direct
# import would.
_LAZY_SUBMODULES = frozenset({"runner", "model_stub"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
