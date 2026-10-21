"""The serving layer for the virtual accelerator.

:class:`~osprey.services.virtual_accelerator.serving.runner.ModelRunner`
serves a composite on Channel Access and PVAccess and answers the model RPC.

**Importing this package must stay cheap.** The runner module imports the
serving package (``lume_pva_apg``) and the Channel Access server extension at
module level, so ``ModelRunner`` is resolved lazily, PEP 562 style: it pays
that import when first touched, never at package import.
"""

from typing import Any

__all__ = ["ModelRunner"]


def __getattr__(name: str) -> Any:
    if name == "ModelRunner":
        import importlib

        return importlib.import_module(f"{__name__}.runner").ModelRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
