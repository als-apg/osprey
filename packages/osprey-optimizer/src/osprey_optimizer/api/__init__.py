"""The optimizer's HTTP service. Needs the ``service`` extra."""

import importlib.util

_SERVICE_ROOTS = ("fastapi", "redis", "polars", "xopt", "torch")

_missing = [name for name in _SERVICE_ROOTS if importlib.util.find_spec(name) is None]
if _missing:
    raise ImportError(
        f"osprey_optimizer.api needs the optimizer's service extra (missing: {', '.join(_missing)}). "
        'Install it with: pip install "osprey-optimizer[service]"'
    )
