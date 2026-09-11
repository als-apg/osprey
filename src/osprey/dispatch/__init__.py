"""OSPREY event dispatch package — pool, registry, trigger configuration, and worker client.

Importing this package, or any one leaf of it, costs only that leaf. The four
pieces gathered here have unrelated dependencies: the trigger-configuration
reader needs a YAML parser, while the worker client speaks HTTP. Re-exporting
them eagerly would make reading ``triggers.yml`` pull the HTTP worker client in
behind it, so every public name is resolved on first attribute access instead.
"""

from __future__ import annotations

from typing import Any

#: Public name -> the submodule of this package that defines it. Entries are
#: resolved on first attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {
    "DispatchPool": ".pool",
    "QueueFullError": ".pool",
    "TriggerRegistry": ".registry",
    "DispatcherConfig": ".trigger_config",
    "TriggerConfig": ".trigger_config",
    "load_triggers": ".trigger_config",
    "WorkerAuthRejectedError": ".worker_client",
    "WorkerRejectedRequestError": ".worker_client",
    "WorkerRequestError": ".worker_client",
    "WorkerUnavailableError": ".worker_client",
    "WorkerUnreachableError": ".worker_client",
    "cancel_worker_run": ".worker_client",
    "dispatch_to_worker": ".worker_client",
    "fetch_worker_runs": ".worker_client",
    "proxy_worker_stream": ".worker_client",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
