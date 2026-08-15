"""OSPREY event dispatch package — pool, registry, trigger configuration, and worker client."""

from osprey.dispatch.pool import DispatchPool, QueueFullError
from osprey.dispatch.registry import TriggerRegistry
from osprey.dispatch.trigger_config import DispatcherConfig, TriggerConfig, load_triggers
from osprey.dispatch.worker_client import (
    WorkerAuthRejectedError,
    WorkerRejectedRequestError,
    WorkerRequestError,
    WorkerUnavailableError,
    WorkerUnreachableError,
    cancel_worker_run,
    dispatch_to_worker,
    fetch_worker_runs,
    proxy_worker_stream,
)

__all__ = [
    "DispatchPool",
    "DispatcherConfig",
    "QueueFullError",
    "TriggerConfig",
    "TriggerRegistry",
    "WorkerAuthRejectedError",
    "WorkerRejectedRequestError",
    "WorkerRequestError",
    "WorkerUnavailableError",
    "WorkerUnreachableError",
    "cancel_worker_run",
    "dispatch_to_worker",
    "fetch_worker_runs",
    "load_triggers",
    "proxy_worker_stream",
]
