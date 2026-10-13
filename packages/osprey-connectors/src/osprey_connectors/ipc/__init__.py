"""Inter-process plumbing for the connector-host child.

The child process owns the control-system client library; the parent talks to
it over a pipe pair. This namespace holds what a supervisor needs: the frame
codec, launching and terminating a child, the proxy that drives one, and the
endpoint derivation and verification of what it reports.

Names are resolved on first access (PEP 562) rather than imported here. The
child runs as ``python -m osprey_connectors.ipc._host``, and a package that
imported ``_host`` eagerly would load it a second time under ``__main__``.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey_connectors.ipc._frames import (
        ErrorFrame,
        FrameReader,
        encode_request,
        new_request_id,
    )
    from osprey_connectors.ipc._host import (
        scrub_epics_env,
    )
    from osprey_connectors.ipc._launch import (
        CHILD_MODULE,
        AttributedReader,
        host_env,
        spawn_host,
        terminate_host,
    )
    from osprey_connectors.ipc._proxy import (
        ConnectorHostProxy,
    )
    from osprey_connectors.ipc._verification import (
        MODE_NAME_SERVER,
        ROLE_READ_ONLY,
        ROLE_WRITE_ACCESS,
        Endpoint,
        TargetDerivation,
        Verification,
        config_writes_enabled,
        connector_block,
        derive_endpoints,
        live_collision,
        verify_host_report,
    )

__all__ = [
    "AttributedReader",
    "CHILD_MODULE",
    "ConnectorHostProxy",
    "Endpoint",
    "ErrorFrame",
    "FrameReader",
    "MODE_NAME_SERVER",
    "ROLE_READ_ONLY",
    "ROLE_WRITE_ACCESS",
    "TargetDerivation",
    "Verification",
    "config_writes_enabled",
    "connector_block",
    "derive_endpoints",
    "encode_request",
    "host_env",
    "live_collision",
    "new_request_id",
    "scrub_epics_env",
    "spawn_host",
    "terminate_host",
    "verify_host_report",
]

#: Public name -> the module that defines it.
_LAZY_EXPORTS: dict[str, str] = {
    "ErrorFrame": "osprey_connectors.ipc._frames",
    "FrameReader": "osprey_connectors.ipc._frames",
    "encode_request": "osprey_connectors.ipc._frames",
    "new_request_id": "osprey_connectors.ipc._frames",
    "scrub_epics_env": "osprey_connectors.ipc._host",
    "AttributedReader": "osprey_connectors.ipc._launch",
    "CHILD_MODULE": "osprey_connectors.ipc._launch",
    "host_env": "osprey_connectors.ipc._launch",
    "spawn_host": "osprey_connectors.ipc._launch",
    "terminate_host": "osprey_connectors.ipc._launch",
    "ConnectorHostProxy": "osprey_connectors.ipc._proxy",
    "Endpoint": "osprey_connectors.ipc._verification",
    "MODE_NAME_SERVER": "osprey_connectors.ipc._verification",
    "ROLE_READ_ONLY": "osprey_connectors.ipc._verification",
    "ROLE_WRITE_ACCESS": "osprey_connectors.ipc._verification",
    "TargetDerivation": "osprey_connectors.ipc._verification",
    "Verification": "osprey_connectors.ipc._verification",
    "config_writes_enabled": "osprey_connectors.ipc._verification",
    "connector_block": "osprey_connectors.ipc._verification",
    "derive_endpoints": "osprey_connectors.ipc._verification",
    "live_collision": "osprey_connectors.ipc._verification",
    "verify_host_report": "osprey_connectors.ipc._verification",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
