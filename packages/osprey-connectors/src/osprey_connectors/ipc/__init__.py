"""Inter-process plumbing for the connector-host child.

The child process owns the control-system client library; the parent talks to
it over a pipe pair. This namespace holds what a supervisor needs: the wire
format both sides speak (a pure codec, with no process management in it) and
the typed-exception registry it uses, so a failure crosses the boundary with its
fields intact; launching and stopping a child, the proxy that drives one and the
pool that keeps them; and the endpoint derivation and verification of what a
child reports.

The codec and the exception registry are imported here. Every other name is
resolved on access (PEP 562): the child runs as
``python -m osprey_connectors.ipc._host``, and a package that imported
``_host`` eagerly would load it a second time under ``__main__``.
"""

from typing import TYPE_CHECKING, Any

from osprey_connectors.ipc._exceptions import (
    EXCEPTION_SPECS,
    ExceptionSpec,
    decode_exception,
    encode_exception,
    exception_fields,
    reconstruct_exception,
)
from osprey_connectors.ipc._frames import (
    ErrorFrame,
    FrameDecodeError,
    FrameEncodeError,
    FrameReader,
    RequestFrame,
    ResultFrame,
    decode_frame,
    encode_error,
    encode_request,
    encode_result,
    new_request_id,
)

if TYPE_CHECKING:
    from osprey_connectors.ipc._host import (
        scrub_epics_env,
    )
    from osprey_connectors.ipc._launch import (
        CHILD_MODULE,
        SETTLE_TIMEOUT_S,
        AttributedReader,
        host_env,
        kill_host,
        spawn_host,
        stop_host,
    )
    from osprey_connectors.ipc._pool import (
        DEFAULT_CALL_DEADLINE_S,
        DEFAULT_PING_TIMEOUT_S,
        READONLY,
        ConnectorHostError,
        ConnectorHostLostError,
        ConnectorHostPool,
        ConnectorHostStartError,
        ConnectorHostUnresponsiveError,
        ConnectorHostWriteTimeoutError,
        PooledConnector,
    )
    from osprey_connectors.ipc._proxy import (
        CHILD,
        ChildUnresponsiveError,
        ConnectorHostProxy,
        raised_by_child,
    )
    from osprey_connectors.ipc._verification import (
        DEFAULT_CA_PORT,
        DEFAULT_PVA_PORT,
        MODE_ADDR_LIST,
        MODE_NAME_SERVER,
        NEVER_LIVE_TYPES,
        ROLE_PVA,
        ROLE_READ_ONLY,
        ROLE_WRITE_ACCESS,
        Endpoint,
        LiveCollision,
        TargetDerivation,
        Verification,
        config_section,
        config_writes_enabled,
        connector_block,
        derive_endpoints,
        live_collision,
        same_endpoint,
        subsection,
        verify_child_report,
        verify_host_report,
    )

__all__ = [
    "scrub_epics_env",
    "CHILD_MODULE",
    "SETTLE_TIMEOUT_S",
    "AttributedReader",
    "host_env",
    "kill_host",
    "spawn_host",
    "stop_host",
    "DEFAULT_CALL_DEADLINE_S",
    "DEFAULT_PING_TIMEOUT_S",
    "READONLY",
    "ConnectorHostError",
    "ConnectorHostLostError",
    "ConnectorHostPool",
    "ConnectorHostStartError",
    "ConnectorHostUnresponsiveError",
    "ConnectorHostWriteTimeoutError",
    "PooledConnector",
    "CHILD",
    "ChildUnresponsiveError",
    "ConnectorHostProxy",
    "raised_by_child",
    "DEFAULT_CA_PORT",
    "DEFAULT_PVA_PORT",
    "MODE_ADDR_LIST",
    "MODE_NAME_SERVER",
    "NEVER_LIVE_TYPES",
    "ROLE_PVA",
    "ROLE_READ_ONLY",
    "ROLE_WRITE_ACCESS",
    "Endpoint",
    "LiveCollision",
    "TargetDerivation",
    "Verification",
    "config_section",
    "config_writes_enabled",
    "connector_block",
    "derive_endpoints",
    "live_collision",
    "same_endpoint",
    "subsection",
    "verify_child_report",
    "verify_host_report",
    "EXCEPTION_SPECS",
    "ErrorFrame",
    "ExceptionSpec",
    "FrameDecodeError",
    "FrameEncodeError",
    "FrameReader",
    "RequestFrame",
    "ResultFrame",
    "decode_exception",
    "decode_frame",
    "encode_error",
    "encode_exception",
    "encode_request",
    "encode_result",
    "exception_fields",
    "new_request_id",
    "reconstruct_exception",
]

#: Public name -> the module that defines it, for the names resolved lazily.
_LAZY_EXPORTS: dict[str, str] = {
    "scrub_epics_env": "osprey_connectors.ipc._host",
    "CHILD_MODULE": "osprey_connectors.ipc._launch",
    "SETTLE_TIMEOUT_S": "osprey_connectors.ipc._launch",
    "AttributedReader": "osprey_connectors.ipc._launch",
    "host_env": "osprey_connectors.ipc._launch",
    "kill_host": "osprey_connectors.ipc._launch",
    "spawn_host": "osprey_connectors.ipc._launch",
    "stop_host": "osprey_connectors.ipc._launch",
    "DEFAULT_CALL_DEADLINE_S": "osprey_connectors.ipc._pool",
    "DEFAULT_PING_TIMEOUT_S": "osprey_connectors.ipc._pool",
    "READONLY": "osprey_connectors.ipc._pool",
    "ConnectorHostError": "osprey_connectors.ipc._pool",
    "ConnectorHostLostError": "osprey_connectors.ipc._pool",
    "ConnectorHostPool": "osprey_connectors.ipc._pool",
    "ConnectorHostStartError": "osprey_connectors.ipc._pool",
    "ConnectorHostUnresponsiveError": "osprey_connectors.ipc._pool",
    "ConnectorHostWriteTimeoutError": "osprey_connectors.ipc._pool",
    "PooledConnector": "osprey_connectors.ipc._pool",
    "CHILD": "osprey_connectors.ipc._proxy",
    "ChildUnresponsiveError": "osprey_connectors.ipc._proxy",
    "ConnectorHostProxy": "osprey_connectors.ipc._proxy",
    "raised_by_child": "osprey_connectors.ipc._proxy",
    "DEFAULT_CA_PORT": "osprey_connectors.ipc._verification",
    "DEFAULT_PVA_PORT": "osprey_connectors.ipc._verification",
    "MODE_ADDR_LIST": "osprey_connectors.ipc._verification",
    "MODE_NAME_SERVER": "osprey_connectors.ipc._verification",
    "NEVER_LIVE_TYPES": "osprey_connectors.ipc._verification",
    "ROLE_PVA": "osprey_connectors.ipc._verification",
    "ROLE_READ_ONLY": "osprey_connectors.ipc._verification",
    "ROLE_WRITE_ACCESS": "osprey_connectors.ipc._verification",
    "Endpoint": "osprey_connectors.ipc._verification",
    "LiveCollision": "osprey_connectors.ipc._verification",
    "TargetDerivation": "osprey_connectors.ipc._verification",
    "Verification": "osprey_connectors.ipc._verification",
    "config_section": "osprey_connectors.ipc._verification",
    "config_writes_enabled": "osprey_connectors.ipc._verification",
    "connector_block": "osprey_connectors.ipc._verification",
    "derive_endpoints": "osprey_connectors.ipc._verification",
    "live_collision": "osprey_connectors.ipc._verification",
    "same_endpoint": "osprey_connectors.ipc._verification",
    "subsection": "osprey_connectors.ipc._verification",
    "verify_child_report": "osprey_connectors.ipc._verification",
    "verify_host_report": "osprey_connectors.ipc._verification",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module, on every access.

    Not cached here: the defining module stays the one place a name is bound,
    so a name replaced there (a test's monkeypatch, say) is what every later
    lookup through this namespace gets.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
