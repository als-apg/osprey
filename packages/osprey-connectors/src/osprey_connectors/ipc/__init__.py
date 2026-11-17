"""Inter-process plumbing for the connector-host child.

The child process owns the control-system client library; the parent talks to
it over a pipe pair. This namespace holds what a supervisor needs: the handle
that starts, asks after and stops one child (:class:`ConnectorHost`), the
records its ``init`` frame and its answer cross the pipe as, the proxy that
drives a child and the pool that keeps them, the endpoint derivation and
verification of what a child reports, and the two
errors the wire format raises when a frame cannot be written or read. The wire
codec itself and the typed-exception registry that carries a failure across
the boundary are internal to this package (``_frames``, ``_exceptions``).

The frame errors are imported here. Every other name is resolved on access
(PEP 562): the child runs as ``python -m osprey_connectors.ipc._host``, and a
package that imported ``_host`` eagerly would load it a second time under
``__main__``.
"""

from typing import TYPE_CHECKING

from osprey_connectors.ipc._frames import (
    FrameDecodeError,
    FrameEncodeError,
)

if TYPE_CHECKING:
    from osprey_connectors.ipc._handle import (
        ConnectorHost,
        ConnectorHostError,
        ConnectorHostStartError,
        StartMark,
    )
    from osprey_connectors.ipc._host import (
        HostInit,
        HostReport,
        scrub_epics_env,
    )
    from osprey_connectors.ipc._pool import (
        DEFAULT_CALL_TIMEOUT_S,
        DEFAULT_PING_TIMEOUT_S,
        ConnectorHostLostError,
        ConnectorHostPool,
        ConnectorHostSlowCallError,
        ConnectorHostUnresponsiveError,
        ConnectorHostWriteTimeoutError,
        HostKey,
        PooledConnector,
    )
    from osprey_connectors.ipc._proxy import (
        ChildLostError,
        ChildUnresponsiveError,
        ConnectorHostProxy,
    )
    from osprey_connectors.ipc._verification import (
        GATEWAYS_MISSING,
        GATEWAYS_NONE_DECLARED,
        GATEWAYS_ROWS,
        MODE_NAME_SERVER,
        ROLE_READ_ONLY,
        ROLE_WRITE_ACCESS,
        Endpoint,
        LiveCollision,
        SelectedRole,
        TargetDerivation,
        Verification,
        config_section,
        connector_block,
        declared_gateways,
        derive_endpoints,
        live_collision,
        subsection,
        verify_host_report,
    )

__all__ = [
    "ConnectorHost",
    "ConnectorHostError",
    "ConnectorHostStartError",
    "StartMark",
    "HostInit",
    "HostReport",
    "scrub_epics_env",
    "DEFAULT_CALL_TIMEOUT_S",
    "DEFAULT_PING_TIMEOUT_S",
    "ConnectorHostLostError",
    "ConnectorHostPool",
    "ConnectorHostSlowCallError",
    "ConnectorHostUnresponsiveError",
    "ConnectorHostWriteTimeoutError",
    "HostKey",
    "PooledConnector",
    "ChildLostError",
    "ChildUnresponsiveError",
    "ConnectorHostProxy",
    "GATEWAYS_MISSING",
    "GATEWAYS_NONE_DECLARED",
    "GATEWAYS_ROWS",
    "MODE_NAME_SERVER",
    "ROLE_READ_ONLY",
    "ROLE_WRITE_ACCESS",
    "Endpoint",
    "LiveCollision",
    "SelectedRole",
    "TargetDerivation",
    "Verification",
    "config_section",
    "connector_block",
    "declared_gateways",
    "derive_endpoints",
    "live_collision",
    "subsection",
    "verify_host_report",
    "FrameDecodeError",
    "FrameEncodeError",
]

#: Public name -> the module that defines it, for the names resolved lazily.
_LAZY_EXPORTS: dict[str, str] = {
    "ConnectorHost": "osprey_connectors.ipc._handle",
    "ConnectorHostError": "osprey_connectors.ipc._handle",
    "ConnectorHostStartError": "osprey_connectors.ipc._handle",
    "StartMark": "osprey_connectors.ipc._handle",
    "HostInit": "osprey_connectors.ipc._host",
    "HostReport": "osprey_connectors.ipc._host",
    "scrub_epics_env": "osprey_connectors.ipc._host",
    "DEFAULT_CALL_TIMEOUT_S": "osprey_connectors.ipc._pool",
    "DEFAULT_PING_TIMEOUT_S": "osprey_connectors.ipc._pool",
    "ConnectorHostLostError": "osprey_connectors.ipc._pool",
    "ConnectorHostPool": "osprey_connectors.ipc._pool",
    "ConnectorHostSlowCallError": "osprey_connectors.ipc._pool",
    "ConnectorHostUnresponsiveError": "osprey_connectors.ipc._pool",
    "ConnectorHostWriteTimeoutError": "osprey_connectors.ipc._pool",
    "HostKey": "osprey_connectors.ipc._pool",
    "PooledConnector": "osprey_connectors.ipc._pool",
    "ChildLostError": "osprey_connectors.ipc._proxy",
    "ChildUnresponsiveError": "osprey_connectors.ipc._proxy",
    "ConnectorHostProxy": "osprey_connectors.ipc._proxy",
    "GATEWAYS_MISSING": "osprey_connectors.ipc._verification",
    "GATEWAYS_NONE_DECLARED": "osprey_connectors.ipc._verification",
    "GATEWAYS_ROWS": "osprey_connectors.ipc._verification",
    "MODE_NAME_SERVER": "osprey_connectors.ipc._verification",
    "ROLE_READ_ONLY": "osprey_connectors.ipc._verification",
    "ROLE_WRITE_ACCESS": "osprey_connectors.ipc._verification",
    "Endpoint": "osprey_connectors.ipc._verification",
    "LiveCollision": "osprey_connectors.ipc._verification",
    "SelectedRole": "osprey_connectors.ipc._verification",
    "TargetDerivation": "osprey_connectors.ipc._verification",
    "Verification": "osprey_connectors.ipc._verification",
    "config_section": "osprey_connectors.ipc._verification",
    "connector_block": "osprey_connectors.ipc._verification",
    "declared_gateways": "osprey_connectors.ipc._verification",
    "derive_endpoints": "osprey_connectors.ipc._verification",
    "live_collision": "osprey_connectors.ipc._verification",
    "subsection": "osprey_connectors.ipc._verification",
    "verify_host_report": "osprey_connectors.ipc._verification",
}


def __getattr__(name: str) -> object:
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
