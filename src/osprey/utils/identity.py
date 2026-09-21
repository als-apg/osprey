"""The shared acting-identity ladder, under its historical import path.

The ladder itself is :mod:`osprey_connectors.identity`, one package down,
because it is read where ``osprey`` cannot be imported at all: inside
connector-host children, executor sandboxes and notebook kernels, which resolve
their own record directory through it. Living there rather than here is what
lets those readers and the framework's own writers share one implementation
instead of restating the rungs at each end.

Every name is re-exported, so the ``osprey.utils.identity`` call sites — the
audit envelope and writer, the interface middleware, the MCP server session,
the model adapters and spend attribution, the auth sidecar — keep working
unchanged and keep answering from that one implementation.

Read :mod:`osprey_connectors.identity` for the ladder, for what a rung's value
must satisfy to count, for why the hostname is never a rung, and for the
invariant that ``OSPREY_AUDIT_IDENTITY`` may never join a scrub list.
"""

from osprey_connectors.identity import (
    AUDIT_IDENTITY_ENV,
    IDENTITY_ENV_LADDER,
    TERMINAL_USER_ENV,
    UNKNOWN_IDENTITY,
    acting_identity,
)

# Re-exported under its own name: the ladder's tests pin the path-component
# rule directly, because it is the rule a future rung would be checked against
# and some of its cases cannot be reached through ``os.environ``.
from osprey_connectors.identity import _usable as _usable

__all__ = [
    "AUDIT_IDENTITY_ENV",
    "IDENTITY_ENV_LADDER",
    "TERMINAL_USER_ENV",
    "UNKNOWN_IDENTITY",
    "acting_identity",
]
