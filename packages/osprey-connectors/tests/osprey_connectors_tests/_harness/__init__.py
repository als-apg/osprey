"""Framework-free guards and helpers both test suites share.

The control-context writers and the connector-registry bracket are re-exported
here for short imports. ``_guards``
acts at import, so it is never imported from here.
"""

from osprey_connectors_tests._harness._connector_registries import (
    isolated_connector_registries,
)
from osprey_connectors_tests._harness._control_context_fixtures import (
    DEFAULT_GENERATION,
    DEFAULT_TARGET,
    FIXTURE_IDENTITY,
    owner,
    pin_identity,
    state_dir_under,
    write_control_context,
    write_payload,
    write_server_report,
)

__all__ = [
    "DEFAULT_GENERATION",
    "DEFAULT_TARGET",
    "FIXTURE_IDENTITY",
    "isolated_connector_registries",
    "owner",
    "pin_identity",
    "state_dir_under",
    "write_control_context",
    "write_payload",
    "write_server_report",
]
