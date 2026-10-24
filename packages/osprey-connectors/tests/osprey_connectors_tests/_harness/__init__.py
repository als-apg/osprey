"""Framework-free guards and helpers both test suites share.

The control-context writers, P1's verdict-environment helpers and the
``data_root``/``race_record_writers`` fixtures are re-exported here for short
imports. ``data_root`` is a fixture: it is active only in a module that imports
it by name, and a module that defines its own ``data_root`` keeps its own.
No conftest imports it from here, so it never shadows those module fixtures.
``_guards`` acts at import, so it is never imported from here.
"""

from osprey_connectors_tests._harness._control_context_fixtures import (
    DEFAULT_GENERATION,
    DEFAULT_TARGET,
    FIXTURE_IDENTITY,
    VERDICT_ENV_VARS,
    data_root,
    isolate_verdict_environment,
    owner,
    pin_identity,
    race_record_writers,
    reset_verdict_state,
    state_dir_under,
    write_control_context,
    write_payload,
    write_server_report,
)
from osprey_connectors_tests._harness._vocabulary import REFUSAL_BAN_LIST

__all__ = [
    "DEFAULT_GENERATION",
    "DEFAULT_TARGET",
    "FIXTURE_IDENTITY",
    "REFUSAL_BAN_LIST",
    "VERDICT_ENV_VARS",
    "data_root",
    "isolate_verdict_environment",
    "owner",
    "pin_identity",
    "race_record_writers",
    "reset_verdict_state",
    "state_dir_under",
    "write_control_context",
    "write_payload",
    "write_server_report",
]
