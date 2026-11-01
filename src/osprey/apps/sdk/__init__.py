"""The application SDK: what an application's tool server and backend import from OSPREY.

Stability promise: every name in ``__all__`` keeps its name, its import path
(``from osprey.apps.sdk import <name>``) and its meaning across releases; a
change to one is a breaking change of the application contract, announced in
the changelog. Application code imports OSPREY only through this module (and
``osprey.facility.kinds`` and the public ``osprey_connectors`` modules); the
framework modules behind these names are internal and may move.

The write chain an application meets is the framework's own:
:func:`check_approved_call` runs the login's tier, :func:`write_verdict` and
the approval record for one call; :func:`command_write` fires commands under
the run :func:`lock`; :func:`refusal_phrase` words a refusal.
"""

from osprey import port_layout
from osprey.apps.sdk._approval import ApprovalRefused, BoundState, check_approved_call
from osprey.apps.sdk._facts import facility_file, read_facts, tool_keys
from osprey.apps.sdk._serve import serve_tools
from osprey.apps.sdk._target import owner_target
from osprey.apps.sdk._wrappers import agent_panel_headers, queue_is_active
from osprey.audit.call import current_tool_use_id
from osprey.audit.dedup import record_and_mark
from osprey.audit.envelope import DECISION_ALLOWED, DECISION_REFUSED, POSTURE_SOURCE_APP
from osprey.interfaces._app_setup import configure_interface_app as configure_app
from osprey.interfaces.common_middleware import OWNER_STATE_KEY
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import web_terminal_url
from osprey.mcp_server.startup import startup_timer
from osprey.runtime import raw_put_block
from osprey.runtime.guarded_run import (
    ENV_CONTROL_IDENTITY,
    ActionContext,
    GuardedRunDirError,
    OspreyRunBusy,
    OspreyWriteRefused,
    command_write,
    guarded_run_dir,
    lock,
)
from osprey.services.python_executor.execution.wrapper import armed_contract
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER, verify_launch_token
from osprey.utils.module_loading import load_by_path
from osprey.utils.owner_header import owner_from_header
from osprey.utils.seconds import non_negative_seconds, positive_seconds
from osprey_connectors.posture_store import WriteVerdict, bind_owner, write_verdict
from osprey_connectors.refusals import refusal_phrase

__all__ = [
    "DECISION_ALLOWED",
    "DECISION_REFUSED",
    "ENV_CONTROL_IDENTITY",
    "LAUNCH_TOKEN_HEADER",
    "OWNER_STATE_KEY",
    "POSTURE_SOURCE_APP",
    "ActionContext",
    "ApprovalRefused",
    "BoundState",
    "GuardedRunDirError",
    "OspreyRunBusy",
    "OspreyWriteRefused",
    "WriteVerdict",
    "agent_panel_headers",
    "armed_contract",
    "bind_owner",
    "check_approved_call",
    "command_write",
    "configure_app",
    "current_tool_use_id",
    "facility_file",
    "guarded_run_dir",
    "load_by_path",
    "lock",
    "make_error",
    "non_negative_seconds",
    "owner_from_header",
    "owner_target",
    "port_layout",
    "positive_seconds",
    "queue_is_active",
    "raw_put_block",
    "read_facts",
    "record_and_mark",
    "refusal_phrase",
    "serve_tools",
    "startup_timer",
    "tool_keys",
    "verify_launch_token",
    "web_terminal_url",
    "write_verdict",
]
