"""The scratch root the per-server report suites anchor on.

``target_state`` derives its state directory three ways — the
``OSPREY_CONTROL_CONTEXT_DIR`` bind a container carries, the
``OSPREY_AGENT_DATA_ROOT`` stamp a session sets, and the deployment's own
``resolve_shared_data_root`` where there is neither — in that order of
precedence. A test that redirects the root by patching the resolver alone is
inert wherever either environment rung is present, so this clears both as well
as patching the resolver: the directory is pinned whichever rule ``state_dir``
applies.

It is autouse, and a module that imports it shadows the directory's
``state_root`` with it. That one belongs to the connector-host child harness
and *sets* the stamp, because a child has to resolve the same root its parent
did. A suite that runs no child needs the opposite, which is why these are two
fixtures rather than one with a switch.
"""

import pytest

from osprey.mcp_server.control_system import target_state
from osprey_connectors import posture_store


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    """Anchor the state directory in tmp_path instead of a real deployment.

    The container bind outranks the root stamp, so clearing the stamp alone
    would leave a containerized lane resolving the bound directory while this
    fixture believed it had redirected the suite. Both are cleared, named
    through the store rather than spelled, so one variable cannot become two.

    ``OSPREY_POSTURE_SESSION`` is cleared with them: the session a report
    carries is read from the environment, and a test that inherited a real one
    would assert against the machine it ran on.
    """
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)
    return tmp_path
