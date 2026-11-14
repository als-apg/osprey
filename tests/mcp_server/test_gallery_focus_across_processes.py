"""artifact_focus reaches the gallery whichever MCP server of the project hosts it.

The gallery is a thread inside whichever MCP server process saves the first
artifact, and ``artifact_focus`` always runs in the workspace server. Where no
launcher handed the servers a panel token (``osprey query``, an SDK run,
``claude -p``), the two are different processes that share nothing but the
project, and the focus request has to be admitted all the same.

Both sides are real child processes running the real code: one saves an
artifact through the store, which launches the gallery; the other calls the
tool. Nothing is mocked, and no credential is handed to either.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from osprey.interfaces.web_auth import (
    OPERATOR_SECRET_ENV,
    PANEL_TOKEN_ENV,
    PROJECT_PANEL_TOKEN_FILE,
)

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[2]

_HOST = """
import sys
from pathlib import Path

from osprey.stores.artifact_store import initialize_artifact_store

store = initialize_artifact_store(workspace_root=Path(sys.argv[1]))
entry = store.save_file(
    file_content=b"<html></html>",
    filename="plot.html",
    artifact_type="plot_html",
    title="Test plot",
)
print(entry.id, flush=True)
sys.stdin.read()
"""

_CLIENT = """
import asyncio
import json
import sys
from pathlib import Path

from fastmcp.exceptions import ToolError

from osprey.mcp_server.workspace.tools import focus_tools
from osprey.stores.artifact_store import ArtifactStore, initialize_artifact_store

# The workspace server posts; it does not host. Its own saves stay out of this.
ArtifactStore._launch_gallery = lambda self: None
initialize_artifact_store(workspace_root=Path(sys.argv[1]))
artifact_focus = getattr(focus_tools.artifact_focus, "fn", focus_tools.artifact_focus)
try:
    print(asyncio.run(artifact_focus(artifact_id=sys.argv[2])))
except ToolError as exc:
    print(json.dumps({"status": "error", "detail": str(exc)}))
"""


def _await_health(port: int, host: subprocess.Popen, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        assert host.poll() is None, "the hosting process exited before its gallery answered"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                if resp.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise AssertionError(f"the gallery never answered /health on port {port}")


@pytest.fixture
def project(tmp_path, reserve_port):
    """A project directory and the environment both of its processes start with."""
    root = tmp_path / "project"
    data_root = root / "agent_data"
    data_root.mkdir(parents=True)
    config = root / "config.yml"
    config.write_text(
        f"project_name: focus-test\nproject_root: {root}\nagent_data:\n  base_dir: agent_data\n",
        encoding="utf-8",
    )

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in (OPERATOR_SECRET_ENV, PANEL_TOKEN_ENV)
        and not key.startswith(f"{OPERATOR_SECRET_ENV}_")
    }
    env["OSPREY_CONFIG"] = str(config)
    env["OSPREY_ARTIFACT_SERVER_PORT"] = str(reserve_port())
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT / "src"), *filter(None, [env.get("PYTHONPATH")])]
    )
    return root, data_root, env


def test_focus_is_admitted_by_a_gallery_another_process_hosts(project):
    root, data_root, env = project
    port = int(env["OSPREY_ARTIFACT_SERVER_PORT"])

    host = subprocess.Popen(
        [sys.executable, "-c", _HOST, str(data_root)],
        cwd=root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert host.stdout is not None
        artifact_id = host.stdout.readline().strip()
        assert artifact_id, "the hosting process saved no artifact"
        _await_health(port, host)

        client = subprocess.run(
            [sys.executable, "-c", _CLIENT, str(data_root), artifact_id],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    finally:
        host.kill()
        host.wait(timeout=30)

    assert client.returncode == 0, client.stderr
    result = json.loads(client.stdout.strip().splitlines()[-1])
    assert result["status"] == "success", result
    assert result["artifact_id"] == artifact_id

    token_file = data_root / PROJECT_PANEL_TOKEN_FILE
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
