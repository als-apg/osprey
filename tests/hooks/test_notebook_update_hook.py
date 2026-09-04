"""Tests for the osprey_notebook_update PostToolUse hook.

This hook invalidates cached rendered HTML when a notebook is edited
via NotebookEdit, ensuring the gallery re-renders on next view, and badges the
NOTEBOOKS panel when the edit landed in the agent's own notebooks tree.

The observable contract is small: delete ``_notebook_cache/{stem}_rendered.html``
when it is there, leave the tree alone when it is not, log which of the two
happened under debug, post exactly one agent-activity frame for an edit under
``<agent-data root>/notebooks/`` and none for an edit anywhere else, and never
fail the tool call — whatever the input, the state of the filesystem, or
whether a web terminal is listening.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

HOOK_NAME = "osprey_notebook_update.py"

#: The agent-data subdirectory the badge fires for. Kept in step with the
#: hook's own ``_NOTEBOOKS_SUBDIR`` and the rendered ``NotebookEdit`` allow.
NOTEBOOKS_SUBDIR = "notebooks"


def snapshot(root):
    """Map every path under ``root`` to its bytes (``None`` for directories)."""
    return {
        path.relative_to(root).as_posix(): (path.read_bytes() if path.is_file() else None)
        for path in root.rglob("*")
    }


@pytest.fixture(autouse=True)
def _project_dir_in_tmp_tree(tmp_path, monkeypatch):
    """Pin the hook's project directory at the per-test tree.

    Leak guarded: the curated hook environment forwards ``CLAUDE_PROJECT_DIR``
    and ``OSPREY_HOOK_DEBUG`` when they are set in the parent process. Left
    ambient, ``log_hook`` would resolve ``hooks.debug`` from the real
    checkout's ``config.yml`` and append its JSONL record under the real
    ``.claude/hooks/``. Debug tests opt back in with ``monkeypatch.setenv``.
    """
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)


@pytest.fixture
def run_notebook_update_hook(hook_runner_raw):
    """Run the hook over a tmp_path project tree, returning its stderr.

    Asserts the two invariants shared by every call: the hook exits 0, and it
    writes nothing to stdout. Stdout is the channel Claude Code parses for a
    hook decision, so a PostToolUse hook that only touches the cache must
    leave it empty.
    """

    def _run(tool_input, cwd):
        returncode, stdout, stderr = hook_runner_raw(
            HOOK_NAME,
            "NotebookEdit",
            tool_input,
            cwd=cwd,
        )
        assert returncode == 0, f"Hook failed (exit {returncode}): {stderr}"
        assert stdout == "", f"Hook wrote to stdout: {stdout!r}"
        return stderr

    return _run


@pytest.mark.unit
def test_notebook_update_deletes_cache(tmp_path, run_notebook_update_hook):
    """Hook deletes the cached HTML for the edited notebook, and only that."""
    nb_path = tmp_path / "test_notebook.ipynb"
    nb_path.write_text("{}")
    cache_dir = tmp_path / "_notebook_cache"
    cache_dir.mkdir()
    cached_html = cache_dir / "test_notebook_rendered.html"
    cached_html.write_text("<html>cached</html>")
    other_html = cache_dir / "other_notebook_rendered.html"
    other_html.write_text("<html>other</html>")

    run_notebook_update_hook({"notebook_path": str(nb_path)}, cwd=tmp_path)

    assert not cached_html.exists()
    # A different notebook's cache and the notebook itself survive.
    assert other_html.read_text() == "<html>other</html>"
    assert nb_path.read_text() == "{}"


@pytest.mark.unit
def test_notebook_update_logs_invalidated_on_cache_hit(
    tmp_path, monkeypatch, run_notebook_update_hook
):
    """Debug logging names the deletion: ``status=invalidated`` plus the path."""
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")
    nb_path = tmp_path / "run_log.ipynb"
    nb_path.write_text("{}")
    cache_dir = tmp_path / "_notebook_cache"
    cache_dir.mkdir()
    (cache_dir / "run_log_rendered.html").write_text("<html>cached</html>")

    stderr = run_notebook_update_hook({"notebook_path": str(nb_path)}, cwd=tmp_path)

    assert "[notebook-update]" in stderr
    assert "status=invalidated" in stderr
    assert f"path={nb_path}" in stderr


@pytest.mark.unit
def test_notebook_update_no_cache_leaves_tree_untouched(tmp_path, run_notebook_update_hook):
    """The cache-miss path is a pure no-op: nothing created, nothing removed."""
    nb_path = tmp_path / "uncached.ipynb"
    nb_path.write_text("{}")
    cache_dir = tmp_path / "_notebook_cache"
    cache_dir.mkdir()
    # Cache exists but holds no entry for this notebook.
    (cache_dir / "different_notebook_rendered.html").write_text("<html>other</html>")
    before = snapshot(tmp_path)

    run_notebook_update_hook({"notebook_path": str(nb_path)}, cwd=tmp_path)

    assert snapshot(tmp_path) == before


@pytest.mark.unit
def test_notebook_update_logs_no_cache_on_cache_miss(
    tmp_path, monkeypatch, run_notebook_update_hook
):
    """Debug logging distinguishes the miss: ``status=no-cache``."""
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")
    nb_path = tmp_path / "uncached.ipynb"
    nb_path.write_text("{}")

    stderr = run_notebook_update_hook({"notebook_path": str(nb_path)}, cwd=tmp_path)

    assert "status=no-cache" in stderr
    assert "status=invalidated" not in stderr


@pytest.mark.unit
def test_notebook_update_empty_path_exits_before_logging(
    tmp_path, monkeypatch, run_notebook_update_hook
):
    """An empty notebook_path returns early — no cache lookup, no log line."""
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")

    stderr = run_notebook_update_hook({"notebook_path": ""}, cwd=tmp_path)

    assert "[notebook-update]" not in stderr


@pytest.mark.unit
def test_notebook_update_missing_key_no_error(tmp_path, run_notebook_update_hook):
    """Hook handles a tool_input with no notebook_path key at all."""
    run_notebook_update_hook({}, cwd=tmp_path)


@pytest.mark.unit
def test_notebook_update_swallows_unlink_failure(tmp_path, run_notebook_update_hook):
    """A cache entry that cannot be unlinked still exits 0 and blocks nothing.

    The cache path is occupied by a directory rather than a file, so it passes
    the ``exists()`` check and then fails the ``unlink()`` — an OSError raised
    from inside the invalidation block. Using a directory rather than a
    read-only parent keeps the failure deterministic: revoked write permission
    is not enforced against a root-owned test runner.
    """
    nb_path = tmp_path / "wedged.ipynb"
    nb_path.write_text("{}")
    blocked = tmp_path / "_notebook_cache" / "wedged_rendered.html"
    blocked.mkdir(parents=True)
    (blocked / "keep.txt").write_text("not a cache file")
    before = snapshot(tmp_path)

    run_notebook_update_hook({"notebook_path": str(nb_path)}, cwd=tmp_path)

    assert snapshot(tmp_path) == before


@pytest.mark.unit
@pytest.mark.parametrize(
    "stdin",
    ["", "{nope", "[]", "[1,2,3]"],
    ids=["empty", "invalid-json", "wrong-shape", "wrong-shape-truthy"],
)
def test_malformed_stdin_fails_open(tmp_path, hook_runner_raw, stdin):
    """Unusable stdin leaves the tree alone and the tool call intact.

    A closed pipe, a truncated write and a non-object payload — falsy (``[]``)
    or truthy (``[1,2,3]``) — name no notebook, so there is no cache entry to
    invalidate: exit 0, nothing on stdout, and not a byte changed under the
    project directory. The truthy payload is the one an emptiness check lets
    through, so it has to be rejected on shape.
    """
    before = snapshot(tmp_path)

    returncode, stdout, stderr = hook_runner_raw(
        HOOK_NAME,
        tool_name=None,
        tool_input=None,
        cwd=tmp_path,
        stdin_override=stdin,
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "Traceback" not in stderr
    assert snapshot(tmp_path) == before


# --- NOTEBOOKS panel badge -------------------------------------------------
#
# An edit inside ``<agent-data root>/notebooks/`` is reported to the web
# terminal as an agent-activity frame, so the NOTEBOOKS rail entry glows and
# badges. The contract is the payload, the tree it fires for, and that a web
# terminal which is not there costs a badge and nothing else.


@pytest.fixture
def activity_server():
    """Stub web terminal recording POST /api/agent-activity bodies.

    Yields ``(port, received)``; ``received`` grows a decoded body per POST,
    appended before the response is written, so a hook subprocess that has
    exited has necessarily already been recorded.
    """
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length", 0))
            received.append(
                {
                    "path": self.path,
                    "body": json.loads(self.rfile.read(length) or b"{}"),
                    "content_type": self.headers.get("Content-Type"),
                }
            )
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):  # silence test output
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], received
    finally:
        server.shutdown()
        server.server_close()


def closed_port():
    """A port with nothing listening (bound momentarily, then released)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_notebook(tmp_path, relpath, subdir=NOTEBOOKS_SUBDIR):
    """Create an empty notebook at ``<agent-data root>/<subdir>/<relpath>``."""
    path = tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / subdir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path


@pytest.mark.unit
@pytest.mark.parametrize(
    "relpath, detail",
    [("scan.ipynb", "scan.ipynb"), ("runs/day2/scan.ipynb", "runs/day2/scan.ipynb")],
    ids=["top-level", "nested"],
)
def test_notebooks_edit_posts_one_activity_frame(
    tmp_path, monkeypatch, run_notebook_update_hook, activity_server, relpath, detail
):
    """An edit under ``notebooks/`` badges the jupyter panel, once, with the path.

    The frame is the route's fixed contract, and ``detail`` is the path
    *relative to* the notebooks root — an absolute path would name a container
    directory the operator never sees.
    """
    port, received = activity_server
    monkeypatch.setenv("OSPREY_WEB_PORT", str(port))
    nb_path = make_notebook(tmp_path, relpath)

    run_notebook_update_hook({"notebook_path": str(nb_path)}, cwd=tmp_path)

    assert len(received) == 1
    assert received[0]["path"] == "/api/agent-activity"
    assert received[0]["content_type"] == "application/json"
    assert received[0]["body"] == {
        "tool": "NotebookEdit",
        "target": {"kind": "panel", "panel": "jupyter", "detail": detail},
    }


@pytest.mark.unit
def test_artifacts_edit_posts_nothing(
    tmp_path, monkeypatch, run_notebook_update_hook, activity_server
):
    """An edit under ``artifacts/`` badges nothing.

    Notebooks the agent writes into the gallery tree belong to WORKSPACE;
    glowing the NOTEBOOKS entry for one would point the operator at a panel
    that does not hold the file.
    """
    port, received = activity_server
    monkeypatch.setenv("OSPREY_WEB_PORT", str(port))
    nb_path = make_notebook(tmp_path, "plot.ipynb", subdir="artifacts")

    run_notebook_update_hook({"notebook_path": str(nb_path)}, cwd=tmp_path)

    assert received == []


@pytest.mark.unit
def test_notebook_outside_agent_data_posts_nothing(
    tmp_path, monkeypatch, run_notebook_update_hook, activity_server
):
    """A notebook edited anywhere else in the project badges nothing either."""
    port, received = activity_server
    monkeypatch.setenv("OSPREY_WEB_PORT", str(port))
    nb_path = tmp_path / "loose.ipynb"
    nb_path.write_text("{}")

    run_notebook_update_hook({"notebook_path": str(nb_path)}, cwd=tmp_path)

    assert received == []


@pytest.mark.unit
def test_badge_survives_down_web_terminal(tmp_path, monkeypatch, run_notebook_update_hook):
    """No web terminal costs a badge, not the tool call — and not the cache.

    The invalidation runs before the emit, so the one observable effect the
    hook owns unconditionally still happens with nothing listening.
    """
    monkeypatch.setenv("OSPREY_WEB_PORT", str(closed_port()))
    nb_path = make_notebook(tmp_path, "scan.ipynb")
    cached_html = nb_path.parent / "_notebook_cache" / "scan_rendered.html"
    cached_html.parent.mkdir()
    cached_html.write_text("<html>cached</html>")

    run_notebook_update_hook({"notebook_path": str(nb_path)}, cwd=tmp_path)

    assert not cached_html.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "token, expected",
    [("panel-secret", "Bearer panel-secret"), ("  ", None), ("", None)],
    ids=["token", "whitespace-only", "empty"],
)
def test_activity_request_carries_the_panel_token(hook_module, monkeypatch, token, expected):
    """The bearer rides along only when the carrier holds a real value.

    A blank ``OSPREY_PANEL_TOKEN`` — an uninterpolated compose variable, a
    hand-edited one — is a credential this hook cannot back, so the header is
    omitted entirely rather than sent empty. Exercised in-process because the
    hook subprocess runs on a curated environment that does not forward the
    carrier.
    """
    hook = hook_module("osprey_notebook_update")
    monkeypatch.setenv("OSPREY_PANEL_TOKEN", token)

    request = hook._activity_request("scan.ipynb")

    assert request.headers.get("Authorization") == expected


@pytest.mark.unit
def test_activity_request_targets_the_configured_web_port(hook_module, monkeypatch):
    """The emit dials ``OSPREY_WEB_PORT`` on loopback."""
    hook = hook_module("osprey_notebook_update")
    monkeypatch.setenv("OSPREY_WEB_PORT", "10999")

    request = hook._activity_request("scan.ipynb")

    assert request.full_url == "http://127.0.0.1:10999/api/agent-activity"
