"""Tests for the osprey_notebook_update PostToolUse hook.

This hook invalidates cached rendered HTML when a notebook is edited
via NotebookEdit, ensuring the gallery re-renders on next view.

The observable contract is small: delete ``_notebook_cache/{stem}_rendered.html``
when it is there, leave the tree alone when it is not, log which of the two
happened under debug, and never fail the tool call — whatever the input or the
state of the filesystem.
"""

import pytest

HOOK_NAME = "osprey_notebook_update.py"


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
