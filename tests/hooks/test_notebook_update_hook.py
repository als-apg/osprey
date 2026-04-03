"""Tests for the osprey_notebook_update PostToolUse hook.

This hook invalidates cached rendered HTML when a notebook is edited
via NotebookEdit, ensuring the gallery re-renders on next view.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = (
    Path(__file__).parents[2] / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
)


def run_notebook_update_hook(tool_input, cwd=None):
    """Run the notebook update hook as a subprocess."""
    hook_script = HOOKS_DIR / "osprey_notebook_update.py"
    stdin_data = json.dumps(
        {
            "tool_name": "NotebookEdit",
            "tool_input": tool_input,
        }
    )
    result = subprocess.run(
        [sys.executable, str(hook_script)],
        input=stdin_data,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=str(cwd) if cwd else None,
    )
    assert result.returncode == 0, f"Hook failed (exit {result.returncode}): {result.stderr}"
    return result


@pytest.mark.unit
def test_notebook_update_deletes_cache(tmp_path):
    """Hook deletes cached HTML for the edited notebook."""
    # Create a fake notebook and its cached HTML
    nb_path = tmp_path / "test_notebook.ipynb"
    nb_path.write_text("{}")
    cache_dir = tmp_path / "_notebook_cache"
    cache_dir.mkdir()
    cached_html = cache_dir / "test_notebook_rendered.html"
    cached_html.write_text("<html>cached</html>")

    assert cached_html.exists()

    run_notebook_update_hook(
        {"notebook_path": str(nb_path)},
        cwd=tmp_path,
    )

    assert not cached_html.exists()


@pytest.mark.unit
def test_notebook_update_no_cache_no_error(tmp_path):
    """Hook succeeds even when no cache file exists."""
    nb_path = tmp_path / "uncached.ipynb"
    nb_path.write_text("{}")

    result = run_notebook_update_hook(
        {"notebook_path": str(nb_path)},
        cwd=tmp_path,
    )

    assert result.returncode == 0


@pytest.mark.unit
def test_notebook_update_empty_path_no_error(tmp_path):
    """Hook handles empty notebook_path gracefully."""
    result = run_notebook_update_hook(
        {"notebook_path": ""},
        cwd=tmp_path,
    )

    assert result.returncode == 0


@pytest.mark.unit
def test_notebook_update_missing_key_no_error(tmp_path):
    """Hook handles missing notebook_path key gracefully."""
    result = run_notebook_update_hook(
        {},
        cwd=tmp_path,
    )

    assert result.returncode == 0
