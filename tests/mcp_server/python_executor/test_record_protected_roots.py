"""Executed python code cannot write under the agent-record roots.

``var/spool`` (what the record is ingested from), ``var/agent_record`` (the
embedded record), ``var/state`` (per-container state, the sidecar's
revocations among it) and ``var/archive`` (the lasting half of the audit
trail, holding the same facts as the record) are written by OSPREY processes
only. Executed code that could write there could forge or erase record
entries, so each is a protected root, refused by both layers: the static path
policy before the run, and the runtime filesystem guard inside the child.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.mcp_server.python_executor.executor import (
    resolve_permitted_roots,
    resolve_protected_roots,
)
from osprey.services.python_executor.execution.path_policy import path_policy_issues
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey.utils.workspace import (
    ARCHIVE_DIR_RELPATH,
    RECORD_DIR_RELPATH,
    SPOOL_DIR_RELPATH,
    STATE_DIR_RELPATH,
)
from osprey_connectors.workspace import BUILD_DIR_NAME

RECORD_ROOTS = (SPOOL_DIR_RELPATH, RECORD_DIR_RELPATH, STATE_DIR_RELPATH, ARCHIVE_DIR_RELPATH)

# The deployment repo root inside the container; the same resolver anchors on it.
CONTAINER_PROJECT_ROOT = Path("/app/demo")


def test_relpaths_are_under_the_state_zone():
    assert SPOOL_DIR_RELPATH == "var/spool"
    assert RECORD_DIR_RELPATH == "var/agent_record"
    assert STATE_DIR_RELPATH == "var/state"
    assert ARCHIVE_DIR_RELPATH == "var/archive"


def test_existing_roots_stay_protected(repo: Path):
    protected = resolve_protected_roots(repo)
    for relpath in (BUILD_DIR_NAME, SPOOL_DIR_RELPATH, RECORD_DIR_RELPATH, STATE_DIR_RELPATH):
        assert (repo / relpath).resolve() in protected


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = (tmp_path / "repo").resolve()
    (root / "var" / "agent_data").mkdir(parents=True)
    (root / "profile.yml").write_text("project: demo\n", encoding="utf-8")
    for relpath in RECORD_ROOTS:
        (root / relpath / "alice").mkdir(parents=True)
        (root / relpath / "alice" / "victim.txt").write_text("original\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("relpath", RECORD_ROOTS)
def test_root_is_protected(repo: Path, relpath: str):
    assert (repo / relpath).resolve() in resolve_protected_roots(repo)


@pytest.mark.parametrize("relpath", RECORD_ROOTS)
def test_root_is_not_permitted(repo: Path, relpath: str):
    target = (repo / relpath).resolve()
    for permitted in resolve_permitted_roots(repo, {}):
        assert not target.is_relative_to(permitted)


@pytest.mark.parametrize("relpath", RECORD_ROOTS)
def test_static_policy_refuses_a_write(repo: Path, relpath: str):
    target = repo / relpath / "alice" / "forged.jsonl"
    issues = path_policy_issues(
        f"open({str(target)!r}, 'a').write('x')",
        protected_roots=resolve_protected_roots(repo),
        permitted_roots=resolve_permitted_roots(repo, {}),
    )
    assert len(issues) == 1
    assert relpath in issues[0]


@pytest.mark.parametrize("relpath", RECORD_ROOTS)
def test_container_root_is_protected(relpath: str):
    root = CONTAINER_PROJECT_ROOT.resolve()
    assert (root / relpath).resolve() in resolve_protected_roots(root)


@pytest.mark.parametrize("relpath", RECORD_ROOTS)
def test_static_policy_refuses_a_container_write(relpath: str):
    root = CONTAINER_PROJECT_ROOT.resolve()
    target = root / relpath / "alice" / "forged.jsonl"
    issues = path_policy_issues(
        f"open({str(target)!r}, 'a').write('x')",
        protected_roots=resolve_protected_roots(root),
        permitted_roots=resolve_permitted_roots(root, {}),
    )
    assert len(issues) == 1
    assert relpath in issues[0]


def _probe(repo: Path, statement: str, execution_mode: str) -> str:
    """Run the real executor guard plus *statement* in a child; return stdout."""
    agent_data = repo / "var" / "agent_data"
    execution_folder = agent_data / "run_0001"
    execution_folder.mkdir(parents=True, exist_ok=True)
    wrapper = ExecutionWrapper(
        execution_mode=execution_mode,
        protected_roots=resolve_protected_roots(repo),
        permitted_roots=(agent_data,),
    )
    body = textwrap.dedent(
        """
        import os
        from pathlib import Path

        try:
            {statement}
        except PermissionError as exc:
            print("REFUSED:", exc)
        else:
            print("ALLOWED")
        """
    ).format(statement=statement)
    script = repo / "probe_script.py"
    script.write_text(
        wrapper._get_filesystem_guard(execution_folder) + "\n" + body + "\n", encoding="utf-8"
    )
    proc = subprocess.run(  # fixed argv, test-authored script
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(repo),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


_WRITE_FORMS = {
    "append": "open({target!r}, 'a').write('forged\\n')",
    "create": "Path({new!r}).write_text('forged')",
    "remove": "os.remove({target!r})",
    "rename": "os.rename({target!r}, {new!r})",
}


@pytest.mark.parametrize("execution_mode", ["readonly", "readwrite"])
@pytest.mark.parametrize("form", sorted(_WRITE_FORMS))
@pytest.mark.parametrize("relpath", RECORD_ROOTS)
def test_runtime_guard_refuses_a_write(repo: Path, relpath: str, form: str, execution_mode: str):
    target = repo / relpath / "alice" / "victim.txt"
    new = repo / relpath / "alice" / "forged.sealed"
    statement = _WRITE_FORMS[form].format(target=str(target), new=str(new))

    out = _probe(repo, statement, execution_mode)

    assert out.startswith("REFUSED"), out
    assert target.read_text(encoding="utf-8") == "original\n"
    assert not new.exists()


def test_runtime_guard_still_allows_the_agent_data_zone(repo: Path):
    target = repo / "var" / "agent_data" / "ok.txt"
    out = _probe(repo, f"Path({str(target)!r}).write_text('fine')", "readwrite")
    assert out == "ALLOWED"
    assert target.read_text(encoding="utf-8") == "fine"
