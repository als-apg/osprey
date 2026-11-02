"""The review notebook an ``execute`` ask leaves, and how each answer settles it.

An ``execute`` ask saves the code under review as a notebook that says "Pending
approval". The answer settles that notebook: ``PostToolUse`` says it ran,
``PostToolUseFailure`` says it failed, and a ``Stop`` that finds the ask stamp
still unconsumed says it did not run. The reviewed code and mode are kept; only
the status line and the gallery description change.

The hook runs as a subprocess, the way the harness runs it. The store is read back
from its index on disk, so what is asserted is what the gallery would show.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

from osprey.stores.artifact_store import ArtifactStore
from osprey_connectors_tests._harness import (
    pin_identity,
    state_dir_under,
    write_control_context,
)

HOOK = "osprey_approval.py"
EXECUTE = "mcp__python__execute"
CODE = "x = 1\nprint(x)\n"
REVIEW_TITLE = "Approval preview"
NOT_RUN = "Not run: declined or interrupted at the prompt"

HOOK_CONFIG = {
    "server_prefixes": ["mcp__python__"],
    "approval_prefixes": ["mcp__python__"],
}

APPROVAL_CONFIG = {
    "approval": {"enabled": True, "default_policy": "always"},
    "control_system": {"writes": "allowed"},
}

CONVERSATION = "conv-1111"
OTHER_CONVERSATION = "conv-2222"


@pytest.fixture
def deployment(tmp_path, monkeypatch, make_config):
    """A config, an agent-data root and one pinned identity for the hook and the reader."""
    pin_identity(monkeypatch)
    root = tmp_path / "agent_data"
    state = state_dir_under(root)
    state.mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    # The write verdict asks for a switch-on record; without one every write
    # defers as switch_off before the approval prompt this suite reads.
    write_control_context(root, target="va", switch={"va": "on"})
    config = make_config(APPROVAL_CONFIG)

    class _Deployment:
        state_dir = state
        config_path = config
        cwd = tmp_path

    return _Deployment


def _execute_input(exec_mode: str = "readwrite") -> dict:
    return {"code": CODE, "execution_mode": exec_mode}


def _ask(
    hook_runner,
    deployment,
    tool_use_id,
    *,
    permission_mode="default",
    session=CONVERSATION,
):
    result = hook_runner(
        HOOK,
        EXECUTE,
        _execute_input(),
        config_path=deployment.config_path,
        cwd=deployment.cwd,
        hook_config=HOOK_CONFIG,
        hook_input_extra={
            "hook_event_name": "PreToolUse",
            "tool_use_id": tool_use_id,
            "session_id": session,
            "permission_mode": permission_mode,
        },
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"
    return result


def _after(hook_runner, deployment, event, tool_use_id, *, session=CONVERSATION):
    return hook_runner(
        HOOK,
        EXECUTE,
        _execute_input(),
        config_path=deployment.config_path,
        cwd=deployment.cwd,
        hook_config=HOOK_CONFIG,
        tool_response={"content": "ok"},
        hook_input_extra={
            "hook_event_name": event,
            "tool_use_id": tool_use_id,
            "session_id": session,
        },
    )


def _stop(hook_runner, deployment, *, session=CONVERSATION):
    return hook_runner(
        HOOK,
        None,
        None,
        config_path=deployment.config_path,
        cwd=deployment.cwd,
        hook_config=HOOK_CONFIG,
        hook_input_extra={"hook_event_name": "Stop", "session_id": session},
    )


def _reviews(deployment) -> list[tuple[dict, nbformat.NotebookNode]]:
    """``(index entry, notebook)`` for every review notebook the store holds."""
    found = []
    for index in sorted(Path(deployment.cwd).rglob("artifacts.json")):
        for entry in json.loads(index.read_text(encoding="utf-8"))["entries"]:
            if entry["title"] == REVIEW_TITLE:
                notebook = nbformat.read(index.parent / entry["filename"], as_version=4)
                found.append((entry, notebook))
    return found


def _only_review(deployment) -> tuple[dict, nbformat.NotebookNode]:
    (review,) = _reviews(deployment)
    return review


def _status(notebook: nbformat.NotebookNode) -> str:
    (line,) = [
        line for line in notebook.cells[0].source.splitlines() if line.startswith("**Status:** ")
    ]
    return line.removeprefix("**Status:** ").rstrip()


@pytest.mark.parametrize(
    ("event", "status", "description"),
    [
        pytest.param(
            "PostToolUse",
            "Ran after operator approval",
            "Code ran after operator approval (mode: readwrite)",
            id="ran",
        ),
        pytest.param(
            "PostToolUseFailure",
            "Failed after operator approval",
            "Code failed after operator approval (mode: readwrite)",
            id="failed",
        ),
    ],
)
def test_an_ask_saves_a_pending_review_that_its_outcome_settles_in_place(
    hook_runner, deployment, event, status, description
):
    """The ask saves the review pending; a run or a failed run then rewrites its status.

    The ask stamp names the review, which is how the outcome finds it. Settling changes
    only the status line and the gallery line: the entry keeps its id, timestamp and
    title, and the notebook keeps the reviewed code and mode.
    """
    _ask(hook_runner, deployment, "toolu_review")
    before, pending = _only_review(deployment)
    assert before["description"] == "Code pending approval (mode: readwrite)"
    assert _status(pending) == "Pending approval"
    (stamp_path,) = deployment.state_dir.glob("approval_ask_*.json")
    assert json.loads(stamp_path.read_text(encoding="utf-8"))["review_artifact_id"] == before["id"]

    _after(hook_runner, deployment, event, "toolu_review")

    after, notebook = _only_review(deployment)
    assert _status(notebook) == status
    assert after["description"] == description
    assert (after["id"], after["timestamp"]) == (before["id"], before["timestamp"])
    assert after["title"] == REVIEW_TITLE
    assert notebook.cells[1].cell_type == "code"
    assert notebook.cells[1].source == CODE
    assert "**Mode:** `readwrite`" in notebook.cells[0].source


def test_a_stop_settles_this_conversations_unrun_reviews_as_not_run_and_no_others(
    hook_runner, deployment
):
    """``Stop`` settles the reviews its own conversation left unrun; another's stays pending."""
    _ask(hook_runner, deployment, "toolu_mine")
    _ask(hook_runner, deployment, "toolu_theirs", session=OTHER_CONVERSATION)

    assert _stop(hook_runner, deployment) is None

    settled = sorted(
        (_status(notebook), entry["description"]) for entry, notebook in _reviews(deployment)
    )
    assert settled == [
        (NOT_RUN, "Code not run: declined or interrupted at the prompt (mode: readwrite)"),
        ("Pending approval", "Code pending approval (mode: readwrite)"),
    ]
    assert [p.name for p in deployment.state_dir.glob("approval_ask_*.json")] == [
        "approval_ask_toolu_theirs.json"
    ]


def test_a_dispatch_run_settles_its_review_under_the_policy_approval(
    hook_runner, deployment, monkeypatch
):
    """A dispatch run's ask is answered by the dispatch policy, and its review says so."""
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    # A dispatch run with no owner is refused owner_unscheduled before it is asked.
    monkeypatch.setenv("OSPREY_CONTROL_OWNER", pin_identity(monkeypatch))
    _ask(hook_runner, deployment, "toolu_dispatch")
    result = _after(hook_runner, deployment, "PostToolUse", "toolu_dispatch")

    assert result is None, "a dispatch run's approval is not the operator's, so no note"
    entry, notebook = _only_review(deployment)
    assert _status(notebook) == "Ran after dispatch-policy approval"
    assert entry["description"] == "Code ran after dispatch-policy approval (mode: readwrite)"


def test_a_mode_that_shows_no_prompt_settles_its_review_without_claiming_approval(
    hook_runner, deployment
):
    """Under a permission mode that puts no prompt to a person, the review claims no approver.

    Every mode but ``default`` takes the same branch in the hook (it is not in
    ``_PROMPTED_PERMISSION_MODES``), so ``bypassPermissions`` stands for them all.
    """
    _ask(hook_runner, deployment, "toolu_unprompted", permission_mode="bypassPermissions")
    result = _after(hook_runner, deployment, "PostToolUse", "toolu_unprompted")

    assert result is None
    entry, notebook = _only_review(deployment)
    assert _status(notebook) == "Ran"
    assert entry["description"] == "Code ran (mode: readwrite)"


def test_replace_file_rewrites_the_bytes_and_keeps_the_entry(tmp_path):
    store = ArtifactStore(workspace_root=tmp_path, auto_launch=False)
    saved = store.save_file(b"old", "note.txt", "notebook", "Title", description="before")

    replaced = store.replace_file(
        saved.id, b"new bytes", description="after", metadata={"exec_mode": "readwrite"}
    )

    assert replaced is not None
    assert replaced.id == saved.id
    assert replaced.title == "Title"
    assert replaced.timestamp == saved.timestamp
    assert replaced.size_bytes == len(b"new bytes")
    assert replaced.description == "after"
    assert replaced.metadata == {"exec_mode": "readwrite"}
    assert store.get_file_path(saved.id).read_bytes() == b"new bytes"
    reloaded = ArtifactStore(workspace_root=tmp_path, auto_launch=False).get_entry(saved.id)
    assert reloaded.description == "after"


def test_replace_file_for_an_unknown_id_changes_nothing(tmp_path):
    store = ArtifactStore(workspace_root=tmp_path, auto_launch=False)
    saved = store.save_file(b"keep", "note.txt", "notebook", "Title")

    assert store.replace_file("no-such-id", b"x") is None
    assert store.get_file_path(saved.id).read_bytes() == b"keep"
