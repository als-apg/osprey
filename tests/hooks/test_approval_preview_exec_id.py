"""The approval preview as a run's ``preview`` part, and how its settle is stored.

An ``execute`` ask saves the code under review as a notebook. The hook files it
under the asked call's tool-use id (``exec_id``) with ``metadata.run_part ==
"preview"`` and ``metadata.call_id``, so the executor's later parts join the same
run group. A missing or invalid id leaves ``exec_id`` empty: the preview stands
alone. The answer settles the preview: ``metadata.settled`` is merged in (other
writers' keys survive), and when the executor already stored the run's result
head, the head's ``run_status`` decides what the preview says. A hard deny saves
no preview at all.

The hook runs as a subprocess, the way the harness runs it, with fake hook input.
The store is read back from its index on disk.
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
CONVERSATION = "conv-3333"

HOOK_CONFIG = {
    "server_prefixes": ["mcp__python__"],
    "approval_prefixes": ["mcp__python__"],
}

APPROVAL_CONFIG = {
    "approval": {"enabled": True, "default_policy": "always"},
    "control_system": {"writes": "allowed"},
}


@pytest.fixture
def deployment(tmp_path, monkeypatch, make_config):
    """A config, an agent-data root and one pinned identity for the hook and the reader."""
    pin_identity(monkeypatch)
    root = tmp_path / "agent_data"
    state = state_dir_under(root)
    state.mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    write_control_context(root, target="va", switch={"va": "on"})

    class _Deployment:
        state_dir = state
        cwd = tmp_path
        config_path = make_config(APPROVAL_CONFIG)

    return _Deployment


def _run(hook_runner, deployment, event, tool_use_id, *, code=CODE, config_path=None):
    extra = {"hook_event_name": event, "session_id": CONVERSATION}
    if event == "PreToolUse":
        extra["permission_mode"] = "default"
    if tool_use_id is not None:
        extra["tool_use_id"] = tool_use_id
    return hook_runner(
        HOOK,
        EXECUTE,
        {"code": code, "execution_mode": "readwrite"},
        config_path=config_path or deployment.config_path,
        cwd=deployment.cwd,
        hook_config=HOOK_CONFIG,
        tool_response=None if event == "PreToolUse" else {"content": "ok"},
        hook_input_extra=extra,
    )


def _ask(hook_runner, deployment, tool_use_id):
    result = _run(hook_runner, deployment, "PreToolUse", tool_use_id)
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"
    return result


def _stop(hook_runner, deployment):
    return hook_runner(
        HOOK,
        None,
        None,
        config_path=deployment.config_path,
        cwd=deployment.cwd,
        hook_config=HOOK_CONFIG,
        hook_input_extra={"hook_event_name": "Stop", "session_id": CONVERSATION},
    )


def _indexes(deployment) -> list[Path]:
    return sorted(Path(deployment.cwd).rglob("artifacts.json"))


def _store(deployment) -> ArtifactStore:
    """The store the hook wrote to, opened in this process."""
    (index,) = _indexes(deployment)
    store = ArtifactStore(workspace_root=index.parent.parent, auto_launch=False)
    assert store._index_file == index
    return store


def _previews(deployment) -> list[tuple[dict, nbformat.NotebookNode]]:
    found = []
    for index in _indexes(deployment):
        for entry in json.loads(index.read_text(encoding="utf-8"))["entries"]:
            if entry["title"] == REVIEW_TITLE:
                notebook = nbformat.read(index.parent / entry["filename"], as_version=4)
                found.append((entry, notebook))
    return found


def _only_preview(deployment) -> tuple[dict, nbformat.NotebookNode]:
    (preview,) = _previews(deployment)
    return preview


def _status(notebook: nbformat.NotebookNode) -> str:
    (line,) = [
        line for line in notebook.cells[0].source.splitlines() if line.startswith("**Status:** ")
    ]
    return line.removeprefix("**Status:** ").rstrip()


def _ask_stamp(deployment) -> dict:
    (path,) = deployment.state_dir.glob("approval_ask_*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_an_ask_files_its_preview_under_the_calls_tool_use_id(hook_runner, deployment):
    """The preview carries the call's id as ``exec_id`` and ``call_id``, role ``preview``."""
    _ask(hook_runner, deployment, "toolu_preview1")

    entry, notebook = _only_preview(deployment)
    assert entry["exec_id"] == "toolu_preview1"
    assert entry["metadata"]["call_id"] == "toolu_preview1"
    assert entry["metadata"]["run_part"] == "preview"
    assert "settled" not in entry["metadata"]
    assert _status(notebook) == "Pending approval"

    group = _store(deployment).run_group("toolu_preview1")
    assert group is not None
    assert group.roles == {entry["id"]: "preview"}
    assert group.head_id == entry["id"]
    assert group.in_flight


def test_the_ask_stamp_payload_is_unchanged(hook_runner, deployment):
    """The stamp still names the preview by id and grows no run fields."""
    _ask(hook_runner, deployment, "toolu_stamp1")
    entry, _ = _only_preview(deployment)
    stamp = _ask_stamp(deployment)
    assert stamp["review_artifact_id"] == entry["id"]
    assert stamp["tool_use_id"] == "toolu_stamp1"
    assert "exec_id" not in stamp
    assert "run_part" not in stamp


@pytest.mark.parametrize(
    "tool_use_id",
    [pytest.param(None, id="missing"), pytest.param("../../etc/passwd", id="invalid")],
)
def test_a_missing_or_invalid_tool_use_id_leaves_the_preview_standing_alone(
    hook_runner, deployment, tool_use_id
):
    """No valid id: ``exec_id`` is empty and no ``call_id`` is claimed."""
    _run(hook_runner, deployment, "PreToolUse", tool_use_id)
    entry, _ = _only_preview(deployment)
    assert entry["exec_id"] == ""
    assert "call_id" not in entry["metadata"]
    assert entry["metadata"]["run_part"] == "preview"


@pytest.mark.parametrize(
    ("event", "settled", "status_prefix"),
    [
        pytest.param("PostToolUse", "ran", "Ran", id="ran"),
        pytest.param("PostToolUseFailure", "failed", "Failed", id="failed"),
        pytest.param("Stop", "not_run", "Not run", id="not_run"),
    ],
)
def test_the_settle_stores_settled_per_outcome(
    hook_runner, deployment, event, settled, status_prefix
):
    """Each answer merges its own ``settled`` value and rewrites the status line."""
    _ask(hook_runner, deployment, "toolu_settle1")
    if event == "Stop":
        _stop(hook_runner, deployment)
    else:
        _run(hook_runner, deployment, event, "toolu_settle1")

    entry, notebook = _only_preview(deployment)
    assert entry["metadata"]["settled"] == settled
    assert entry["exec_id"] == "toolu_settle1"
    assert entry["metadata"]["run_part"] == "preview"
    assert _status(notebook).startswith(status_prefix)


def test_a_failure_event_follows_a_stored_not_run_head(hook_runner, deployment):
    """A refusal before launch reports PostToolUseFailure, but the head says not_run.

    The executor stored the run's result head with ``run_status == "not_run"``; the
    preview follows it, in its text and in ``settled``, instead of claiming a failure.
    """
    _ask(hook_runner, deployment, "toolu_refused1")
    store = _store(deployment)
    head = store.save_data(
        tool="execute",
        data={"code": CODE, "error": "refused before launch"},
        title="Execution refused",
        metadata={"run_part": "result", "run_status": "not_run"},
        exec_id="toolu_refused1",
    )

    _run(hook_runner, deployment, "PostToolUseFailure", "toolu_refused1")

    entry, notebook = _only_preview(deployment)
    assert entry["metadata"]["settled"] == "not_run"
    assert _status(notebook) == "Not run: declined or interrupted at the prompt"
    assert entry["description"].startswith("Code not run")
    group = _store(deployment).run_group("toolu_refused1")
    assert group.head_id == head.id
    assert group.run_status == "not_run"


def test_a_success_event_follows_a_stored_failed_head(hook_runner, deployment):
    """A head saying ``failed`` wins over a PostToolUse that reports the call ran."""
    _ask(hook_runner, deployment, "toolu_failedhead")
    _store(deployment).save_data(
        tool="execute",
        data={"code": CODE, "error": "boom"},
        title="Execution failed",
        metadata={"run_part": "result", "run_status": "failed"},
        exec_id="toolu_failedhead",
    )

    _run(hook_runner, deployment, "PostToolUse", "toolu_failedhead")

    entry, notebook = _only_preview(deployment)
    assert entry["metadata"]["settled"] == "failed"
    assert _status(notebook).startswith("Failed")


def test_the_settle_keeps_another_writers_metadata(hook_runner, deployment):
    """``settled`` is merged: keys another writer stamped on the preview survive."""
    _ask(hook_runner, deployment, "toolu_merge1")
    entry, _ = _only_preview(deployment)
    _store(deployment).merge_entry_metadata({entry["id"]: {"transcript_id": "tr-1"}})

    _run(hook_runner, deployment, "PostToolUse", "toolu_merge1")

    after, _ = _only_preview(deployment)
    assert after["metadata"]["transcript_id"] == "tr-1"
    assert after["metadata"]["settled"] == "ran"
    assert after["metadata"]["call_id"] == "toolu_merge1"
    assert after["metadata"]["exec_mode"] == "readwrite"


def test_a_hard_deny_saves_no_preview(hook_runner, deployment, make_config):
    """Store-naming code under ``enabled: false`` is denied outright, and nothing is saved."""
    config_path = make_config(
        {
            "approval": {"enabled": False, "default_policy": "always"},
            "control_system": {"writes": "allowed"},
        }
    )
    result = _run(
        hook_runner,
        deployment,
        "PreToolUse",
        "toolu_denied1",
        code="open('control_context.json')\n",
        config_path=config_path,
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert _previews(deployment) == []
    assert not list(deployment.state_dir.glob("approval_ask_*.json"))
