"""Tests for the artifact-mutation activity listener.

The listener is the single emit site standing in for every artifact-writing
tool, so the properties worth pinning are the ones a per-tool emit would not
have given us for free:

* it registers exactly once even though MCP startup runs repeatedly in-process;
* one ``execute`` run that produces a figure emits ONE frame, not three (the
  auto-saved notebook and the ``code_output`` record are bookkeeping);
* the store callback itself performs no HTTP — the blocking POST happens on the
  worker thread, so ``delete_all`` over a full gallery does not stall the
  caller;
* a process that never registers (gallery, retention sweep) emits nothing.

``notify_agent_activity`` is patched at this module's import site because
``osprey.mcp_server.http`` has two posters and the listener names the notify
helper directly.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import patch

import pytest

from osprey.mcp_server import artifact_activity
from osprey.stores.artifact_store import ArtifactStore

DRAIN_TIMEOUT = 5.0


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Minimal deployed project: config.yml with artifact auto-launch disabled."""
    (tmp_path / "config.yml").write_text(
        "agent_data:\n  base_dir: ./_agent_data\nartifact_server:\n  auto_launch: false\n"
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def clean_registration():
    """Listener registration is process-global — never let it leak between tests."""
    artifact_activity.unregister_artifact_activity_listeners()
    yield
    artifact_activity.unregister_artifact_activity_listeners()


@pytest.fixture
def notified():
    """Patched notify recording ``(kwargs, thread_ident)`` per emitted frame."""
    calls: list[tuple[dict, int]] = []

    def record(**kwargs):
        calls.append((kwargs, threading.get_ident()))

    with patch.object(artifact_activity, "notify_agent_activity", side_effect=record):
        yield calls


def wait_drained(timeout: float = DRAIN_TIMEOUT) -> None:
    """Block until the worker has processed every queued event.

    ``Queue.join()`` has no timeout, so it runs on a throwaway thread and the
    test fails loudly instead of hanging the suite if the worker ever wedges.
    """
    done = threading.Event()

    def join_then_signal():
        artifact_activity._pending.join()
        done.set()

    threading.Thread(target=join_then_signal, daemon=True).start()
    assert done.wait(timeout), "artifact activity queue never drained"


def save_figure(store: ArtifactStore, title: str = "Beam profile") -> None:
    """A deliberate save: the figure the executed code chose to keep."""
    store.save_file(
        file_content=b"\x89PNG fake",
        filename="beam.png",
        artifact_type="image",
        title=title,
        mime_type="image/png",
        tool_source="execute",
    )


def save_run_bookkeeping(store: ArtifactStore) -> None:
    """The two records every ``execute`` run writes on its own."""
    store.save_file(
        file_content=b"{}",
        filename="run.ipynb",
        artifact_type="notebook",
        title="Notebook: plot the orbit",
        mime_type="application/x-ipynb+json",
        tool_source="execute",
    )
    store.save_data(
        tool="execute",
        data={"stdout": "done"},
        title="plot the orbit",
        category="code_output",
    )


@pytest.mark.unit
def test_repeated_startup_registers_exactly_once(project):
    """Startup runs 3x in-process under test; the store appends unconditionally."""
    from osprey.mcp_server.startup import initialize_workspace_singletons

    for _ in range(3):
        initialize_workspace_singletons()

    assert ArtifactStore._listeners.count(artifact_activity._on_artifact_saved) == 1
    assert ArtifactStore._delete_listeners.count(artifact_activity._on_artifact_deleted) == 1


@pytest.mark.unit
def test_execute_run_with_figure_emits_one_frame(project, notified):
    """The flood case: figure + auto-notebook + code_output record → ONE frame."""
    from osprey.mcp_server.startup import initialize_workspace_singletons

    initialize_workspace_singletons()
    store = ArtifactStore(workspace_root=project / "_agent_data")

    save_figure(store)
    save_run_bookkeeping(store)
    wait_drained()

    assert len(notified) == 1
    kwargs, _ = notified[0]
    assert kwargs["kind"] == "artifact"
    assert kwargs["tool"] == "artifact_save"
    assert "Beam profile" in kwargs["detail"]
    assert "execute" in kwargs["detail"]


@pytest.mark.unit
def test_deliberately_saved_notebook_still_emits(project, notified):
    """The filter keys on the auto-save sources, not on the notebook type alone."""
    from osprey.mcp_server.startup import initialize_workspace_singletons

    initialize_workspace_singletons()
    store = ArtifactStore(workspace_root=project / "_agent_data")

    store.save_file(
        file_content=b"{}",
        filename="analysis.ipynb",
        artifact_type="notebook",
        title="Orbit analysis",
        mime_type="application/x-ipynb+json",
        tool_source="artifact_save",
    )
    wait_drained()

    assert len(notified) == 1
    assert "Orbit analysis" in notified[0][0]["detail"]


@pytest.mark.unit
def test_code_output_record_never_emits(project, notified):
    """A ``code_output`` record is bookkeeping whatever wrote it."""
    from osprey.mcp_server.startup import initialize_workspace_singletons

    initialize_workspace_singletons()
    store = ArtifactStore(workspace_root=project / "_agent_data")

    store.save_data(
        tool="execute_file",
        data={"stdout": ""},
        title="script run",
        category="code_output",
    )
    wait_drained()

    assert notified == []


@pytest.mark.unit
def test_delete_all_emits_per_surviving_entry(project, notified):
    """Deletes emit one frame per entry, under the same bookkeeping filter."""
    from osprey.mcp_server.startup import initialize_workspace_singletons

    initialize_workspace_singletons()
    store = ArtifactStore(workspace_root=project / "_agent_data")

    save_figure(store, title="Orbit X")
    save_figure(store, title="Orbit Y")
    save_run_bookkeeping(store)
    wait_drained()
    notified.clear()

    store.delete_all()
    wait_drained()

    assert [kwargs["tool"] for kwargs, _ in notified] == ["artifact_delete"] * 2
    details = " ".join(kwargs["detail"] for kwargs, _ in notified)
    assert "Orbit X" in details and "Orbit Y" in details


@pytest.mark.unit
def test_notify_runs_on_the_worker_not_the_caller(project, notified):
    """The store callback must not do HTTP on the thread that saved."""
    from osprey.mcp_server.startup import initialize_workspace_singletons

    initialize_workspace_singletons()
    store = ArtifactStore(workspace_root=project / "_agent_data")

    save_figure(store)
    wait_drained()

    assert len(notified) == 1
    _, ident = notified[0]
    assert ident != threading.get_ident()


@pytest.mark.unit
def test_delete_all_returns_while_notifies_are_still_blocked(project):
    """A slow web terminal must not stall a gallery-wide delete."""
    from osprey.mcp_server.startup import initialize_workspace_singletons

    initialize_workspace_singletons()
    store = ArtifactStore(workspace_root=project / "_agent_data")

    release = threading.Event()
    entered = threading.Event()

    def block(**_kwargs):
        entered.set()
        release.wait(DRAIN_TIMEOUT)

    with patch.object(artifact_activity, "notify_agent_activity", side_effect=block):
        for i in range(5):
            save_figure(store, title=f"Orbit {i}")
        assert entered.wait(DRAIN_TIMEOUT), "worker never picked up the first event"

        t0 = time.perf_counter()
        store.delete_all()
        elapsed = time.perf_counter() - t0

        release.set()
        # Drain before the patch lifts, so no leftover event reaches the real
        # (network-touching) notify.
        wait_drained()

    # The worker is parked inside a notify for the whole call; delete_all fires
    # the listener five more times and must still return promptly.
    assert elapsed < 1.0, f"delete_all blocked on the notify path ({elapsed:.2f}s)"


@pytest.mark.unit
@pytest.mark.timeout(10)
def test_full_backlog_drops_instead_of_waiting(monkeypatch):
    """A stalled worker must never turn the enqueue into a blocking put.

    The timeout is the real assertion for a ``put_nowait`` → ``put`` regression:
    a blocking put on a full queue with no worker draining it would hang here
    forever, and the guard turns that into a failure. ``pytest-timeout`` is a
    declared dev dependency (pyproject.toml), so the marker is always live.
    """
    from osprey.stores.artifact_store import ArtifactEntry

    monkeypatch.setattr(artifact_activity, "_pending", queue.Queue(maxsize=1))
    entry = ArtifactEntry(
        id="abc",
        artifact_type="image",
        title="Orbit",
        description="",
        filename="abc_orbit.png",
        mime_type="image/png",
        size_bytes=1,
        timestamp="2026-01-01T00:00:00Z",
        tool_source="execute",
    )

    for _ in range(3):
        artifact_activity._on_artifact_saved(entry)

    assert artifact_activity._pending.qsize() == 1


@pytest.mark.unit
def test_unregistered_process_emits_nothing(project, notified):
    """Gallery / retention processes build their own store and never register."""
    store = ArtifactStore(workspace_root=project / "_agent_data")

    save_figure(store)
    store.delete_all()
    wait_drained()

    assert notified == []
