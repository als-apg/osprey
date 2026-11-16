"""Dispatch runs expose the artifacts they produced — created-by isolation.

A dispatched agent writes plots into the shared artifact store. The worker must
say *which* artifacts a run produced and hand back their (renderable) bytes, so
a bridge can republish them — WITHOUT one run ever reading another run's output.

Association is **created-by**: ``ArtifactEntry.run_id`` is stamped from
``OSPREY_DISPATCH_RUN_ID`` at write time inside the trusted store, and every
artifact surface (the run-status ``artifacts`` list, the list route, and the
byte route) bottoms out in the single strict predicate ``entry.run_id ==
run_id``. Authorization never reads agent-controllable data, so a prompt-injected
run cannot widen its artifact set or reach another run's bytes. This is
deliberately NOT ``OSPREY_SESSION_ID`` (which relocates the whole store root).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.agent_runner import artifact_resolve as resolve_mod
from osprey.stores.artifact_store import ArtifactStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
PDF_BYTES = b"%PDF-1.4\nfake-pdf-body\n%%EOF"
HTML_BYTES = b"<html><body><h1>plot</h1></body></html>"
CSV_BYTES = b"time,state\n2026-09-24T10:00:00,OPEN\n"
TSV_BYTES = b"time\tstate\n2026-09-24T10:00:00\tOPEN\n"
TABLES = [
    pytest.param("text/csv", "t.csv", CSV_BYTES, id="csv"),
    pytest.param("text/tab-separated-values", "t.tsv", TSV_BYTES, id="tsv"),
]
TOKEN = "test-worker-token"


@pytest.fixture(autouse=True)
def _no_artifact_server_autolaunch(monkeypatch):
    """Keep ``save_file``'s companion-server auto-launch out of this module.

    Every save below goes through ``ArtifactStore.save_file``, whose last act is
    to start the artifacts backend if it is not already up. That is a real
    uvicorn on a daemon thread, and its lifespan resolves an agent-data root of
    its own -- not the ``workspace_root`` these tests hand the store. Unstamped,
    that resolves to the checkout, so the thread writes ``var/agent_data`` into
    the repository, outliving the test that started it and failing the session
    on ``no_agent_data_in_the_repo`` under whichever test happened to be running
    at the time. Nothing here asserts on the server, so it is not started.
    """
    from osprey.infrastructure import server_launcher

    monkeypatch.setattr(server_launcher, "ensure_web_server", lambda *a, **k: None)


def _save(
    store: ArtifactStore,
    run_id: str,
    *,
    title: str,
    filename: str,
    mime: str,
    body: bytes,
) -> str:
    """Save one artifact made for the person, tagged with ``run_id`` (via the env the store reads)."""
    with pytest.MonkeyPatch.context() as mp:
        if run_id:
            mp.setenv("OSPREY_DISPATCH_RUN_ID", run_id)
        else:
            mp.delenv("OSPREY_DISPATCH_RUN_ID", raising=False)
        entry = store.save_file(
            file_content=body,
            filename=filename,
            title=title,
            artifact_type="image",
            mime_type=mime,
            deliver=True,
        )
    return entry.id


def _rooted_store(tmp_path: Path, monkeypatch) -> ArtifactStore:
    """Store rooted so ``resolve._get_store()`` reads the same place.

    ``_get_store()`` roots at the ``agent_data.base_dir`` of the config the
    process was pointed at, anchored on ``$OSPREY_PROJECT_DIR`` — with no config
    in reach that is the default ``var/agent_data``. The byte/list/describe
    surfaces all go through it, so tests must write there too.
    """
    project = tmp_path / "project"
    (project / "var" / "agent_data").mkdir(parents=True)
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(project))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    return ArtifactStore(workspace_root=project / "var" / "agent_data")


# ---------------------------------------------------------------------------
# Store-level provenance (Slot 1)
# ---------------------------------------------------------------------------


class TestArtifactEntryRunId:
    def test_save_file_stamps_run_id_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "run-abc")
        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_file(
            file_content=PNG_BYTES,
            filename="p.png",
            title="p",
            artifact_type="image",
            mime_type="image/png",
        )
        assert entry.run_id == "run-abc"

    def test_run_id_defaults_empty_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_DISPATCH_RUN_ID", raising=False)
        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_file(
            file_content=PNG_BYTES,
            filename="p.png",
            title="p",
            artifact_type="image",
            mime_type="image/png",
        )
        assert entry.run_id == ""

    def test_old_index_without_run_id_still_loads(self, tmp_path):
        d = tmp_path / "artifacts"
        d.mkdir(parents=True)
        (d / "artifacts.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "updated": "2026-01-01T00:00:00Z",
                    "entry_count": 1,
                    "entries": [
                        {
                            "id": "old1",
                            "artifact_type": "image",
                            "title": "t",
                            "description": "",
                            "filename": "t.png",
                            "mime_type": "image/png",
                            "size_bytes": 3,
                            "timestamp": "2026-01-01T00:00:00Z",
                            "tool_source": "x",
                        }
                    ],
                }
            )
        )
        store = ArtifactStore(workspace_root=tmp_path)
        entries = store.list_entries()
        assert len(entries) == 1
        assert entries[0].run_id == ""


class TestRunFilter:
    """``list_entries(run_filter=...)`` is the strict created-by boundary."""

    @pytest.fixture
    def store(self, tmp_path) -> ArtifactStore:
        s = ArtifactStore(workspace_root=tmp_path)
        _save(s, "run-1", title="mine", filename="a.png", mime="image/png", body=PNG_BYTES)
        _save(s, "run-1", title="mine2", filename="b.png", mime="image/png", body=PNG_BYTES)
        _save(s, "run-2", title="theirs", filename="c.png", mime="image/png", body=PNG_BYTES)
        _save(s, "", title="legacy", filename="d.png", mime="image/png", body=PNG_BYTES)
        return s

    def test_returns_only_this_runs_entries(self, store):
        got = {e.title for e in store.list_entries(run_filter="run-1")}
        assert got == {"mine", "mine2"}

    def test_excludes_other_runs(self, store):
        titles = {e.title for e in store.list_entries(run_filter="run-1")}
        assert "theirs" not in titles

    def test_excludes_untagged_legacy(self, store):
        titles = {e.title for e in store.list_entries(run_filter="run-1")}
        assert "legacy" not in titles

    def test_empty_run_filter_matches_nothing(self, store):
        # Foot-gun guard: an empty run tag must NOT match untagged artifacts;
        # the run boundary is strict.
        assert store.list_entries(run_filter="") == []

    def test_none_run_filter_returns_all(self, store):
        assert len(store.list_entries(run_filter=None)) == 4


class TestGetRunEntry:
    """The single centralized cross-run gate."""

    @pytest.fixture
    def store(self, tmp_path):
        s = ArtifactStore(workspace_root=tmp_path)
        a1 = _save(s, "run-1", title="a", filename="a.png", mime="image/png", body=PNG_BYTES)
        a2 = _save(s, "run-2", title="b", filename="b.png", mime="image/png", body=PNG_BYTES)
        leg = _save(s, "", title="c", filename="c.png", mime="image/png", body=PNG_BYTES)
        return s, a1, a2, leg

    def test_exact_match_returns_entry(self, store):
        s, a1, _, _ = store
        assert s.get_run_entry("run-1", a1).id == a1

    def test_wrong_run_returns_none(self, store):
        s, _, a2, _ = store
        assert s.get_run_entry("run-1", a2) is None

    def test_untagged_entry_returns_none(self, store):
        s, _, _, leg = store
        assert s.get_run_entry("run-1", leg) is None

    def test_empty_run_id_returns_none(self, store):
        s, _, _, leg = store
        # Even against the untagged entry, an empty run_id must never match.
        assert s.get_run_entry("", leg) is None

    def test_unknown_id_returns_none(self, store):
        s, _, _, _ = store
        assert s.get_run_entry("run-1", "does-not-exist") is None


class TestStoreRooting:
    """resolve._get_store() must root where the agent's tools write."""

    def test_root_follows_project_dir_not_cwd(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        (project / "var" / "agent_data" / "artifacts").mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setenv("OSPREY_PROJECT_DIR", str(project))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        monkeypatch.chdir(elsewhere)
        assert resolve_mod._get_store()._store_dir == project / "var" / "agent_data" / "artifacts"

    def test_root_follows_the_configs_relocated_agent_data_dir(self, tmp_path, monkeypatch):
        """A project that moved ``agent_data.base_dir`` moves the store with it.

        The compose generator renders the worker's volume mount target from this
        same key, so a resolver that ignored it would read a directory the mount
        never covers.
        """
        project = tmp_path / "project"
        (project / "build").mkdir(parents=True)
        (project / "build" / "config.yml").write_text("agent_data:\n  base_dir: state/data\n")
        monkeypatch.setenv("OSPREY_PROJECT_DIR", str(project))
        monkeypatch.setenv("CONFIG_FILE", str(project / "build" / "config.yml"))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        assert resolve_mod._get_store()._store_dir == project / "state" / "data" / "artifacts"


# ---------------------------------------------------------------------------
# Render-free descriptors (Slot 2)
# ---------------------------------------------------------------------------


class TestPredictDelivery:
    def test_image_passthrough(self):
        assert resolve_mod.predict_delivery("image/png") == ("image/png", True)

    def test_pdf_passthrough(self):
        assert resolve_mod.predict_delivery("application/pdf") == ("application/pdf", True)

    def test_html_renders_to_png(self):
        assert resolve_mod.predict_delivery("text/html") == ("image/png", True)

    def test_unknown_renders_to_png(self):
        assert resolve_mod.predict_delivery("application/whatever") == ("image/png", True)


class TestDescribeRunArtifacts:
    def test_unknown_run_is_empty(self, tmp_path, monkeypatch):
        _rooted_store(tmp_path, monkeypatch)
        assert resolve_mod.describe_run_artifacts("nope") == []

    def test_empty_run_id_is_empty(self, tmp_path, monkeypatch):
        _rooted_store(tmp_path, monkeypatch)
        assert resolve_mod.describe_run_artifacts("") == []

    def test_descriptor_shape_and_source_mime(self, tmp_path, monkeypatch):
        store = _rooted_store(tmp_path, monkeypatch)
        _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)
        [d] = resolve_mod.describe_run_artifacts("run-1")
        assert set(d) == {"artifact_id", "filename", "source_mime", "delivered_mime", "convertible"}
        assert d["source_mime"] == "text/html"
        assert d["delivered_mime"] == "image/png"  # rendered
        assert d["filename"].endswith(".png")  # predicted delivered name

    def test_render_free_never_invokes_converter(self, tmp_path, monkeypatch):
        store = _rooted_store(tmp_path, monkeypatch)
        _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)

        async def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("describe must not render")

        import osprey.stores.artifact_converters as conv

        monkeypatch.setitem(conv.CONVERTER_REGISTRY, "text/html", _boom)
        # describe only classifies (get_converter lookup), never calls the converter
        [d] = resolve_mod.describe_run_artifacts("run-1")
        assert d["delivered_mime"] == "image/png"


def _as_run(store: ArtifactStore, run_id: str) -> pytest.MonkeyPatch:
    """A patch context in which the tools' store is *store* and saves are tagged *run_id*."""
    from osprey.stores import artifact_store as store_mod

    mp = pytest.MonkeyPatch()
    mp.setenv("OSPREY_DISPATCH_RUN_ID", run_id)
    mp.setattr(store_mod, "_artifact_store", store)
    return mp


def _execute_call(
    store: ArtifactStore,
    run_id: str,
    description: str,
    *figures: Path,
    saved: tuple[dict, ...] = (),
) -> None:
    """One ``execute`` call, saved by the executor's own response builder.

    It leaves what a real call leaves: a notebook and a result record, one
    entry per figure the code drew and one per ``save_artifact()`` call (*saved*).
    """
    from osprey.mcp_server.python_executor.executor import ExecutionResult
    from osprey.mcp_server.python_executor.tools._response_builder import (
        build_execution_response,
    )

    mp = _as_run(store, run_id)
    try:
        asyncio.run(
            build_execution_response(
                code="print('ok')",
                description=description,
                execution_mode="readonly",
                exec_result=ExecutionResult(
                    success=True,
                    stdout="ok\n",
                    stderr="",
                    figures=list(figures),
                    artifacts=list(saved),
                ),
                patterns={},
                save_output=True,
            )
        )
    finally:
        mp.undo()


def _call_tool(store: ArtifactStore, run_id: str, tool, **arguments) -> dict:
    """Call a workspace tool as the run's agent would; returns its reply."""
    from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

    mp = _as_run(store, run_id)
    try:
        return extract_response_dict(asyncio.run(get_tool_fn(tool)(**arguments)))
    finally:
        mp.undo()


def _approval_preview(store: ArtifactStore, run_id: str) -> str:
    """The review the approval hook saves before a call runs, stamped as the hook stamps it."""
    from osprey.stores.artifact_store import RUN_RECORD_KEY

    mp = _as_run(store, run_id)
    try:
        entry = store.save_file(
            file_content=b"# Review\n",
            filename="review.md",
            title="Review: plot",
            artifact_type="markdown",
            mime_type="text/markdown",
            tool_source="osprey_approval",
            metadata={"exec_mode": "readonly", RUN_RECORD_KEY: True, "run_part": "preview"},
        )
    finally:
        mp.undo()
    return entry.id


def _tool_data_record(store: ArtifactStore, run_id: str, tool: str) -> str:
    """A data record saved the way ``archiver_read`` and its peers save theirs."""
    mp = _as_run(store, run_id)
    try:
        entry = store.save_data(
            tool=tool,
            data={"series": {"SR:DCCT": {"timestamps": [0, 1], "values": [499.8, 500.1]}}},
            title=f"{tool} record",
            description=f"{tool} record",
            summary={"channels": 1},
            access_details={"access_pattern": 'json_data["series"]'},
            metadata={"data_type": "timeseries"},
        )
    finally:
        mp.undo()
    return entry.id


def _figure_file(tmp_path: Path, name: str = "current_vs_time.png") -> Path:
    folder = tmp_path / "execution"
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_bytes(PNG_BYTES)
    return path


def _listed(run_id: str) -> list[str]:
    return [d["artifact_id"] for d in resolve_mod.describe_run_artifacts(run_id)]


def _run_ids(store: ArtifactStore, run_id: str) -> list[str]:
    return [e.id for e in store.list_entries(run_filter=run_id)]


def _plot_question_run(store: ArtifactStore, run_id: str, tmp_path: Path) -> str:
    """The run of #2499: read the archiver, try twice, draw one plot. Returns the plot's id."""
    _tool_data_record(store, run_id, "archiver_read")
    _approval_preview(store, run_id)
    _execute_call(store, run_id, "test tool availability")
    _execute_call(store, run_id, "inspect the series")
    _execute_call(store, run_id, "plot current vs time", _figure_file(tmp_path))
    [plot] = [
        e.id
        for e in store.list_entries(run_filter=run_id)
        if e.metadata.get("run_part") == "output"
    ]
    return plot


class TestOnlyWhatIsForThePersonIsListed:
    """A run lists the entries whose writer said they are for the person, and no other (#2499)."""

    def test_an_archiver_read_and_three_execute_calls_list_exactly_the_plot(
        self, tmp_path, monkeypatch
    ):
        store = _rooted_store(tmp_path, monkeypatch)
        plot = _plot_question_run(store, "run-1", tmp_path)

        # archiver record + preview + 3 x (notebook, result record) + the plot
        assert len(_run_ids(store, "run-1")) == 9
        assert _listed("run-1") == [plot]

    def test_a_writer_that_states_nothing_is_not_listed(self, tmp_path, monkeypatch):
        store = _rooted_store(tmp_path, monkeypatch)
        mp = _as_run(store, "run-1")
        try:
            silent = store.save_file(
                file_content=PNG_BYTES,
                filename="orbit.png",
                title="Orbit",
                artifact_type="image",
                mime_type="image/png",
                tool_source="some_new_tool",
            )
            stated = store.save_file(
                file_content=PNG_BYTES,
                filename="orbit.png",
                title="Orbit",
                artifact_type="image",
                mime_type="image/png",
                tool_source="some_new_tool",
                deliver=True,
            )
        finally:
            mp.undo()

        assert silent.deliver is False
        assert _listed("run-1") == [stated.id]

    @pytest.mark.parametrize(
        "tool", ["archiver_read", "channel_read", "screenshot_capture", "phoebus_snapshot"]
    )
    def test_a_tools_data_record_is_not_listed(self, tmp_path, monkeypatch, tool):
        store = _rooted_store(tmp_path, monkeypatch)
        _tool_data_record(store, "run-1", tool)

        assert _listed("run-1") == []

    def test_no_oversized_channel_reading_is_listed(self, tmp_path, monkeypatch):
        """All three shapes ``channel_read`` stores: a series, an image, a bare array."""
        import numpy as np

        from osprey.mcp_server.control_system.tools import channel_read

        store = _rooted_store(tmp_path, monkeypatch)
        frame = np.zeros((4, 4), dtype=np.uint8)
        mp = _as_run(store, "run-1")
        try:
            channel_read._save_series_reading(store, "RING:WF", np.arange(8.0), {}, np)
            channel_read._save_image_reading(store, "RING:CAM", frame, {}, np, PNG_BYTES)
            channel_read._save_data_only_reading(store, "RING:CUBE", np.zeros((2, 2, 2, 2)), "4-D")
        finally:
            mp.undo()

        assert len(_run_ids(store, "run-1")) == 3
        assert _listed("run-1") == []

    def test_the_filed_answer_text_is_not_listed(self, tmp_path, monkeypatch):
        """``submit_response`` files the answer; the person gets it as the reply's text."""
        from osprey.mcp_server.workspace.tools.submit_response import submit_response

        store = _rooted_store(tmp_path, monkeypatch)
        reply = _call_tool(
            store, "run-1", submit_response, title="Beam current", content="It is 500 mA."
        )

        assert store.get_entry(reply["artifact_id"]) is not None
        assert _listed("run-1") == []

    def test_a_document_lists_its_pdf_and_not_its_source(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch

        from osprey.mcp_server.workspace.tools.create_document import create_document

        store = _rooted_store(tmp_path, monkeypatch)

        def pdflatex(cmd, **kwargs):  # noqa: ARG001 - subprocess.run fixes this signature
            (Path(kwargs.get("cwd", ".")) / "document.pdf").write_bytes(PDF_BYTES)
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("shutil.which", return_value="/usr/bin/pdflatex"),
            patch("subprocess.run", side_effect=pdflatex),
        ):
            reply = _call_tool(
                store,
                "run-1",
                create_document,
                latex_source="\\documentclass{article}\\begin{document}x\\end{document}",
                title="Shift report",
            )

        pdf, source = reply["artifact_ids"]
        assert store.get_entry(source).mime_type == "application/x-tex"
        assert _listed("run-1") == [pdf]

    def test_what_the_code_saved_on_purpose_is_listed(self, tmp_path, monkeypatch):
        """A ``save_artifact()`` notebook is listed; the call's own notebook is not."""
        store = _rooted_store(tmp_path, monkeypatch)
        folder = tmp_path / "execution"
        folder.mkdir()
        kept = folder / "analysis.ipynb"
        kept.write_bytes(b'{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}')
        _execute_call(
            store,
            "run-1",
            "fit the tune",
            saved=(
                {
                    "path": kept,
                    "artifact_type": "notebook",
                    "title": "Notebook: analysis",
                    "description": "",
                    "mime_type": "application/x-ipynb+json",
                },
            ),
        )

        [listed] = _listed("run-1")
        assert store.get_entry(listed).title == "Notebook: analysis"
        assert len(_run_ids(store, "run-1")) == 3

    def test_status_body_and_list_route_name_exactly_the_plot(self, client, tmp_path):
        c, store, _ = client
        plot = _plot_question_run(store, "run-3", tmp_path)

        status_ids = [d["artifact_id"] for d in _list_via_status(c, "run-3")]
        listed = c.get("/dispatch/run-3/artifacts", headers=_auth()).json()

        assert status_ids == [plot]
        assert [d["artifact_id"] for d in listed] == [plot]

    def test_a_record_stays_fetchable_by_id(self, client, monkeypatch):
        """Left out of the list, not locked away: the byte route still serves it."""
        c, store, _ = client
        record = _tool_data_record(store, "run-1", "archiver_read")

        import osprey.stores.artifact_converters as conv

        monkeypatch.setitem(conv.CONVERTER_REGISTRY, "application/json", conv.passthrough)
        r = c.get(f"/dispatch/run-1/artifacts/{record}", headers=_auth())

        assert r.status_code == 200


def _slack_delivery(c, run_id: str, monkeypatch) -> tuple[list[str], list[tuple]]:
    """Deliver *run_id*'s status body through ``SlackOps.deliver_files``.

    The bytes come over the worker's real byte route. Returns the ids fetched
    and the uploads made, each ``(channel, thread_ts, name, data, mime)``.
    """
    from osprey.bridges.core import FetchedArtifact
    from osprey.bridges.core.artifacts import bare_mime
    from osprey.bridges.slack import ops as slack_ops
    from tests.bridges.slack._posting import RecordingClient, make_entry, make_ops

    class UploadClient(RecordingClient):
        def __init__(self) -> None:
            super().__init__()
            self.uploads: list[tuple] = []

        def upload(self, channel, thread_ts, name, data, mime):
            self.uploads.append((channel, thread_ts, name, data, mime))
            return {"ok": True}

    fetched: list[str] = []

    def fetch_from_the_worker(_http, _cfg, run_id, artifact_id, **_):
        fetched.append(artifact_id)
        r = c.get(f"/dispatch/{run_id}/artifacts/{artifact_id}", headers=_auth())
        assert r.status_code == 200
        return FetchedArtifact(r.content, bare_mime(r.headers.get("content-type")))

    monkeypatch.setattr(slack_ops, "fetch_artifact", fetch_from_the_worker)
    slack = UploadClient()
    ops, _ = make_ops(slack)
    ops.deliver_files(
        make_entry(),
        {
            "status": "completed",
            "text_output": "done",
            "run_id": run_id,
            "artifacts": _list_via_status(c, run_id),
        },
    )
    return fetched, slack.uploads


class TestABridgePostsOnlyWhatWasAskedFor:
    """Worker status body -> ``SlackOps.deliver_files`` -> uploads, over the real byte route."""

    def test_slack_uploads_the_plot_and_nothing_else(self, client, tmp_path, monkeypatch):
        from tests.bridges.slack._posting import CHAN, TS

        c, store, _ = client
        plot = _plot_question_run(store, "run-3", tmp_path)

        fetched, uploads = _slack_delivery(c, "run-3", monkeypatch)

        assert fetched == [plot]
        [(channel, thread_ts, name, data, mime)] = uploads
        assert (channel, thread_ts, data, mime) == (CHAN, TS, PNG_BYTES, "image/png")
        assert name.endswith(".png")

    def test_a_camera_image_is_posted_once_the_agent_registers_it(self, client, monkeypatch):
        """Asked to send the camera image: the reading is a record until the agent says send it."""
        import numpy as np

        from osprey.mcp_server.control_system.tools import channel_read
        from osprey.mcp_server.workspace.tools.artifact_register import artifact_register

        c, store, _ = client
        frame = np.zeros((4, 4), dtype=np.uint8)
        mp = _as_run(store, "run-3")
        try:
            handle = channel_read._save_image_reading(store, "RING:CAM", frame, {}, np, PNG_BYTES)
        finally:
            mp.undo()
        picture = store.get_file_path(handle["artifact_id"])

        assert _listed("run-3") == []

        reply = _call_tool(
            store, "run-3", artifact_register, title="Camera image", file_path=str(picture)
        )
        fetched, uploads = _slack_delivery(c, "run-3", monkeypatch)

        assert fetched == [reply["artifact_id"]]
        [(_, _, name, data, mime)] = uploads
        assert (data, mime) == (PNG_BYTES, "image/png")
        assert name.endswith(".png")

    def test_data_is_posted_as_a_file_once_the_agent_registers_a_csv(
        self, client, tmp_path, monkeypatch
    ):
        """The guidance for "send me the data": write a CSV, register it."""
        from osprey.mcp_server.workspace.tools.artifact_register import artifact_register

        c, store, _ = client
        table = tmp_path / "beam_current.csv"
        table.write_text("time,current\n0,500.1\n1,500.2\n")

        reply = _call_tool(
            store, "run-3", artifact_register, title="Beam current", file_path=str(table)
        )
        fetched, uploads = _slack_delivery(c, "run-3", monkeypatch)

        assert fetched == [reply["artifact_id"]]
        [(_, _, name, data, mime)] = uploads
        assert (data, mime) == (table.read_bytes(), "text/csv")
        assert name.endswith(".csv")


class TestResolveSingleRunArtifact:
    async def test_missing_file_is_none(self, tmp_path, monkeypatch):
        store = _rooted_store(tmp_path, monkeypatch)
        gone = _save(store, "run-1", title="g", filename="g.png", mime="image/png", body=PNG_BYTES)
        (store.get_file_path(gone)).unlink()  # tag survives, file removed
        assert await resolve_mod.resolve_single_run_artifact("run-1", gone) is None


# ---------------------------------------------------------------------------
# HTTP surfaces (Slot 3): client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(
    tmp_path,
    monkeypatch,
    worker_record,  # noqa: ARG001 - the worker reads its record
) -> Iterator[tuple[TestClient, ArtifactStore, dict]]:
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", TOKEN)
    store = _rooted_store(tmp_path, monkeypatch)
    ids = {
        "a1": _save(
            store, "run-1", title="mine", filename="a.png", mime="image/png", body=PNG_BYTES
        ),
        "a2": _save(
            store, "run-1", title="mine2", filename="b.png", mime="image/png", body=PNG_BYTES
        ),
        "other": _save(
            store, "run-2", title="theirs", filename="c.png", mime="image/png", body=PNG_BYTES
        ),
        "legacy": _save(
            store, "", title="legacy", filename="d.png", mime="image/png", body=PNG_BYTES
        ),
    }
    from osprey.mcp_server.dispatch_worker import dispatch_api

    monkeypatch.setattr(
        dispatch_api, "_runs", {"run-1": {"status": "completed"}, "run-2": {"status": "completed"}}
    )
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    with TestClient(dispatch_api.app) as c:
        yield c, store, ids


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


class TestByteRouteIsolation:
    """The byte route: identical, non-distinguishing 404 for every deny reason."""

    def test_serves_own_png_bytes(self, client):
        c, _, ids = client
        r = c.get(f"/dispatch/run-1/artifacts/{ids['a1']}", headers=_auth())
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert r.content == PNG_BYTES

    @pytest.mark.parametrize(
        ("run", "artifact"),
        [
            pytest.param("run-1", "other", id="cross-run"),
            pytest.param("run-1", "legacy", id="untagged-legacy"),
            pytest.param("no-such-run", "a1", id="unknown-run"),
        ],
    )
    def test_a_denied_artifact_is_404(self, client, run, artifact):
        c, _, ids = client
        r = c.get(f"/dispatch/{run}/artifacts/{ids[artifact]}", headers=_auth())
        assert r.status_code == 404

    def test_unknown_id_is_404(self, client):
        c, _, _ = client
        r = c.get("/dispatch/run-1/artifacts/does-not-exist", headers=_auth())
        assert r.status_code == 404

    def test_traversal_shaped_id_is_404(self, client):
        c, _, _ = client
        r = c.get("/dispatch/run-1/artifacts/..%2f..%2fetc%2fpasswd", headers=_auth())
        assert r.status_code == 404

    def test_deny_reasons_are_indistinguishable(self, client):
        """Cross-run, unknown-id, and unknown-run all return the SAME 404 body."""
        c, _, ids = client
        bodies = {
            c.get(f"/dispatch/run-1/artifacts/{ids['other']}", headers=_auth()).text,
            c.get("/dispatch/run-1/artifacts/nope", headers=_auth()).text,
            c.get(f"/dispatch/no-run/artifacts/{ids['a1']}", headers=_auth()).text,
        }
        assert len(bodies) == 1  # no existence oracle


class TestInjectionResistance:
    """HEADLINE: a run referencing another run's artifact-id cannot fetch it.

    Under the old referenced-by model, an artifact-id appearing in a run's
    tool_calls[].result would authorize the fetch. Created-by ignores the
    reference entirely — only the write-time tag counts.
    """

    def test_referenced_but_not_created_is_404(self, client, tmp_path):
        c, _, ids = client
        # Plant a persisted run record for run-1 whose results "reference" run-2's
        # artifact (as a prompt-injected agent might emit).
        log_dir = tmp_path / "project" / "var" / "agent_data" / "dispatch"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run-1.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "tool_calls": [{"result": json.dumps({"artifact_id": ids["other"]})}],
                }
            )
        )
        r = c.get(f"/dispatch/run-1/artifacts/{ids['other']}", headers=_auth())
        assert r.status_code == 404  # created-by wins; the reference is irrelevant


class TestListRoute:
    def test_lists_only_own_descriptors(self, client):
        c, _, ids = client
        r = c.get("/dispatch/run-1/artifacts", headers=_auth())
        assert r.status_code == 200
        got = {d["artifact_id"] for d in r.json()}
        assert got == {ids["a1"], ids["a2"]}

    def test_unknown_run_is_empty_not_404(self, client):
        c, _, _ = client
        r = c.get("/dispatch/no-such-run/artifacts", headers=_auth())
        assert r.status_code == 200
        assert r.json() == []

    def test_metadata_only_no_render(self, client, monkeypatch):
        """Listing must not invoke a converter even for renderable artifacts."""
        c, store, _ = client
        _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)
        import osprey.stores.artifact_converters as conv

        async def _boom(*a, **k):  # pragma: no cover
            raise AssertionError("list must not render")

        monkeypatch.setitem(conv.CONVERTER_REGISTRY, "text/html", _boom)
        r = c.get("/dispatch/run-1/artifacts", headers=_auth())
        assert r.status_code == 200
        html_desc = [d for d in r.json() if d["source_mime"] == "text/html"]
        assert html_desc and html_desc[0]["delivered_mime"] == "image/png"

    def test_requires_auth(self, client):
        c, _, _ = client
        assert c.get("/dispatch/run-1/artifacts").status_code in (401, 403)


class TestCoherence:
    """KEYSTONE: status body, list route, and byte route agree on one set."""

    def test_all_surfaces_agree(self, client):
        c, _, ids = client
        # status body carries descriptors for run-1's artifacts
        status_ids = {d["artifact_id"] for d in _list_via_status(c, "run-1")}
        list_ids = {
            d["artifact_id"] for d in c.get("/dispatch/run-1/artifacts", headers=_auth()).json()
        }
        served = {
            aid
            for aid in (ids["a1"], ids["a2"], ids["other"])
            if c.get(f"/dispatch/run-1/artifacts/{aid}", headers=_auth()).status_code == 200
        }
        assert status_ids == list_ids == served == {ids["a1"], ids["a2"]}
        assert ids["other"] not in status_ids  # run-2's artifact appears nowhere under run-1


def _list_via_status(c, run_id):
    """Read the artifact descriptors from the run-status body."""

    # Record the run the way completion does, then read it back over HTTP.
    _record_finished_run(run_id, resolve_mod.describe_run_artifacts(run_id))
    return c.get(f"/dispatch/{run_id}", headers=_auth()).json()["artifacts"]


def _record_finished_run(run_id: str, artifacts: list) -> None:
    """Write the run's accept and terminal entries, as a finished run leaves them."""
    from osprey.mcp_server.dispatch_worker import dispatch_api

    now = time.time()
    dispatch_api._write_run_accept(run_id, {"created_at": now, "session_id": f"s-{run_id}"})
    dispatch_api._write_run_terminal(
        run_id, {"status": "completed", "completed_at": now, "artifacts": artifacts}
    )
    dispatch_api._runs.pop(run_id, None)


class TestStatusBodyFromTheRecord:
    def test_a_run_this_process_no_longer_holds_is_served_from_the_record(self, client):
        c, _, ids = client
        _record_finished_run("run-1", resolve_mod.describe_run_artifacts("run-1"))

        r = c.get("/dispatch/run-1", headers=_auth())

        assert r.status_code == 200
        assert r.json()["status"] == "completed"
        assert {d["artifact_id"] for d in r.json()["artifacts"]} == {ids["a1"], ids["a2"]}

    def test_truly_unknown_run_is_404(self, client):
        c, _, _ = client
        assert c.get("/dispatch/ghost", headers=_auth()).status_code == 404


class TestConversionFidelity:
    def test_pdf_passthrough(self, client):
        c, store, _ = client
        pid = _save(
            store, "run-1", title="doc", filename="d.pdf", mime="application/pdf", body=PDF_BYTES
        )
        r = c.get(f"/dispatch/run-1/artifacts/{pid}", headers=_auth())
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")
        assert r.content == PDF_BYTES

    def test_html_renders_to_png(self, client, monkeypatch):
        c, store, _ = client
        hid = _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)
        import osprey.stores.artifact_converters as conv

        async def _fake_png(source: Path, output_dir: Path) -> Path:
            out = output_dir / f"{source.stem}.png"
            out.write_bytes(PNG_BYTES)
            return out

        monkeypatch.setitem(conv.CONVERTER_REGISTRY, "text/html", _fake_png)
        r = c.get(f"/dispatch/run-1/artifacts/{hid}", headers=_auth())
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert r.content == PNG_BYTES

    def test_converter_failure_falls_back_to_original(self, client, monkeypatch):
        c, store, _ = client
        hid = _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)
        import osprey.stores.artifact_converters as conv

        async def _broken(*a, **k):
            raise RuntimeError("no playwright")

        monkeypatch.setitem(conv.CONVERTER_REGISTRY, "text/html", _broken)
        r = c.get(f"/dispatch/run-1/artifacts/{hid}", headers=_auth())
        assert r.status_code == 200  # graceful: original bytes, not a 500
        assert r.content == HTML_BYTES
        assert r.headers["content-type"].startswith("text/html")


async def _rendering_png(source: Path, output_dir: Path) -> Path:
    out = output_dir / f"{source.stem}.png"
    out.write_bytes(PNG_BYTES)
    return out


class TestTablesAreDeliveredAsFiles:
    @pytest.mark.parametrize("mime", ["text/csv", "text/tab-separated-values"])
    def test_predicted_as_the_file_itself(self, mime):
        assert resolve_mod.predict_delivery(mime) == (mime, True)

    @pytest.mark.parametrize("mime,filename,body", TABLES)
    def test_descriptor_keeps_the_stored_name(self, tmp_path, monkeypatch, mime, filename, body):
        store = _rooted_store(tmp_path, monkeypatch)
        _save(store, "run-1", title="table", filename=filename, mime=mime, body=body)
        [d] = resolve_mod.describe_run_artifacts("run-1")
        assert d["delivered_mime"] == mime
        assert d["filename"].endswith(filename)

    @pytest.mark.parametrize("mime,filename,body", TABLES)
    def test_byte_route_serves_the_original_bytes(self, client, monkeypatch, mime, filename, body):
        import osprey.stores.artifact_converters as conv

        # A working renderer, so a table that reached it would come back as a PNG.
        monkeypatch.setattr(conv, "text_to_png", _rendering_png)
        c, store, _ = client
        tid = _save(store, "run-1", title="table", filename=filename, mime=mime, body=body)
        r = c.get(f"/dispatch/run-1/artifacts/{tid}", headers=_auth())
        assert r.status_code == 200
        assert r.headers["content-type"].split(";")[0] == mime
        assert r.content == body

    async def test_a_logbook_attachment_still_renders_a_table(self, tmp_path, monkeypatch):
        import osprey.stores.artifact_converters as conv

        monkeypatch.setattr(conv, "text_to_png", _rendering_png)
        store = _rooted_store(tmp_path, monkeypatch)
        tid = _save(
            store, "run-1", title="table", filename="t.csv", mime="text/csv", body=CSV_BYTES
        )
        path = await resolve_mod.resolve_artifact_path(store, tid, tmp_path)
        assert path.endswith(".png")


class TestAuthMatrix:
    def test_all_artifact_routes_require_token(self, client):
        c, _, ids = client
        for path in (
            "/dispatch/run-1",
            "/dispatch/run-1/artifacts",
            f"/dispatch/run-1/artifacts/{ids['a1']}",
        ):
            assert c.get(path).status_code in (401, 403), path

    def test_unset_worker_token_fails_closed(self, client, monkeypatch):
        c, _, ids = client
        monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
        r = c.get(f"/dispatch/run-1/artifacts/{ids['a1']}", headers={"Authorization": "Bearer x"})
        assert r.status_code == 500  # never serve when the gate is misconfigured

    def test_health_is_open(self, client):
        c, _, _ = client
        assert c.get("/health").status_code == 200


class TestDeniedToolsInvariant:
    """The created-by tag is only spoof-proof while the agent has no shell.

    Binds that load-bearing assumption to a test: if a future change re-enabled
    a shell/exec tool for dispatch, an agent could forge OSPREY_DISPATCH_RUN_ID
    and mis-tag its artifacts. Keep these denied.
    """

    def test_shell_tools_denied(self):
        from osprey.agent_runner.tool_names import DISPATCH_DENIED_TOOLS

        assert {"Bash", "TaskStop"} <= set(DISPATCH_DENIED_TOOLS)


# ---------------------------------------------------------------------------
# Terminal-state contract: every finished result carries artifacts (Slot 3)
# ---------------------------------------------------------------------------


async def _drive_failing_task(monkeypatch, tmp_path, *, kind: str) -> dict:
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    from osprey.mcp_server.dispatch_worker import dispatch_api

    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})

    if kind == "exception":

        async def _run(**kw):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _run)
        await dispatch_api._run_dispatch_task(
            "run-x", dispatch_api.DispatchRequest(prompt="p", allowed_tools=[])
        )
    elif kind == "timeout":
        monkeypatch.setattr(dispatch_api, "DISPATCH_TIMEOUT_SEC", 0)

        async def _slow(**kw):
            await asyncio.sleep(1)

        monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _slow)
        await dispatch_api._run_dispatch_task(
            "run-x", dispatch_api.DispatchRequest(prompt="p", allowed_tools=[])
        )
    elif kind == "cancel":

        async def _hang(**kw):
            await asyncio.sleep(10)

        monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _hang)
        task = asyncio.create_task(
            dispatch_api._run_dispatch_task(
                "run-x", dispatch_api.DispatchRequest(prompt="p", allowed_tools=[])
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    return dispatch_api._runs["run-x"]


class TestTerminalStateContract:
    @pytest.mark.parametrize("kind", ["exception", "timeout", "cancel"])
    async def test_failed_result_has_empty_artifacts(self, monkeypatch, tmp_path, kind):
        result = await _drive_failing_task(monkeypatch, tmp_path, kind=kind)
        assert result["status"] == "error"
        assert result["artifacts"] == []
