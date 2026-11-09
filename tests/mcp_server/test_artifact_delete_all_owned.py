"""``artifact_delete_all`` deletes only what the current session owns.

At every scope, ``"everything"`` included, another session's artifacts, the
shipped example, watchdog judgements and dispatch-run output survive. A form
spares its extension types. A process with no current transcript (a dispatch
run, or no session at all) deletes nothing and answers success with a reason.
The scope is resolved from the store's own workspace root.
"""

from __future__ import annotations

import json
import queue

import pytest

from osprey.stores import session_ledger, type_registry
from osprey.stores.artifact_store import (
    EXAMPLE_ORIGIN,
    WATCHDOG_ORIGIN,
    initialize_artifact_store,
)
from osprey.stores.type_registry import ExtensionType
from tests.mcp_server.conftest import get_tool_fn

EXT_KEY = "beam_study"
OTHER_KEY = "22222222-2222-4222-8222-222222222222"
OTHER_STAMPED = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
EXEC = "toolu_run1"

SCOPES = ["everything", "plots", "documents", "runs", "trends", "data", EXT_KEY]


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    types = {EXT_KEY: ExtensionType(key=EXT_KEY, label="Beam studies", base_form="documents")}
    monkeypatch.setattr(type_registry, "_EXTENSION_TYPES", types)
    return types


@pytest.fixture(autouse=True)
def _drop_queued_frames():
    """Drop the summary frames these deletes queue on the process-global queue.

    ``notify_bulk_delete`` enqueues whether or not the activity worker runs; a
    frame left behind would be drained into a later test's patched notify.
    """
    from osprey.mcp_server import artifact_activity

    yield
    while True:
        try:
            artifact_activity._pending.get_nowait()
        except queue.Empty:
            break
        artifact_activity._pending.task_done()


@pytest.fixture
def store(tmp_path):
    ws = tmp_path / "_agent_data"
    (ws / "artifacts").mkdir(parents=True)
    return initialize_artifact_store(workspace_root=ws)


@pytest.fixture
def delete_all():
    from osprey.mcp_server.workspace.tools.artifact_register import artifact_delete_all

    fn = get_tool_fn(artifact_delete_all)

    async def call(scope: str) -> dict:
        return json.loads(await fn(scope))

    return call


def _one_of_each_form(store, stamp: str | None = None) -> dict[str, str]:
    """A plot, a document, an extension-type document, a trend, a dataset and a run.

    Saved under the ambient session tag; *stamp* names another owning transcript.
    """
    meta = {"transcript_id": stamp} if stamp else {}
    exec_id = f"{EXEC}-{stamp or 'mine'}"
    ids = {}
    ids["plot"] = store.save_file(
        b"png", "o.png", "image", "Orbit", mime_type="image/png", metadata=dict(meta)
    ).id
    ids["doc"] = store.save_file(
        b"# n", "n.md", "markdown", "Note", mime_type="text/markdown", metadata=dict(meta)
    ).id
    ids["ext"] = store.save_file(
        b"# s",
        "s.md",
        "markdown",
        "Study",
        mime_type="text/markdown",
        kind=EXT_KEY,
        metadata=dict(meta),
    ).id
    ids["trend"] = store.save_data(
        "archiver_read", {"series": {}}, title="Trend", metadata=dict(meta)
    ).id
    ids["data"] = store.save_data("lookup", {"series": {}}, title="Dataset", metadata=dict(meta)).id
    ids["head"] = store.save_data(
        "execute",
        {"stdout": "ok"},
        title="Run",
        exec_id=exec_id,
        metadata={**meta, "run_part": "result", "run_status": "succeeded"},
    ).id
    ids["figure"] = store.save_file(
        b"png",
        "f.png",
        "image",
        "Figure",
        mime_type="image/png",
        exec_id=exec_id,
        metadata={**meta, "run_part": "output", "figure": True},
    ).id
    return ids


def _foreign(store, web_session) -> set[str]:
    """Everything the current session does not own."""
    assert session_ledger.append(web_session.state_dir, OTHER_KEY, OTHER_KEY, session_ledger.EPOCH)
    ids = set(_one_of_each_form(store, stamp=OTHER_STAMPED).values())
    ids.add(store.save_file(b"png", "k.png", "image", "Other key", session_id=OTHER_KEY).id)
    ids.add(
        store.save_file(
            b"<h1/>", "ex.html", "html", "Example", origin=EXAMPLE_ORIGIN, session_id=""
        ).id
    )
    ids.add(store.save_file(b"{}", "judge.json", "json", "Judgement", origin=WATCHDOG_ORIGIN).id)
    ids.add(store.save_file(b"png", "d.png", "image", "Dispatched", run_id="run-1").id)
    return ids


def _left(store) -> set[str]:
    return {e.id for e in store.list_entries()}


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", SCOPES)
async def test_every_scope_spares_what_this_session_does_not_own(
    store, web_session, delete_all, scope
):
    mine = _one_of_each_form(store)
    foreign = _foreign(store, web_session)

    out = await delete_all(scope)

    assert out["status"] == "success"
    assert out["scope"] == scope
    assert out["session"] == web_session.transcript
    assert "reason" not in out
    assert foreign <= _left(store)
    assert set(out["artifact_ids"]) <= set(mine.values())
    assert out["deleted_count"] == len(out["artifact_ids"]) > 0


@pytest.mark.asyncio
async def test_everything_takes_every_form_this_session_owns(store, web_session, delete_all):
    mine = _one_of_each_form(store)
    foreign = _foreign(store, web_session)

    out = await delete_all("everything")

    assert set(out["artifact_ids"]) == set(mine.values())
    assert _left(store) == foreign


@pytest.mark.asyncio
@pytest.mark.usefixtures("web_session")
async def test_a_form_spares_its_extension_types(store, delete_all):
    mine = _one_of_each_form(store)

    out = await delete_all("documents")

    assert out["artifact_ids"] == [mine["doc"]]
    assert mine["ext"] in _left(store)


@pytest.mark.asyncio
async def test_an_extension_key_takes_only_this_sessions_type(store, web_session, delete_all):
    mine = _one_of_each_form(store)
    foreign = _foreign(store, web_session)

    out = await delete_all(EXT_KEY)

    assert out["artifact_ids"] == [mine["ext"]]
    assert foreign <= _left(store)


@pytest.mark.asyncio
@pytest.mark.usefixtures("web_session")
async def test_a_dispatch_process_deletes_nothing(store, delete_all, monkeypatch):
    mine = _one_of_each_form(store)
    monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "run-7")
    before = _left(store)

    out = await delete_all("everything")

    assert out["status"] == "success"
    assert out["deleted_count"] == 0
    assert out["artifact_ids"] == []
    assert out["session"] is None
    assert "dispatch" in out["reason"]
    assert _left(store) == before
    assert set(mine.values()) <= before


@pytest.mark.asyncio
@pytest.mark.usefixtures("web_session")
async def test_no_session_deletes_nothing_and_says_why(store, delete_all, monkeypatch):
    _one_of_each_form(store)
    monkeypatch.delenv("OSPREY_SESSION_ID")
    before = _left(store)

    out = await delete_all("everything")

    assert out["status"] == "success"
    assert out["deleted_count"] == 0
    assert out["session"] is None
    assert out["reason"]
    assert _left(store) == before


@pytest.mark.asyncio
@pytest.mark.usefixtures("web_session")
async def test_a_key_without_a_ledger_row_deletes_nothing(store, delete_all, monkeypatch):
    _one_of_each_form(store)
    monkeypatch.setenv("OSPREY_SESSION_ID", "99999999-9999-4999-8999-999999999999")
    before = _left(store)

    out = await delete_all("plots")

    assert out["deleted_count"] == 0
    assert out["reason"]
    assert _left(store) == before


@pytest.mark.asyncio
async def test_a_cli_conversation_resolves_from_the_stores_root(store, cli_session, delete_all):
    """The pointer lives under the store's root, not the default agent-data root."""
    tag = "cli:" + cli_session.transcript
    mine = store.save_file(b"png", "o.png", "image", "Orbit", session_id=tag)
    other = store.save_file(b"png", "p.png", "image", "Other", session_id="cli:someone-else")

    out = await delete_all("everything")

    assert out["session"] == cli_session.transcript
    assert out["artifact_ids"] == [mine.id]
    assert _left(store) == {other.id}


@pytest.mark.asyncio
async def test_the_docstring_names_the_by_id_route():
    from osprey.mcp_server.workspace.server import mcp
    from osprey.mcp_server.workspace.tools import artifact_register  # noqa: F401  (registers)

    description = (await mcp.get_tool("artifact_delete_all")).description
    assert "this session" in description.lower()
    assert "artifact_delete" in description
