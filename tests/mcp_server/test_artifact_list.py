"""Tests for the artifact_list MCP tool (contract C5).

``scope="session"`` lists exactly what the current transcript owns (the
sidebar row); ``scope="all"`` lists every operator-visible row except the
shipped example and names each row's owner. Rows are compact, one per run,
newest first, capped by ``last_n``.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from osprey.stores.artifact_store import EXAMPLE_ORIGIN, WATCHDOG_ORIGIN, initialize_artifact_store
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

OTHER_KEY = "22222222-2222-4222-8222-222222222222"
NEW_TRANSCRIPT = "33333333-3333-4333-8333-333333333333"


@pytest.fixture
def store(workspace_dir):
    """A store whose root is the CLI pointer root (``tmp_path/_agent_data``)."""
    return initialize_artifact_store(workspace_root=workspace_dir)


@pytest.fixture
def list_tool():
    from osprey.mcp_server.workspace.tools.artifact_query import artifact_list

    return get_tool_fn(artifact_list)


def _plot(store, title="Orbit", **kw):
    return store.save_file(
        b"png",
        f"{title}.png",
        "image",
        title,
        mime_type="image/png",
        tool_source="create_static_plot",
        **kw,
    )


def _data(store, title="Reading", tool="channel_read", **kw):
    return store.save_data(
        tool=tool,
        data={"value": 42},
        title=title,
        description="first line\nsecond line with detail",
        summary={"channels": ["SR:DCCT"]},
        **kw,
    )


def _run(store, exec_id, stdout="ok"):
    """A run: head (stdout in its data), notebook, figure."""
    head = store.save_data(
        "execute",
        {"stdout": stdout, "code": "plt.plot(x)"},
        title=f"Run {exec_id}",
        exec_id=exec_id,
        metadata={"run_part": "result", "run_status": "succeeded"},
    )
    store.save_file(
        b"{}",
        "run.ipynb",
        "notebook",
        "Notebook",
        tool_source="execute",
        exec_id=exec_id,
        metadata={"run_part": "code"},
    )
    store.save_file(
        b"png",
        "fig.png",
        "image",
        "Figure",
        mime_type="image/png",
        tool_source="execute",
        exec_id=exec_id,
        metadata={"run_part": "output", "figure": True},
    )
    return head


def _ids(out):
    return [e["artifact_id"] for e in out["entries"]]


async def _list(fn, **kw):
    return json.loads(await fn(**kw))


# ---- web session: the default listing is the sidebar row (reqs 1, 2) --------


@pytest.mark.asyncio
async def test_session_lists_only_this_transcripts_rows(store, list_tool, web_session):
    """Req 1: another session's artifact is not in the default listing."""
    mine = _plot(store, "Mine")
    _plot(store, "Theirs", session_id=OTHER_KEY)
    out = await _list(list_tool)
    assert out["scope"] == "session"
    assert out["session"] == web_session.transcript
    assert out["ledger"] == "ok"
    assert "reason" not in out
    assert _ids(out) == [mine.id]


@pytest.mark.asyncio
async def test_after_clear_the_listing_starts_empty_then_lists_the_first_turn(
    store, list_tool, web_session
):
    """Req 2: /clear empties the default listing; the next save is listed; the old is in all."""
    before = _plot(store, "Before")
    web_session.clear(NEW_TRANSCRIPT)

    assert _ids(await _list(list_tool)) == []

    after = _plot(store, "After")
    out = await _list(list_tool)
    assert out["session"] == NEW_TRANSCRIPT
    assert _ids(out) == [after.id]

    everything = await _list(list_tool, scope="all")
    owners = {e["artifact_id"]: e["owner"] for e in everything["entries"]}
    assert owners[before.id] == web_session.key  # a key's first transcript id is the key
    assert owners[after.id] == NEW_TRANSCRIPT


@pytest.mark.asyncio
async def test_a_resumed_seeded_session_lists_its_seeded_artifacts(store, list_tool, web_session):
    """Req 3: seeded artifacts stamped with this transcript are this session's;
    another session sees them only under scope="all", owned by the seed's transcript."""
    seeded = _plot(store, "Seeded", metadata={"transcript_id": web_session.transcript})
    assert _ids(await _list(list_tool)) == [seeded.id]

    web_session.clear(NEW_TRANSCRIPT)
    assert _ids(await _list(list_tool)) == []
    row = (await _list(list_tool, scope="all"))["entries"][0]
    assert row["artifact_id"] == seeded.id
    assert row["owner"] == web_session.key


@pytest.mark.usefixtures("web_session")
@pytest.mark.asyncio
async def test_watchdog_dispatch_and_example_never_list(store, list_tool):
    """Reqs 4 and 5: neither scope lists watchdog, dispatch or the shipped example."""
    mine = _plot(store, "Mine")
    _plot(store, "Watchdog", origin=WATCHDOG_ORIGIN)
    _plot(store, "Dispatch", run_id="run-1")
    _plot(store, "Example", origin=EXAMPLE_ORIGIN)

    assert _ids(await _list(list_tool)) == [mine.id]
    assert _ids(await _list(list_tool, scope="all")) == [mine.id]


@pytest.mark.asyncio
async def test_all_names_every_owner_and_null_for_unowned(store, list_tool, web_session):
    """Req 5: scope="all" lists every visible artifact, each naming its owner or null."""
    from osprey.stores import session_ledger

    session_ledger.append(web_session.state_dir, OTHER_KEY, OTHER_KEY, session_ledger.EPOCH)
    from osprey_connectors import posture_store

    posture_store.invalidate_cache()
    mine = _plot(store, "Mine")
    theirs = _plot(store, "Theirs", session_id=OTHER_KEY)
    untagged = _plot(store, "Untagged", session_id="")
    cli = _plot(store, "Cli", session_id="cli:dddddddd-dddd-4ddd-8ddd-dddddddddddd")

    out = await _list(list_tool, scope="all")
    assert out["scope"] == "all"
    assert out["ledger"] == "ok"
    owners = {e["artifact_id"]: e["owner"] for e in out["entries"]}
    assert owners == {
        mine.id: web_session.transcript,
        theirs.id: OTHER_KEY,
        untagged.id: None,
        cli.id: None,  # req 16: a CLI conversation has no owner in a web listing
    }


# ---- unresolved session (req 6) ----------------------------------------------


@pytest.mark.usefixtures("web_session")
@pytest.mark.asyncio
async def test_no_session_answers_empty_with_a_reason(store, list_tool, monkeypatch):
    """Req 6: a process with no session key and no CLI conversation lists nothing and says why."""
    _plot(store, "Something")
    monkeypatch.delenv("OSPREY_SESSION_ID")
    out = await _list(list_tool)
    assert out["entries"] == []
    assert out["session"] is None
    assert out["reason"]
    assert out["total"] == 0 and out["form_totals"] == []


@pytest.mark.usefixtures("web_session")
@pytest.mark.asyncio
async def test_a_key_without_a_ledger_row_answers_empty(store, list_tool, monkeypatch):
    """Req 6: a key the ledger does not know never falls back to another transcript."""
    _plot(store, "Mine")
    monkeypatch.setenv("OSPREY_SESSION_ID", OTHER_KEY)
    out = await _list(list_tool)
    assert out["entries"] == [] and out["reason"]


@pytest.mark.usefixtures("web_session")
@pytest.mark.asyncio
async def test_a_dispatch_run_is_told_to_use_its_ids(store, list_tool, monkeypatch):
    """Req 6: a dispatch run answers empty; the reason points it at the ids it was given."""
    _plot(store, "Mine")
    monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "run-7")
    out = await _list(list_tool)
    assert out["entries"] == []
    assert "ids" in out["reason"]


@pytest.mark.asyncio
async def test_an_unreadable_ledger_answers_unavailable(store, list_tool, web_session):
    """An unreadable ledger lists nothing for the session and says so; all still lists rows."""
    entry = _plot(store, "Mine")
    from osprey.stores import session_ledger
    from osprey_connectors import posture_store

    ledger_file = web_session.state_dir / session_ledger.STORE_FILENAME
    assert ledger_file.exists()
    ledger_file.write_text("not json", encoding="utf-8")
    posture_store.invalidate_cache()

    out = await _list(list_tool)
    assert out["ledger"] == "unavailable"
    assert out["entries"] == [] and out["reason"]

    everything = await _list(list_tool, scope="all")
    assert everything["ledger"] == "unavailable"
    assert everything["entries"][0]["artifact_id"] == entry.id
    assert everything["entries"][0]["owner"] is None


# ---- rows, cap, totals (reqs 9, 11) -----------------------------------------


@pytest.mark.usefixtures("web_session")
@pytest.mark.asyncio
async def test_rows_are_compact_one_per_run_newest_first(store, list_tool):
    """Req 9: rows carry no code or stdout; a run lists once (head), its parts fold away."""
    first = _data(store, "First")
    head = _run(store, "toolu_run1", stdout="SECRET-STDOUT")
    out = await _list(list_tool)

    # Newest first: the run's figure (saved last), its head, then the reading.
    assert _ids(out)[1:] == [head.id, first.id]
    forms = {e["artifact_id"]: e["form"] for e in out["entries"]}
    assert forms[head.id] == "runs"
    assert forms[first.id] == "trends"
    # The run's figure is one of its outputs and is listed; its notebook is a part.
    assert len(out["entries"]) == 3
    text = json.dumps(out)
    assert "SECRET-STDOUT" not in text and "plt.plot" not in text

    row = next(e for e in out["entries"] if e["artifact_id"] == first.id)
    assert set(row) == {
        "artifact_id",
        "title",
        "form",
        "form_label",
        "tool",
        "timestamp",
        "description",
        "channels",
    }
    assert row["description"] == "first line"
    assert row["channels"] == ["SR:DCCT"]
    assert row["form_label"] == "Trends"
    assert row["tool"] == "channel_read"


@pytest.mark.usefixtures("web_session")
@pytest.mark.asyncio
async def test_default_cap_is_twenty_runs_and_reports_the_cut(store, list_tool):
    """Req 9: 25 runs list 20 by default; total/omitted report the cut; last_n raises it."""
    heads = [
        store.save_data(
            "execute",
            {"stdout": "x"},
            title=f"Run {i}",
            exec_id=f"toolu_{i}",
            metadata={"run_part": "result", "run_status": "succeeded"},
        )
        for i in range(25)
    ]
    out = await _list(list_tool)
    assert len(out["entries"]) == 20
    assert out["total"] == 25 and out["omitted"] == 5
    assert _ids(out)[0] == heads[-1].id
    assert out["form_totals"] == [{"key": "runs", "label": "Runs", "count": 25}]

    raised = await _list(list_tool, last_n=200)
    assert len(raised["entries"]) == 25 and raised["omitted"] == 0


@pytest.mark.usefixtures("web_session")
@pytest.mark.asyncio
async def test_form_totals_count_the_whole_scope_before_narrowing(store, list_tool):
    """Totals count every row in scope; total/omitted count after form/tool and the cap."""
    _plot(store, "P1")
    _plot(store, "P2")
    _data(store, "D1")
    _data(store, "D2", tool="archiver_read")

    out = await _list(list_tool, form="plots", last_n=1)
    totals = {t["key"]: t["count"] for t in out["form_totals"]}
    assert totals == {"plots": 2, "trends": 2}
    assert out["total"] == 2 and out["omitted"] == 1
    assert len(out["entries"]) == 1

    by_tool = await _list(list_tool, tool="archiver_read")
    assert [e["title"] for e in by_tool["entries"]] == ["D2"]
    assert by_tool["total"] == 1 and by_tool["omitted"] == 0


@pytest.mark.usefixtures("store", "web_session")
@pytest.mark.asyncio
@pytest.mark.parametrize("last_n", [0, -1, 201])
async def test_last_n_out_of_range_is_a_validation_error(list_tool, last_n):
    with assert_raises_error(error_type="validation_error"):
        await list_tool(last_n=last_n)


@pytest.mark.usefixtures("store", "web_session")
@pytest.mark.asyncio
async def test_an_unknown_scope_is_a_validation_error(list_tool):
    with assert_raises_error(error_type="validation_error"):
        await list_tool(scope="everything")


def test_dropped_filters_are_gone():
    import inspect

    from osprey.mcp_server.workspace.tools.artifact_query import artifact_list

    params = set(inspect.signature(get_tool_fn(artifact_list)).parameters)
    assert params == {"scope", "form", "tool", "last_n"}


# ---- CLI conversation (req 16) ----------------------------------------------


@pytest.mark.asyncio
async def test_cli_conversation_lists_only_its_current_transcript(store, list_tool, cli_session):
    """Save, /clear (pointer rewritten), save a first-turn artifact: only that one lists."""
    first = cli_session.transcript
    _plot(store, "Before", session_id=f"cli:{first}")
    cli_session.clear(NEW_TRANSCRIPT)
    after = _plot(store, "After", session_id=f"cli:{NEW_TRANSCRIPT}")

    out = await _list(list_tool)
    assert out["session"] == NEW_TRANSCRIPT
    assert _ids(out) == [after.id]
    assert len((await _list(list_tool, scope="all"))["entries"]) == 2


@pytest.mark.asyncio
async def test_cli_lists_from_the_store_roots_pointer(
    store, list_tool, cli_session, tmp_path, monkeypatch
):
    """The default agent-data root differs from the store's: the store root's pointer wins."""
    from osprey_connectors import workspace

    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "cli-transcripts").mkdir(parents=True)
    (elsewhere / "cli-transcripts" / cli_session.launch).write_text(OTHER_KEY + "\n")
    monkeypatch.setattr(workspace, "resolve_shared_data_root", lambda *a, **k: elsewhere)

    mine = _plot(store, "Mine", session_id=f"cli:{cli_session.transcript}")
    _plot(store, "Elsewhere", session_id=f"cli:{OTHER_KEY}")

    out = await _list(list_tool)
    assert out["session"] == cli_session.transcript
    assert _ids(out) == [mine.id]


# ---- error handling ----------------------------------------------------------


@pytest.mark.usefixtures("web_session")
@pytest.mark.asyncio
async def test_list_reraises_tool_error_unwrapped(store, list_tool, monkeypatch):
    """A ToolError from inside the store passes through, never re-wrapped as internal_error."""
    envelope = json.dumps(
        {"error": True, "error_type": "custom_error", "error_message": "x", "suggestions": []}
    )

    def _boom(**kwargs):
        raise ToolError(envelope)

    monkeypatch.setattr(store, "list_entries", _boom)

    with assert_raises_error(error_type="custom_error"):
        await list_tool()


@pytest.mark.usefixtures("web_session")
@pytest.mark.asyncio
async def test_unexpected_store_failure_is_internal_error(store, list_tool, monkeypatch):
    """An unexpected store exception surfaces as internal_error, not a raw traceback."""

    def _boom(*args, **kw):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(store, "list_entries", _boom)

    with assert_raises_error(error_type="internal_error") as ctx:
        await list_tool()
    assert "index unreadable" in ctx["envelope"]["error_message"]
