"""The agent's artifact tools speak forms, never categories.

Category is retired: the save tools (``submit_response``, ``artifact_register``)
take no category or data type, the gallery groups every artifact by its form
(a base form, or a registered extension type's key), and ``artifact_list`` /
``artifact_delete_all`` scope by that same form. ``artifact_delete_all`` keeps
its explicit ``"everything"`` scope; ``artifact_list`` keeps its tool, agent,
type and recency filters.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from osprey.stores import type_registry
from osprey.stores.artifact_store import initialize_artifact_store
from osprey.stores.type_registry import ExtensionType
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

#: A registered extension type conforming to the ``documents`` base form.
EXT_KEY = "beam_study"


@pytest.fixture
def store(tmp_path):
    ws = tmp_path / "_agent_data"
    (ws / "artifacts").mkdir(parents=True)
    return initialize_artifact_store(workspace_root=ws)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Install a one-type extension registry for the process."""
    types = {EXT_KEY: ExtensionType(key=EXT_KEY, label="Beam studies", base_form="documents")}
    monkeypatch.setattr(type_registry, "_EXTENSION_TYPES", types)
    return types


def _tools():
    from osprey.mcp_server.workspace.tools import artifact_query, artifact_register, submit_response

    return {
        "submit_response": submit_response.submit_response,
        "artifact_register": artifact_register.artifact_register,
        "artifact_list": artifact_query.artifact_list,
        "artifact_delete_all": artifact_register.artifact_delete_all,
    }


def _registered(name: str):
    """The tool as the MCP server offers it to the agent (schema, description)."""
    from osprey.mcp_server.workspace.server import mcp

    _tools()  # importing the modules registers the tools
    return asyncio.run(mcp.get_tool(name))


def _params(name: str) -> set[str]:
    return set(_registered(name).parameters.get("properties", {}))


def _seed(store):
    """One artifact per interesting form, plus a run (head, notebook, figure)."""
    ids = {}
    ids["plot"] = store.save_file(
        b"png",
        "orbit.png",
        "image",
        "Orbit",
        mime_type="image/png",
        tool_source="create_static_plot",
    ).id
    ids["doc"] = store.save_file(
        b"# hi",
        "note.md",
        "markdown",
        "Note",
        mime_type="text/markdown",
        tool_source="artifact_register",
    ).id
    ids["ext"] = store.save_file(
        b"# study",
        "study.md",
        "markdown",
        "Study",
        mime_type="text/markdown",
        tool_source="beam_tool",
        kind=EXT_KEY,
    ).id
    ids["head"] = store.save_data(
        "execute",
        {"stdout": "ok"},
        title="Run",
        exec_id="toolu_run1",
        metadata={"run_part": "result", "run_status": "succeeded"},
    ).id
    ids["notebook"] = store.save_file(
        b"{}",
        "run.ipynb",
        "notebook",
        "Notebook: run",
        tool_source="execute",
        exec_id="toolu_run1",
        metadata={"run_part": "code"},
    ).id
    ids["figure"] = store.save_file(
        b"png",
        "fig.png",
        "image",
        "Figure",
        mime_type="image/png",
        tool_source="execute",
        exec_id="toolu_run1",
        metadata={"run_part": "output", "figure": True},
    ).id
    return ids


# ---- schemas ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["submit_response", "artifact_register", "artifact_list"])
def test_no_artifact_tool_exposes_a_category_or_data_type(name):
    params = _params(name)
    assert "category" not in params
    assert "data_type" not in params


def test_artifact_list_takes_a_form_and_keeps_its_other_filters():
    params = _params("artifact_list")
    assert {"form", "tool", "last_n", "source_agent", "artifact_type"} <= params


def test_no_tool_docstring_tells_the_agent_about_categories():
    for name in _tools():
        doc = (_registered(name).description or "").lower()
        assert "category" not in doc, name
        assert "data_type" not in doc, name


# ---- save tools ------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_response_saves_without_a_category(store):
    fn = get_tool_fn(_tools()["submit_response"])
    out = json.loads(
        await fn(title="Answer", content="# body", source_agent="logbook-search", entry_ids=["e1"])
    )
    entry = store.get_entry(out["artifact_id"])
    assert entry.category == ""
    assert "data_type" not in entry.metadata
    assert entry.source_agent == "logbook-search"
    assert entry.description == "logbook-search"


@pytest.mark.asyncio
async def test_submit_response_without_an_agent_describes_itself_as_an_agent_answer(store):
    fn = get_tool_fn(_tools()["submit_response"])
    out = json.loads(await fn(title="Answer", content="body"))
    assert store.get_entry(out["artifact_id"]).description == "Agent response"


@pytest.mark.asyncio
@pytest.mark.usefixtures("store")
async def test_submit_response_rejects_a_data_type_keyword():
    fn = get_tool_fn(_tools()["submit_response"])
    with pytest.raises(TypeError):
        await fn(title="A", content="b", data_type="logbook_research")


@pytest.mark.asyncio
async def test_artifact_register_saves_without_a_category(store):
    fn = get_tool_fn(_tools()["artifact_register"])
    out = json.loads(await fn(title="Summary", content="# x", content_type="markdown"))
    assert store.get_entry(out["artifact_id"]).category == ""
    with pytest.raises(TypeError):
        await fn(title="S", content="x", content_type="markdown", category="visualization")


# ---- artifact_list by form -------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_list_by_base_form(store):
    ids = _seed(store)
    fn = get_tool_fn(_tools()["artifact_list"])

    plots = json.loads(await fn(form="plots"))
    assert {e["artifact_id"] for e in plots["entries"]} == {ids["plot"], ids["figure"]}
    assert plots["filters_applied"]["form"] == "plots"

    runs = json.loads(await fn(form="runs"))
    assert [e["artifact_id"] for e in runs["entries"]] == [ids["head"]]

    docs = json.loads(await fn(form="documents"))
    # An artifact sits in exactly one group: the extension entry is under its
    # own key, not its base form, and the run's notebook is a part of its run.
    assert [e["artifact_id"] for e in docs["entries"]] == [ids["doc"]]


@pytest.mark.asyncio
async def test_artifact_list_by_extension_key(store):
    ids = _seed(store)
    fn = get_tool_fn(_tools()["artifact_list"])
    out = json.loads(await fn(form=EXT_KEY))
    assert [e["artifact_id"] for e in out["entries"]] == [ids["ext"]]


@pytest.mark.asyncio
async def test_artifact_list_entries_carry_their_form(store):
    ids = _seed(store)
    fn = get_tool_fn(_tools()["artifact_list"])
    forms = {e["artifact_id"]: e["form"] for e in json.loads(await fn())["entries"]}
    assert forms[ids["plot"]] == "plots"
    assert forms[ids["head"]] == "runs"
    assert forms[ids["ext"]] == EXT_KEY
    assert forms[ids["notebook"]] is None


@pytest.mark.asyncio
async def test_artifact_list_keeps_its_other_filters(store):
    ids = _seed(store)
    fn = get_tool_fn(_tools()["artifact_list"])
    out = json.loads(await fn(form="plots", tool="execute"))
    assert [e["artifact_id"] for e in out["entries"]] == [ids["figure"]]


@pytest.mark.asyncio
@pytest.mark.usefixtures("store")
async def test_artifact_list_rejects_an_unknown_form():
    fn = get_tool_fn(_tools()["artifact_list"])
    with assert_raises_error(error_type="validation_error") as ctx:
        await fn(form="visualization")
    assert EXT_KEY in json.dumps(ctx["envelope"])


# ---- artifact_delete_all by form -------------------------------------------


@pytest.mark.asyncio
async def test_delete_all_by_base_form_leaves_other_forms(store):
    ids = _seed(store)
    fn = get_tool_fn(_tools()["artifact_delete_all"])
    out = json.loads(await fn(scope="plots"))
    assert out["scope"] == "plots"
    assert set(out["artifact_ids"]) == {ids["plot"], ids["figure"]}
    left = {e.id for e in store.list_entries()}
    assert left == {ids["doc"], ids["ext"], ids["head"], ids["notebook"]}


@pytest.mark.asyncio
async def test_delete_all_runs_takes_each_run_with_its_parts(store):
    ids = _seed(store)
    fn = get_tool_fn(_tools()["artifact_delete_all"])
    out = json.loads(await fn(scope="runs"))
    assert set(out["artifact_ids"]) == {ids["head"], ids["notebook"], ids["figure"]}
    assert {e.id for e in store.list_entries()} == {ids["plot"], ids["doc"], ids["ext"]}


@pytest.mark.asyncio
async def test_delete_all_by_base_form_spares_its_extension_types(store):
    ids = _seed(store)
    fn = get_tool_fn(_tools()["artifact_delete_all"])
    out = json.loads(await fn(scope="documents"))
    assert out["artifact_ids"] == [ids["doc"]]
    left = {e.id for e in store.list_entries()}
    assert ids["ext"] in left and ids["notebook"] in left


@pytest.mark.asyncio
async def test_delete_all_by_extension_key_takes_only_that_type(store):
    ids = _seed(store)
    fn = get_tool_fn(_tools()["artifact_delete_all"])
    out = json.loads(await fn(scope=EXT_KEY))
    assert out["artifact_ids"] == [ids["ext"]]
    assert out["deleted_count"] == 1
    assert ids["doc"] in {e.id for e in store.list_entries()}


@pytest.mark.asyncio
async def test_delete_all_everything_is_kept(store):
    _seed(store)
    fn = get_tool_fn(_tools()["artifact_delete_all"])
    out = json.loads(await fn(scope="everything"))
    assert out["deleted_count"] == 6
    assert store.list_entries() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["uncategorized", "visualization", ""])
async def test_delete_all_rejects_category_scopes(store, scope):
    ids = _seed(store)
    fn = get_tool_fn(_tools()["artifact_delete_all"])
    with assert_raises_error(error_type="validation_error"):
        await fn(scope=scope)
    assert len(store.list_entries()) == len(ids)
