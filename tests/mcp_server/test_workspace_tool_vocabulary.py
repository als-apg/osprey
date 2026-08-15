"""Guards on the workspace MCP server's agent-facing vocabulary.

The workspace server exposes ONE store (``ArtifactStore``). Two prefixes over
it (``data_*`` and ``artifact_*``) and two spellings of the same identifier
(``entry_id`` and ``artifact_id``) made the agent re-key values it already held
and hid the blast radius of the bulk delete. These tests pin the single
vocabulary: the record is an *artifact*, its id is ``artifact_id``, and the
data views are a ``category`` argument rather than a parallel tool family.

They also pin **allowlist parity**: a tool renamed on the server but not in
``registry.mcp`` does not raise anywhere — the rendered permission list simply
names a tool that no longer exists and the agent quietly loses the capability.
"""

import asyncio

import pytest


@pytest.fixture(scope="module")
def workspace_tools() -> dict:
    """Every tool registered on the workspace MCP server, keyed by name."""
    from osprey.mcp_server.workspace.server import mcp
    from osprey.mcp_server.workspace.tools import (  # noqa: F401  (import = register)
        archiver_downsample,
        artifact_export,
        artifact_query,
        artifact_save,
        create_dashboard,
        create_document,
        create_interactive_plot,
        create_static_plot,
        facility_description,
        focus_tools,
        lattice_tools,
        panel_tools,
        provenance_locator,
        screen_capture,
        session_log,
        session_summary,
        setup,
        submit_response,
    )

    return {t.name: t for t in asyncio.run(mcp.list_tools())}


def test_no_tool_uses_the_data_prefix(workspace_tools):
    """``data_*`` was a second noun for the same artifact records."""
    offenders = sorted(n for n in workspace_tools if n.startswith("data_"))
    assert offenders == [], (
        f"{offenders} name artifact-store records 'data'; the store's own noun is 'artifact'"
    )


def test_store_record_tools_exist_under_the_artifact_noun(workspace_tools):
    assert {"artifact_list", "artifact_read", "artifact_get"} <= set(workspace_tools)


def test_no_tool_spells_the_record_id_entry_id(workspace_tools):
    """One record, one id spelling — the agent must not have to re-key it."""
    offenders = sorted(
        name
        for name, tool in workspace_tools.items()
        if "entry_id" in tool.parameters["properties"]
    )
    assert offenders == [], f"{offenders} spell the artifact id 'entry_id'; use 'artifact_id'"


def test_artifact_list_filters_by_category_not_a_separate_tool_family(workspace_tools):
    """The data views are a ``category`` argument on the one list tool."""
    props = workspace_tools["artifact_list"].parameters["properties"]
    # The filtering data_list had must survive the rename.
    assert {"category", "tool", "last_n", "source_agent"} <= set(props), props


#: The panel verbs, by the axis each one moves. Rail membership is "can the
#: operator launch this in one click"; on-screen is "is there a tile". The two
#: used to be straddled by one verb pair: ``show_panel`` moved rail membership
#: despite its name, ``hide_panel`` moved both, and ``switch_panel`` was the only
#: verb that put anything on screen — so ``show_panel``/``hide_panel`` was not a
#: round trip and the agent had no way to close a tile without also making the
#: panel unlaunchable.
PANEL_VERBS_BY_AXIS = {
    "rail membership": ("add_panel_to_rail", "remove_panel_from_rail"),
    "on screen": ("open_panel", "close_panel"),
}


def test_each_panel_axis_has_both_halves_of_its_round_trip(workspace_tools):
    for axis, verbs in PANEL_VERBS_BY_AXIS.items():
        missing = sorted(set(verbs) - set(workspace_tools))
        assert missing == [], f"the {axis} axis is missing {missing}"


def test_no_panel_verb_names_an_axis_it_does_not_move(workspace_tools):
    """The straddling names must not come back under a new implementation."""
    offenders = sorted({"show_panel", "hide_panel", "switch_panel"} & set(workspace_tools))
    assert offenders == [], (
        f"{offenders} name the on-screen axis but move rail membership; "
        f"the honest verbs are {PANEL_VERBS_BY_AXIS}"
    )


def test_registry_allowlist_names_only_tools_the_server_registers(workspace_tools):
    """Allowlist parity — the failure mode a rename hits and nothing reports.

    A stale name in ``permissions_allow``/``permissions_ask`` renders into the
    agent's settings as an allow rule for a tool that does not exist, so the
    real tool falls through to a prompt (or a denial) instead of being
    auto-approved. Nothing errors; the capability just goes dark.
    """
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    workspace = FRAMEWORK_SERVERS["osprey_workspace"]
    listed = set(workspace.permissions_allow) | set(workspace.permissions_ask)
    stale = sorted(listed - set(workspace_tools))
    assert stale == [], f"registry.mcp lists workspace tools that are not registered: {stale}"
