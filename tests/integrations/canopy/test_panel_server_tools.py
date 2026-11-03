"""The CANOPY panel's declared hop permissions equal the tools CANOPY registers.

Needs CANOPY installed: the gate installs the pinned export. ``mcp_allow`` and
``mcp_ask`` are what the agent's settings render from; a tool CANOPY offers that
neither list names would reach the agent with no permission entry at all.
"""

from __future__ import annotations

from canopy_mcp import server as canopy_server

from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID, SIDECAR_PANELS


def _registered() -> set[str]:
    """Every tool CANOPY can offer: authoring on, and the workspace the sidecar opts into."""
    canopy_server.register_all()
    canopy_server.set_authoring(True)
    canopy_server.set_workspace_tools(True)
    return set(canopy_server.registered_tools())


def test_panel_server_tools_match_canopy() -> None:
    panel = SIDECAR_PANELS[CANOPY_PANEL_ID]
    allow, ask = set(panel.mcp_allow), set(panel.mcp_ask)
    assert not allow & ask
    assert _registered() == allow | ask
    assert ask == set(canopy_server.AUTHORING_TOOLS)
    assert len(canopy_server.AUTHORING_TOOLS) == 7
    assert len(panel.mcp_ask) == 7 and len(panel.mcp_allow) == 12


def test_the_read_tools_are_the_ones_left_with_authoring_off() -> None:
    panel = SIDECAR_PANELS[CANOPY_PANEL_ID]
    _registered()
    canopy_server.set_authoring(False)
    try:
        left = set(canopy_server.registered_tools())
    finally:
        canopy_server.set_authoring(True)
    assert left == set(panel.mcp_allow)


def test_the_workspace_tools_ride_the_allow_list() -> None:
    """Allowed without a prompt, and exactly the two the workspace opt-in offers."""
    panel = SIDECAR_PANELS[CANOPY_PANEL_ID]
    workspace = set(canopy_server.WORKSPACE_TOOLS)
    assert workspace == {"get_workspace", "set_workspace"}
    assert workspace <= set(panel.mcp_allow)
    assert not workspace & set(panel.mcp_ask)
    assert not workspace & set(canopy_server.AUTHORING_TOOLS)
    _registered()
    canopy_server.set_workspace_tools(False)
    try:
        left = set(canopy_server.registered_tools())
    finally:
        canopy_server.set_workspace_tools(True)
    assert left == set(panel.mcp_allow) - workspace | set(panel.mcp_ask)
