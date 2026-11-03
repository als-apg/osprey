The agent in a web terminal reaches that user's own CANOPY. Every project that
shows the CANOPY panel renders a `canopy` entry in `.mcp.json` that goes
through the terminal's own panel proxy (`/panel/canopy/mcp`). The read tools
(`check`, `read_pv`, `search_pvs`, `screenshot`, `get_screen`, `list_drafts`,
`list_widgets`, `get_widget`, `get_schema`, `list_collections`) and the
workspace tools (`get_workspace`, `set_workspace`) run without a prompt; the
authoring tools (`new_screen`, `patch_node`, `add_node`, `remove_node`,
`convert_legacy`, `delete_draft`, `request_push`) ask on every call and write
only the user's personal store. No workspace command writes a PV or presses a
widget. Each `set_workspace` batch is a row in the activity log with the
status the tool returned. A CANOPY url, whether given to `open_panel` or posted
to the terminal's panel ops or panel registration, is refused with
`agent_hop_panel_url`. Such a project carries the `canopy-panel-builder` and
`canopy-operator` skills, CANOPY's own text bundled byte for byte.
