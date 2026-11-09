The web terminal's work pane holds one panel per tile again, and the tile
header's tab strip is gone. Opening a panel replaces the panel in the service
tile you used last, which stays in the sidebar one click away; Open in a new
tile and drops on a tile's edge or the work pane's edge still split. A layout
saved with several panels in one tile comes back showing the panel that was in
front, the others in the sidebar. For the agent, `close_panel_tab` and
`close_panel_tile` become `close_panel`, `open_panel_set` opens panels side by
side, `drop_panel` takes edge, border and outer targets, and a layout, preset
or report with several panels in one tile is refused `one_panel_per_tile`.
