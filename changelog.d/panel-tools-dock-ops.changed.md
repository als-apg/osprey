The agent changes the dock through one op path. Every layout tool is relayed
to the browser that holds the dock and answers with the resulting layout and
its `rev`, or with a named refusal that changed nothing; `list_panels` reports
the layout, the panels in pop-out windows and the `rev`. `close_panel` is
replaced by `close_panel_tab` and `close_panel_tile`; `drop_panel` and
`open_panel_set` are new; `arrange_workspace` takes a preset or an exact
layout. `POST /api/panel-close`, `POST /api/panel-arrange` and the agent form
of `POST /api/panel-focus` are gone; the panel tier gets `/api/panel-op/*`. The
terminal is never an agent drop source or close target, and a panel in its own
window is brought forward, never moved or closed.
