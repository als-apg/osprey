Web terminal: the agent's `open_panel` opens a panel in a new tile split off the
service tile used last instead of replacing the panel there. `place="replace"`
replaces it, and `tile`/`edge` choose the target and side; the result reports
how the panel was placed and notes when no room for a new tile forced a
replace. Panels the Logbook and Phoebus tools open follow the same default.
Opens from the sidebar and from links are unchanged.
