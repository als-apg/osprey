When the browser does not answer an agent's dock change within five seconds,
the agent now gets `no_dock_answer`. The browser can still make the change,
so the agent reads the panels again before it tries again. Before, the agent
got `no_dock_client`, which says that no browser shows the workspace and that
nothing changed. The web terminal now records a late answer from the browser
as the layout on screen. Before, it refused that answer.
