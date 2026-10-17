The `pyat-specialist` agent names each lattice deck and `variables.json` by the
path `mcp__python__execute` resolves, relative to the deployment root, so it can
load its deck. The artifacts rule leaves `artifact_focus` to the session that
answers the user; a subagent hands back the artifact id instead.
