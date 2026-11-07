# Sourced by Claude Code before every Bash and Monitor command an OSPREY agent
# runs (CLAUDE_ENV_FILE, set by osprey.agent_runner.clean_env.add_panel_token).
# The panel token is for the agent's MCP panel tools and hooks, which Claude
# Code starts itself; a shell command never inherits it.
unset OSPREY_PANEL_TOKEN
