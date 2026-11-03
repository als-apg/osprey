An agent's `Bash` and `Monitor` commands start without the panel token. OSPREY
points `CLAUDE_ENV_FILE` at a script that unsets it, so only the agent's MCP
client and hooks hold it, and a shell command can no longer fire a dispatcher
job or call a panel's MCP tools around their approval prompts. The build's
warning for a profile-declared shell gate no longer names those routes.
