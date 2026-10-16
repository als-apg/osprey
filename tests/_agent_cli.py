"""The stand-in for the Claude Code a test's headless runs are pointed at."""

#: Where ``tests/conftest.py`` points every project's headless runs unless a
#: test asks for the real lookup. It names no file, so a test that spawns
#: through it fails at the spawn instead of running whatever the host has
#: installed.
STUB_AGENT_CLI = "/osprey-tests/no-claude-code-here/claude"
