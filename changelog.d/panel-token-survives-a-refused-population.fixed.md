In a multi-user deployment the agent's artifact-focus call no longer fails with
`HTTP 401`. Saving an artifact used to start the artifact gallery from inside
the agent's own MCP server process, which consumed that process's panel token
and left every later panel call unauthenticated; the gallery is now started only
by a process that can authenticate it, and a refused credential population
leaves the process the panel token it was handed.
