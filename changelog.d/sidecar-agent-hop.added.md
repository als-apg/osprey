A sidecar panel that declares an MCP transport (`mcp_path`) can opt in to an
agent hop with `agent_hop=True`. The terminal then admits exactly
`GET|POST|DELETE /panel/<id><mcp_path>` on the panel token
(`web_auth.panel_tier_routes()`), and the panel proxy relays that request
unrewritten, without the caller's `Authorization`, `Cookie` or
`X-Osprey-Terminal-Secret` and without re-issuing the operator secret. A call
to one of the panel's ask tools is refused while the calling session is held,
as a job fired on the dispatcher hop is. A browser session or the operator
secret gets 403 on that path, every other spelling of it stays refused, and a
panel that declares `mcp_path` alone is closed to every caller.
