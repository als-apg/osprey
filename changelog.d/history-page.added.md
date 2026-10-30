Under web-terminal sign-on, the history page at `/history/` shows each person
the agent record of their own sessions, or every user's with `history: all` on
their roster entry or role. It is served on the history page's origin
(`modules.web_terminals.history_port`, or `modules.web_terminals.history_origin`
behind a load balancer), so pages on the terminals' origin cannot read from it.
People named under `agent_record.readers` read there too, each limited to the
entry kinds and days their entry allows.
