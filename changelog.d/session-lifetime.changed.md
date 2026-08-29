The terminal session cookie now honours the configured
`modules.web_terminals.auth.session_lifetime` in both single-user
(`osprey web`) and multi-user deployments, and carries a matching
`Max-Age` so a session survives closing and reopening the browser.
Sessions also survive a web-terminal restart on the same port.
