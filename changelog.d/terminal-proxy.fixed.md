Every per-user web terminal now receives `HTTP_PROXY`, `HTTPS_PROXY` and
`NO_PROXY` from the deploy env chain, as the login service already did. On a
proxied site the agent in a terminal could not reach the model provider, and
nothing said so — the stack started and the health check was green. The
lowercase-spelling warning `osprey up` prints now covers any deployment with
web terminals, not only one using OIDC logins.
