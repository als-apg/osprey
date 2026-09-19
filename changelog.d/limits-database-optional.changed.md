A channel-limits database is required at build time only when limits checking
is enabled for a target that arms writes. With checking off, an armed
deployment builds and runs without one, as the connector already allowed, and
the compose files it renders mount no limits database.
