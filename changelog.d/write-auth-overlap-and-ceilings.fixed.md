An optimizer run that finished, failed or crashed under a person-approved
write authorization no longer blocks approving another program on its channels
for the rest of the eight-hour expiry; it stops blocking about a minute after
its lease is last renewed. A paused run still renews, so it still blocks.
`OPTIMIZER_AUTHORIZATION_EXPIRES_S` is held to at most 24 hours.
