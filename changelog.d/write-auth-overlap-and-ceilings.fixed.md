An optimizer run that finished, failed or crashed under a person-approved
write authorization no longer blocks approving another program on its channels
for the rest of the eight-hour expiry; it stops blocking about a minute after
its lease is last renewed, when the approver can see the run's ledger (with
none in sight it still blocks until it expires). A paused run still renews, so
it still blocks. `OPTIMIZER_AUTHORIZATION_EXPIRES_S` is held just under 24
hours.
