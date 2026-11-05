Write authorization follow-ups. An optimizer run cancelled while waiting for
approval no longer reports `awaiting`; a manual write with the switch at `asked`
is refused `switch_not_allowed` rather than `auth_revoked`; and a run's request
is labelled `<environment> optimization (<job id>)`. The open chip popover keeps
its programs current, the machine name is no longer cut off by the program
badge, and the confirmation for `allowed` says programs then authorize
themselves. `osprey write-authorization approve` on a request that is no longer
pending says whether it was already approved. `.env.example` leaves
`ANTHROPIC_API_KEY` commented out, so copying it to `.env` no longer overrides a
Claude login with a placeholder key.
