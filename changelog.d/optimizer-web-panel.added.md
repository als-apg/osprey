A build profile's `optimizer_web:` block, beside its `optimizer:` block, deploys
the OPTIMIZER web-terminal tab: a sidecar at the layout's `optimizer_web` slot
that serves the panel and relays to the optimizer service. The agent drafts a
run; a person reviews it in the panel and starts it there. Start sends the
draft revision the panel displayed, so a draft changed since then is refused
rather than run; a different draft replacing the one on screen, or a draft
whose environment has not loaded to verify its objective direction, holds
Start until the operator has looked. The sidecar alone holds `OPTIMIZER_LAUNCH_TOKEN`, and attaches
it only to start and resume; every write it relays names the operator who made
it, from the roster secret their terminal presented, and the panel warns before
Start when the session names no account, since the service may then refuse the
run's writes (`owner_unknown`).
The build writes `web.panels.optimizer.read_timeout_s` (the sidecar's 80 s relay
budget plus a 10 s margin; an explicit value wins), and attached renders are
told the host's value with the tab's url, so a Start that waits on machine reads
is not cut by the panel proxy's 30 s default. Cancelling stops the run where it
is: the machine keeps the last point written, which may not have been evaluated
yet; restoring the starting setpoints stays a manual step. While a draft waits,
the approval card re-reads its readbacks every 5 s without asking for another
review.
