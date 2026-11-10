A build profile's `optimizer_web:` block, beside its `optimizer:` block, deploys
the OPTIMIZER web-terminal tab: a sidecar at the layout's `optimizer_web` slot
that serves the panel and relays to the optimizer service. The agent drafts a
run; a person reviews it in the panel and starts it there. Start sends the
draft revision the panel displayed, so a draft changed since then is refused
rather than run; a different draft replacing the one on screen, or a draft
whose environment has not loaded to verify its objective direction, holds
Start until the operator has looked. The sidecar alone holds `OPTIMIZER_LAUNCH_TOKEN`, and attaches
it only to start and resume; every write it relays names the operator who made
it, from the roster secret their terminal presented.
