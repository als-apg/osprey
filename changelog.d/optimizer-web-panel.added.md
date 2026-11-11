A new OPTIMIZER tab in the web terminal, where a person starts the optimization
run an agent drafted. Turn it on with an `optimizer_web:` block beside the
profile's `optimizer:` block; the build deploys an `optimizer-web` sidecar that
serves the panel and relays to the optimizer service. Start runs exactly the
draft shown: a draft changed since the panel displayed it is refused, and a
replaced draft holds Start until the operator has looked at it. The panel warns
before Start when the session names no operator account, since the service may
then refuse the run's writes (`owner_unknown`). Cancelling stops the run where it
is: the machine keeps the last point written, and restoring the starting
setpoints stays a manual step. The build also raises
`web.panels.optimizer.read_timeout_s` above the panel proxy's 30 s default, so a
slow Start is not cut off; an explicit value wins.
