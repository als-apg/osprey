A `control_system.target_switch.probe_timeout_s` set to an integer beyond
float range now falls back to the default with a warning, as
`web.sidecar_ready_timeout_s` already did, instead of raising `OverflowError`
during a target switch.
