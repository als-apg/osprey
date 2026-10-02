A `control_system.target_switch.probe_interval_s` that is an integer beyond
float range, infinity or `true` now falls back to the 30 s default, instead
of failing the controls server's start, never refreshing the reachability
roster, or probing every second.
