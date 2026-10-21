The model RPC's `status` reply carries `instance`, `endpoint`, `last_cycle_ms`, `queue_depth`, `uptime_s` and `last_refused_write`, and its `info` reply carries `variables` alone; neither reports a backend, a lattice source or an update rate.
`info` names the side of each variable as `served` or `model`, and over a simulator view it lists each physics model's own variables as `<model>/<name>` after the served addresses.
`reset` over a simulator view writes each drifted model variable back to the value its model held when it was built; no setpoint, held value or session write is touched.
