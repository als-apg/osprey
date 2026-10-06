The simulation engine no longer reads a single-name `active_scenario` state
file. The active set is read from `active_scenarios` only, which
`osprey sim apply` has always written.
