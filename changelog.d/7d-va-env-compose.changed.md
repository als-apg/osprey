The virtual accelerator's tick comes from `simulation.tick_s` (default 1.0 s); a `VA_POLL_INTERVAL_S` in the project `.env` is not read.
The virtual-accelerator compose blocks no longer read `VA_BPM_ERRORS`, `VA_STANDIN_BPM_ERRORS`, `VA_CORR_GAIN`, `VA_STUCK_SETPOINTS`, `VA_CHANNELS_FILE` or `VA_LATTICE`.
`osprey build` no longer writes the virtual accelerator's channel manifest or the derived `VA_CHANNELS_FILE`/`VA_LATTICE` keys into `.env`.
`var/simulator/` and `var/simulator/standin/` are provisioned setgid and group-writable before compose runs; each virtual-accelerator instance mounts its own, and web terminals and dispatch workers mount `var/simulator/` whenever a simulated target is configured.
