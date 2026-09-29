`control_system.target_switch.probe_timeout_s` (default 5) sets how long a
target switch waits for the new connection to read the target's
`probe_channel` before it refuses the switch and keeps the current target. A
value that is not a positive number is logged and the default is used.
