The Virtual Accelerator's telemetry cadence and noise level are now set from
the deployment's `.env`: `VA_POLL_INTERVAL_S` (default 1.0 seconds) and
`VA_NOISE_LEVEL` (default 0.01). The noise level previously could not be
changed at all — the IOC never read the one it was given. Both are refused at
boot if they are not a number or are out of range, rather than being clamped.
