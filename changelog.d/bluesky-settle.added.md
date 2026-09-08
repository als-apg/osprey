Two new build-profile keys bound how long a Bluesky plan waits for a device to
settle after a write: `bluesky.settle_timeout_s` (default 5.0 seconds) and
`bluesky.settle_tolerance` (default `1e-9`, an absolute difference). The
previous values were fixed in the code and suited only devices whose readback
echoes the setpoint exactly; a magnet or an insertion-device gap can now be
given the room it needs. Running out of budget still fails the plan, and a
value the settle loop cannot use fails the worker's start rather than the first
plan that writes.
