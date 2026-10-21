**Breaking change:** `osprey build` writes each render's
`data/channel_limits.json` from `data/facility/limits.yaml`: `_version: "4.0"`,
one entry per limits record with `writable` and `confirm` stated, and no
`defaults` block. A channel with no record has no entry and follows
`control_system.limits_checking.mode`. The build derives
`control_system.limits_checking.database_path`, so a profile no longer states
it, and the `hello-world` preset no longer ships a hand-written
`data/channel_limits.json`.
