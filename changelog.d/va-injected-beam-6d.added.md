A turn-by-turn ring launches its bunch in six dimensions. The `turn_by_turn`
block's `jitter` takes all six coordinates (`x`, `px`, `y`, `py`, `delta`,
`ct`), `twiss` states the optics the beam arrives with and `mismatch` a factor
on their beta functions. New wiring signals `offset_<coordinate>` and
`mismatch_<plane>` hold a static injection error and the mismatch as
setpoints, so a scenario's `overrides` state them and a client moves them
between shots. A block that states none of this launches as before.
