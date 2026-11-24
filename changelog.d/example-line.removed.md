**Breaking change:** the example facility no longer holds the synthetic
transfer line `LINE`: its place, 20 devices, 40 `LINE:` channels, groups,
limits, model, deck and measurement file are gone, and the simulator serves
`[BTS, LTB, SR]`. The BTS deck's quadrupoles now state a 20 mm aperture, so a
large corrector kick loses the beam there, and the `line-server-stopped`
scenario is replaced by `bts-server-stopped`, which stops `sim-BTS`.
