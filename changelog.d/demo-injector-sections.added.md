The demo facility now covers the whole injector: the gun-to-linac line `GTL`,
the linac `LN`, the linac-to-booster line `LTB` and the booster-to-storage line
`BTS` gain their magnets, monitors, screens, gun and RF stand-ins, and the
booster gains its injection kicker, extraction septa, screens and energy-ramp
setpoints. One readback gate chain runs in beam order from the gun to storage
ring injection: while the gun's high voltage or heater is off, or any upstream
screen paddle is in, every BPM sum and charge reading downstream drops to a
hundredth of its value and every BPM `STATUS:VALID` reads `FALSE`; a screen's
lamp follows its own paddle, and the "all paddles out" count follows every
paddle. `scripts/facility_demo/panel_rename.py` re-maps the GTL and BTS CANOPY
panels from legacy names to the demo channels: `check --facility` confirms
every mapped channel exists, and `apply --library` rewrites the panels'
bindings in place.
