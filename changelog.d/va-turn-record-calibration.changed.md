A `calibration` on a Virtual Accelerator turn-by-turn wiring record now stops
`osprey facility validate` and `osprey build`. The engine never read it, so
the record was served unconverted while the facility file said otherwise.
Remove it; a monitor's own sum scale is the `gain` fault of its `turn_sum`
record.
