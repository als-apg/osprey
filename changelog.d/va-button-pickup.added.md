A Virtual Accelerator turn-by-turn monitor may now state a button pickup. The
ring's `turn_by_turn` settings state `pickup` for every monitor, or `pickups`
for single ones: a round pipe's radius, button angles and button width, or a
table of button responses on a grid for any other chamber, with the buttons
each plane adds and subtracts and an optional scale. The monitor's sum and
positions are then formed from the button signals at the beam position each
turn, so the sum depends on where the beam is and a position far from the
centre shows the pickup's own distortion. Each button takes a gain,
`button_<n>` on the monitor's `turn_sum` record, so uneven or dead buttons
show as a position offset. A monitor that states no pickup reads as before,
bit for bit. With a pickup, a centred beam reads the same sum as before to
floating-point rounding. The served positions are the raw difference over sum
times the scale; no linearisation is applied.
