A Virtual Accelerator turn-by-turn monitor's sum now takes its own gain. A
scenario states `gain` on a monitor's `turn_sum` record, and that monitor's
sum reads the injected bunch and the stored beam times it, before the sum's
noise and floor, so monitors no longer all read the same sum for the same
charge. Positions are untouched.
