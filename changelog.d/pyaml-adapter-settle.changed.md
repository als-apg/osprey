The pyAML adapter's `set_and_wait` and OSPREY's stepped measurements now wait
for each setpoint's readback to come within the tolerance the facility states
for it, and fail when it does not.
