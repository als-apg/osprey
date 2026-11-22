The `pyat` engine wires a cavity's `Voltage` and `TimeLag` as setpoints with
readbacks, beside `Frequency`: a phase setpoint is a time lag through the
record's calibration, a voltage is shared over `slices` by their weights, and
the RF switch holds the commanded voltage while it is off. Each RF setpoint
carries one fault: `cal_offset` on a frequency and on a time lag, `cal_factor`
on a voltage.
