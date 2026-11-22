A `pyat` readback of a deck property (`engine: {attribute: energy}`, no
element) may state its own calibration: it then reads the value of the energy
setpoint's curve through its own, so a bending-magnet string can serve its
voltage beside its current. Such a readback states `energy_scaling: none`, and
its computed `default` is the value it starts with.
