"""The hello-world facility channels the model-backed tests ask an agent to use."""

#: A seeded measurement: a read returns a number near its nominal.
READ_CHANNEL = "SR:BEAM:CURRENT"

#: A writable setpoint with a limits record; WRITE_VALUE lies inside its band.
WRITE_CHANNEL = "SR:MAG:QF:01:CURRENT:SP"
WRITE_VALUE = 1.5
