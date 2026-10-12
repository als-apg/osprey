The `pyaml_measure` approval prompt lists every setpoint the call steps, no
longer the first twenty and a count of the rest. A call that would step more
than 500 setpoints, the most a prompt lists, is refused without asking.

The pyAML view writes `trm.json` and `crm.json` whenever every member of the
`quad` or `sext` group has a modelled column, so `pyaml_measure` steps such a
group as one knob, `crm` by group included, even when the group is a single
supply. The tune or chromaticity correction tool that loads the matrix is still
added only when the matrix has rank two. The example's `SR` view now carries
`crm.json`.
