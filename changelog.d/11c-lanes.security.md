`pyaml_measure` refuses a measurement before any setpoint moves when a magnet's
strength, a hardware conversion, an RF write or a `max_step` it sizes a step
from is not a finite number, or when a `max_step` is not above zero. Such a
value used to pass every comparison and hand the run the full requested step.

`pyaml_measure` saves to the artifact store only the measurement file its run
left in the run's own execution folder under the script's fixed name. A run that
names any other path saves nothing; the answer carries `data_file: null` and a
`data_file_refused` line saying what was named.
