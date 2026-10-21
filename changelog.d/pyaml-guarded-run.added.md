A pyAML write outside a `run_tool` call, `execute` included, is refused as
`OspreyWriteRefused` ("pyAML writes run only inside pyaml_measure") before
anything is read or written.

`pyaml_cs_osprey.run_tool.run_tool` calls a pyAML tool method under the
target's run lock, joining the lock the process already holds: a second run on
the target is refused busy before the tool starts, each setpoint's value is
read before its first write, and the call returns a `ToolRun` whose `moved`
lists the setpoints the runtime sent or landed, each with its value before the
call; a refused write lists nothing, and a write whose outcome is unknown (the
call raised for another reason, or was interrupted, before its outcome came
back) is listed. An escaping exception carries the same
list as `moved_setpoints`. Nothing is written back.
A Ctrl-C stops every tool of the run that takes a callback, in any thread,
through pyAML's own abort path. A run that reaches the execution timeout is
killed like any other, with no early stop.

`pyaml_cs_osprey.measure.StepMeasurement` measures what pyAML's own tools step
only bipolar or one magnet at a time: a unipolar orbit response with a step per
corrector, a unipolar dispersion, and tune and chromaticity responses that step
each group of magnets as one knob, every write inside its `measure()` call.

The `pyaml-cs-osprey` package is built, installed and released with the
framework: images, `--dev` wheel staging and a source checkout's venv install
every workspace member from the same checkout, and a release uploads the member
wheels before the framework.
