A pyAML write outside a journaled guarded run, `execute` included, is refused as
`OspreyWriteRefused` ("pyAML writes run only inside pyaml_measure") before
anything is read or written.

`pyaml_cs_osprey.run_tool.run_tool` calls a pyAML tool method inside
`journaled_run` on the process's control target: a second run on the target is
refused busy before the tool starts, each setpoint is journaled before its first
write, a tool that stops part-way is written back before the call returns and
reported on one `OSPREY_PYAML_RESTORE` line, and a killed run is written back by
the next guarded run. Under `OSPREY_EXECUTION_DEADLINE` a tool that takes a
callback is stopped while the write-back still fits before the sandbox is killed.

`pyaml_cs_osprey.measure.StepMeasurement` measures what pyAML's own tools step
only bipolar or one magnet at a time: a unipolar orbit response with a step per
corrector, a unipolar dispersion, and tune and chromaticity responses that step
each group of magnets as one knob, every write inside its `measure()` call.

The `pyaml-cs-osprey` package is built, installed and released with the
framework: images, `--dev` wheel staging and a source checkout's venv install
every workspace member from the same checkout, and a release uploads the member
wheels before the framework.
