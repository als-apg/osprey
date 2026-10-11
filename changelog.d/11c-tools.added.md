A new read-only tool on the `python` server, `model_response_matrix(model, correctors)`,
solves a served model's orbit response to its corrector setpoints on the model's own
deck. Each corrector is stepped by the measurement file's `corrector_delta`, converted
to the corrector's hardware unit, and every row and column names its address and
device. The solve runs in a sandbox child stopped at the executor's timeout.

A new tool on the `python` server, `pyaml_measure(kind, model, correctors, modulation,
by)`, measures an orbit, dispersion, tune or chromaticity response, or the chromaticity,
with pyAML on the control target. Every write runs inside one guarded run that writes
back what a stopped run moved, each step is sized so no setpoint's largest single write
exceeds half its limits.yaml `max_step`, and the run starts only on the pyAML view the
build recorded and the approval prompt showed. A group stepped as one knob moves every
member by the same integrated strength, sized so the tune or chromaticity moves no more
than under the strongest single-magnet step; its column is per unit integrated strength,
the sum of its members' columns. Both tools register when the render carries a pyAML
view.
