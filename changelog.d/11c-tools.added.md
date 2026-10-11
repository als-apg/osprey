A new read-only tool on the `python` server, `model_response_matrix(model, correctors)`,
solves a served model's orbit response to its corrector setpoints on the model's own
deck. Each corrector is stepped by the measurement file's `corrector_delta`, converted
to the corrector's hardware unit, and every row and column names its address and
device. The solve runs in a sandbox child stopped at the executor's timeout.
