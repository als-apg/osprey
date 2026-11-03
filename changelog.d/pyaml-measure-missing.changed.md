Every `pyaml_measure` result carries `missing`: `{}` for a completed run; a
stopped run maps what it did not measure to the first line of what stopped it,
or `null` when nothing raised. The top-level `remainder` and `error` keys are
gone.
