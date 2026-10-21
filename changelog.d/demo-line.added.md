The control-assistant demo gains `LINE`, a synthetic transfer line solved
single-pass: 20 devices and 40 `LINE:` channels beside the periodic model `SR`.
The simulator now serves `[LINE, SR]`, so it publishes one status channel per
model (`<code>:SIM:LINE:STATUS` and `<code>:SIM:SR:STATUS`), `osprey sim status`
prints a line for each, and the lattice dashboard lists `LINE` first and opens
on it. The hierarchical and middle-layer channel-finder indexes gain the 40
`LINE:` addresses; the in-context index and `data/facility/limits.yaml` are
unchanged, so a `LINE` setpoint is written with no limits record. The benchmark
query "Show me the transfer line corrector setpoints for both planes" now names
the booster-to-storage transfer line; the ambiguous-category query "Show me the
second focusing quad setpoint in the transfer line" is kept as is.
