Each simulator instance of a deployment has an active scenario set of its own,
so the live stand-in can run a set of errors while the virtual accelerator runs
the clean model. `osprey sim apply` and `osprey sim list` take
`--target va|standin`, defaulting to the deployment's own target, and
`simulation.default_scenarios` also accepts `{va: [...], standin: [...]}`; a
list still starts every instance in the set. On a deployment that runs the
stand-in, `osprey sim apply` without `--target` no longer moves both machines:
it moves that one target and names it. The first `osprey up` gives a running
stand-in its own start set and prints the set it followed until then. The
logbook and the stored archive follow the `va` set, so a set applied to the
stand-in changes telemetry only. A deployment with one instance behaves and
renders as before.
