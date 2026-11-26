Each simulator instance of a deployment has an active scenario set of its own,
so the live stand-in can run a set of errors while the virtual accelerator runs
the clean model. `osprey sim apply` and `osprey sim list` take
`--target va|standin`, defaulting to the deployment's own target when it runs
that instance and else to `va`, and
`simulation.default_scenarios` also accepts `{va: [...], standin: [...]}`; a
list still starts every instance in the set. On a deployment that runs the
stand-in, `osprey sim apply` without `--target` no longer moves both machines:
it moves that one target and names it. The logbook and the stored archive
follow the `va` set, so a set applied to the stand-in changes telemetry only.
On a deployment whose own target is the stand-in, `osprey sim apply` without
`--target` therefore no longer reseeds the logbook or the stored archive; use
`--target va` to change the set they follow. The split takes effect once the
deployment is built again: run `osprey build`, then `osprey up`. The first
`osprey up` after that gives a stand-in that was already running its own start
set and prints the set it followed until then. Until the build, both instances
follow the `va` set, the commands say so, and `--target standin` is refused. A
deployment with one instance behaves and renders as before.
