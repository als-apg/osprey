The scenarios a deployment that has never chosen a set starts in are now the
profile key `simulation.default_scenarios` (a list of scenario names; absent or
`[]` starts in `nominal` alone) instead of `default_scenarios` in the
simulation `machine.json`, which is no longer read. The control-assistant
preset states `[rf-thermal]`, so its deployments start where they did.

Upgrade notes: a profile written from the control-assistant preset before this
change gains no start set until `simulation.default_scenarios: [rf-thermal]` is
added to its `config:` block.
