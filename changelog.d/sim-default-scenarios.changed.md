A fresh control-assistant deployment now starts in the `rf-thermal` scenario:
`osprey up` activates the scenarios the new profile key
`simulation.default_scenarios` names (a list of scenario names; absent or `[]`
starts in `nominal` alone) when the deployment has never chosen a set, and seeds
that set's archived history and logbook narrative. The control-assistant preset
states `[rf-thermal]`; a value that is not a list of names stops the build
with `profile-invalid`. `osprey sim apply` still means exactly the set it names,
and a deployment that already chose a set (including `nominal` alone) keeps it.
