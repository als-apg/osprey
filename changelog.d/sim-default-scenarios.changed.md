A fresh control-assistant deployment now starts in the `rf-thermal` scenario:
`osprey up` activates the machine model's new `default_scenarios` list when the
deployment has never chosen a set, and seeds that set's archived history and
logbook narrative. `osprey sim apply` still means exactly the set it names, and a
deployment that already chose a set (including `nominal` alone) keeps it.
