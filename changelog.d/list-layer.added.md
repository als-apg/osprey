`osprey facility import list FILE` imports a CSV channel list as the list layer's sources under
`data/facility/imported/list/`: a header row names `address` and any of `role`, `pair`, `device`,
`place`, `unit`, `description`, `tags`, `s` and `model`, and a file without a header holds one
address per line. A row's `s` and `model` are its device's position, which `osprey build` places
by the span that contains it.
