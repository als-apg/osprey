The framework types the connectors' closed vocabularies with their named types
(control targets, machine kinds, switch positions, processing modes, directory
status and the simulation view's words). `osprey.facility.TEXTURE` is removed:
the texture model's name is `osprey_connectors.simulation.TEXTURE`. The PV
directory's `directory_search` tool now states `status` as `"Active"` or
`"Inactive"` in its schema and refuses any other word, which used to reach the
directory and match nothing.
