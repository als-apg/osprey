The bundled `archive` service (`template: osprey.archive`) runs `osprey archive
--watch` once a day into the host directory `var/archive/`, and `osprey health`
warns when its last pass is stale. `var/archive/` is set to mode 2770 and
shares the agent record's group, and only the archive service and the agent
record mount it. The control-assistant preset deploys it.
