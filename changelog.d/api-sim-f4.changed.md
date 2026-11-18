`osprey.facility.views.simulator` no longer re-exports the simulator view's
file and schema constants, `DECKS_DIR` or `SCENARIOS_DIR`: take a file's name
and schema from its document in `osprey_connectors.simulation.records`
(`VariablesDocument.FILE`, `VariablesDocument.SCHEMA`) and the directories from
`osprey_connectors.simulation.view`.
