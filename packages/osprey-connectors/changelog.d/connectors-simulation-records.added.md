`osprey_connectors.simulation.records` names the shapes of the records a
simulator view carries (channels, seeds, scenarios, archiver events, topoff
records and the documents that hold them). `SimulatorView.document()`,
`seed()` and `scenarios()` return them, and the simulation functions that read
those records take them.
