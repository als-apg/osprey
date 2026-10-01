`osprey build` now writes the simulator view's skeleton into `data/simulator/`
(`served_models.json`, `addresses.json` and `decks/<model>.json`), and the
config key `simulation.models` selects the models that view serves. No reader
consumes the view or the key yet.
