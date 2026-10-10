Every render of `osprey build` now writes the simulator view into
`data/simulator/`: `served_models.json`, `addresses.json` and a byte copy of
each deck-bearing model's deck under `decks/<model>.json`. No reader consumes
them yet.
