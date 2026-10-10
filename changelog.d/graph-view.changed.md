**Breaking change:** the knowledge graph is the graph view `osprey build` writes from the facility file, and three things differ from the hand-written corpus it replaces.

- Every node IRI is minted from the facility identity's `code` rather than the word `demo`: a device's local name is `<code>_device_<escaped id>` under `https://narad.example.org/device/`, and places, channels and groups follow the same `<code>_<kind>_` form. A query or a saved link that spells an old `demo_` IRI no longer matches; `deviceId` and `bindingId` carry the raw ids.
- A device's ordinals are `ordinalInPlace` (its position among the devices of its class in its place) and `ordinalInModel` (among the devices of its class in its model); `ordinalInFacility` and `ordinalInSection` are gone.
- `sPositionM` and `lengthM` are the device's position and length in metres along the lattice, written only when the facility file carries them, where the hand corpus carried placeholder positions.
