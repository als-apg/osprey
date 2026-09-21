A deployment's Virtual Accelerator is now built from the facility's MATLAB
Middle Layer instead of from physics written into OSPREY, demo ring included.

- `mml_export.m` 2.0 adds `<machine>.<submachine>.lattice.mat`, `.va.json` and
  `.response.json`; `osprey mml import` files one deck per system as
  `data/mml/lattice/<system>.mat`, keys the other two by system, and replaces
  the last import whole, naming every sibling it removed.
- `osprey mml map --init` proposes a `virtual_accelerator:` verdict per family
  and leaves a question open where no rule decides: units or a lattice type the
  table cannot read, a field more than one family drives, a family the Middle
  Layer reaches through a special function or a parameter group.
- `osprey mml emit` writes `data/simulation/lattice.json`, `va_bindings.json`
  and `machine.json`, `data/machine_state_channels.json` and
  `data/channel_limits.json`. In that shared limits file it stamps each band it
  writes, keeps every other entry byte-for-byte, and refuses an address the
  file already bands differently without that stamp.
- `osprey mml verify` compares the emitted model's orbit response with the
  facility's measured one and writes `data/mml/VA-REPORT.md`.
- `VA_LATTICE` now names the lattice file to serve, relative to the data
  directory, or `none`. `VA_LATTICE=builtin` is gone and `osprey-connectors`
  defaults to `none`: a deployment that set `builtin` names its lattice file
  or sets `none`, and `osprey build` derives the value from the built tree.
- The model RPC's model-only variables are the served tree's own: a monitor
  the bindings publish a reading for carries the nine reading-error fields, a
  magnet they drive carries a calibration factor and offset, and each is named
  `<element>.<field>` for the element the deck spells it at. `VA_BPM_ERRORS`
  accepts either spelling of a device — the address its reading goes out on or
  the element it sits at — and refuses a name the tree knows under neither.
- OSPREY requires `lume-pyat` 0.2.0 for sliced magnets and the energy knob.
