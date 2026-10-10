pyAML (`accelerator-middle-layer` 0.3.1) ships with OSPREY as the `pyaml-cs-osprey`
control-system backend: every pyAML read and write goes through the OSPREY
connector, so channel limits, write posture, control-target pinning and audit
apply to it. A pyAML write runs only inside a journaled guarded run; anywhere
else it is refused and nothing is written.

`osprey build` writes a pyAML view for every served model that names a deck and
has a `measurement/<model>.yaml`: `data/pyaml/<model>/configuration.yaml`, with
the deck beside it as the design simulator's lattice (a `single_pass` model has
no design simulator), one array per measurement group, one tool per measurement
kind and, for a periodic model, the tune and chromaticity response matrices of
its design optics. Magnets are named after their setpoint addresses, BPMs and
arrays after their device and group ids. A served model without a view is named
in a note on stderr. `data/pyaml/` is written by the build only; a profile's
`project/` mirror may not carry it.

A measurement file is held to the kinds it allows: `orm` needs groups `bpm`,
`hcor` and `vcor`; `dispersion` adds instrument `rf`; `trm` needs group `quad`
and instrument `tune`; `crm` needs group `sext` and instrument `chromaticity`;
`chromaticity_monitor` needs instruments `tune` and `rf`. A member the file does
not name or the model does not wire, or any kind but `orm` on a `single_pass`
model, stops the build (`reference-missing`); a step or settle key the kind's
tool takes, missing, stops it (`value-invalid`). Group `hcor` stands for the
setpoints its model steers horizontally and `vcor` for those it steers
vertically, so a corrector steered in both planes is split between them.

`data/facility_facts.json` records each pyAML view the render wrote under
`measurement_models` (the configuration's path and digest, and the digest of
each file beside it), and the rendered `.claude/hooks/hook_config.json` gains
`measurement`: per model, the kinds the measurement file allows with the arrays
each steps or reads, the addresses each group stands for, and the view's digest.
