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
