The control-assistant, ARIEL and channel-finder presets and hello-world now ship
their facility description as committed sources under `data/facility/`, and
`osprey build` writes the facility file built from them as `facility.json` at
every render root. `osprey facility validate` checks that tree and renders the
facility file without writing anything into the repo.

The control-assistant demo serves four addresses beyond its previous set: the readbacks
`SR:DIAG:TUNE:X`, `SR:DIAG:TUNE:Y`, `SR:DIAG:CHROM:X` and `SR:DIAG:CHROM:Y`,
computed from the deck's optics. The first cavity's existing
`SR:RF:CAVITY:01:FREQUENCY:SP` and `SR:RF:CAVITY:01:FREQUENCY:RB` are now wired
to the deck's cavity frequency (MHz on the channel, Hz in the deck); the second
cavity stays unwired.

The demo's `data/facility/limits.yaml` holds exactly three teaching records and
no defaults block: `SR:MAG:HCM:01:CURRENT:SP` bounded to [-12.0, 12.0];
`SR:RF:CAVITY:01:FREQUENCY:SP` bounded to [500.0, 500.8] MHz with a largest
single step of 0.01 MHz, a band re-based around the deck's 500.417 MHz where
the packaged `channel_limits.json` carries [499.0, 500.3]; and
`SR:VAC:ION-PUMP:01:VOLTAGE:SP` not writable. Every other channel follows the
deployment's limits setting. The build still ships the packaged
`channel_limits.json` until the limits view renders `limits.yaml`, so the
re-based cavity band, and any record a deployment adds to `limits.yaml`,
reaches a render only from that view.

The channel-finder standalone preset serves more channels. New SR families
`SR:MAG:QFA`, `SR:MAG:SHD` and `SR:MAG:SHF` (devices 01-24, each with
`CURRENT:SP/RB/GOLDEN` and `STATUS:ON/READY/FAULT`) and the `SR:DIAG:TUNE:X/Y`
and `SR:DIAG:CHROM:X/Y` readbacks join it; `SR:DIAG:BPM`, `SR:MAG:HCM` and
`SR:MAG:VCM` grow from 20 to 72 devices, `SR:MAG:DIPOLE` from 24 to 36,
`SR:MAG:QD` and `SR:MAG:QF` from 16 to 24, and `SR:MAG:SD` and `SR:MAG:SF` from
12 to 24. Every address the preset served before is still served.
