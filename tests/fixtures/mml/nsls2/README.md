# NSLS-II export

A real Middle Layer (MML) export of NSLS-II, the 3 GeV light source at Brookhaven
National Laboratory: the storage ring and the linac-to-booster transfer line (LTB),
each as the five files the shipped `mml_export.m` writes for one sub-machine.
`mapping.yaml` is the reviewed mapping; every slot is `stated`, so it passes
`osprey mml map --check --no-derived`.

| File | What it is |
|------|------------|
| `nsls2.storagering.ao.json` | Storage ring Accelerator Objects: 25 families, 3 GeV operational mode |
| `nsls2.storagering.ad.json` | Storage ring Accelerator Data (machine scalars, lattice name, directories) |
| `nsls2.storagering.lattice.mat` | The deck the ring export was sampled over: `nsls2atlat2014March`, 3510 elements, no `RingParam` |
| `nsls2.storagering.va.json` | Per family: the conversion between supply current and physics, the nominal each device sits at, and every refusal |
| `nsls2.storagering.response.json` | The stored orbit-response matrix, four blocks of 180 monitors against 180 correctors |
| `nsls2.ltb.*` | The same five files for the transfer line: 7 families at 200 MeV, a 121-element deck, one family with no channels (`Screen`) |
| `mapping.yaml` | Reviewed mapping for both systems; `SQ` and `TUNE` are new classes under `Quadrupole` and `Instrumentation`, and the `virtual_accelerator` block decides the 25 storage-ring families |

Import both systems at once:

```
osprey mml import nsls2.storagering.ao.json nsls2.ltb.ao.json
```

The eight siblings are never named on the command line. `mml import` pairs each of
them with the `ao.json` it sits beside, and files them under their own sub-machine.

## Shapes worth knowing

The export asks its reviewer for thirteen judgments. Ten families share
power-supply records between magnets and are answered `keep_all`, so every device
keeps its reading. `DCCT` lists one device but stages three `Monitor` channels, and
the two beyond it become fields of their own, `Lifetime` and `Total`. Both carry
the export's `mA`, because the unit sits on the whole `Monitor` field and the export
states no other for them, which is true of the total current and not of a lifetime.
`TUNE` lists three devices and stages channels for two, and the third is dropped.

The `virtual_accelerator` block names one sub-machine, and here that is
`StorageRing`. The LTB's own `va.json` and deck are imported and kept, and a
mapping naming `LTB` instead would serve the transfer line from them.

## What the virtual accelerator does with it

17 storage-ring families are driven and 8 stand still. The export leaves the
reviewer **one** question, which is the cavity voltage below.

Three families stand still for reasons worth knowing:

- **`BEND`** — the facility's `bend2gev` answers the same 3 GeV at every current, so
  the ramp is a constant and not an energy knob. The export also states a NaN
  nominal for it at 3 GeV.
- **`SM1`, `SH3`, `SH4`, `SL1`, `SL3`** — the facility's `k2amp` answers nothing
  usable for a negative sextupole strength, so these 150 devices carry no hardware
  nominal and the model has nowhere to start them. `map --init` proposes them as
  `latch` with that reason written beside the verdict; a reviewer may overrule it.

### The cavity is built, not exported

This deck carries no cavity element at all, so `osprey mml emit` builds one: a
single zero-length cavity on harmonic 1320, which is the number the Accelerator
Data states, at the deck's own revolution frequency. The `RF` family then couples
to it the way it would on any ring, and the served model solves in six dimensions
instead of four.

Two frequencies appear in the emit line and they are not the same number. The
export states 499 680 000 Hz, rounded to five figures, and the deck closes 1320
waves at 499 680 595 Hz. The built cavity uses the deck's figure. A frequency
rounded to five figures cannot set a ring's closed orbit: building at the stated
one puts the beam at a momentum offset of about 3e-3 and rescales the whole orbit
response by some three per cent. The exported figure is what triggers the build and
what seeds the channel, and it is printed beside the built one so the gap is visible.

The voltage is the reviewer's answer, not an export fact. The export states none,
so the committed mapping answers 3 MV, which is what a storage ring of this size
runs at. It sets the synchrotron tune and never the orbit, which is why the export
not stating it costs the orbit-response comparison nothing.

85 supplies feed their magnets in series, two to six magnets each; every string is
one knob with a fixed share per magnet.

The deck marks each girder's start and end with a zero-length element the facility
built with the beam-position-monitor type: 180 named `GE` and 180 named `GS`. No
family reads them and the names repeat, so `osprey mml emit` serves all 360 as plain
markers, same name, same place, nothing to read off them. Bind a family to one of
those names and they come back as monitors.

The LTB's seven `BPMx` and seven `BPMy` monitors drive nothing either. The export
could sample neither, and says why in its own words, quoted in the refusals table
below. The transfer line is served with those positions as markers too.

## How the export was made

- **Exporter:** `mml_export 2.0.0`, the script shipped at
  `src/osprey/templates/apps/control_assistant/data/mml/mml_export.m`.
- **MATLAB:** 26.1.0.3276743 (R2026a) Update 3, Linux x86_64, on the host `appsml`.
- **Middle Layer:** the `MML-prod` tree as synced 2026-05-28. It is a plain synced
  folder with no version history, so there is no commit id to quote.
- **Accelerator Toolbox:** the AT 2.0 that `MML-prod` bundles under
  `simulators/at2.0`, which the machine's own `setpathmml` puts on the path before
  it initialises the Accelerator Objects.
- **Commands:** `~/mml-reexport/run_reexport.sh nsls2-sr` and
  `~/mml-reexport/run_reexport.sh nsls2-ltb`, one fresh MATLAB each. The full
  runbook is `.claude/scratch/handoffs/2026-09-18-mml-reexport-runbook.md`; the
  console and probe logs of this run are under
  `.claude/scratch/handoffs/reexport-logs/`.
- **Date:** 2026-09-19.

NSLS-II ships no physics-data file, so none of its numbers come from one. The link
method defaulted to LabCA, which is not installed on that host, so the Middle Layer
warned once and every family was switched to simulator mode before the export.

`tests/templates/test_mml_export_parity.py` re-runs both programs when a MATLAB is
available and compares the fresh exports with these files.

## What the probes found

- The storage ring's orbit-response matrix names its file,
  `machine/NSLS2/StorageRingOpsData/User/GoldenBPMResp`, and the matrix in that file
  is itself **model**-made, stamped 2014-03-23 in m/rad at 3 GeV. Its monitor mode
  reads `Model` and its actuator mode `SIMULATOR`.
- The LTB's matrix comes from `machine/NSLS2/LTBOpsData/50MEV/GoldenBPMResp_LTB`,
  also **model**, stamped 2012-11-15, six monitors against eight correctors at
  0.2 GeV. The file is filed under `50MEV` while the deck runs at 200 MeV.
- `getpvmodel(..., 'Struct')` errors on this facility ("dot indexing is not
  supported for variables of type double"), so the export takes the numeric path.
- The ring-name digest agrees three ways on both sub-machines: MATLAB, the committed
  `va.json`, and Python reading `lattice.mat`. The storage ring answers
  `5a06e13f0c31cf2701700401834597a39a438bdc104449fa810b3587fe2f31e2`, the LTB
  `2e83d52efd0cf28cb0733356d8092a3f7585a17258a2afe120de8a7b9ff20ca3`.
- Every quadrupole conversion round-trips to about 1e-16.
- The top of every storage-ring quadrupole `Range` converts to NaN: a `Range` of
  0 to 200 A reaches physics `[0, NaN]`. The conversion is sampled only as far as
  it answers, so the grid stops short of the stated top.

## Refusals that are facility oddities, not bugs

| Sub-machine | Family | What the export could not take |
|-------------|--------|-------------------------------|
| StorageRing | `BEND` | "Index exceeds the number of array elements. Index must not exceed 2", and the nominal at 3 GeV is NaN, so the ramp has no nominal |
| StorageRing | `TUNE` | "The length of ValPhysics must match the number of devices" |
| StorageRing | `SM1`, `SH3`, `SH4`, `SL1`, `SL3` | `k2amp` answered with nothing usable over the negative-strength grid, for every device |
| LTB | `BPMx`, `BPMy` | `getpvmodel` answered with 350 values for 7 devices — a 50-turn history where one reading per device was asked for |

None of these is repaired in the Middle Layer. Each is carried as what the facility
states, and the verdict rules decide what the model does about it.
