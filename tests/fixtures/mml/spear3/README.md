# SPEAR3 export

A real Middle Layer (MML) export of SPEAR3, the 3 GeV light source of SSRL at SLAC
National Accelerator Laboratory: the storage ring, as the five files the shipped
`mml_export.m` writes for one sub-machine. `mapping.yaml` is the reviewed mapping;
every slot is `stated`, so it passes `osprey mml map --check --no-derived`.

| File | What it is |
|------|------------|
| `spear3.storagering.ao.json` | Storage ring Accelerator Objects: 43 families in 3 GeV user mode, including the transfer-line BPMs, beamline signals, vacuum instrumentation and injection elements the Middle Layer stages under the ring |
| `spear3.storagering.ad.json` | Storage ring Accelerator Data |
| `spear3.storagering.lattice.mat` | The deck the export was sampled over: `sp3v82`, 876 elements, no `RingParam` |
| `spear3.storagering.va.json` | Per family: the conversion between supply current and physics, the nominal each device sits at, the dipole ramp, and every refusal |
| `spear3.storagering.response.json` | The stored orbit-response matrix, four blocks of BPM against corrector |
| `mapping.yaml` | Reviewed mapping; nine new classes are declared under packaged parents (`BendTrim`, `BeamlineMonitor`, `OrbitInterlock`, `CorrectorCurrentReference`, `InjectionKicker`, `InjectionSeptum`, `MachineStatus`, `QuadrupoleShunt`, `SkewQuadrupole`, `TuneMonitor`), and a `virtual_accelerator` block deciding all 43 families |

Import it with:

```
osprey mml import spear3.storagering.ao.json
```

The four siblings are never named on the command line. `mml import` pairs each of
them with the `ao.json` it sits beside.

## Shapes worth knowing

`TUNE` broadcasts one channel to every device, and `MachineParameters`,
`ShuntCurrent` and `ShuntRelay` each stage the same record as monitor and setpoint.
`RF.Setpoint` and `RF.Monitor` are the same address, `SPEAR:RFFreqSetpt`, so the
cavity is read back through the record it is written to. `HCM` and `VCM` list 78
devices but stage channels for 76 and 74 of them. Seventeen families share
power-supply records between magnets, which is every judgment the export asks for;
all seventeen are answered `keep_all`, so every device keeps its reading.

`Q9S` device `[9, 2]` binds only one lattice element where the other two bind a
pair: its `at_index` row reads `[440, "NaN"]`. The Middle Layer pads a stale
entry of the facility's physics-data file rather than failing, and the export
carries the padding faithfully.

The same device sits at −52.6 A against a stated `Range` of 0 to 98 A, so the
band the emitted tree writes into was widened to hold it (−52.5739 to 98). That
is a fact about the facility's file, not about the exporter.

## What the virtual accelerator does with it

21 families are driven and 22 stand still. `BEND` is the energy knob, `RF` the
cavity, `BPMx`/`BPMy` the monitors, and the quadrupole and sextupole families drive
strengths. Fifteen supplies feed their magnets in series, the widest being `SD` and
`SF` at 28 magnets each; every string is one knob with a fixed share per magnet.

Three families the type table does not know are answered `latch` in the mapping,
with the reason written beside the answer: `KickerAmp`, `KickerDelay` and `Septum`
are injection elements and are not in the stored-beam model.

## How the export was made

- **Exporter:** `mml_export 2.0.0`, the script shipped at
  `src/osprey/templates/apps/control_assistant/data/mml/mml_export.m`.
- **MATLAB:** 26.1.0.3276743 (R2026a) Update 3, Linux x86_64, on the host `appsml`.
- **Middle Layer:** the `MML-prod` tree as synced 2026-05-28. It is a plain synced
  folder with no version history, so there is no commit id to quote.
- **Accelerator Toolbox:** the AT 2.0 that `MML-prod` bundles under
  `simulators/at2.0`, which the machine's own `setpathmml` puts on the path before
  it initialises the Accelerator Objects.
- **Command:** `~/mml-reexport/run_reexport.sh spear3`, which runs one fresh MATLAB
  for the sub-machine. The full runbook is
  `.claude/scratch/handoffs/2026-09-18-mml-reexport-runbook.md`; the console and
  probe logs of this run are under `.claude/scratch/handoffs/reexport-logs/`.
- **Two symlinks the host needed:** Linux is case-sensitive and the Middle Layer
  asks for `machine/SPEAR3/...` and `SPEAR3physdata.mat` where the checkout spells
  both `Spear3`. Without `machine/SPEAR3 -> Spear3` and
  `Spear3/StorageRingOpsData/SPEAR3physdata.mat -> Spear3physdata.mat` the golden
  response file and the physics data are skipped silently and the export measures
  the model instead. A fresh checkout needs them again.
- **Date:** 2026-09-19.

The link method defaulted to LabCA, which is not installed on that host, so the
Middle Layer warned once and every family was switched to simulator mode before the
export.

`tests/templates/test_mml_export_parity.py` re-runs this program when a MATLAB is
available and compares the fresh export with these files.

## What the probes found

- The orbit-response matrix is **measured**, read from
  `machine/SPEAR3/StorageRingOpsData/User/GoldenBPMResp` and stamped 2006-10-24. Its
  four blocks are 57 monitors against 58 (`HCM`) and 56 (`VCM`) correctors, in
  meter/radian at 2.99988 GeV against a deck built for 3 GeV.
- `getpvmodel(..., 'Struct')` errors on this facility ("dot indexing is not
  supported for variables of type double"), so the export takes the numeric path.
- The ring-name digest agrees three ways: MATLAB, the committed `va.json`, and
  Python reading `lattice.mat` all answer
  `b07cf1f2aaa089464f31603526db1ce5653dfe19fe549007c86fb73e43cc3694`.
- Every quadrupole conversion round-trips to about 1e-16. The widest deviation over
  all families is 7.9e-15, on `BEND`.

## Refusals that are facility oddities, not bugs

Four families cost the export something, each for a reason that lives in the
facility's own files:

| Family | What the export could not take |
|--------|-------------------------------|
| `TUNE` | "The length of ValPhysics must match the number of devices" |
| `MachineParameters` | "Index exceeds array bounds" |
| `HCMCurrReference` | "Index exceeds the number of array elements. Index must not exceed 1" |
| `VCMCurrReference` | the same |

None of them is a family the model drives, and none is worth repairing in the
Middle Layer.
