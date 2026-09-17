# SPEAR3 export

A real Middle Layer (MML) export of SPEAR3, the 3 GeV light source of SSRL at SLAC
National Accelerator Laboratory: the storage ring, as the paired
`spear3.storagering.ao.json` / `.ad.json` files the shipped `mml_export.m` writes.
`mapping.yaml` is the reviewed mapping; every slot is `stated`, so it passes
`osprey mml map --check --no-derived`.

| File | What it is |
|------|------------|
| `spear3.storagering.ao.json` | Storage ring Accelerator Objects: 43 families in 3 GeV user mode, including the transfer-line BPMs, beamline signals, vacuum instrumentation and injection elements the Middle Layer stages under the ring |
| `spear3.storagering.ad.json` | Storage ring Accelerator Data |
| `mapping.yaml` | Reviewed mapping; nine new classes are declared under packaged parents (`BendTrim`, `BeamlineMonitor`, `OrbitInterlock`, `CorrectorCurrentReference`, `InjectionKicker`, `InjectionSeptum`, `MachineStatus`, `QuadrupoleShunt`, `SkewQuadrupole`, `TuneMonitor`) |

Shapes worth knowing: `TUNE` broadcasts one channel to every device, `RF.Monitor` and
`RF.Setpoint` share one record, and `MachineParameters`, `ShuntCurrent` and `ShuntRelay`
each stage the same record as monitor and setpoint. `HCM` and `VCM` list 78 devices but
stage channels for 76 and 74 of them. Seventeen families share power-supply records
between magnets, which is every judgment the export asks for; all seventeen are
answered `keep_all`, so every device keeps its reading.

Import it with:

```
osprey mml import spear3.storagering.ao.json
```

## How the export was made

- **Exporter:** `mml_export 1.0.0`, the script shipped at
  `src/osprey/templates/apps/control_assistant/data/mml/mml_export.m`.
- **MATLAB:** 26.1.0.3276743 (R2026a) Update 3, Linux x86_64.
- **Middle Layer:** the `MML-prod` tree (no git metadata in the copy used; the tree's
  `mml/setpathmml.m` has MD5 `f1d2c1ed6bd2648de6785f5a221fd9d5` and
  `machine/SPEAR3/StorageRing/spear3init.m` MD5 `78b56764e59f6fd22ab1f8e0f46eac17`).
- **Accelerator Toolbox:** the AT 2.0 that `MML-prod` bundles under `simulators/at2.0`,
  which the machine's own `setpathmml` puts on the path before it initialises the
  Accelerator Objects. Its integrators were recompiled with `atmexall` for the MATLAB
  above, because the bundled binaries predate AT 2.0's tracking API.
- **Program:** `addpath(<mml>); setpathspear3; setpathat(<at>); switch2sim;
  addpath(<exporter dir>); mml_export(<outdir>)`. The link method defaulted to LabCA,
  which was not installed, so the Middle Layer warned once and every family was
  switched to simulator mode before the export.
- **Date:** 2026-09-16.

`tests/templates/test_mml_export_parity.py` re-runs this program when a MATLAB is
available and compares the fresh export with these files.
