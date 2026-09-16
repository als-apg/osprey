# NSLS-II export

A real Middle Layer (MML) export of NSLS-II, the 3 GeV light source at Brookhaven
National Laboratory: the storage ring and the linac-to-booster transfer line (LTB),
each as the paired `<machine>.<submachine>.ao.json` / `.ad.json` files the shipped
`mml_export.m` writes. `mapping.yaml` is the reviewed mapping; every slot is
`stated`, so it passes `osprey mml map --check --no-derived`.

| File | What it is |
|------|------------|
| `nsls2.storagering.ao.json` | Storage ring Accelerator Objects: 25 families, 3 GeV operational mode |
| `nsls2.storagering.ad.json` | Storage ring Accelerator Data (machine scalars, lattice name, directories) |
| `nsls2.ltb.ao.json` | LTB Accelerator Objects: 7 families at 200 MeV, one with no channels (`Screen`) |
| `nsls2.ltb.ad.json` | LTB Accelerator Data |
| `mapping.yaml` | Reviewed mapping for both systems; `SQ` and `TUNE` are new classes under `Quadrupole` and `Instrumentation` |

Import both systems at once:

```
osprey mml import nsls2.storagering.ao.json nsls2.ltb.ao.json
```

## How the export was made

- **Exporter:** `mml_export 1.0.0`, the script shipped at
  `src/osprey/templates/apps/control_assistant/data/mml/mml_export.m`.
- **MATLAB:** 26.1.0.3276743 (R2026a) Update 3, Linux x86_64.
- **Middle Layer:** the `MML-prod` tree (no git metadata in the copy used; the tree's
  `mml/setpathmml.m` has MD5 `f1d2c1ed6bd2648de6785f5a221fd9d5` and
  `machine/NSLS2/StorageRing/nsls2init.m` MD5 `a47495ead2ff202e0ffc4fc128a4e913`).
- **Accelerator Toolbox:** the AT 2.0 that `MML-prod` bundles under `simulators/at2.0`,
  which the machine's own `setpathmml` puts on the path before it initialises the
  Accelerator Objects. Its integrators were recompiled with `atmexall` for the MATLAB
  above, because the initialisation computes the momentum compaction factor and the
  bundled binaries predate AT 2.0's tracking API.
- **Program:** `addpath(<mml>); setpathnsls2('StorageRing'); setpathat(<at>); switch2sim;
  addpath(<exporter dir>); mml_export(<outdir>)`, and the same with `setpathnsls2('LTB')`.
  The link method defaulted to LabCA, which was not installed, so the Middle Layer warned
  once and every family was switched to simulator mode before the export.
- **Date:** 2026-09-16.

`tests/templates/test_mml_export_parity.py` re-runs this program when a MATLAB is
available and compares the fresh export with these files.
