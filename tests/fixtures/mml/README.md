# MML fixtures

Middle Layer (MML) exports for the mml layer's tests, one directory per fixture. Seven
are invented (below); two are real facility exports made with the shipped exporter
(last section). A tree that carries a whole export keeps its reviewed mapping at
`imported/mml/mapping.yaml`, and `_trees.py` states what each such tree is held to.

## Synthetic fixtures

Every name in the invented export files (the facility "Quokka", its systems, families
and channel names) is made up; no facility's real data lives in them.

Every fixture carries the same five census and hazard cases, so those code paths run
on every input form:

- at least one **zero-channel family** (a `DeviceList` but no channel key on any field);
- at least one **broadcast row** (a one-slot channel list on a family with several devices);
- at least one **blank slot** (an empty or whitespace entry inside a channel list);
- a **shared PV** bound by two different devices (not counting broadcast rows);
- `HWUnits` in all three shapes: `[]`, a plain string such as `"Amps"`, and a per-device list.

`tests/facility/test_mml_fixtures_wellformed.py` checks all of the above.

| Directory | Form | System token | What it is for |
|-----------|------|--------------|----------------|
| `tango/` | flat `export.json` | given by the caller: `RING` | A Tango facility: every field carries `TangoNames` only, never `ChannelNames`. Also has a one-device family whose `DeviceList` is a flat pair and whose channel list is a bare string. |
| `dualkey/` | flat `export.json` | given by the caller: `STOR` | `ChannelNames` and `TangoNames` staged on the same field (`SF.Monitor`, and a broadcast pair on `SF.Setpoint`), the way Elettra's `elettrainit.m` stages both on one field. |
| `casedup/` | flat `export.json` | given by the caller: `MAIN` | Two families in one system whose names differ only by case, `BPMx` and `bpmx`, the shape SIRIUS uses. `CH.Monitor` also lists four channels for two devices, so its last two rows, `MN-CH:Sum-Mon` and `MN-CH:Spare-Mon`, are judgments. |
| `wrapped/` | `{"ao": {...}}` | given by the caller: `INJ` | A flat AO wrapped under a single `ao` key. Also has the MML `On` / `OnControl` field pair with no `MemberOf` tags, so the direction vote has an undecided field. `QM.Monitor` reads `IJ:QM2:RB` on devices 2 and 3, a supply group a reviewer answers with `keep_all` or an owner map. |
| `dialect/` | system-keyed `export.json` (at most 30 lines) | its own keys `RING`, `BOOST`; refuses a caller's token | The system-keyed JSON dialect: quoted `"inf"` / `"-inf"` / `"NaN"` strings (bare `inf` is not JSON), bare `NaN` and `Infinity` tokens, `HW2PhysicsFcn: 1`, bare-string function handles, a `function_handle` record under the typo key `HW2PhysicSDcn`, a `Handles` key, whitespace blanks, JSON booleans in `Status`, family arrays under `setup` with one family keeping them at family level, and a system `_description`. `TUNE` in `BOOST` lists two channels for three devices, so its third device is a judgment. |
| `paired/` | `quokka.ring.ao.json` + `quokka.ring.ad.json` | none needed: `AD.SubMachine` = `RING` | The exporter's paired output. The `_export` block names no sub-machine, so the system token must come from the sibling `.ad.json`. The AD file carries the machine scalars (`Machine`, energy, circumference, harmonic number, MCF, lattice). `SQ.Monitor` lists four channels for three devices, so its last row, `QK:SQ4:RB`, is a judgment. |
| `mat/` | `quokka_booster.mat` (MATLAB v7) | none needed: `AD.SubMachine` = `BOOSTER` | A `saveao`-style `.mat` with `AO` and `AD` variables: padded char matrices with an all-blank row, cell arrays, a single-row char matrix as a broadcast list, an empty double `[]`, logical `Status` and a non-finite `Range`. `build_mat.py` wrote it with `scipy.io.savemat`; rerun it to regenerate, so no test needs MATLAB. |

## Real facility exports

Made with the shipped `mml_export.m` from an `MML-prod` checkout in simulator mode;
each directory's own `README.md` records the MATLAB, the tree and the program used.
`tests/templates/test_mml_export_parity.py` re-runs the export when a MATLAB is on
PATH and compares it with the committed files.

| Directory | Form | System token | What it is for |
|-----------|------|--------------|----------------|
| `nsls2/` | `nsls2.storagering.ao.json` + `.ad.json`, `nsls2.ltb.ao.json` + `.ad.json` | none needed: each `.ad.json` names its sub-machine | NSLS-II storage ring and LTB transfer line: two systems in one import, a zero-channel `Screen` family, `RBKL`/`SPKL` strength fields beside the current fields, turn-by-turn BPM fields. |
| `spear3/` | `spear3.storagering.ao.json` + `.ad.json` | none needed: `AD.SubMachine` = `StorageRing` | SPEAR3 storage ring: 43 families including beamline, vacuum and injection signals; a broadcast `TUNE` channel; monitor and setpoint on one record (`RF`, `ShuntCurrent`); nine new classes declared under packaged parents. |
