# Synthetic MML fixtures

Invented Middle Layer (MML) exports for the `osprey mml` tests. Every name in the
export files (the facility "Quokka", its systems, families and channel names) is made
up; no facility's real data lives here. Each fixture sits in its own directory, and
each directory later gains a hand-filled `mapping.yaml` beside the export.

Every fixture carries the same five census and hazard cases, so those code paths run
on every input form:

- at least one **zero-channel family** (a `DeviceList` but no channel key on any field);
- at least one **broadcast row** (a one-slot channel list on a family with several devices);
- at least one **blank slot** (an empty or whitespace entry inside a channel list);
- a **shared PV** bound by two different devices (not counting broadcast rows);
- `HWUnits` in all three shapes: `[]`, a plain string such as `"Amps"`, and a per-device list.

`tests/services/mml/test_fixtures_wellformed.py` checks all of the above.

| Directory | Form | System token | What it is for |
|-----------|------|--------------|----------------|
| `tango/` | flat `export.json` | `--system RING` | A Tango facility: every field carries `TangoNames` only, never `ChannelNames`. Also has a one-device family whose `DeviceList` is a flat pair and whose channel list is a bare string. |
| `dualkey/` | flat `export.json` | `--system STOR` | `ChannelNames` and `TangoNames` staged on the same field (`SF.Monitor`, and a broadcast pair on `SF.Setpoint`), the way Elettra's `elettrainit.m` stages both on one field. |
| `casedup/` | flat `export.json` | `--system MAIN` | Two families in one system whose names differ only by case, `BPMx` and `bpmx`, the shape SIRIUS uses. The mapping resolves them with `rename`. |
| `wrapped/` | `{"ao": {...}}` | `--system INJ` | A flat AO wrapped under a single `ao` key. Also has the MML `On` / `OnControl` field pair with no `MemberOf` tags, so the direction vote has an undecided field. |
| `dialect/` | system-keyed `export.json` (at most 30 lines) | its own keys `RING`, `BOOST`; refuses `--system` | The system-keyed JSON dialect: quoted `"inf"` / `"-inf"` / `"NaN"` strings (bare `inf` is not JSON), bare `NaN` and `Infinity` tokens, `HW2PhysicsFcn: 1`, bare-string function handles, a `function_handle` record under the typo key `HW2PhysicSDcn`, a `Handles` key, whitespace blanks, JSON booleans in `Status`, family arrays under `setup` with one family keeping them at family level, and a system `_description`. |
| `paired/` | `quokka.ring.ao.json` + `quokka.ring.ad.json` | none needed: `AD.SubMachine` = `RING` | The exporter's paired output. The `_export` block names no sub-machine, so the system token must come from the sibling `.ad.json`. The AD file carries the machine scalars (`Machine`, energy, circumference, harmonic number, MCF, lattice). |
| `mat/` | `quokka_booster.mat` (MATLAB v7) | none needed: `AD.SubMachine` = `BOOSTER` | A `saveao`-style `.mat` with `AO` and `AD` variables: padded char matrices with an all-blank row, cell arrays, a single-row char matrix as a broadcast list, an empty double `[]`, logical `Status` and a non-finite `Range`. `build_mat.py` wrote it with `scipy.io.savemat`; rerun it to regenerate, so no test needs MATLAB. |
