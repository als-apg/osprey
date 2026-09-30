# MATLAB Middle Layer Export

`mml_export.m` writes a facility's MATLAB Middle Layer (MML) — the Accelerator
Objects from `getao`, the Accelerator Data from `getad`, the simulator's own
model ring, and what the Middle Layer's conversions say that ring is set to —
as files that `osprey mml import` reads with no flags. Nobody has to write
their own exporter.

## Get the script

```bash
osprey scaffold pull control-assistant:data/mml/mml_export.m
```

It lands at `data/mml/mml_export.m` in your deployment repo. Copy it anywhere on
the MATLAB path of the machine that runs your Middle Layer.

## Run it once per sub-machine

The script exports whichever sub-machine the Middle Layer is currently set up
for, so run your usual MML setpath for a sub-machine, load its simulator model,
then one command:

```matlab
mml_export
```

That writes six files into the current folder:

| File | Holds |
|------|-------|
| `<machine>.<submachine>.lattice.mat` | the model ring (`THERING`) |
| `<machine>.<submachine>.ao.json` | the Accelerator Objects (`getao`) |
| `<machine>.<submachine>.ad.json` | the Accelerator Data (`getad`) |
| `<machine>.<submachine>.va.json` | per-family calibrations, energy facts and nominals |
| `<machine>.<submachine>.response.json` | the orbit response matrix |
| `<machine>.<submachine>.model.json` | what the Middle Layer's model answers: tune, chromaticity, dispersion and their responses |

`<machine>` and `<submachine>` are `AD.Machine` and `AD.SubMachine`, lowercased.
To write somewhere else, pass the folder: `mml_export('/path/to/exports')`.

Repeat for every sub-machine you want in the deployment — set up the next one,
run `mml_export` again. Each run writes its own set of six, so nothing is
overwritten.

The lattice is saved first, before the export samples anything. Reading a
nominal reaches its answer through the ring's own closed orbit and turns
radiation off or the cavity on to get one, leaving the ring in whichever state
it needed, so a lattice saved afterwards would no longer be the ring the rest
of the export describes. A model measurement of the response matrix steps its
correctors on a copy, so what it leaves behind is what its own readings needed
rather than the columns it built.

## Import the files

Import the `.ao.json` files; the four siblings beside each one are read
automatically, and the sub-machine name recorded in the file becomes the system
name, so no `--system` flag is needed:

```bash
osprey mml import mymachine.storagering.ao.json mymachine.ltb.ao.json
```

(Use your own file names — the ones the script printed.)

The model file is not imported: it is what a check of OSPREY's model against
the Middle Layer's reads.

## What the files contain

Each file opens with an `_export` block recording the exporter version, the
MATLAB version, the machine, the sub-machine and when the export ran. After it
come the AO families (or the AD fields) as the Middle Layer holds them, with a
few values rewritten so JSON can carry them faithfully:

- function handles become `{"$fn": "<name>", "file": "<path>"}`;
- padded char matrices become one trimmed string per row;
- `Inf`, `-Inf` and `NaN` are written as the strings `"Inf"`, `"-Inf"` and
  `"NaN"` — never as `null`, which would lose the value;
- logicals become `0`/`1`, and the `Handles` field (plot handles) is left out.

Everything else is written as is. The script only reads from the Middle Layer;
it sets nothing on the machine.

`va.json` is the part the Middle Layer cannot state as stored data. It opens
with a fingerprint of the ring the file belongs to — element count, a digest of
the family names in ring order, the model energy, where any ring-parameter
element sits — which the importer recomputes from the lattice file and refuses
on disagreement. Then one block per family: the Setpoint's and the Monitor's own
hardware→physics calibrations, and the Monitor's physics→hardware inverse, each
sampled through the facility's own conversion functions rather than read out of
their stored parameters; whether the Setpoint's conversion carries the beam's rigidity; the
per-device nominal the Middle Layer's model read gives; what the family's own
readings are corrected by, where it states it; and, for a family the ring's
energy is read from, the energy table its current maps to. The exporter's own
header lists every key.

The corrections — a gain, an offset, a roll and a crunch, one number per device
— belong to whichever family states them, not to the beam monitors alone. A
facility that calibrates its magnets and correctors from an orbit measurement
keeps their gains and rolls under the same four names, so a ring-sized export
usually carries a block of them on a dozen or more families. They are four
numbers per device and nothing is resampled for them, so what that costs the
file is negligible. The offset is in the family's own **hardware** units — what
its readback answers in, millimetres on most beam monitors — and not the
physics units of the conversion beside it. A number the facility states nowhere
is left out rather than defaulted, and one written in a shape that is not one
per device is refused by name and costs that family only that number.

The numbers are looked up in the three places the Middle Layer looks: the
family's `Monitor` field, the family itself, then the facility's physics data
file, which a facility fills from an orbit fit and copies into the Accelerator
Objects at operating-mode set-up. Run the export from a session where that
set-up has run. An answer that came from the physics data can hold `"NaN"` for
single devices, because that file is stored against its own device list and a
device it does not cover comes back as no number, named on the console as it
happens. Such an entry means the facility states nothing for that one device.

A family whose conversions refuse a sample keeps the facts already in hand and
records the MATLAB message under `refused`; it costs the export nothing else.
The same holds for the response matrix. A file the Accelerator Data names but
the Middle Layer cannot find ends in its own measurement of the model, and so
does an Accelerator Data that names no file at all: naming nothing is the one
state the Middle Layer answers with a file-chooser dialog, and an export runs
unattended, so the measurement is asked for directly instead. Each block's
`origin` says whether the matrix was measured on the machine or computed from a
model — which is not the same question as whether a file answered, since a
facility may keep a computed matrix in a file like any other. A matrix computed
from a model says nothing about the machine: `osprey mml verify` then compares a
deck against a deck, and its report says so and names the file the matrix was
read from.

## Requirements

- A MATLAB release with `jsonencode` (R2016b or newer), started with its Java
  runtime — the lattice digest is taken through it.
- The Middle Layer on the path and set up for the sub-machine being exported,
  so `getao` and `getad` return its data.
- That sub-machine's simulator model loaded, so `THERING` holds its ring. The
  export refuses without it: the lattice is what every calibration in the
  export was sampled against.

## Re-running the export on a MATLAB host

The same steps produce a fresh export on any Linux host with MATLAB and an
MML-prod checkout; `tests/templates/test_mml_export_parity.py` runs exactly this
program when a MATLAB is on PATH.

1. **Compile the bundled AT once.** The Accelerator Toolbox that MML-prod ships
   under `simulators/at2.0` needs integrators built for the MATLAB that runs it:
   run `atmexall` in its `atmat/` folder.
2. **Make the two case symlinks SPEAR3 needs.** Linux is case-sensitive, and the
   Middle Layer asks for `machine/SPEAR3/...` and `SPEAR3physdata.mat` where the
   checkout spells both `Spear3`. Without `machine/SPEAR3 -> Spear3` and
   `Spear3/StorageRingOpsData/SPEAR3physdata.mat -> Spear3physdata.mat` the golden
   response file and the physics data are skipped silently and the export measures
   the model instead. A fresh checkout needs them again.
3. **Start one MATLAB per sub-machine.** Nothing carries over between
   sub-machines, so each gets its own session. The machine's setpath
   (`setpathspear3`, `setpathnsls2('StorageRing')`, `setpathnsls2('LTB')`) calls
   `setpathmml`, and that has to run before anything else is put on the path: it
   puts the bundled AT on the path and initialises the Accelerator Objects through
   it. One line for `matlab -batch`, which runs only the first line of a
   multi-line statement:

   ```matlab
   addpath('<MML-prod>/mml'); setpathspear3; setpathat('<AT>'); switch2sim; addpath('<folder of mml_export.m>'); mml_export('<outdir>')
   ```

4. **Expect the LabCA warning.** The link method defaults to LabCA; on a host
   without it the Middle Layer warns once. `switch2sim` then puts every family in
   simulator mode, which is what the export samples.

A 2.1.0 run writes six files per sub-machine: `.ao.json`, `.ad.json`, `.va.json`,
`.response.json`, `.lattice.mat` and `.model.json`.

### Refreshing the committed SPEAR3 and NSLS-II exports

This step is the owner's; no automated change touches those files.

- Commit all six files of one run per sub-machine, never a mix of runs.
- Compare the five older files with the committed ones first. If any differs
  beyond `_export.exporter`, `timestamp` or `matlab`, stop: that is a change in
  the facility or the exporter, not a refresh. Two differences are expected
  and are not a stop: floating-point values that move in the last digits
  (about 1e-15 relative) when the run is on a different host, and the `DCCT`
  nominal, which the Middle Layer's simulator derives from the time of day.
- Update the file counts in `tests/fixtures/mml/spear3/README.md` and
  `tests/fixtures/mml/nsls2/README.md`.
