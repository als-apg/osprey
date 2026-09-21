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

That writes five files into the current folder:

| File | Holds |
|------|-------|
| `<machine>.<submachine>.lattice.mat` | the model ring (`THERING`) |
| `<machine>.<submachine>.ao.json` | the Accelerator Objects (`getao`) |
| `<machine>.<submachine>.ad.json` | the Accelerator Data (`getad`) |
| `<machine>.<submachine>.va.json` | per-family calibrations, energy facts and nominals |
| `<machine>.<submachine>.response.json` | the orbit response matrix |

`<machine>` and `<submachine>` are `AD.Machine` and `AD.SubMachine`, lowercased.
To write somewhere else, pass the folder: `mml_export('/path/to/exports')`.

Repeat for every sub-machine you want in the deployment — set up the next one,
run `mml_export` again. Each run writes its own set of five, so nothing is
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
