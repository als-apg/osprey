# MATLAB Middle Layer Export

`mml_export.m` writes a facility's MATLAB Middle Layer (MML) — the Accelerator
Objects from `getao` and the Accelerator Data from `getad` — as JSON that
`osprey mml import` reads with no flags. Nobody has to write their own exporter.

## Get the script

```bash
osprey scaffold pull control-assistant:data/mml/mml_export.m
```

It lands at `data/mml/mml_export.m` in your deployment repo. Copy it anywhere on
the MATLAB path of the machine that runs your Middle Layer.

## Run it once per sub-machine

The script exports whichever sub-machine the Middle Layer is currently set up
for, so run your usual MML setpath for a sub-machine, then one command:

```matlab
mml_export
```

That writes two files into the current folder:

| File | Holds |
|------|-------|
| `<machine>.<submachine>.ao.json` | the Accelerator Objects (`getao`) |
| `<machine>.<submachine>.ad.json` | the Accelerator Data (`getad`) |

`<machine>` and `<submachine>` are `AD.Machine` and `AD.SubMachine`, lowercased.
To write somewhere else, pass the folder: `mml_export('/path/to/exports')`.

Repeat for every sub-machine you want in the deployment — set up the next one,
run `mml_export` again. Each run writes its own pair, so nothing is overwritten.

## Import the files

Import the `.ao.json` files; the `.ad.json` beside each one is read
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

## Requirements

- A MATLAB release with `jsonencode` (R2016b or newer).
- The Middle Layer on the path and set up for the sub-machine being exported,
  so `getao` and `getad` return its data.
