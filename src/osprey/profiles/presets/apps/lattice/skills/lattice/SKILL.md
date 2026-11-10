---
name: lattice
description: >
  Read the lattice model results the Lattice panel shows: optics (Twiss
  functions, tunes, chromaticity), tune against momentum offset, dynamic
  aperture, tune footprint and local momentum aperture, per configured model
  and energy. lattice_state lists the models and what is computed,
  lattice_result returns one analysis's numbers, lattice_export returns them
  as a CSV or SDDS file. Use when asked for the "model tunes", the "design
  chromaticity", "beta functions", "dispersion", the "dynamic aperture",
  "momentum aperture", "tune footprint", "what does the lattice panel show",
  or to "export the twiss table". NOT for measured machine values, and not
  for changing the lattice: the app is read-only.
---

# Lattice

The agent and the Lattice panel read one backend. Each analysis is computed
once per deployment for a given deck, energy, engine and settings, and cached;
the number you read is the number every operator's panel shows. Nothing here
writes to the machine.

## The tools

| Tool | Panel equivalent |
|---|---|
| `lattice_state(model?, energy_gev?)` | the model and energy pickers, the analysis toggle and its status |
| `lattice_result(analysis, model?, energy_gev?, wait_s?, retry?, max_rows?)` | the charts and the Results list of one analysis |
| `lattice_export(analysis, format?, model?, energy_gev?, table?)` | the CSV and SDDS buttons |

## Look first

Call `lattice_state`. It computes nothing and answers:

- `models`: each configured model with its `engine` (`pyat`, `elegant`, or a
  facility engine), `engine_settings`, `energies_gev` (empty when the model has
  one energy) and the `analyses` it offers.
- `selected`: the model and energy the rest of the answer is for (the first
  model and its first energy when you name none).
- `results`: per analysis its `settings`, `deadline_s` and `status`: `ready`
  (with `scalars`), `running`, `queued`, `failed` (with `error`) or `absent`.

Scalars such as the tunes and chromaticities are already in `lattice_state`
for a ready analysis; you only need `lattice_result` for the tables.

## Read a result

`lattice_result` starts the analysis when it is absent and waits up to
`wait_s` (45 s at most per call). A status of `running` or `queued` is not an
error: say it is computing, and call again. Long analyses (dynamic aperture,
momentum aperture) can take minutes; `deadline_s` is the facility's bound.

- `failed`: report `error` as it stands. A deadline failure means the facility
  must raise `deadline_s` or reduce the analysis's settings in the
  deployment's configuration; do not retry it in a loop. Pass `retry=true`
  once, and only when the operator asks or the cause was fixed.
- Tables are thinned to `max_rows` evenly spaced rows (`rows_total` tells you
  when). Use `lattice_export` when every row matters.

| Analysis | Scalars | Table |
|---|---|---|
| `optics` | `tune_x/y`, `chromaticity_x/y` | `twiss`: `s`, `beta_x`, `beta_y`, `eta_x` |
| `chromaticity` | fitted chromaticity orders | `tune_vs_dp`: `dp_pct`, `tune_x`, `tune_y` |
| `da` | `area_mm2` | `aperture`: `x_mm`, `y_mm` |
| `footprint` | particle counts | `tunes`: `x_mm`, `y_mm`, `tune_x`, `tune_y` |
| `lma` | smallest acceptance | `aperture`: `s`, `dp_plus_pct`, `dp_minus_pct` |

An engine may add scalars, columns and tables beyond these.

## Say where a number comes from

Engines differ: pyAT's default bend pass and finite-difference steps give a
chromaticity that differs from elegant's. Always name the `engine`, and the
`engine_settings` and `settings` when they are not empty, with a number you
quote. Never mix numbers from two models or two energies in one statement
without saying so.

## Files

`lattice_export` returns `filename` and `content` (CSV, or ASCII SDDS with
`format="sdds"`). Both carry the model, analysis, engine, deck hash and the
scalars in their header. To hand the operator a file, write `content` to a
file of that name with the workspace tools; if `truncated` is true, tell the
operator to use the panel's download button instead.
