# synthetic/ — a 2.1 export of an invented ring

The six files `mml_export.m` 2.1 writes for one sub-machine, for a machine small
enough to read by eye, plus a seventh that pairs with none of them:

| File | What it is |
|------|------------|
| `quokka.sr.lattice.mat` | the ring, saved as `THERING` the way `at.save_mat(use='THERING')` saves it |
| `quokka.sr.ao.json` | the Accelerator Objects |
| `quokka.sr.ad.json` | the Accelerator Data |
| `quokka.sr.va.json` | per-family calibration, nominals and energy facts |
| `quokka.sr.response.json` | the orbit response matrix |
| `quokka.sr.model.json` | what the ring's model answers — tune, chromaticity, dispersion, orbit response — asked of pyAT directly |
| `mismatched.lattice.mat` | the same ring with one quadrupole renamed — the fingerprint counter-example |

Every name in them is invented. The contract they are written against is the
frozen key set in the header of
`src/osprey/templates/apps/control_assistant/data/mml/mml_export.m`.

## Provenance

`build.py` writes all seven, and the committed files are its output. It runs no
MATLAB: the export pipeline of `mml_export.m` is ported into it function for
function — the hardware grid and its anchor word, the line-or-table test, the
rigidity probe, the energy table, the model read and the joining of refusals —
and the conversions a real export calls out to the facility for are small Python
functions at the top of the file. The ring is pyAT's, saved with
`at.save_mat(use='THERING')` — the variable name `mml_export.m` saves and the
importer loads; `osprey.simulation.lattice.artifact` makes the same call for the
demo ring under `use='RING'`.

`--check` passes with pyAT (`accelerator-toolbox`) 0.8.0 and NumPy 2.4.2, with
SciPy 1.18.0 on Python 3.12 and 3.13 and SciPy 1.17.1 on Python 3.11, on macOS
arm64 and Linux x86_64. Rebuild it, or check that the committed files are still exactly what
it writes, from the repository root:

    uv run python tests/fixtures/mml/synthetic/build.py
    uv run python tests/fixtures/mml/synthetic/build.py --check

`--check` compares every byte, the `.mat` files included. Three things a clock or
a machine would otherwise decide are pinned so that holds: the `_export`
timestamp, the MATLAB version the `_export` block names, and the 116-byte
descriptive header `scipy.io.savemat` writes into a MAT-file, which carries the
build clock and which nothing reads — the format version and the endian marker
live in the twelve bytes after it. The arithmetic is pinned the same way: the
corrector's curved conversion is summed as a power series of products and sums,
which IEEE 754 rounds identically everywhere, and the bend ramp's `exp` and `log`
are the C library's, called one value at a time — so a rebuild on Linux writes the
bytes one on macOS does. Numbers are written at fifteen significant
digits, the spelling `jsonencode` gives a double, with whole numbers written
without a fractional part the way it writes those; digits past the fifteenth
would be the arithmetic's own noise rather than a value a reader can use.

The ring is 4D, closed, and boots: 43 elements including the leading `RingParam`,
four cells with tunes near (0.61, 0.58), one cavity, and a pair of `GE` markers
ahead of the cavity. Its nominal state is the
document's: the settings in `NOMINAL_AMPS` drive the element strengths and
kicks, and the nominals `va.json` records are those same settings read back, with
the beam monitors' read off the closed orbit the ring actually has. `ATIndex` is
1-based into `THERING`, so the `RingParam` at index 1 is included and every
element index is one further along than its position in the pyAT lattice.

## The ring

Four cells, one dipole each, bending a quarter turn apiece. Cells 1–3 carry two
corrector elements, cell 4 carries one — that is where the ragged index list
comes from. The last girder's two ends are marked by a pair of `GE` elements:
monitor type, zero length, bound by no family, and served as plain markers.

## What each family exercises

| Family | Devices | Carries |
|--------|---------|---------|
| `QF` | 4 | A strength family: linear Setpoint calibration, `energy_scaling` `brho`. Its `monitor_inverse` is linear with gain 95 where the calibration's 0.01 would invert to 100 — a consumer that derives the inverse instead of reading it gets this family wrong by five per cent. |
| `QD` | 4 | A second strength family whose readback conversion has a curve in it, so its `monitor_inverse` is a table. |
| `SF` | 4 | A sextupole family, Setpoint only. |
| `SQ` | 4 | A skew quadrupole on the same elements as `SF`: one element, two lattice fields. |
| `HC` | 4 | The sliced kick family. Its `at_index` is 4×2 with one `"NaN"` — cell 4 has one element where the others have two, so that device has one slice and the rest have two. Its first device is set above its `Range`, so that device is sampled over its band stretched up to its nominal while the other three are sampled over the band itself: one field whose rows are not all the same span. Its conversion bends, so both fields are tables, and their grid rows record the stretched device's span (-1, 1.5) beside the others' (-1, 1). `grid_source` is still `range`, because every device had a band to stretch. |
| `VC` | 4 | The vertical plane on the same corrector elements, with a `Range` that holds every one of its nominals, so no band is stretched (`grid_source` `range`). |
| `BPMx` | 4 | A monitor-only family with a `Range`, so its `monitor_inverse` is sampled over the Monitor's own image (`grid_source` `range`). |
| `BPMy` | 4 | A monitor-only family with no `Range`: the calibration falls back to a grid about the nominal, and the inverse to the ±10 mm beam-position span (`grid_source` `fallback`). |
| `BEND` | 4 | The energy candidate with a knob. Its conversion is a measured ramp that stops at 500 A, so the Setpoint calibration is a **table** with a `"NaN"` tail and a `finite_span` shorter than its grid, and so is the energy table. |
| `BDM` | 2 | A bend trim: the same elements and the same `ATType` as `BEND`, but `MemberOf` names `COR`, so it is not an energy candidate. Its `Range` is per device with one non-finite row: the first device is sampled over its band, the second over the fallback grid about its nominal, so the field's `grid_source` is `fallback`. |
| `BSOFT` | 2 | An energy candidate by membership alone — no `AT` block at all, `MemberOf` naming `BEND`. Its energy table is **flat**: the conversion hands back the deck energy at every current. |
| `RF` | 1 | The cavity. One device, so every per-device value is written flat and its `at_index` is a bare number. `energy_scaling` is `none`: the conversion ignores the energy it is handed. |
| `IDGAP` | 2 | The escape hatch: `AT.SpecialFunctionSet` and `AT.ATParameterGroup` beside an `ATType` no branch of the model read knows, so its nominals are `"NaN"` and `synthetic` is 1. Its `ATIndex` is empty and its anchor word is `range_midpoint`. |
| `SEPTUM` | 1 | `ATType` `Septum`, so `synthetic` is 1. Its model read answers in physics units, which is a refusal recorded **beside** the nominal it had already read, and with no `Range` its anchor word is `zero`. |
| `DCCT` | 1 | Synthetic by family name, with no `AT` block and no element of its own name. |
| `TUNE` | 0 | A family with no devices: two steps refuse, and `refused` carries both, joined. |
| `Version` | — | Not a family at all — a stray text entry of the kind a facility's init leaves in its Accelerator Objects. The export walks every field, so this one gets a block that is a refusal and nothing else. |

Between them the closed vocabularies are covered: `kind` both ways,
`grid_source` `range`/`setpoint`/`fallback`, `anchor` `nominal`/`range_midpoint`/`zero`,
and `energy_scaling` `brho`/`none`.

## `response.json`

Four blocks — two monitor families against two corrector families — measured on
the model by stepping each corrector both ways and reading the closed orbit, so
every `origin` is `model` and every `mode` is `Simulator`. `file` is empty,
because the model answered rather than a file. The numbers are at six significant
digits, the width the exporter writes measurements at; device lists and status
flags are verbatim.

`BPMy`'s third device is one the file does not hold: its status flag is down, its
row of the matrix is `"NaN"` throughout and so is its operating point. The
vertical corrector side kept no settings, so each device's operating point is
the `"NaN"` the exporter writes for an absent number. `actuator_delta` is one
width per corrector device.

## `model.json`

What `mml_export.m` 2.1 asks the Middle Layer's model, answered here by pyAT
directly and never by replaying the Middle Layer's recipes, so a test that
replays a recipe holds it to an independent answer. Each section records the
pyAT call it was taken by:

- `state` — the deck's cavity (`Off`: the ring is saved 4D), energy,
  circumference, momentum compaction, harmonic number and the closed orbit at
  the monitors, solved by `find_orbit6` with the cavity on.
- `tune` — `get_tune` on the cavity-on ring (`cavity_on`) and on a 4D copy
  (`fixed_momentum`); `method` is `get_tune`.
- `chromaticity` — `get_chrom` on the 4D copy, in physics units only; the
  hardware entry is refused in the words a cavity-less ring refuses it with.
- `dispersion` — `avlinopt` at the monitors on the 4D copy, in metres and in
  millimetres per MHz of RF.
- `orbit_response` — the same columns `response.json` is built from, in physics
  and in hardware units, laid out block for block like it at six significant
  digits, `Linear` at `FixedPathLength`, with no `origin`.
- `tune_response`, `chromaticity_response` — refused in the Middle Layer's own
  words, because no family is tagged as a tune or chromaticity corrector.

The RF step both RF sections record is a builder constant, one per mille of
momentum: `delta_rf_hz = f · mcf · 1e-3`. Every other tracking-derived number is
written at twelve significant digits, and one below 1e-15 is written as zero.

## `mismatched.lattice.mat`

The same ring with `QF1` renamed. Its element count still matches
`lattice.elements`, so a consumer that pairs a lattice with this export by
counting alone accepts it; the `famname_sha256` is what refuses it.
