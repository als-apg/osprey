# osprey-panels

The OSPREY panel engine: the control-room screens generated from a built
`facility.json` and the facility's panel rule tables, written as Screen-IR
documents with an index. It carries none of the OSPREY framework (agents,
views, web UI, etc.) and depends on `jsonschema` only.

## Install

```
pip install osprey-panels
```

## Usage

Every input is named on the command line; the command reads no OSPREY profile
or checkout. `--facility` is a built `facility.json`, `--tables` the directory
of panel rule tables, `--out` the directory the screens go to. `--draw` defaults
to `draw.json` beside the tables; `--theme` names a theme file whose state
colours the drawing colours are held against.

```
osprey-panels write --facility facility.json --tables panels --theme theme.json --out screens
osprey-panels check --facility facility.json --tables panels --theme theme.json --out screens
osprey-panels classes --out screens
```

`write` writes the screens and `index.json`; when any screen fails the vendored
Screen-IR schema it exits 1 and leaves `--out` as it was. `check` compares the written screens with
what the engine generates now and exits 1 on a failing problem. `classes`
prints the screen classes the index records. `osprey-panels <command> --help`
lists every option. `python -m osprey_panels` runs the same command.

From Python, `osprey_panels.load_graph(facility, tables)` reads the facility
file and the rule tables into the panel graph every screen is drawn from, and
`osprey_panels.screen_ir.problems(document)` checks one screen against the
vendored schema.

## Public API

The modules below are the package's public API; every other module is
private. Each names the one consumer it serves: a *fork/tool* that uses the
engine without the framework, the *command* (`osprey-panels`), or a
*contract test* (the root test that checks the written screens), named by
its file. Each public module lists its public names in `__all__`.

| Module | Consumer |
|---|---|
| `osprey_panels` | fork/tool |
| `osprey_panels.cli` | command |
| `osprey_panels.describe` | contract test: `tests/templates/canopy/test_panel_screens.py` |
| `osprey_panels.document` | contract test: `tests/facility/test_facility_json_schema.py` |
| `osprey_panels.layout` | contract test: `tests/templates/canopy/test_panel_screens.py` |
| `osprey_panels.model` | contract test: `tests/templates/canopy/test_panel_screens.py` |
| `osprey_panels.screen_ir` | fork/tool |
| `osprey_panels.symbols` | contract test: `tests/templates/canopy/test_panel_screens.py` |
| `osprey_panels.view` | contract test: `tests/templates/canopy/test_panel_screens.py` |
