First version of `osprey-facility`: the facility description and its build,
moved out of the framework under the import root `osprey_facility`. It declares
its public modules, registers the five built-in measurement kinds under the
`osprey_facility.kinds` entry-point group, ships the published `facility.json`
JSON Schema (`osprey_facility.schema.published_schema()`), and installs and
tests without the framework; the MML layer needs the `mml` extra.
