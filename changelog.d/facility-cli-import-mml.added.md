`osprey facility import mml EXPORT...` writes MML exports as the mml layer's
sources under `data/facility/imported/mml/` and seeds each authored file that
does not exist yet. The mapping is checked against the exports first: a mapping
with a problem prints one `<key>: <message>` line per problem and
`<n> problems in <path>; fix each and check again.`, writes nothing and exits 1.
The verb takes every export the mapping names: a system the mapping names and
no given export carries is one such problem, `models.<system>: <system> is no
exported system`. A mapping with the wrong structure, or a profile that does
not resolve, prints a one-line failure with its cause and exits 1.
While `data/facility/` holds an authored record source that would merge
against the layer — a file of `records/` or `decks/`, `models.yaml`, or a
`seeds.yaml`, `limits.yaml`, `identity.yaml` or `measurement/` file that does
not open with the layer's header line — the verb prints
`import mml: authored-present: <n> files` (`1 file` for one) and one
`rm <path>` line per file and exits 1; `fixes.yaml`, `classes.yaml` and
`knowledge/` are never named.
`--print-exporter` prints the MATLAB exporter the layer ships and needs neither
a repo nor an export. A wired setpoint whose export `Range` lacks a finite edge
is seeded into `limits.yaml` as `writable: false` beside whichever edge is
stated, and a channel a write field retypes as a setpoint takes that field's
unit and description.
