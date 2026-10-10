# Cards

Every card in this skill is drawn from this file. One grammar, so a run reads as one
instrument from the first question to the wrap-up.

## Rendering rule

- **The card is in the chat message, before its question.** Never inside the question
  text of an AskUserQuestion. The question that follows is one line: "Is this correct?
  yes / modify", "Adopt as shown, or change an area?".
- **One card per question.** A card is never combined with an unrelated question in the
  same AskUserQuestion call.
- **Boxed panels, one per group.** A box has a title bar with the group name and, after
  `──`, the counts or the one-line verdict for that group. Facts go inside as
  `label  value` lines. A group with nothing to say is omitted, not drawn empty.
- **Width 72, no wrapping.** A value longer than the line continues on the next line,
  indented under its value column, never mid-word. Lists inside a box are separated by
  ` · `.
- **Prose around a card is two sentences at most.** Evidence paths are shown on request
  (the Sources list), never inline.
- **Symbols.** `?` opens a line that needs the user's answer. `⚠` opens an advisory the
  tooling raised. Nothing else decorates a line.

Box drawing:

```
 ┌ TITLE ── counts or verdict ─────────────────────────────────────────┐
 │ label      value                                                     │
 │            continuation of a long value                              │
 └──────────────────────────────────────────────────────────────────────┘
```

## STATUS QUO

Header line, then one box per group in this order, then the question.

```
 STATUS QUO — <name>                                  generation: <current|overlay|early>

 ┌ FRAMEWORK ───────────────────────────────────────────────────────────┐
 │ requires <floor> · built with <version|?> · preset <name|none>       │
 │ drift <n> unmarked differences (measured with OSPREY <version>)      │
 │ artifacts framework-managed ×<n> · claimed ×<n|? (never built here)> │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ CONTROL ── <type> · writes <ON|OFF> · archiver <type> ───────────────┐
 │ limits     <n> records in data/facility/limits.yaml                  │
 │ archiver   <endpoint or ?>                                           │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ AGENT ── <provider> / <model> · <n> hooks · <n> agents · <n> skills ─┐
 │ agents     <native list>                                             │
 │ skills     <native list>                                             │
 │ custom     <list, each `custom (shadows <name>)` where it shadows>   │
 │ models     main <id> · pinned <agent> <id> · … | none pinned         │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ WEB ── <n> panels · <n> users · <n> personas · auth <method> ────────┐
 │ panels     <native list> · configured: <list>                        │
 │ personas   <list>                                                    │
 │ mcp        osprey-native ×<n> · own: <keys>                          │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ DATA ── <mode> finder ───────────────────────────────────────────────┐
 │ OKF <n> docs · decks ×<n> · ARIEL vocabulary · triggers <set|path>   │
 │ reference facility: <paths still describing the demo facility>       │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ ENV ─────────────────────────────────────────────────────────────────┐
 │ names      <list> · from <where read>                                │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ CUSTOM ──────────────────────────────────────────────────────────────┐
 │ <dir>/ ×<n> · one entry per non-empty convention directory           │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ OPEN ────────────────────────────────────────────────────────────────┐
 │ ? <path>   <what is unclear, as one question>                        │
 │ ⚠ validate <advisory that belongs to no box above>                   │
 └──────────────────────────────────────────────────────────────────────┘
 Is this correct?  yes / modify
```

An era repo draws FRAMEWORK as `requires n/a · built with ? · extends <preset|none>`,
`drift n/a (era repo)`, `artifacts ? (never built here)`. An early-era repo adds an
OBSOLETE box, one line per path with why it is gone. The `reference facility` line in
DATA is what MAP turns into `placeholder` rows; omit it when there are none.

A facility with no OSPREY replaces the boxes with one REFERENCES box and one box per
kind, each line carrying its source and state:

```
 STATUS QUO — <facility|?>                                       generation: none

 ┌ REFERENCES ── <n> named ─────────────────────────────────────────────┐
 │ <name>       <path|endpoint|user said>            <state>            │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ CONTROL ─────────────────────────────────────────────────────────────┐
 │ system       <type|?>          <source>            <state>           │
 │ archiver     <type|?>          <source>            <state>           │
 │ logbook      <type|?>          <source>            <state>           │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ DATA ────────────────────────────────────────────────────────────────┐
 │ channels     <n|?> named       <source>            <state>           │
 │ documents    <what|?>          <source>            <state>           │
 │ lattice      <what|?>          <source>            <state>           │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ OWNERS ──────────────────────────────────────────────────────────────┐
 │ <who owns what|?>              <source>            <state>           │
 └──────────────────────────────────────────────────────────────────────┘
 Is this correct?  yes / modify
```

`<state>` is one of `verified`, `verified from files`, `reported, not verified`,
`not reachable`, `no access`. "Nothing yet" draws the same boxes with every value `?` and
every state blank, so the user sees that nothing was assumed.

## PORTING MAP

`?` rows first, then grouped by verdict. `placeholder` rows carry their answer.

```
 PORTING MAP — <name>                                  <n> elements · <n> open
 ┌ OPEN ── <n> rows need an answer ─────────────────────────────────────┐
 │ ? <old element>             <what is unclear, as one question>       │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ PORT ── still needed, no native equivalent ──────────────────────────┐
 │ <old element>               → <path it lands at, or the command>     │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ NATIVE ── OSPREY covers it now ──────────────────────────────────────┐
 │ <old element>               → <native artifact selected instead>     │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ PLACEHOLDER ── reference-facility material ──────────────────────────┐
 │ <old element>               refresh  · <current skeleton or template>│
 │ <old element>               localize · <facility-named stub>         │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ OBSOLETE ── dropped ─────────────────────────────────────────────────┐
 │ <old element>               <one-line reason>                        │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ GAP ── upstream candidates ──────────────────────────────────────────┐
 │ <old element>               <what cannot be said> · <short-id>       │
 └──────────────────────────────────────────────────────────────────────┘
 Is this the map?  yes / modify
```

## MAP FACTS

```
 MAP FACTS — <name>
 ┌──────────────────────────────────────────────────────────────────────┐
 │ facility     <name>                     <user said | file | command> │
 │ timezone     <IANA name>                <…>                          │
 │ project      <repo directory>           <…>                          │
 └──────────────────────────────────────────────────────────────────────┘
 Is this correct?  yes / modify
```

## MODEL WIRING

Drawn once per model from its `wiring` block in
`data/facility/imported/mml/mapping.yaml`, after the first
`osprey facility import mml` run wrote the draft, and again after every answer. `?`
rows first, then the wired families, then the rest. The export is not editable from
the card.

```
 MODEL WIRING — <facility> · <model>          <n> families · <n> open
 ┌ OPEN ── <n> slots need an answer ────────────────────────────────────┐
 │ ? <FAMILY>  <element_field|engine|engine.index|calibration>          │
 │             <what the import asks for in that slot, verbatim>        │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ WIRED ── <n> families the model drives or reads ─────────────────────┐
 │ <FAMILY>    <element_field> · <engine> · <calibration>               │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ NOT WIRED ── <n> families the model leaves to their seeds ───────────┐
 │ <FAMILY> · <FAMILY> · <FAMILY>                                       │
 └──────────────────────────────────────────────────────────────────────┘
 Is this the wiring?  yes / answer <family> … / modify
```

Where each line comes from:

- **Header.** `<facility>` is the mapping's `facility.code`, or `code:` in
  `data/facility/identity.yaml` once the first import has moved it there; `<model>` is
  `models.<raw>.name`. `<n> families` counts the keys of `families:`; `<n> open` counts
  the `null` slots of this model's `wiring` block.
- **OPEN** is one `?` row per `null` slot of the block, in the block's order: a
  family's `element_field`, its `engine`, an `engine.index` left open, or its
  `calibration`. The second line is what the import prints for that slot when it stops
  with `import mml: mapping-undecided` — run the import and copy the line. Never
  re-type an answer list: the vocabularies are closed and the import refuses a word
  outside them by name.
- **WIRED** is one line per family of the block with no open slot: its
  `element_field`, then its `engine` block written `<attribute>[<index>]`, the bare
  attribute where it carries no index, or the `axis` of a monitor, then its
  `calibration` (`linear` or `table`). In the block's order.
- **NOT WIRED** names every key of `families:` the block does not list, in the
  mapping's order. The simulator serves their channels from their seeds, with no
  physics behind them. A family the draft proposed that the model should not drive is
  moved here by deleting its entry from the block.

A family with an open slot is drawn in OPEN only. It joins WIRED once its answer is
written and the card is drawn again.

The card's words are the mapping's words: `element_field`, `engine`, `attribute`,
`index`, `axis`, `calibration`, `linear`, `table`. Nothing is paraphrased into a
friendlier word.

One AskUserQuestion follows the card: `Is this the wiring?  yes / answer <family> … /
modify`. An answer is written into that family's entry of the block; `modify` edits
the block by hand. Units and the hooks of a family are the reviewer's to check: the
draft reads neither.

Draw no card when there is nothing to draw: a model whose draft carries no `wiring`
block. Say that in one line instead.

A worked example — the synthetic `quokka` export, one model, two slots open:

```
 MODEL WIRING — Quokka · SR                    17 families · 2 open
 ┌ OPEN ── 2 slots need an answer ──────────────────────────────────────┐
 │ ? BDM       engine.index                                             │
 │             name the plane: 0 for x, 1 for y                         │
 │ ? SEPTUM    engine                                                   │
 │             name the attribute and index, or the axis, the model     │
 │             wires                                                    │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ WIRED ── 10 families the model drives or reads ──────────────────────┐
 │ BEND        Setpoint · energy · table                                │
 │ RF          Setpoint · Frequency · linear                            │
 │ QD          Setpoint · PolynomB[1] · linear                          │
 │ QF          Setpoint · PolynomB[1] · linear                          │
 │ SF          Setpoint · PolynomB[2] · linear                          │
 │ SQ          Setpoint · PolynomA[1] · linear                          │
 │ HC          Setpoint · KickAngle[0] · linear                         │
 │ VC          Setpoint · KickAngle[1] · linear                         │
 │ BPMx        Monitor · x · linear                                     │
 │ BPMy        Monitor · y · linear                                     │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ NOT WIRED ── 5 families the model leaves to their seeds ─────────────┐
 │ BSOFT · DCCT · IDGAP · TUNE · Version                                │
 └──────────────────────────────────────────────────────────────────────┘
 Is this the wiring?  yes / answer <family> … / modify
```

## FEATURES

One box per feature area of the reference example, in the order of
`references/map.md`. The title bar carries the verdict and the reason. `brings` lists
what an adopted area lands; `leaves` lists what a `later` or `never` area leaves out.

```
 FEATURES — what the reference example offers, decided per area

 ┌ LOGBOOK ── adopt ── <reason from DISCOVER> ──────────────────────────┐
 │ brings     ARIEL service + postgres · logbook agent · logbook persona│
 │            ingest via <adapter> (<native | Generic JSON + candidate>)│
 └──────────────────────────────────────────────────────────────────────┘
 ┌ KNOWLEDGE ── adopt ── <reason> ──────────────────────────────────────┐
 │ brings     OKF bundle + knowledge panel · graph store · agents       │
 │            harvest of <n> named sources                              │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ SIMULATION ── never ── <reason> ─────────────────────────────────────┐
 │ leaves     virtual accelerator · bluesky · pyat agent · sim skills   │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ EVENT DISPATCH ── later ── <what unblocks it> ───────────────────────┐
 │ leaves     dispatch service · events panel                           │
 └──────────────────────────────────────────────────────────────────────┘
 Adopt as shown, or change an area?
```

## MODELS

Drawn in BUILD step 9, once the ports in step 5 have made the agent list final, and again
after every answer.

```
 MODELS — <name> · <provider> · main <model id>
 ┌ AGENTS ── <n> enabled · <n> pinned ──────────────────────────────────┐
 │ <agent>                    main model                                │
 │                            <purpose>                                 │
 │ <agent>                    <pinned id>                               │
 │                            <purpose>                                 │
 │                            ⚠ not in <provider>'s models list         │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ SERVED ── <provider> ────────────────────────────────────────────────┐
 │ <id> · <id> · <id>                                                   │
 └──────────────────────────────────────────────────────────────────────┘
 Every agent on the main model?  yes / pin <agent>=<id> … / modify
```

Where each line comes from:

- The header's provider is the profile's `provider:`. Its model is `model:`, else that
  entry's `default_model` in `providers.yml` beside the profile.
- The AGENTS names are the profile's `agents:` list plus the `agents/` directory.
  `<purpose>` is the agent's line in `osprey profile artifacts`, or a deployment file's
  `description:`. The model is `main model`, a `claude_code.agent_models.<agent>` value,
  or a deployment file's `model:` line. The `⚠` continuation marks a pinned id the
  entry's `models` list lacks.
- SERVED is the entry's `models` list, verbatim.

The question:

- `yes` is the default and writes nothing.
- A pin is written with `osprey set config.claude_code.agent_models.<agent>=<id>`, which
  refuses a name that is no agent and a bare alias word, and notes an unlisted id.
- Offer ids from SERVED. An id outside it is the user's to name.
- A deployment's own agent file takes no pin: its `model:` line is edited instead.
- Then `osprey validate --drift=warn`, and the card is drawn again.

Worked example:

```
 MODELS — quokka · cborg · main claude-haiku-4-5
 ┌ AGENTS ── 3 enabled · 1 pinned ──────────────────────────────────────┐
 │ channel-finder             main model                                │
 │                            Channel-finder sub-agent                  │
 │ logbook-deep-research      claude-opus-5                             │
 │                            Logbook deep-research sub-agent           │
 │ logbook-search             main model                                │
 │                            Logbook search sub-agent                  │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ SERVED ── cborg ─────────────────────────────────────────────────────┐
 │ claude-opus-5 · claude-sonnet-5 · claude-haiku-4-5                   │
 └──────────────────────────────────────────────────────────────────────┘
 Every agent on the main model?  yes / pin <agent>=<id> … / modify
```

## LEDGER GATE

Drawn at CLOSE from `## Ledger` against the file tree. Only blocking rows are drawn;
a clean gate is one line: `LEDGER GATE — <n> paths, all accounted for`.

```
 LEDGER GATE — <n> paths · <n> blocking

 ┌ NO ROW ── landed, never recorded ────────────────────────────────────┐
 │ <path>                      localize / refresh / remove / keep — why │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ REFERENCE FACILITY ── still the demo's content ──────────────────────┐
 │ <path>                      <provenance> · localize / refresh / keep │
 └──────────────────────────────────────────────────────────────────────┘
 Resolve each row, then the gate re-runs.
```

## SCOUT

Drawn when a finished scout is surfaced, one box per candidate, followed by the
disposition question from `/osprey:upstream-scout`.

```
 ┌ SCOUT ── <short-id> ── <UPSTREAM|DEPLOYMENT_LOCAL|UNCLEAR> · <MECHANICAL|ARCHITECTURAL> ┐
 │ fix lives   <owning subsystem path> · <abstraction level>            │
 │ blast       <n> files · extension point <exists|missing>             │
 │ prior art   <#n title (state)> | none (<n> queries)                  │
 │ write-up    upstream/<short-id>.md                                   │
 └──────────────────────────────────────────────────────────────────────┘
```

## WRAP-UP

```
 DONE — <name>                                         osprey <version>
 ┌ NEXT ────────────────────────────────────────────────────────────────┐
 │ osprey build          render build/ from the profile                 │
 │ osprey web            web dashboard on this machine                  │
 │ osprey up -d          start the services                             │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ OWED ── <n> items ───────────────────────────────────────────────────┐
 │ <curation, later features, open candidates — one line each>         │
 └──────────────────────────────────────────────────────────────────────┘
```

Every line in NEXT is read from the repo's README or `osprey <command> --help`.
