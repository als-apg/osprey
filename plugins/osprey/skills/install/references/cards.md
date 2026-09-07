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
 │ limits     <n> channels in data/channel_limits.json                  │
 │ archiver   <endpoint or ?>                                           │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ AGENT ── <provider> / <model> · <n> hooks · <n> agents · <n> skills ─┐
 │ agents     <native list>                                             │
 │ skills     <native list>                                             │
 │ custom     <list, each `custom (shadows <name>)` where it shadows>   │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ WEB ── <n> panels · <n> users · <n> personas · auth <method> ────────┐
 │ panels     <native list> · configured: <list>                        │
 │ personas   <list>                                                    │
 │ mcp        osprey-native ×<n> · own: <keys>                          │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ DATA ── <mode> finder · tier <n> ────────────────────────────────────┐
 │ OKF <n> docs · lattice ×<n> · ARIEL vocabulary · triggers <set|path> │
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
 │ prefix       <facility.prefix>          <…>                          │
 │ timezone     <IANA name>                <…>                          │
 │ project      <repo directory>           <…>                          │
 └──────────────────────────────────────────────────────────────────────┘
 Is this correct?  yes / modify
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
