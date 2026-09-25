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
 │ models     main <id> · pinned <agent> <id> · … | none pinned         │
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

## VA MAP

Drawn after `osprey mml map --init` appends the `virtual_accelerator:` block, and again
after every answer. `?` rows first, then the families grouped by verdict. The export is
not editable from the card, so its facts ride in the EXPORT box and there is no second
card for them.

```
 VA MAP — <facility> · <system>      export 2.0 · <n> families · <n> open
 ┌ EXPORT ── <exporter> · <n> refused ──────────────────────────────────┐
 │ deck        <deck> · <n> elements · <n> GeV · <n|unstated> cavities  │
 │ calibrate   <kind> <n> · <kind> <n>                                  │
 │ nominals    <n> · <n> synthetic                                      │
 │ response    <origin> · <n> blocks of <rows>×<cols> · <mons> × <acts> │
 │ others      <system>: <n> families, <n> elements, <n> GeV            │
 │ ⚠ <FAMILY>  <the MATLAB reason, verbatim>                            │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ OPEN ── <n> slots need an answer ────────────────────────────────────┐
 │ ? <FAMILY>  <attype|shared_field|escape_hatch>                       │
 │             <the slot's question, verbatim>                          │
 │             answers: <the `# answers:` comment beside the null slot> │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ COUPLE ── <n> families the model drives ─────────────────────────────┐
 │ <FAMILY>    <kind> · <field> · <calib> · <nominal> · <n> devices     │
 │             ⚠ <the verdict's reason, where it carries one>           │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ LATCH ── <n> families the model does not drive ──────────────────────┐
 │ <FAMILY> · <FAMILY>     <the reason they share>                      │
 └──────────────────────────────────────────────────────────────────────┘
 Is this the map?  yes / answer <family> … / modify
```

Where each line comes from:

- **Header.** `<facility>` is `facility.token` in `mapping.yaml`, `<system>` is
  `virtual_accelerator.system`. `<n> families` counts the block's `families`; `<n> open`
  counts its null slots — the number `map --init` already reported.
- **EXPORT** reads PROFILE.md's `### Virtual accelerator` heading for that system, which
  sits last in the system's section and carries six level-4 headings: `Export facts`,
  `Refused families`, `Response`, `Family coverage`, `Sampled fields`, `Other systems`.
  `deck`, `calibrate` and `nominals` are rows of the `| Fact | Value |` table under
  `#### Export facts`. `response` collapses the `#### Response` table: its origin, its
  block count, the shape every block shares, then the monitor families and the actuator
  families — `<mons>` and `<acts>` are each `/`-joined in the table's row order.
  `others` is `#### Other systems`, one line per system the VA lane ignores, drawn only
  when the export carried more than one; a system that carried no block of its own reads
  `<system>: no 2.0 export`. A fact the export left unstated reads `unstated`, never
  `0` and never blank. Each `⚠` line is one `#### Refused families` bullet with the
  MATLAB reason verbatim — that reason is what sends the reviewer back to MATLAB.
- **OPEN** is one `?` row per null slot, in the block's order, carrying the slot's
  `kind`, its `question` verbatim, and the `# answers:` comment `map --init` wrote under
  it. Never re-type that answer list: the vocabularies are closed and `map --check`
  refuses a word outside them by name. An undecided `system:` is a slot too — draw it as
  `? system` and offer the system tokens the export carried.
- **COUPLE** is one line per `verdict: couple` family: `kind`, `element_field`,
  `calibration`, `nominal_source`, then the device count from `#### Family coverage`.
  Sorted by kind — energy, rf, strength, kick, monitor — and within a kind in the block's
  order. The energy knob and the cavity bind no element field, so those lines drop that
  term. Where the verdict carries a `reason` — the sibling field whose units disagree —
  it goes on a `⚠` continuation under the family's line.
- **LATCH** is one line per distinct `reason`, naming every family that shares it. The
  largest group goes first; groups of equal size follow the block's order of the first
  family in each. A family whose export block is a refusal and nothing else latches on
  `no lattice element` and groups on that line with every other family that binds
  nothing; its MATLAB reason stays in the EXPORT box.

A family with an open slot is drawn in OPEN only. It joins COUPLE or LATCH once its
answer is written and the card is drawn again.

The card's words are the mapping's words. Verdicts are `couple` and `latch`; kinds are
`strength`, `kick`, `monitor`, `energy`, `rf`; slot kinds are `attype`, `shared_field`,
`escape_hatch`. Nothing is paraphrased into a friendlier word.

One AskUserQuestion follows the card: `Is this the map?  yes / answer <family> … /
modify`. An answer is written into that family's `slot.answer`; `modify` edits the block
by hand. `map --init` refuses to touch a block that may already hold reviewed answers —
`--force-va` replaces it and discards every answer in it.

Draw no card when there is nothing to draw: PROFILE's heading reads `no 2.0 export`, or
`map --init` reported `no lattice deck for <system> in <dir>; VA block not written`. Say
that in one line instead.

A worked example — the synthetic `quokka` export, one storage ring, two slots open:

```
 VA MAP — Quokka · SR                   export 2.0 · 17 families · 2 open
 ┌ EXPORT ── mml_export 2.0.0 · 3 refused ──────────────────────────────┐
 │ deck        quokka_sr_deck · 41 elements · 2 GeV · unstated cavities │
 │ calibrate   linear 20 · table 2                                      │
 │ nominals    15 · 4 synthetic                                         │
 │ response    model · 4 blocks of 4×4 · BPMx/BPMy × HC/VC              │
 │ ⚠ SEPTUM    SEPTUM.Monitor: getpvmodel answered the nominal in       │
 │             Physics units, not the hardware units it was asked in.   │
 │ ⚠ TUNE      Family TUNE lists no devices to read a nominal for.;     │
 │             Family TUNE lists no devices to sample.                  │
 │ ⚠ Version   Invalid input argument of type 'char'. Input must be a   │
 │             structure array or an object.                            │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ OPEN ── 2 slots need an answer ──────────────────────────────────────┐
 │ ? IDGAP     escape_hatch                                             │
 │             the AT block reaches this family through                 │
 │             SpecialFunctionSet; does the model drive it?             │
 │             answers: latch, ignore_hook                              │
 │ ? SEPTUM    attype                                                   │
 │             ATType Septum is not one the table knows; what does this │
 │             family drive?                                            │
 │             answers: latch, strength:<PolynomB|PolynomA>[<i>],       │
 │             kick:<0|1>, energy, rf, monitor:<x|y>                    │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ COUPLE ── 10 families the model drives ──────────────────────────────┐
 │ BEND        energy · table · Setpoint · 4 devices                    │
 │ RF          rf · linear · Setpoint · 1 device                        │
 │ QD          strength · PolynomB[1] · linear · Setpoint · 4 devices   │
 │ QF          strength · PolynomB[1] · linear · Setpoint · 4 devices   │
 │ SF          strength · PolynomB[2] · linear · Setpoint · 4 devices   │
 │ SQ          strength · PolynomA[1] · linear · Setpoint · 4 devices   │
 │ HC          kick · KickAngle[0] · linear · Setpoint · 4 devices      │
 │ VC          kick · KickAngle[1] · linear · Setpoint · 4 devices      │
 │ BPMx        monitor · x · linear · Monitor · 4 devices               │
 │ BPMy        monitor · y · linear · Monitor · 4 devices               │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ LATCH ── 5 families the model does not drive ────────────────────────┐
 │ DCCT · TUNE · Version   no lattice element                           │
 │ BDM                     element BD1 (BndMPoleSymplectic4Pass) takes  │
 │                         no KickAngle                                 │
 │ BSOFT                   bend2gev is constant at this facility        │
 └──────────────────────────────────────────────────────────────────────┘
 Is this the map?  yes / answer <family> … / modify
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
