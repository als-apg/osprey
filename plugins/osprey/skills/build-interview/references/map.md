# Porting map and build wiring

MAP turns the locked status quo into one verdict per element. BUILD executes those
verdicts against a `hello-world` base. This file carries the vocabulary, the two cards,
and the recipes that make a verdict actually land.

Read the map from live sources, never from memory: the locked status quo, `osprey profile
artifacts`, the emitted `profile.yml` (its commented blocks are the catalog of optional
features), and `src/osprey/profiles/presets/control-assistant.yml`, the reference
package.

Two menus are not in the profile. The provider list is in `osprey config --defaults`, in
the comment above `claude_code.provider`. The connector list is in `osprey init --help`,
under `--set connector`. The emitted profile's comments explain the one value each key
carries, not the menu it was chosen from.

## Verdict vocabulary (closed)

One verdict per element. Nothing outside this list.

| Verdict | Meaning | Lands as |
| --- | --- | --- |
| `port` | Still needed, no native equivalent | A `scaffold pull`, a copy from the old repo, or a convention directory |
| `native` | OSPREY now covers it | Dropped; the native artifact is selected in `profile.yml` instead |
| `obsolete` | No longer serves a purpose, such as era machinery or demo content | Dropped, with a one-line reason |
| `gap` | The facility needs it, OSPREY cannot express it | Upstream candidate; hand to `references/upstream-scout.md` |
| `unknown` | Purpose unclear | One question to the user, then a real verdict. Never ships as a verdict |

## The porting-map card

Old element on the left, verdict in the middle, new home on the right. Group by verdict.
Put every `?` row first, because those are the rows the user has to answer.

```
 PORTING MAP — <name>                                     <n> elements · <n> open
 ───────────────────────────────────────────────────────────────────────────────
 ?         <old element>          ──?──▶  <what is unclear, as one question>
 ───────────────────────────────────────────────────────────────────────────────
 port      <old element>          ─────▶  <path it lands at, or the command>
 native    <old element>          ─────▶  <native artifact selected instead>
 obsolete  <old element>          ─────▶  dropped, <one-line reason>
 gap       <old element>          ─────▶  upstream candidate, <what cannot be said>
 ───────────────────────────────────────────────────────────────────────────────
 Is this the map?  yes / modify
```

Confirmed, it is written to INTERVIEW.md under `## Porting map (locked)` and is not
reopened for the rest of the interview. `modify` edits rows and re-renders the card.

A status quo with no elements (`generation: none`) has nothing to map: render no
porting-map card, and lock `## Porting map (locked)` as `none`. The four facts at the end
of MAP are then this phase's card. Render them as MAP FACTS and confirm them the same
way, so every phase still ends in one card and one AskUserQuestion.

```
 MAP FACTS — <name>
 ───────────────────────────────────────────────────────────────────────────────
 facility     <name>                    <where it came from>
 prefix       <facility.prefix>         <where it came from>
 timezone     <IANA name>               <where it came from>
 project      <repo directory>          <where it came from>
 ───────────────────────────────────────────────────────────────────────────────
 Is this correct?  yes / modify
```

The right column is `user said`, or the file or command the fact was read from. Confirmed,
the card goes under `## Porting map (locked)`, below the word `none`.

## Wiring rule: a data path is not ported until its keys are set

A `port` verdict on a data path, and a `close` on a gap-card data row, is complete only
when the config keys that bind that path are set. Files on disk that nothing points at
are invisible to the build.

Recipe, per data path:

1. Open `templates/apps/control_assistant/config.yml.j2`. From a source checkout it sits
   under `src/osprey/`; from a pip install, join the same path onto the installed package
   root using the wheel-path recipe in `references/osprey-map.md`.
2. Grep that template for the path you pulled, for example `data/facility_knowledge`.
3. Port the whole top-level section that encloses the hit, not the single line. Set each
   key with `osprey set config.<dotted key>=<value>`.
4. If the path has a web panel, add that panel to the profile's top-level `web_panels`.
5. `osprey validate --drift=warn`.

Worked example, the facility knowledge bundle:

```
osprey scaffold pull control-assistant:data/facility_knowledge
osprey set config.facility_knowledge.bundle_path=data/facility_knowledge
```

plus `okf` in the profile's top-level `web_panels` list, which is what renders the
KNOWLEDGE tab against that bundle. The pull brings the directory skeleton and the
`index.md` files only. Stub authoring and `osprey knowledge regen-index` are in
`references/knowledge-starter.md`.

## The web-terminal recipe

Multi-user web terminals are absent from hello-world's emission, so they are copied from
the control-assistant preset. Copying only the `modules.web_terminals` block does not
validate and does not build on a hello-world base. The persona deltas in that preset are
deltas against the control-assistant base, and several of that base's facts leak out of
the block. Run all eight steps, in order.

1. `osprey init <name> --preset hello-world`.
2. From `src/osprey/profiles/presets/control-assistant.yml`, copy the
   `modules.web_terminals:` block **and** its two floor keys into `profile.yml`. They go
   inside `config:`, after its last key, before the top-level `data:`.

   ```yaml
   claude_code.permissions.deny:
     - mcp__osprey_workspace__setup_patch
   web.config_panel.enabled: false
   ```

   Why: without that floor the shared logbook and knowledge personas resolve as
   privileged, and validate refuses them. The preset sets a third key at the same
   floor, `web.scaffold_gallery.write_enabled: false`. That one is posture rather than
   a validate requirement, so carry it, or record the decision to drop it in
   INTERVIEW.md.

   Two things arrive inside the pasted block that do not belong on this base:

   - **`landing.notices:`** names `data/landing/working-safely.md`, which hello-world does
     not ship. Nothing reports the missing file — validate and build both exit 0 and
     render the dead path into every persona config. Delete the key. Dropping it is what
     yields OSPREY's built-in safety notice, and the alternative,
     `osprey scaffold pull control-assistant:data/landing/working-safely.md`, brings the
     reference package's own demo text, which the facility would rewrite anyway.
   - **The demo logins `alice`, `bob` and `carol`** under `users:`. This is the step where
     they leave; the rule is `references/knowledge-starter.md` §5.
3. `osprey set config.facility.prefix=<prefix>`, with the prefix from the MAP-end
   questions. Why: hello-world carries none, and the web container names are built from
   it (`<prefix>-nginx`), so validate refuses the profile without it. Setting it here
   rather than at the end is what makes `osprey validate --drift=warn` exit 0 from this
   step on, so "validate after every change" holds for the whole recipe.
4. `osprey scaffold personas --from control-assistant`. It writes one
   `personas/<name>.yml` per catalogued persona and repoints the catalog by appending
   dotted `modules.web_terminals.personas.<name>.<key>` keys under `config:`. The dotted
   keys win, so the nested `personas:` mapping still sitting inside the pasted block is
   now dead text stating every catalog fact a second time with the preset's values.
   **Delete that nested mapping**, so each fact is stated once. Validate and build both
   stay at 0 without it.
5. Prune the persona set to the roles the user named. That ordering is one rule and it
   lives in `references/knowledge-starter.md` §5. Dropping a persona is four deletions
   that go together: `personas/<name>.yml`, its dotted catalog keys, its roster entry
   under `users:`, and, once a build has seeded it, `web-terminal-context/<name>/`
   (step 8).

   **The `logbook` persona is dropped on any base with no ARIEL service.** A hello-world
   base declares `services: {}`, so its landing card and the `ariel` panel it opens on
   would have nothing behind them. Add the persona, its roster entry and the `ariel`
   panel back only once the profile declares an ARIEL service block.
6. Open every surviving `personas/<name>.yml` and delete the whole `web_panels:` key
   wherever it names only panels this deployment does not have. Delete the key rather
   than emptying it, so the delta carries no panel selection at all and the host's list
   stands.
   Why: the readwrite and admin deltas list `events` and `bluesky`, which need the
   `dispatch:` and `bluesky:` service blocks a hello-world base does not deploy.
   The general rule is to remove from each delta every panel this deployment lacks.
   The same pruning applies to the `exclude.web_panels` list in the knowledge delta, and
   in the logbook delta where that one survives step 5.
7. Check `default_panel` in each surviving delta. It must name a panel that is in the
   profile's top-level `web_panels`. A persona whose `default_panel` names a panel the
   host does not select fails `osprey build`. Validate does not catch it; build does.
   So the host's minimum list is exactly the `default_panel` values the surviving deltas
   name: `okf` for knowledge, `ariel` for logbook. With logbook dropped at step 5,
   `web_panels: [okf]` validates and builds. Extra panels are legal — `[okf,
   system-health]` builds too — but each one is a tab this deployment has to serve, so
   add them through the wiring rule above, not by habit.
8. `osprey validate --drift=warn`, then `osprey build`. Drift warnings are expected here
   and are not marked.

   The build writes into the SOURCE zone as well as `build/`: it seeds
   `web-terminal-context/<roster user>/.gitkeep`, one directory per roster entry. Delete
   a roster entry afterwards and its directory is left behind — remove it in the same
   edit, or every later build warns that `web-terminal-context/` holds context for users
   not on the roster.

## Gap card

After the porting map is applied, check what the reference package has that this
deployment does not. One row per artifact, feature block, data item and persona, each
answered `close` or `skip`.

The first two rows are fixed, and they come from hello-world rather than
control-assistant. Both are demo material the base emitted, so neither is a `close`/`skip`
question — each has its own two answers:

```
 GAP CARD — against the reference package
 ───────────────────────────────────────────────────────────────────────────────
 mcp_servers/example_server   hello-world's worked MCP example       keep / remove
 data/channel_limits.json     hello-world's demo write limits        keep / empty
 ───────────────────────────────────────────────────────────────────────────────
 <artifact>                   <what it gives this deployment>        close / skip
 <feature block>              <what it gives this deployment>        close / skip
 <data item>                  <what it gives this deployment>        close / skip
 <persona>                    <what it gives this deployment>        close / skip
 ───────────────────────────────────────────────────────────────────────────────
```

`remove` on the first row deletes the `example_server` entry under `mcp_servers:` in
`profile.yml` and the `mcp_servers/example_server/` directory together. One without the
other costs every session a 20 second wait for a server that cannot start. `keep` leaves
it as the worked example the facility edits into its own first server.

The second row exists because `osprey init` writes `data/channel_limits.json` already
populated: demo storage-ring channels with hand-written `min_value` and `max_value`. Those
bounds are the demo machine's, not this facility's, and the limits hook checks every write
against them. `keep` is only honest once the facility's own channels are the ones in the
file, which they never are on a fresh build. Otherwise `empty` it to `_version` plus
`defaults`, or replace it with the facility's own file. The states are
`references/knowledge-starter.md` §4. This row is fixed because
`osprey scaffold pull control-assistant --list` reads the reference package, so it never
surfaces what hello-world itself emitted.

Derive the other rows live:

| Row source | How to get it |
| --- | --- |
| Artifacts | `osprey profile artifacts`, against the six profile lists in each file |
| Feature blocks | The top-level blocks in `src/osprey/profiles/presets/control-assistant.yml` |
| Data and context files | `osprey scaffold pull control-assistant --list` |
| Personas | The `modules.web_terminals.personas` catalog in that preset |

`close` runs the pull or the edit, then the wiring rule for anything with a data path.
`skip` is a decision, so it goes to INTERVIEW.md under Decided as `skipped — <reason>`.
No row is left blank.

## Questions at the end of MAP

MAP ends with four facts. Ask only for the ones the inventory did not yield, and say
where each one came from when it did:

- **Facility name**, for `facility.name`.
- **`facility.prefix`**, the short name the web container names are built from.
- **Timezone**, as an IANA name for `system.timezone`.
- **Project name**, which becomes the repo directory.

Only the last is an `osprey init` argument. `osprey init <project name>` takes no other
fact; the other three are `osprey set` keys applied to the emitted profile afterwards.
They are asked here because BUILD needs them from its first steps: `facility.prefix` at
step 3 of the web-terminal recipe above, and validate refuses the profile without it.

## Hand-off to the devil's advocate

CLOSE passes both cards, the status quo and the porting map, marked
**reference, do not reopen**. They are locked decisions, not proposals. The brief is
otherwise unchanged.

Add one check to that brief: **demo artifacts left in a real deployment**. Anything that
arrived from a preset and still describes the reference package rather than this facility
counts, including unedited stubs, template channel databases, demo personas and logins,
and the worked MCP example when the facility did not choose to keep it.
