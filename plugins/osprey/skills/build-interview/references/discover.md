# DISCOVER reference

Loaded on every answer to the first question, "nothing yet" included. Covers the generation fingerprint, the inventory recipe per generation, the
exploration protocol for a facility with no OSPREY, and the status-quo card.

Two rules hold over everything below.

- **Every card line names its evidence.** A file path or a command, on request. A line
  with no evidence reads `?`. A `?` is never resolved by guessing.
- **Fingerprint before speaking.** Decide the generation from files first. Load no
  legacy knowledge unless its fingerprint matched.

## 1. Fingerprint

On "a deployment exists", look for these files at the repo root before asking anything.
On "a facility exists", ask for references first (section 6), then fingerprint each named
repo the same way.

| Fingerprint | Generation | Loads |
| --- | --- | --- |
| `profile.yml` carrying `provenance:` or `requires_osprey_version:` | current | nothing extra |
| `<name>-base.yml` plus `overlays/` plus `docker-compose.yml`, and no `profile.yml` | overlay era | § 4 of this file |
| `config.yml` plus a `registry.py`, or any `langgraph` / `StateGraph` import | early | `references/migration-legacy.md` |
| none of the above | none | nothing |

Fingerprint files match anywhere under the repo, not only at its root: an early-era project
keeps its package under `src/<name>/`. So the early test is `**/registry.py` and a grep for
`langgraph|StateGraph` over `**/*.py`, skipping `.venv*`, `__pycache__` and `node_modules`.

`profile.yml` wins. A repo that has both `profile.yml` and an `overlays/` directory is
current generation. Only a repo with no `profile.yml` at all is overlay era.

A current-generation repo's host overlays live in `profiles/<name>.yml`, and the choice
between them is one line in the git-ignored `.env.variant` (`OSPREY_PROFILE_VARIANT=<name>`).
When it selects one, `osprey validate` merges that overlay first and names it in its output.
With `profiles/` present and nothing selected — the normal state of a fresh clone — the
tracked `profile.yml` is what builds. List the `profiles/*.yml` names as the variants the
repo defines, and record the selection as `none selected`, not `?`.

`references/migration-legacy.md` loads on the early fingerprint and on nothing else. It
describes a frozen architecture, so loading it against a current repo teaches the wrong
map.

## 2. Framework version

One evidence order, no substitutes.

1. `build/.osprey-manifest.json`, key `creation.osprey_version`. This is the version that
   rendered the build zone.
2. No manifest, no version. The card's version reads `?`.

`requires_osprey_version` in `profile.yml` is a schema floor written as `>=<value>`. Show
it on the card as the floor it is, in its own field, never as the installed version.
Pins in `pyproject.toml`, lock files or compose image tags say what an environment may
install, not what built this repo, so they are quoted only when the user asks where the
`?` came from.

Two more strings name a version and are not it. `osprey validate` prints `profile.yml was
materialized from preset <name> at <hash> by OSPREY <a>; the preset bundled with OSPREY <b>
is <hash>`: `<a>` emitted `profile.yml`, `<b>` is the installed framework. `profile.yml`'s
own header comment `# emitted by OSPREY <version>` is that same `<a>`. Neither built the
repo.

Early era has neither a manifest nor a `profile.yml`, so the only version fact in the tree
is the osprey pin in `pyproject.toml` or `requirements.txt`. Show it as `pinned floor
<specifier>` and leave `built with` at `?`.

## 3. Inventory: current generation

Run these from the repo. None writes anything. They run on a current-generation repo only:
sections 4 and 5 run no verb at all.

| Command | What it yields |
| --- | --- |
| `osprey profile card --json` | Users, personas, provider, control system, panels, services |
| `osprey validate --drift=warn` | Validity, advisory warnings, plus one `⚠ preset drift:` line per unmarked difference |
| `osprey scaffold list` | Framework-managed and claimed artifacts. Run **only** when `build/.osprey-manifest.json` exists |
| `osprey profile artifacts` | The menu the six profile lists can name. Installation-wide: it takes no `--repo` and reads no repo |

**`osprey profile card --json`** emits one JSON array of `{group, label, value}` objects
on stdout, in card order. Up to four groups appear: `web terminal`, `agent`, `machine`
and `services`. A group with nothing to say is omitted, so a missing group is a fact
about the profile, not an error. A hello-world deployment emits only `agent` and
`machine`: no web tier means no `web terminal` group, no service blocks mean no
`services` group. Every value is a display string, so read it for the card and re-read
`profile.yml` for structure. Warnings go to stderr, one per persona whose delta could not
be read, for example a catalog entry whose `build_profile` is not a file in this repo.
Each such warning is a card line, not a silent omission.

**`osprey validate --drift=warn`** passes and reports. Drift is a difference from the
preset the profile came from, so a profile with no `provenance:` has nothing to drift
from and reports none. The comparison is against the preset bundled with the osprey that
runs the command, so the count moves when the installed framework does: the `drift` row
names that version. Put the count on the card and offer the list.

Validate also prints advisory `⚠` lines that are not drift — a privileged terminal with no
login wall, a bar item whose panel is not selected. They are findings, not failures. Append
each to the card row it concerns, and give it the `warnings` row when it concerns none.

**`osprey scaffold list`** reads the build zone. On a repo that was never built it still
prints the whole framework catalog with an empty claimed list, which reads as fact and is
not one. So check for `build/.osprey-manifest.json` first. When it is absent the artifacts
row reads exactly:

```
 claimed  ? (never built here)
```

**Native versus custom** is computed here, not by a command. For the five `.claude` kinds
(`hooks`, `rules`, `skills`, `agents`, `output_styles`), compare each list in `profile.yml`
against `osprey profile artifacts`: a name the menu offers is native, a name it does not
offer is custom.

**Panels have two classes, not three.** Native: on the menu, or a universal panel such as
`artifacts` that the menu omits because every profile carries it. Configured, not
catalogued: a `web_panels:` entry backed by `web.panels.<id>.url` in `config:`, or `events`
with a `dispatch:` block, or `bluesky` with a `bluesky_web:` block — the build derives those
two addresses after validation. Name what backs each configured one. Anything else fails
validation, so there is no custom panel class.

**MCP servers are not one of the six lists.** The card's `mcp` row is the framework
registry's enabled servers plus every key of `profile.yml`'s `mcp_servers:`. Compare keys,
not directory names: a key under `mcp_servers:` is the deployment's own, and a card entry
that is no such key is `osprey-native`. Read every file under `personas/` too — a delta can
declare servers the host profile does not, and the card row shows the host profile only. The
deployment's own servers split by how they start: `command:` is command-launched, and a
Python one keeps its package at `mcp_servers/<key>/` while a third-party one launched by
`uvx` or `npx` has no directory at all; `url:` or `port:` is a remote server, also with no
directory.

**Shadowing.** A file under a convention directory carrying a native artifact's name
(`rules/facility.md`, `agents/channel-finder.md`, `skills/diagnose/`) replaces the native
one in the build. The list keeps selecting that name and the repo's file is what renders, so
report it as `custom (shadows <name>)` — one artifact, not two.

**Convention directories** are declared by their own existence, so list them and count
entries: `rules/`, `skills/`, `agents/`, `commands/`, `output-styles/`, `hooks/`,
`web-terminal-context/`, `mcp_servers/`, `services/`, `project/`. A root directory that is
none of those and none of `profile.yml`, `triggers.yml`, `data/`, `personas/`, `profiles/`,
`scripts/`, `ci-extra.yml`, `osprey.service`, `build/`, `var/`, `docs/`, `README*`,
`LICENSE*` or a dot-entry
is not a zone: nothing reads it, and `osprey build` names it in its unrecognized-root-entry
warning. `overlays/` in a current-generation repo is exactly that. Give it an `unknown` row
and one question: what reads this?

**`data/`** is counted, not summarized: channel-database files and their tiers,
`channel_limits.json` entries, documents under the facility-knowledge bundle, lattice
files, ARIEL vocabulary, simulation scenarios, benchmark sets. Also count `personas/` and
`triggers.yml`. The card JSON's machine group gives `<mode> finder · tier <n>`; in `graph`
mode that is the whole channel-store fact, because graph ships no tiered database — its
store is a seeded service, not a file, and the tier there selects only the tier-3 benchmark
query set. Do not look for `data/channel_databases/` to count in graph mode.

**Environment variables** are names, never values. The source of truth is `profile.yml`'s
`env:` block: `required`, `pinned`, and the keys of `defaults`. A deployment's env chain is
`.env.shared` then `.env`, and `env.file:` names a profile-relative file copied as `.env`.
Any other `.env*` file in the repo may supply NAMES only. Never read a value out of one.

**Custom code** is anything under `mcp_servers/`, `services/`, `hooks/` or `project/`
that no framework artifact matches. Read each README or module docstring. When the
purpose is still unclear after reading, it is a `?` row on the card with the path and
what is unclear, and it becomes an `unknown` element in MAP.

Everything inventoried here has one row in section 7: personas and logins on `users`,
triggers and the `data/` counts on `data`, variable names on `env`, convention directories
on `custom`, validate's advisories on the row they concern or on `warnings`. Nothing is left
to prose.

## 4. Inventory: overlay era

Same card shape, derived by scanning. No OSPREY verb runs against these repos.

Read the base file's `overlay:` map first. It is the repo's own index of what it ships, and
it names files the patterns below would otherwise leave unmatched.

```
<name>-base.yml, overlays/**/*.yml, overlays/**/*.md, overlays/**/SKILL.md
docker-compose*.yml, Dockerfile*
data/**/*.json, data/**/*.csv
.claude/rules/**, .claude/skills/**, .claude/agents/**, .claude/hooks/**
services/**, mcp_servers/**, project/**
.env*, requirements.txt, pyproject.toml
```

Read the base file **and every child overlay**. The base says what the deployment is, but
provider and model may live in a child, and `extends: <preset>` inherits the rest from an
era preset that is not in this repo. A value that comes from that preset reads `inherited
(era preset)`, not `?`, and MAP treats it as a `native` candidate.

Variable names come from the file each overlay's `env.file:` names, since era repos ship no
`.env.example`. Names only, as in section 3.

Native versus custom has no verb here. The base file's own comments are the source; mark
those rows `verified from files`.

**Which overlay ran** takes an evidence order: a `build/` manifest first; failing that, a
README or a script that claims it, and then the row reads `reported, not verified`.

**Unmatched files are grouped, not listed.** Group by top-level directory, skip tool caches
and checkouts (`.venv*`, `.pytest_cache`, `__pycache__`, `node_modules`, `_agent_data`,
`.superpowers`, `.claude/plans`, `.git`), and give each remaining group one `unknown` row.
Unmatched is `unknown`, never `obsolete`.

## 5. Inventory: early era

Read `references/migration-legacy.md` and follow its scan patterns and reading rules. It
supplies the classification, the destinations, and the early-era row map — which part of the
old config answers which card row. Bring back the same card rows as § 3, with everything it
marks EVALUATE rendered as a `?` row until the user answers, and everything it marks
OBSOLETE on the `obsolete` row.

## 6. A facility with no OSPREY

Nothing is probed before the user names it. The skill never lists an endpoint, repo or
channel the user did not name.

**Ask once for references**: repos (control-system definitions, IOC databases, lattice,
tooling), documents, endpoints (logbook, archiver, ARIEL), channel lists, and who owns
what.

**Then, per reference, propose one exploration plan** as a single multiple-choice
question. The options:

| Option | What it does |
| --- | --- |
| local path or git repo | Read it, cloning first when it is a URL. List, read, count. No writes, no hooks run |
| HTTP endpoint the user named | One existence probe of the root. No auth handling beyond what the user gave |
| control-system read | Only when the user said reads are possible from this machine. One read of one channel the user named. Never a channel the skill chose |
| describe it to me instead | Available for every reference |

**Sequence per reference**: confirm the plan, probe for existence, report the result in
one line (`exists`, `not reachable`, or `no access`), then ask once whether to go deeper.
Anything not probed is marked `reported, not verified` on the card.

The result is the facility card in section 7: one row per reference, plus the `data` row
for lattice files, channel lists and simulators found inside a repo. A channel count from
code is `<n> <how counted>` (for example `<n> literal names, grep`), never a bare number. Go deeper
only on the user's yes; on no, every row stays `reported, not verified`.

## 7. The status-quo card

Existing deployment:

```
 STATUS QUO — <name>                        generation: <current|overlay|early>
 ─────────────────────────────────────────────────────────────────────────────
 OSPREY      requires <floor> · built with <version|?> · preset <name|none>
 drift       <n> unmarked differences from preset, measured with OSPREY <version>
 artifacts   framework-managed ×<n> · claimed ×<n>
 control     <type> · writes <ON|OFF> · limits <n> channels · archiver <type>
 provider    <provider> / <model>
 agents      <native list> · custom: <list>
 panels      <native list> · configured: <list>
 mcp         osprey-native ×<n> · own: <keys>
 data        channel db <mode> finder tier <n> · OKF <n> docs · lattice ×<n> · <…>
 env         names: <list> · from <where read>
 users       <n> entries · personas <list> · auth <method> (<wall|no wall>)
 custom      <dir>/ ×<n>, one per non-empty convention directory
 warnings    <validate advisory that belongs to no row above>
 unknown     ? <path> · <what is unclear>
 ─────────────────────────────────────────────────────────────────────────────
 Is this correct?  yes / modify
```

`auth` stands a login wall on `password` and `oidc` only; `token` (the default, and what an
absent `auth:` means) and `none` stand none.

An era repo has no verb behind three of those rows, so there they read:

```
 OSPREY      requires n/a · built with ? · extends <preset|none>
 drift       n/a (era repo)
 artifacts   ? (never built here)
```

An early-era repo adds one row for what the migration discards:

```
 obsolete    <path> · <why it is gone>
```

Facility with no OSPREY: replace the OSPREY rows with one row per reference, and add the
`source` and state columns.

```
 STATUS QUO — <facility>                                    generation: none
 ─────────────────────────────────────────────────────────────────────────────
 reference   <name>              <path|endpoint|user said>   <state>
 control     <type|?>            <source>                    <state>
 archiver    <type|?>            <source>                    <state>
 logbook     <type|?>            <source>                    <state>
 channels    <n|?> named         <source>                    <state>
 data        <lattice, lists|?>  <source>                    <state>
 owners      <who owns what|?>   <source>                    <state>
 ─────────────────────────────────────────────────────────────────────────────
 Is this correct?  yes / modify
```

`<state>` is one of `verified`, `verified from files` (named in a file, endpoint not probed),
`reported, not verified`, `not reachable`, `no access`. `<source>` is a short label; the full
path or `file:line` goes in the Sources list shown on request.

`<facility>` is the name the inventory yielded, else `?`.

"Nothing yet" has no repo: skip the fingerprint, `generation: none` comes from the answer.
Still ask once for references (section 6): a design report or a planned channel list counts.
Then render this same card with every value `?` and every state blank, so the user sees that
nothing was assumed, and ask the confirm question as usual; a "modify" here is how the user
adds what they do have.

The header then reads `STATUS QUO — ?`, and it stays that way. MAP asks the facility name
one phase later, but a locked card is never re-derived, so that answer does not back-fill
this header. The `?` is the record of what was known at DISCOVER, which is what the locked
card is for. Do not edit it after the fact, and say so if the user asks.

On `yes`, copy the card verbatim into `INTERVIEW.md` under `## Status quo (locked)`, and put
`generation:` and `phase: map` in the file **header**, above `## Coverage`. `INTERVIEW.md`
exists only after `osprey init` in BUILD; until then the locked card is held verbatim and
written the moment the file is created. A locked card is input to MAP, BUILD and the devil's
advocate, and is never re-derived. On `modify`, take the correction, note which line
changed and what the user said, and re-render the card.
