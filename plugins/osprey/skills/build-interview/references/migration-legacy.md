# Migration reference — legacy OSPREY / LangGraph-era projects

Loaded from `references/discover.md` on the early-era fingerprint only. The OLD
architecture described here is frozen, so this file's facts are safe to
hardcode. Everything about the NEW side comes from the live deployment repo:
its profile.yml comments, its convention directories, and `osprey validate`.

## Classification

Every file in the old project gets one of four categories:

| Category  | Meaning                                   | Action                                    |
|-----------|-------------------------------------------|-------------------------------------------|
| SALVAGE   | Directly reusable                         | Confirm with user, place in the new repo  |
| OBSOLETE  | LangGraph-era machinery — discard         | Mention briefly, explain why unneeded     |
| TRANSFORM | Reusable content, wrong shape             | Extract values, re-express in the profile |
| EVALUATE  | Custom Python — may work, needs review    | Walk through with the user, one by one    |

When in doubt, EVALUATE — surface it rather than silently discard.

## Architecture mapping (old → classification)

| Old (LangGraph-era)                        | Why                                        | Category  |
|--------------------------------------------|--------------------------------------------|-----------|
| LangGraph graph definitions                | Claude Code is the orchestrator now        | OBSOLETE  |
| `osprey.context.CapabilityContext`         | Removed                                    | OBSOLETE  |
| `osprey.approval` module                   | Replaced by the approval hook              | OBSOLETE  |
| `osprey.gateway` / pipeline server         | Replaced by direct agent sessions          | OBSOLETE  |
| OpenWebUI pipeline server                  | Was the LangGraph gateway                  | OBSOLETE  |
| `registry.py` (component registry)         | Pattern still exists — check APIs          | EVALUATE  |
| Custom connectors (`connectors/*.py`)      | Connector layer still exists               | EVALUATE  |
| Custom providers (`models/providers/*.py`) | Provider registry still exists             | EVALUATE  |
| Custom prompt builders (`*prompts*/*.py`)  | Customization layer likely still needed    | EVALUATE  |
| `services/channel_finder/` full copies     | Framework-native now — likely redundant    | EVALUATE  |
| `data/channel_databases/*.json`            | Same format                                | SALVAGE   |
| `data/channel_limits.json`                 | Same format                                | SALVAGE   |
| `data/benchmarks/**`, `data/raw/*.csv`     | Same format                                | SALVAGE   |
| `data/tools/*.py`, machine-state JSON      | Utility data                               | SALVAGE   |
| Custom `.claude/rules/`, `.claude/skills/` | Content still valid                        | SALVAGE   |
| Custom `.claude/hooks/`                    | Hook API may differ — review               | TRANSFORM |
| `config.yml` (and variants)                | Values survive, shape changed              | TRANSFORM |
| Multi-role `models:` config                | Single provider+model now (see below)      | TRANSFORM |
| `requirements.txt` / `pyproject.toml`      | Facility deps → profile                    | TRANSFORM |
| `.env` / `.env.example`                    | Variable NAMES → profile `env:`            | TRANSFORM |
| `services/` (Docker, compose)              | Per-service review (see reading rules)     | TRANSFORM |

## Where things land in the NEW repo

Do not memorize target paths — open the live repo and read its profile.yml
comments — but the general destinations are:

- Data files → the repo's `data/` tree (same formats).
- Config values (gateways, archiver URLs, timezone, provider/model) →
  `osprey set key=value` into the live profile.
- Custom code and assets (rules, skills, agents, MCP servers, services,
  arbitrary project files) → the matching convention directory
  (`rules/`, `skills/`, `agents/`, `mcp_servers/`, `services/`,
  `project/`, …) — the directory name is the declaration.
- Environment variable NAMES → the profile's `env:` block (`required`
  vs `defaults`). NEVER copy values — even from a committed file; the
  user may not realize a token is in there. Flag `*_TOKEN`, `*_KEY`,
  `*_PASSWORD`, `*_SECRET` names explicitly in INTERVIEW.md.
- A secret **value** committed in the old repo — in a compose file, a
  template, a config default — is readable by everyone who can read that
  repo's history. Record the name, say once that the value is exposed
  there, and recommend rotating it. Never carry the value forward, not
  even into `.env`.

## Scan patterns

Match anywhere under the repo, not only at its root. These projects usually keep the
package under `src/<name>/` and its data under `src/<name>/data/`, so a root-anchored
pattern finds nothing. Skip `.venv*`, `__pycache__`, `node_modules` and `.git`.

```
**/config.yml, **/config.yaml, **/config.y*ml-*, **/*config*.y*ml
**/data/**/*.json, **/data/**/*.csv, **/data/tools/*.py
.claude/rules/**, .claude/hooks/**, .claude/skills/**
**/registry.py
**/connectors/*.py
**/providers/*.py
*prompts*/*.py, **/prompt_builders/**, **/framework_prompts/**/*.py
services/**, docker-compose*.yml, **/Dockerfile*
scripts/*.py, **/scripts/*.py
requirements.txt, pyproject.toml, .env*
**/*.py        # check for langgraph / StateGraph imports
```

Reading rules: a file whose **module-level** imports include `langgraph` or `StateGraph`,
and that matches no EVALUATE pattern above, is OBSOLETE. A file that matches an EVALUATE
pattern stays EVALUATE even when it imports either name — the path says what the file is
for, the import says what it will need changed, and the second is a note on the row, not a
verdict. An import inside a function is that note too, never the verdict. A
file subclassing a connector/provider base class or defining prompt builders
is EVALUATE. For `.env`, read variable names only. For channel databases,
count entries and note the format; show the user a short preview. For
`services/`, split each service: infrastructure (compose fragments,
Dockerfiles) is TRANSFORM, custom assets (startup scripts, CSS, seed data)
are SALVAGE, and anything referencing LangGraph or `CapabilityContext` is
OBSOLETE.

## Multi-role model config → single provider/model

Old projects assigned models to ~10 roles (orchestrator, response,
classifier, approval, task_extraction, memory, python_code_generator,
time_parsing, channel_write, channel_finder). The new architecture takes one
`provider` + `model`. Pick the dominant pair, record role-level exceptions in
INTERVIEW.md's migration notes, and check whether the old provider name is a
current built-in (the live profile's own comment names the selectable set).
A facility-custom provider module is EVALUATE.

## Config variants

Old projects often carry `config.yml-prod` / `config.yml-mock` variants.
Find all of them, ask which represents the target deployment, extract from
that one, and note the differences in INTERVIEW.md — the user may want a
second deployment repo for the other mode later.

## Early-era card rows

The status-quo card in `references/discover.md` § 7 is derived from verbs an early-era repo
cannot run. `config.yml` answers most of those rows instead. Read the block, then name the
file and line as the row's evidence — a row sourced this way is `verified from files`, not
`?`. A row with no block behind it stays `?`.

| Card row | Where it comes from |
| --- | --- |
| `OSPREY`   | `requires` is the osprey pin in `pyproject.toml` or `requirements.txt`, shown as `pinned floor <specifier>`; `built with` is `?` and `preset` is `none` |
| `provider` | the multi-role `models:` block — the dominant pair, per the section below |
| `control`  | the control-system block: type, whether writes are enabled, the limits file it names, the archiver it names |
| `data`     | the data paths the config names, counted where the files exist |
| `custom`   | the convention directories that exist, plus the package under `src/` |
| `env`      | names from `.env*` and from the `*_KEY`-style variables the config references |
| `obsolete` | one row per OBSOLETE path |
| `unknown`  | one row per EVALUATE item, until the walkthrough below answers it |

A config value that names a path outside the repo, or a file the repo does not contain, is
not a card row. Report it as a repo-side contradiction and ask which is current.

## EVALUATE walkthrough

For each item: say what it does (base class, added functionality, rough
size); check obvious API compatibility (does the base class still exist?
`osprey.context` imports always fail); then ask — port now (place in the
convention directory), port flagged (place it, note needed changes in
INTERVIEW.md), skip, or defer. Record every verdict in INTERVIEW.md.
