# Migration reference — legacy OSPREY / LangGraph-era projects

Loaded when the interview is migrating an existing project. The OLD
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

## Scan patterns

```
config.yml, config.yaml, config.yml-*, config.yaml-*
data/**/*.json, data/**/*.csv, data/tools/*.py
.claude/rules/**, .claude/hooks/**, .claude/skills/**
registry.py, **/registry.py
connectors/*.py, **/connectors/*.py
models/providers/*.py, **/providers/*.py
*prompts*/*.py, **/prompt_builders/**, **/framework_prompts/**/*.py
services/, docker-compose*.yml, **/Dockerfile*
requirements.txt, pyproject.toml, .env, .env.example
src/**/*.py        # check for langgraph / StateGraph imports
```

Reading rules: a file importing `langgraph` or `StateGraph` is OBSOLETE; a
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

## EVALUATE walkthrough

For each item: say what it does (base class, added functionality, rough
size); check obvious API compatibility (does the base class still exist?
`osprey.context` imports always fail); then ask — port now (place in the
convention directory), port flagged (place it, note needed changes in
INTERVIEW.md), skip, or defer. Record every verdict in INTERVIEW.md.
