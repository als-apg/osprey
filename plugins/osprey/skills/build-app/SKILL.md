---
name: build-app
description: >
  Build an OSPREY app: a panel in the web terminal's sidebar, the backend
  container behind it, and the agent tools that reach the same backend, all
  declared in one app.yml. Use whenever someone wants to create, scaffold,
  extend or fix an OSPREY app or application, runs or asks about `osprey app
  new` / `osprey app check`, edits an app.yml, adds a tool, a route, a write
  with an approval card, or a skill to an app, or wants an operator screen
  that the agent can also read and drive, even when they only say "panel
  with a backend" or "give the agent a tool for X". Works in a facility
  deployment repo with OSPREY installed and in an OSPREY checkout. Starts by
  checking itself against the installed OSPREY's app contract. For a static
  themed page with no backend and no tools, use the panel skill instead.
---

# Build an OSPREY app

An app is any of four parts, at least one: a **backend** container (FastAPI)
that holds the state and talks to the machine, a **panel** the operator sees, a
**tool server** the agent calls, and **skills**. A panel needs the backend, and
so does a write tool on a login that writes; without one the tool server only
reads or computes, keeps nothing between calls and opens no data store of the
deployment. `app.yml` declares every part (an empty one is refused); `osprey
build` reads nothing else about the app. The scaffold is the full shape.

This skill carries what does not change from release to release: the order of
work, the product rules, what to verify and the mistakes to avoid. The facts
that do change (manifest keys, rule names, SDK names, command options) are
printed by a script from the OSPREY you are building against. Trust the
script's output over any key or name written here.

## Step 0: check this skill against your OSPREY

Run from the directory you will work in (the deployment repo, or the OSPREY
checkout), with the Python that has OSPREY:

```bash
python3 <this skill's directory>/scripts/app_contract.py
```

In a uv project, `uv run python <...>/scripts/app_contract.py`. It reads source
only (standard library, no network, about a second) and finds OSPREY in this
order: a checkout around the working directory, the interpreter's installed
package, the package behind the `osprey` command on `PATH`. `--root <dir>`
names one explicitly.

The first block is the verdict:

- **contract unchanged since review**: the snapshot and steps below hold.
- **contract CHANGED in N file(s)**: each changed file is listed with what it
  defines and which parts of this skill rest on it. Read those files (full
  paths are printed) before using those parts. The workflow and the product
  rules still stand; keys, rule names and signatures may not. It also says
  whether your OSPREY is older or newer than the one reviewed.
- **no OSPREY found** (exit 2): fix the interpreter or pass `--root`; do not
  build from the snapshot alone.

The rest is the live contract: every `app.yml` key with type and default,
every rule name with its message, the importable modules, the SDK's names, the
`osprey app` commands, the design-system classes, the scaffold's files. Keep
it; steps 3 to 6 refer to it as *the summary*. `--drift-only` and
`--summary-only` print one half.

## Product rules

These are decisions, not conventions. Each says what the checker enforces
today and what is left to you.

1. **At most one backend per app.** With a backend, the panel and every tool
   go through the same backend routes. A tool never reads a file, a store or a channel itself; it
   calls a route through the terminal's panel door (`/panel/<app>/<route>`).
   Enforced in part: the import rule keeps app code off OSPREY internals, and
   the door reaches only the routes `panel.agent_routes` declares. Nothing
   stops a tool from opening a file. Do not.
2. **Agent parity.** Every value the panel shows is in a read tool's answer;
   every control the operator can press is a tool or an argument of one. Not
   checked mechanically. The parity table in step 1 is how you hold to it.
3. **The app's own skill is optional.** `skills:` in `app.yml` defaults to
   empty. The check refuses only a *listed* skill whose
   `skills/<name>/SKILL.md` is missing, and a skill name two apps share. The
   scaffold ships one; keep it when tool docstrings alone do not tell the
   agent the order of work or what a refusal means, drop it (directory and
   `skills:` entry together) when they do.
4. **Panels use the design system.** `index.html` links the terminal's
   `tokens.css`, `base.css` and `primitives.css`, and uses `osprey-*` classes
   for buttons, chips, cards, toggles. `osprey app check` refuses a panel
   that omits a sheet and any class with no rule in the panel's or the linked
   sheets. It does not notice a hand-rolled button where a primitive exists;
   look in the summary's class list first.
5. **The session's control target, the connector, the approval path.** The
   backend answers for the login that asked (`owner_from_header`,
   `owner_target`) and never picks a machine from config. A hardware write is
   a tool under `tools.approval`, checked with `check_approved_call`, sent to
   the approved target, and written by the connector's
   `write_channel_checked` under `bind_owner`. The connector applies limits,
   the write verdict and the audit record. Any other route to hardware (a raw
   EPICS client, a subprocess) bypasses all of that; the import rule refuses
   the obvious ones.
6. **No hook code.** `osprey app check` refuses a `hooks` key in `app.yml`,
   a `hooks/` directory at the app root, a `.claude/` directory anywhere under
   the app, and `hooks` in a listed skill's frontmatter, each with rule
   `no_hooks`. Approval is declared, and OSPREY's own hook reads the rendered
   declaration; it never runs app code.

## Workflow

### 1. Decide before scaffolding

- **Is it an app?** State that the operator and the agent both act on, or a
  write that needs its own approval card: an app. A page that only displays:
  the panel skill. Tools or skills with no screen and no state: an app with
  no `containers:`.
- **Write the parity table.** One row per panel element: what it shows or
  does, the backend route, the tool or argument, and whether it writes a
  channel. A route that writes no channel (compute a plan, clear it) backs a
  `read` tool even when it is a `POST`. A route that writes a channel backs
  an `approval` tool.
- **What does an approval bind?** For each write: the state the operator
  must have seen (a readback, a plan id, a digest of setpoints). That becomes
  `bind` in the manifest and a re-check in the backend.
- **What is the facility's, not the app's?** Channels, device lists,
  tolerances go in `apps.<name>.config` in `profile.yml`, never in code.

### 2. Scaffold and configure

An app lives in a deployment repo (one with `profile.yml`). In a facility
repo that is where you are. In an OSPREY checkout, make a scratch one outside
the tree: `osprey init <dir> --preset demo --no-git`.

```bash
osprey app new <name>     # lower-case letters, digits, dashes
osprey app check          # the scaffold passes as copied
```

`new` copies the scaffold to `apps/<name>/`, renames the app and its Python
package, lists it under `apps:` in `profile.yml` (listing turns it on) and
git-ignores fetched panel files. A name an installed package already gives an
application is refused. Read the copied files before changing them:
they are the reference implementation of every pattern below. The summary
ends with the path of a larger worked example (several routes and tools, a
panel that writes).

Give it its config:

```yaml
apps:
  <name>:
    config:
      channel: <a channel of this facility>
```

The block is free-form; the build writes it to the file named by
`OSPREY_APP_CONFIG`, which the backend reads at start. An app with no
container has nothing to read it, so its `config:` is refused.

### 3. Manifest

Edit `app.yml` last-to-first as you add things, but understand it first: a
route the tools call must be in `panel.agent_routes`; a tool must be under
`tools.read` or `tools.approval`; the backend container that writes needs
`drives_target: true`. Take keys and defaults from the summary.

### 4. Backend (`<package>/server.py`)

- One state route (`GET`) answers the whole screen in the panel's terms,
  including why a control is disabled. The panel and the read tool both show
  exactly this.
- Resolve the machine per request from the owner; keep one connector per
  target as the scaffold does.
- A write route verifies the launch token, re-reads the state the approval
  was bound to and refuses with 409 when it moved, then writes through the
  connector under the owner. Pass the connector's refusal sentence on.
- Facility knowledge (elements, channels, model members, response matrices)
  comes from the SDK, not from files the app carries.
- `create_app(connector_for=..., config=...)` takes a fake connector and a
  config, so routes are testable without a machine.

### 5. Tool server (`<package>/tools.py`)

- One function per tool inside `register`. The docstring is the description
  the agent reads: what it does, each argument in the facility's terms, what
  it answers.
- Every tool calls the backend through `_panel_call`. It imports nothing from
  the backend module.
- A write tool begins with `check_approved_call` and sends the backend the
  values that returned (`bound.approved_target`, `bound.state_fields`), never
  the agent's own. Its `approved_*` parameters default to `None`; the hook
  fills them.
- Report with `make_error` (unreachable) and `ApprovalRefused` (refused), so
  the agent gets the manifest's `errors` sentences.

### 6. Panel and its tests (`panel/`, `tests/`)

- Plain HTML, CSS and ES modules. Addresses are relative (`api/state`), since
  the page is served under `/panel/<id>/`; the panel's files are served from
  `static/`.
- The page says when it can be driven: `index.html` carries
  `<html data-page-ready="pending">`, and the script imports `markPageReady`
  from `/design-system/js/page-ready.js` and calls it once the first refresh
  has drawn, error or not. `osprey app check` refuses a panel that never
  imports it; whatever drives the panel waits for that mark.
- Primitives first; `panel.css` holds only the app's own layout, on design
  tokens (`var(--space-2)`), with a rule for every class you introduce.
- Export the functions that render state and test them in
  `tests/*.test.js` (Vitest, DOM). The test builds its own DOM, so a new
  field goes into the test's markup too.
- A panel that offers a write posts to the same write route the tool uses.
  When writes are off, say so in the panel; do not hide the control.
- A third-party JS library is a `vendor` entry (exact version, SHA-256), not
  a committed file.

### 7. The app's skill (optional)

When kept: write it from the parity table. Which tool for which request, each
tool's panel equivalent, the order (read, tell the operator, then write), what
each refusal means, and that `approved_*` arguments are left unset. Update the
table whenever a tool or panel element changes.

### 8. Verify

```bash
osprey app check                  # every change
osprey build && osprey up -d      # to see it in the terminal
osprey health
```

`check` is done when it prints the pass line and exits 0. It needs Node and
npm for the panel's lint and tests; the first run installs a pinned toolchain
into the user cache and needs network once. Then in the terminal: the panel
shows state for the session's machine and follows the control-target chip;
the read tool answers what the panel shows; the write tool is refused with
writes off, shows a card with your prompt at *asked*, and is denied on a
read-only login. Walk the parity table once more against the running panel.

## Writes and approvals

One `tools.approval` entry per write tool. `prompt` is the card's text, with
`{{ input.<arg> }}` for call arguments and `{{ state.<key> }}` for the answer
of `state_path`. `bind.state_fields` copies state values into the call as
`approved_*` fields (a dotted `key`, `step.digest`, is a path into nested
state); `bind.input_digests` binds the arguments themselves;
`require` names the fields a call must carry. Three places then check one
declaration: OSPREY's approval hook, the tool (`check_approved_call`), and the
backend against its current state.

Hold to these, and take their exact rule names from the summary:

- `state_path` is a `GET` route listed in `agent_routes`.
- Bound fields are named `approved_*`; `approved_target` is recorded by the
  approval itself and only needs requiring.
- Every bound input digest is in `require.all_of`; a `one_of` group holds
  state-field names only.
- A write tool that binds fetched state cannot run in a dispatch job. Set
  `deny_in_jobs: true` when no tool of the app should.
- A login arms the app when it shows it and its tier is `asked` or `allowed`;
  write routes and the launch token exist only there.

## Common mistakes

- A panel value computed in JavaScript from several routes, with no single
  tool answer carrying it. Compute it in the backend's state route.
- A second read path "for the agent" (a file, a database, a direct channel
  read in `tools.py`). It drifts from the panel within a week.
- A channel name, device list or machine name in code instead of
  `apps.<name>.config` and the owner's target.
- A write tool that forwards the agent's arguments instead of the bound
  values, or a backend that skips the re-read.
- A new route used by a tool but missing from `agent_routes` (the door
  refuses the call), or a new tool missing from `tools.read` / `tools.approval`.
- An absolute panel address (`/api/state`): works on the bare port, breaks
  behind the terminal.
- A class added in HTML or JS with no CSS rule, or panel markup changed
  without the test's markup.
- Importing `osprey.<anything else>` or an underscore-prefixed name. The
  allowed modules are in the summary; if the SDK lacks what you need, that is
  a request to OSPREY, not a reason to reach in.
- Renaming the app by hand. `name`, the directory, the package, the panel id
  and the tool server share the name; make a new one with `osprey app new`
  and move the code.
- Editing compose concerns (`volumes`, `ports`) into `app.yml`. They are
  refused with the place they belong.

## Contract snapshot

Orientation only, as of the OSPREY this skill was reviewed against (step 0
names it, and says when this has moved). The summary is the authority.

```yaml
name: <name>                  # = directory = panel id = tool server name
summary: <one sentence>
containers:                   # optional; a panel and armed write tools need one
  <name>:
    runtime: osprey           # or image: with a compose fragment
    command: [python, -m, <package>.server]
    health: /health
    drives_target: true       # the container reaches the control system
    state: var/<name>         # kept across restarts
panel:
  label: <sidebar text>
  icon: <glyph>               # optional; so are description: (tooltip) and help: (page)
  container: <name>
  launch_token: true          # write routes demand the terminal's token
  agent_routes:               # what the tools may call; non-GET = write route
    - [GET, /api/state]
    - [POST, /api/write]
tools:
  module: <package>.tools
  read: [<tool>, ...]
  approval:
    - tool: <write tool>
      prompt: Set {{ state.channel }} to {{ input.value }}, now {{ state.readback }}.
      state_path: /api/state
      bind: {state_fields: [{field: approved_readback, key: readback}]}
      require: {all_of: [approved_target, approved_readback]}
  errors: {unreachable: <sentence>, refused: <sentence>}
skills: [<name>]              # optional
```

Files of an app: `app.yml`, `pyproject.toml`, `<package>/{__init__,server,tools}.py`,
`panel/{index.html,panel.js,panel.css}`, `tests/panel.test.js`,
`skills/<name>/SKILL.md`. Application code imports `osprey.apps.sdk`,
`osprey.facility.kinds` and `osprey_connectors`, nothing else of OSPREY.

The narrative version of all this, with the worked example, is the
documentation's *How-to: Apps* section (*Build Your Own App*).

## Maintaining this skill

`contract-fingerprint.json` records a hash of each file the contract lives in
(the list is `WATCHED` in the script). When one changes in the OSPREY
repository, `tests/plugins/test_build_app_contract.py` fails and names it.
Re-read this file against the change, correct what it made wrong, then record
the review:

```bash
uv run python plugins/osprey/skills/build-app/scripts/app_contract.py --refresh
```
