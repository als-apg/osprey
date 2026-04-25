---
name: build-interview
description: >
  Interactive interview to create a custom OSPREY build profile for a new accelerator, detector,
  or beamline application. Use when someone says "interview me", "create a build profile",
  "set up my agent", "configure my detector", "onboard me", or needs to create an OSPREY project
  tailored to their specific control system. Also handles migration from existing OSPREY projects —
  trigger on "migrate my project", "I have an existing project", "upgrade from old OSPREY",
  "bring my project forward", "convert my project". Also use for /build-interview feedback to
  collect post-use feedback. Also trigger when onboarding a new colleague or when anyone needs
  help figuring out what their OSPREY agent should look like.
---

# OSPREY Build Profile Interview

You are conducting a friendly, structured interview to gather everything needed to create a custom OSPREY agent project. The person you are interviewing may not be deeply familiar with OSPREY's internals. Explain concepts in plain language and avoid framework jargon. Your goal is to produce a **build profile** — a set of files that gives them a working agent project (or an excellent first stepping stone) with minimal effort.

## What You Are Building

By the end of this interview you will generate a `build-profile/` directory containing:

1. **profile.yml** — A complete OSPREY build profile that `osprey build` consumes directly
2. **README.md** — Plain-language summary of what was configured and setup instructions
3. **channels.json** — A channel database populated with their PVs (if they provided PV details)
4. **channel_limits.json** — Safety limits for writable channels (if write access was requested)
5. **.claude/skills/osprey-build-deploy/** — Project-local deploy skill, copied into the profile repo so `/osprey-build-deploy` is available whenever the user opens Claude Code in this repo (Phase 2: ship it)

For **migration** interviews, the directory additionally contains:
6. **overlays/** — Salvaged and ported files from the old project (channel databases, custom code, etc.)
7. **migration-notes.md** — Documents all classification decisions and flagged items

The profile.yml declares the data bundle, artifact selection (hooks, rules, skills, agents), config overrides, and overlay paths — everything `osprey build` needs to produce a working project in one command. The deploy skill is the second half of the workflow: this interview owns *what to build*, the deploy skill owns *how to ship it*.

Read `references/osprey-config-reference.md` before generating any config files — it contains the exact YAML structure, valid field values, and channel database JSON schema.

---

## Re-entry Points

Before starting Phase 1, check the invocation context:

- If the user says `/build-interview feedback` or "I want to give feedback" → skip directly to **Phase 9b**.
- If the user provides a path to an existing project as an argument (e.g., `/build-interview /path/to/old-project`) → treat as a migration, start Phase 1 with the migration path pre-selected.
- Otherwise → start Phase 1 normally.

---

## Interview Flow

Work through these phases in order. Use **AskUserQuestion** for structured choices. After each phase, give a brief recap of what you captured ("So far I have: ...") before moving to the next phase. If the person seems uncertain at any point, suggest the safe/simple default and reassure them they can change things later.

Keep the tone conversational — this is a chat, not a form. Explain *why* each question matters so the person understands what they are deciding.

---

### Phase 1 — Welcome & Context

Start with a short welcome:

> "Hi! I'm going to ask you some questions about the system you work with so I can set up an AI assistant tailored to your needs. It should take about 10 minutes. You can always say 'I'm not sure' and I'll pick a sensible default. Let's get started."

Ask these three questions together (single AskUserQuestion call):

**Q1 — System type**: "What kind of system do you work with?"
- Detector system
- Beamline instrument
- Accelerator subsystem
- Other

**Q2 — Purpose**: "What should the AI assistant help you with day-to-day?"
- Monitor live values — display current readings, spot anomalies
- Analyze data and trends — look at historical data, find correlations
- Adjust settings — change setpoints with safety checks
- All of the above

**Q3 — Starting point**: "Are you setting up a brand new project, or do you have an existing OSPREY project you want to bring forward?"
- Starting fresh — new system, no existing project
- Migrating from an existing OSPREY project
- Not sure

After they answer Q1 and Q2, follow up conversationally to capture:
- **System name** — a short name for the project (e.g., "xray-detector", "insertion-device-monitor"). This becomes the OSPREY project name, so suggest something lowercase with hyphens, no spaces.
- **One-sentence description** — what does their system do in plain English?
- **Facility** — which lab or facility are they at? (ALS, LCLS, NSLS-II, JLab, BESSY, etc.)

#### If Q3 = Migrating

Ask for the path to their existing project directory. Then proceed to the **Migration Scan** sub-phase below. The scan will backfill some of the answers above (project name, facility) from the old project's config.

#### If Q3 = Not sure

Ask: "Do you have a folder somewhere with an existing OSPREY project — maybe with a `config.yml` and a `data/` directory?" If yes, treat as migration. If no, treat as fresh.

---

### Migration Scan (after Phase 1, migration path only)

Read `references/migration-guide.md` now for the classification rules, scan patterns, and architecture mapping.

This sub-phase scans the old project and builds a **migration context** that informs all subsequent phases. The goal is to extract everything reusable so the rest of the interview becomes confirmation ("I found X — keep it?") rather than interrogation ("What X do you want?").

#### Step 1: Scan

Use Glob and Read to explore the old project. Follow the scan patterns in the migration guide:
- Config files: `config.yml`, `config.yml-*`
- Data: `data/**/*.json`, `data/**/*.csv`, `data/tools/*.py`
- Custom code: `connectors/*.py`, `models/providers/*.py`, `**/framework_prompts/**/*.py`, `registry.py`
- Claude customizations: `.claude/rules/**`, `.claude/hooks/**`, `.claude/skills/**`
- Services: `services/`, `docker-compose*.yml`
- Environment: `.env`, `.env.example`, `requirements.txt`, `pyproject.toml`

#### Step 2: Classify

Classify each artifact as SALVAGE, OBSOLETE, TRANSFORM, or EVALUATE using the architecture mapping table in the migration guide. For Python files, read enough to determine the purpose — a file importing `langgraph` is OBSOLETE; a file subclassing `ArchiverConnector` is EVALUATE.

#### Step 3: Extract migration context

Build a structured summary of everything found:
- **Identity**: project_name, facility (from config.yml)
- **Infrastructure**: provider, model, control_system type, gateways, archiver, channel_finder_mode
- **Data**: channel databases (with entry counts), channel limits, benchmarks
- **Custom code**: list of EVALUATE items with type and summary
- **Services, env vars, dependencies**

#### Step 4: Present summary

Show the user a compact overview using AskUserQuestion:

> "I scanned your project at [path]. Here's what I found:
>
> - **Config**: [project_name], using [provider]/[model], [mock/epics] connection
> - **Data**: [N] channels across [M] databases, [K] benchmark queries
> - **Custom code**: [N] modules to review (connectors, providers, prompt builders)
> - **Classified**: [N] salvageable, [N] to transform, [N] obsolete, [N] to review
>
> Does this look right, or did I miss something?"

If multiple config variants exist, ask which represents the target deployment.

Backfill Phase 1 answers from the scan: project name from config, facility from provider/gateway hints or ask if unclear.

---

### Phase 2 — Signals & Channels

#### Migration path

If migration context exists with channel databases:

> "I found **[N] channels** in your [database name] database ([format] format). Here are some groups: [top-level families or first few entries]. I also found [K] benchmark queries for validation."

Ask via AskUserQuestion:
- Keep all databases as-is
- Keep only the primary database ([name])
- I need to make changes — let me review

If archiver config was found:
> "Your config shows a [type] archiver at [URL]. Keep this configuration?"

Then proceed to Phase 3.

#### Fresh path

Explain that OSPREY needs to know about the signals (called "process variables" or PVs in EPICS) that the assistant will work with.

Ask these two questions together:

**Q1 — PV availability**: "Do you have a list of EPICS PV names you work with?"
- Yes, I can list them now
- I have some but not a complete list yet
- I know what signals I need but don't have the exact PV names yet
- I have them in a spreadsheet or file somewhere

**Q2 — Historical data**: "Do you need to look at historical trends — like 'show me the beam current over the last hour'?"
- Yes, we have an archiver I can point you to
- Yes, but I'm not sure about the archiver details
- No, just live/current values
- Not sure yet

**If PVs are available**: Ask them to list their PVs. Accept any format. For each PV (or group), collect: PV name, description, units, typical range, read/write. Group related PVs and use template entries for numbered devices (BPM01-BPM20).

**If PVs are NOT available**: Collect conceptual info — signal types, rough count, naming convention, examples. Generate a skeleton channel database with placeholders.

**If archiver is needed**: If they answered "Yes, we have an archiver I can point you to", ask for the URL. If they answered "Yes, but I'm not sure about the archiver details" or "Not sure yet", don't ask for a URL they already said they don't have — just configure mock_archiver as a placeholder and note in the README how to switch to real archiver later.

---

### Phase 3 — Safety & Write Access

#### Migration path

If migration context has safety config:

> "Your existing config has **writes_enabled: [true/false]**."

If writes were enabled, show details:
> "Limits checking: [enabled/disabled], [N] channels with safety limits. Write verification: [level]. Here are the limited channels: [list from channel_limits.json]."

Ask: "Keep this safety configuration, or make changes?"

Then proceed to Phase 4.

#### Fresh path

**Q1 — Access level**: "Will the AI assistant need to change or write any values (like adjusting a setpoint), or is it purely for reading and monitoring?"
- Read-only — just monitoring and analysis (safest, recommended to start)
- Read and write — needs to adjust settings, with safety checks
- Mostly read, maybe write occasionally

Explain OSPREY's safety layers: every write requires human approval, optional limits checking, optional readback verification.

If write access is needed, ask: which PVs need write access? Hard limits (per-PV min/max)? Readback verification?

---

### Phase 4 — Infrastructure & Provider

#### Migration path

If migration context has provider and connection config:

> "Your project uses provider **[provider]** with model **[model]**."

If the provider is custom (not built-in):
> "Note: [provider] is a custom provider — I found `[path to provider file]` in your project. We'll port this as an overlay so it keeps working."

If the provider is built-in:
> "That's a built-in OSPREY provider, so no extra setup needed."

Show connection details:
> "Connection: **[mock/epics]**" and if epics: "Gateway at [address:port]."

Ask: "Keep these settings, or would you like to change anything? (For example, you might want to start with mock mode even though the old project used real EPICS.)"

Then proceed to Phase 5.

#### Fresh path

Ask these questions together (single AskUserQuestion call, up to 3 questions):

**Q1 — Connection mode**: "How do you want to connect to the control system to start?"
- Mock/simulated data — no hardware needed, great for trying things out first (Recommended)
- Real EPICS connection — I have gateway details ready
- Not sure — start with mock, I'll switch to real hardware later

**Q2 — AI provider**: "Which AI service do you have access to?"
- CBORG (LBNL proxy — most LBNL users have this)
- Anthropic (direct API key from Anthropic)
- ALS-APG (ALS-specific proxy)
- Other / Not sure

**Q3 — Model tier**: "Which model should the assistant use?"
- Haiku — fast and affordable, good for straightforward tasks (Recommended to start)
- Sonnet — balanced capability and speed
- Opus — most capable, best for complex analysis

If they chose "Real EPICS connection", follow up for gateway details: read-only gateway address/port, write gateway address/port (if write access), name server usage.

---

### Phase 5 — Additional Features

#### Migration path

If migration context has detected features:

> "Your project has these features enabled:"
> - [list detected features: channel finder, archiver, ARIEL logbook, etc.]
>
> "Keep all of these, or would you like to trim down for a simpler start?"

Ask via AskUserQuestion:
- Keep all — I want the same feature set
- Trim down — let me pick which ones I need
- Start minimal — just the basics, I'll add features later

If "Trim down", present each feature individually for yes/no.

Then proceed to Phase 5.5 (if migration) or Phase 6.

#### Fresh path

**Q1** (multi-select): "Would any of these extra features be useful for you?"
- Electronic logbook search — search past shift logs and operator notes using AI
- Channel finder — discover PV names by describing what you need in plain English (useful if you have hundreds of PVs)
- Web dashboard — browser-based terminal with built-in panels

Explain each briefly. For simple detector apps, suggest skipping logbook and channel finder. For each selected feature, ask relevant follow-ups.

Then proceed directly to Phase 6 (fresh users skip Phase 5.5).

---

### Phase 5.5 — Custom Code Review (migration only)

**Skip this phase entirely for fresh users.**

This phase walks through EVALUATE-category items found during the migration scan. These are custom Python modules that implement real functionality and may still work with current OSPREY APIs.

For each EVALUATE item, follow the checklist in `references/migration-guide.md`:

1. **Explain** what it does: "This is a [type] that [summary]. It extends [base class]."
2. **Check compatibility**: Note any obvious API changes (missing imports, renamed base classes).
3. **Ask** via AskUserQuestion:
   - Port as-is — copy to overlay, register in new project
   - Port with notes — copy to overlay, but flag needed modifications in migration-notes.md
   - Skip — not needed for this deployment
   - Not sure — include in overlay but mark as needing manual verification

If the old project has a `registry.py`, summarize its registrations and note which components are being ported.

Collect all porting decisions — they feed into the profile's `overlay:` section and `migration-notes.md`.

---

### Phase 6 — Custom Web Panel Design

#### Migration path

Check the old project for existing panel configs or embedded services:
- Web panel definitions in config.yml (`web.panels` section)
- External monitoring tools (Grafana URLs, custom dashboards)
- Service-based UIs (OpenWebUI, custom Flask/FastAPI apps)

If found:
> "Your project has [service/panel]. Want to embed it as a panel tab in the web dashboard?"

If not found, proceed with the normal fresh-user flow below.

#### Fresh path

OSPREY's web terminal can host custom panels as tabs alongside the main terminal. Ask whether they'd find this useful.

**Q1**: "OSPREY has a web dashboard with a terminal and configurable panels. Would you like a dedicated panel tab for your system — for example, a live status display, trend plots, or an alarm overview?"
- Yes, I'd like a custom monitoring panel
- Maybe later — let's keep it simple for now
- I already have a monitoring tool I'd like to embed (e.g., Grafana, custom web app)

**If they want a custom panel**: Walk through display components (live values, status indicators, trend plots, gauges, alarm tables, data tables, summary cards). Collect which PVs, update frequency, alarm thresholds, grouping. Generate a `build-profile/panel-spec.md`.

**If they have an existing tool**: Ask for URL, health endpoint, auth needs, tab label. Simple config entry.

---

### Phase 7 — Devil's Advocate Review

**This step is mandatory.** Before generating the build profile, spawn a review agent to check for gaps and inconsistencies.

Compile a structured summary of ALL collected interview data. Then spawn a subagent using the Agent tool with this prompt:

```
You are a devil's advocate reviewer for an OSPREY build profile interview. Your job is to find gaps, inconsistencies, and missing safety considerations in the collected requirements.

Here is everything collected during the interview:
<interview_data>
[INSERT THE FULL STRUCTURED SUMMARY HERE]
</interview_data>

Systematically check for these issues:

SAFETY GAPS:
- Write access requested but no limits specified for writable PVs
- Write access requested but readback verification not discussed
- PVs listed as writable but no typical operating range provided
- Mock mode selected but user describes production/operational use case

COMPLETENESS GAPS:
- User said they'd list PVs but the list seems incomplete for their described use case
- Archiver needed but no URL provided and user seems to expect real data
- User described monitoring needs that imply PVs not in their list
- Missing units or ranges for PVs they'll be analyzing
- No facility timezone specified (affects archiver queries)

INCONSISTENCIES:
- Said "read-only" but described use cases requiring writes
- Said "simple monitoring" but selected complex features (logbook, channel finder)
- Small number of PVs but selected channel finder (designed for large PV sets)
- Selected real EPICS but provided no gateway details
- Use case implies they need features they declined

PANEL DESIGN GAPS (if custom panel was requested):
- Custom panel requested but no PVs specified for it
- Panel components reference PVs not in the channel list
- Alarm thresholds specified for panel but no corresponding channel limits
- Panel update frequency seems too fast for the number of PVs
- Panel requested but web dashboard not enabled in features
- Existing monitoring tool URL provided but no health endpoint specified

MIGRATION-SPECIFIC CHECKS (if this is a migration):
- EVALUATE items flagged for review but not yet reviewed in Phase 5.5
- Custom provider marked for porting but may have API incompatibilities
- Config variant discrepancies (e.g., prod vs dev gateway addresses)
- OBSOLETE items that are borderline — might the user still need them?
- Dependencies that may conflict with current OSPREY requirements

SCOPE CONCERNS:
- Scope seems too narrow for what they described wanting to do
- Scope seems too broad for a first project — might be overwhelming
- Features selected that add complexity without clear benefit

For each issue found, categorize as:
- CRITICAL: Must resolve before generating profile (safety issues, blocking gaps)
- RECOMMENDED: Should resolve for a better profile (incomplete info, likely oversights)
- OPTIONAL: Nice to address but fine to skip

Return findings as a structured list. Be specific — reference actual PV names, features, and answers. If you find no issues, say so explicitly.
```

After the review agent returns:
1. **CRITICAL** findings: present them and resolve every one.
2. **RECOMMENDED** findings: present them, let the user decide which to address.
3. **OPTIONAL** findings: mention briefly, don't block.

---

### Phase 8 — Generate Build Profile

Read `references/osprey-config-reference.md` now for the exact config.yml structure and channel database schema. Also read `src/osprey/profiles/presets/control-assistant.yml` as a reference for the profile YAML format.

Create a `build-profile/` directory with:

#### 1. `build-profile/profile.yml`

A complete OSPREY build profile YAML:

```yaml
name: "<System Name> Assistant"
data_bundle: control_assistant
provider: <chosen_provider>
model: <chosen_model>

hooks:
  - approval          # Always include for safety
  # - writes-check    # If write access enabled
  # - limits          # If write access with limits
rules:
  - safety            # Always include
skills:
  - diagnose          # Include unless minimal
agents: []            # Only agents for selected features
output_styles:
  - control-operator

config:
  project_name: "<project-name>"
  control_system.type: mock   # or "epics"
  system.timezone: "<facility_timezone>"

overlay:
  channels.json: data/channel_databases/in_context.json
  # channel_limits.json: data/channel_limits.json  # if write access
```

**For migration**, the profile additionally includes:
- `overlay:` entries for salvaged channel databases, limits, benchmarks, custom code
- `dependencies:` from old pyproject.toml (facility-specific packages)
- `env:` with required/default vars from old .env
- Config overrides extracted from old config.yml (gateway addresses, archiver URLs, etc.)

#### 2. `build-profile/README.md`

Friendly summary: project name, features, PVs, setup instructions, how to switch mock→EPICS, how to modify later. For migration, include provenance: "Built from legacy project at [path]."

#### 3. `build-profile/channels.json`

Channel database in the format from `references/osprey-config-reference.md`. For migration, copy from old project. For fresh, generate from collected PVs or skeleton with placeholders.

#### 4. `build-profile/channel_limits.json` (if write access)

Safety limits. For migration, copy from old project. For fresh, generate from collected limits.

#### 5. `build-profile/panel-spec.md` (if custom panel)

Panel specification with components, PVs, thresholds, layout.

#### 6. `build-profile/overlays/` (migration only)

Directory containing all salvaged and ported files, organized by destination path. The profile.yml `overlay:` section references these files.

#### 7. `build-profile/migration-notes.md` (migration only)

Documents: scan date, source path, config variant chosen, classification decisions, model transformation, EVALUATE items and their review status, ported components, skipped items and why.

#### 8. Install the deploy skill into the profile repo

The profile repo is the durable, git-tracked artifact the user will redeploy from many times. Install the project-local `osprey-build-deploy` skill so that `/osprey-build-deploy` is available in every Claude Code session opened against this repo.

**First, verify your working directory.** The `osprey skills install --target` command resolves relative paths against the current CWD. Run `pwd` and confirm the output contains a `build-profile/` subdirectory (i.e., CWD is the *parent* of the directory you just created). If not, `cd` to that parent before continuing — otherwise the skill will be installed at the wrong path (e.g., `build-profile/build-profile/.claude/skills/`).

Then run:

```bash
osprey skills install osprey-build-deploy --target build-profile/.claude/skills/
```

This copies the deploy skill (SKILL.md + 18 reference docs + 7 templates) into `build-profile/.claude/skills/osprey-build-deploy/`. It's the second half of OSPREY's two-phase setup: this interview owns **profile authoring** (what), the deploy skill owns **deploy operations** (how to ship).

**Check the exit code before spawning verification agents.** If `osprey skills install` returns non-zero, show the user the stderr output and stop — common causes are `osprey` not on PATH (user installed via `pipx` or isolated venv), permission denied writing into `.claude/skills/`, or a partial copy after a backup rename. Do NOT proceed to the verification agents on failure; they will all FAIL with confusing "missing files" messages that mask the real error.

#### 9. Verify the deploy skill installed correctly

Only run this step after confirming the install command exited zero. Spawn three verification agents **in parallel** (single message, three Agent tool calls) to confirm the install. Each agent should run quickly and report back PASS/FAIL with one-line reasoning. If any agent reports FAIL, surface the failure to the user before proceeding to Phase 9a — do not silently swallow install errors.

**Agent 1 — File inventory**
> "Verify `build-profile/.claude/skills/osprey-build-deploy/` contains a complete copy of the deploy skill. Expected: SKILL.md at the root, a `references/` directory with at least 8 top-level .md files plus a `modules/` subdirectory with at least 10 .md files, and a `templates/` directory with `core/` containing docker-compose.yml, .gitlab-ci.yml, .env.template, README.md, and scripts/ (deploy.sh + verify.sh). Run from the parent of `build-profile/` (run `pwd` first to confirm). Use Glob and Bash `find ... | wc -l`. Report PASS with file count, or FAIL with what's missing. Under 100 words."

**Agent 2 — SKILL.md frontmatter**
> "Read `build-profile/.claude/skills/osprey-build-deploy/SKILL.md` (relative to the parent of `build-profile/` — run `pwd` first to confirm CWD) and verify the YAML frontmatter is intact: opens with `---`, has `name: osprey-build-deploy`, has a `description:` block, closes with `---`. Also verify the body still references the templates and modules (grep for `templates/core/` and for at least one file under `references/`). Report PASS or FAIL. Under 100 words."

**Agent 3 — Discoverability sanity check**
> "From the parent of `build-profile/` (run `pwd` first to confirm CWD), verify what Claude Code will see when a user runs `/osprey-build-deploy` inside the profile repo. Check: `build-profile/.claude/skills/osprey-build-deploy/SKILL.md` exists and is readable, and no broken symlinks under `build-profile/.claude/skills/`. A `.bak.*` directory is EXPECTED and NOT a failure if this was a re-run of the install (backups are created by design); only flag a `.bak.*` directory if it is empty or has a symlink loop. Use `ls -la` and `head` on the SKILL.md. Report PASS or FAIL with one-line reasoning. Under 80 words."

If all three return PASS, continue to the final message. If any FAIL, tell the user exactly what failed and offer to retry the install before proceeding.

#### After generating all files

Tell the user:
> "Your build profile is ready in the `build-profile/` directory. Here's what I created: [list files]. The deploy skill is now installed at `build-profile/.claude/skills/osprey-build-deploy/` — that's Phase 2.
>
> **Phase 1 (now done): Build the assistant.** Run:
> ```
> osprey build <project-name> build-profile/profile.yml
> ```
> Then `cd <project-name> && claude` to start using your agent.
>
> **Phase 2 (when you're ready to deploy): Ship it.** Open Claude Code in the `build-profile/` repo and run `/osprey-build-deploy` — it walks you through CI/CD setup, the deploy server, and ongoing release operations. The deploy skill lives with the repo, so it's always available wherever you cloned this profile."

If mock mode:
> "Everything is set to simulated/mock mode right now, which is perfect for trying things out. When you're ready to connect to real hardware, edit `profile.yml` — change `control_system.type` from `mock` to `epics`, add your gateway addresses, and run `osprey build` again."

Then proceed to Phase 9a.

---

### Phase 9a — Quick Feedback (inline)

Right after the Phase 8 closing message, ask one quick question:

> "One last thing before you go — how did this process feel?"

AskUserQuestion with options:
- Quick and painless — got what I needed
- Fine but a bit long
- Confusing in places — I had to guess at some answers
- Too many questions
- Skip this

If they answer (anything except Skip):
> "Thanks for that. When you've had a chance to build and test your project, you can come back anytime and say `/build-interview feedback` — I'll help you send quick notes to the OSPREY team. No obligation. Good luck with your project!"

If they skip:
> "No problem! If you ever want to send feedback later, just say `/build-interview feedback`. Good luck with your project!"

**That's it for 9a.** No email, no follow-up. They're free to go.

---

### Phase 9b — Standalone Feedback (re-entry)

This phase runs when the user invokes `/build-interview feedback` or says "I want to give feedback on the build interview." It may be a new session with no prior interview context.

> "Welcome back! I'll ask a couple of quick questions, then draft a short email you can send to the OSPREY team. You can skip any question."

**Q1** (AskUserQuestion): "How is your OSPREY project working out?"
- Working well — the generated profile was a good starting point
- Needed some tweaking but got there
- Had trouble getting it running — needed significant changes
- Haven't tried it yet — just wanted to share thoughts on the interview

**Q2** (AskUserQuestion): "What would have made the biggest difference?"
- Better explanations of what each option means
- Fewer questions — get to a working project faster
- More questions — I wish it had asked about something it missed
- Better defaults — too many things I had to configure manually
- A way to import my PV list from a file instead of typing them

If they chose options that suggest specific feedback (like "More questions" or typed a custom response), follow up briefly: "What was missing?" or "What would you change?"

**Q3** (only if Q1 was one of the first three options — meaning they actually used the project): "Anything else the OSPREY team should know? One sentence is plenty, or just skip."

#### Compose the email

Assemble a plain-text email from their answers. **Omit any line entirely where the data is unavailable or was skipped** — don't leave blank fields.

```
Subject: OSPREY Build Interview Feedback — [system_name] ([facility])

Hi,

Feedback from a build interview session.

System: [system_name or "not specified"]
Facility: [facility or "not specified"]
Date: [YYYY-MM-DD]

Interview experience: [9a answer — OMIT THIS LINE if no 9a data exists]
Project status: [Q1 answer — OMIT if skipped]
Suggestion: [Q2 answer plus any free-form — OMIT if skipped]
Additional notes: [Q3 answer — OMIT if skipped or not asked]

---
Sent via OSPREY build interview feedback
```

If the user skipped all questions, don't compose an email — just say "No problem, thanks for coming back!" and end.

#### Present and send

Show the draft in a code block so the user can review it:

> "Here's a draft based on your feedback. Want me to open it in your email client so you can review and send it?"

AskUserQuestion:
- Yes, open it in my email client
- No thanks — I'll copy it myself
- Let me change something first

**If "Yes, open it":**

Detect the platform and open a mailto: link:

```bash
# Detect platform
uname -s
# Darwin → open "mailto:..."
# Linux  → xdg-open "mailto:..."
```

Construct the mailto URL with the email address `thellert@lbl.gov`, URL-encoded subject and body. For URL encoding: spaces become `%20`, newlines become `%0A`, `&` becomes `%26`, `=` becomes `%3D`, `#` becomes `%23`. The format is `mailto:thellert@lbl.gov?subject=...&body=...`.

If the encoded URL exceeds 2000 characters (possible with long free-form answers), save the body to `build-profile/feedback-draft.txt` instead and tell the user to paste it into an email to thellert@lbl.gov.

If the `open`/`xdg-open` command fails (headless server, SSH session), display the mailto: URL as text and suggest copy-paste.

**If "Let me change something":** Ask what they want to change, update the draft, re-present.

---

## Guidelines

- **Be conversational, not interrogative.** Explain why each question matters. "I'm asking about limits because the AI assistant needs to know what values are safe to set — this prevents accidental damage to equipment."
- **Provide defaults for everything.** If they say "I'm not sure", pick the safe/simple option and move on. Note it in the README so they can revisit.
- **Don't overwhelm.** If they seem unsure about multiple things, suggest: "Let's start with a minimal setup — just reading your main signals. You can always add more features later."
- **Summarize after each phase.** A quick "OK so far I have: ..." keeps them oriented and catches misunderstandings early.
- **The devil's advocate is mandatory.** Always run it. It catches real issues.
- **Generate practical output.** The build profile should work as-is for mock mode. The user should be able to follow the README and have a working agent in 5 minutes.
- **Migration = confirmation.** When migration_context exists, phrase questions as confirmations: "I found X — keep it?" not "What X do you want?" The user already made these decisions once; respect that.
- **Explain what's obsolete.** For migration, always briefly explain why old LangGraph code is not needed: "The new architecture uses Claude Code directly as the orchestrator, so the old graph definitions aren't needed anymore."
- **Feedback is never mandatory.** Phase 9a is always offered but never insisted upon. If the user declines or says "skip", respect it immediately. No guilt trips.
