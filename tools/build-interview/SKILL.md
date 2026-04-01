---
name: build-interview
description: Interactive interview to create a custom OSPREY build profile for a new accelerator, detector, or beamline application. Use this skill when someone says "interview me", "create a build profile", "set up my agent", "configure my detector", "onboard me", or needs to create an OSPREY project tailored to their specific control system. Also use when onboarding a new colleague or when anyone needs help figuring out what their OSPREY agent should look like.
---

# OSPREY Build Profile Interview

You are conducting a friendly, structured interview to gather everything needed to create a custom OSPREY agent project. The person you are interviewing may not be deeply familiar with OSPREY's internals. Explain concepts in plain language and avoid framework jargon. Your goal is to produce a **build profile** — a set of files that gives them a working agent project (or an excellent first stepping stone) with minimal effort.

## What You Are Building

By the end of this interview you will generate a `build-profile/` directory containing:

1. **README.md** — Plain-language summary of what was configured and step-by-step setup instructions
2. **config.yml** — A complete OSPREY configuration file tailored to their use case
3. **channels.json** — A channel database populated with their PVs (if they provided PV details)
4. **channel_limits.json** — Safety limits for writable channels (if write access was requested)

Read `references/osprey-config-reference.md` before generating any config files — it contains the exact YAML structure, valid field values, and channel database JSON schema.

---

## Interview Flow

Work through these phases in order. Use **AskUserQuestion** for structured choices. After each phase, give a brief recap of what you captured ("So far I have: ...") before moving to the next phase. If the person seems uncertain at any point, suggest the safe/simple default and reassure them they can change things later.

Keep the tone conversational — this is a chat, not a form. Explain *why* each question matters so the person understands what they are deciding.

---

### Phase 1 — Welcome & Context

Start with a short welcome. Something like:

> "Hi! I'm going to ask you some questions about the system you work with so I can set up an AI assistant tailored to your needs. It should take about 10 minutes. You can always say 'I'm not sure' and I'll pick a sensible default. Let's get started."

Then ask these two questions together (single AskUserQuestion call):

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

After they answer, follow up conversationally to capture:
- **System name** — a short name for the project (e.g., "xray-detector", "insertion-device-monitor"). This becomes the OSPREY project name, so suggest something lowercase with hyphens, no spaces.
- **One-sentence description** — what does their system do in plain English?
- **Facility** — which lab or facility are they at? (ALS, LCLS, NSLS-II, BESSY, etc.)

---

### Phase 2 — Signals & Channels

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

#### If PVs are available

Ask them to list their PVs. Accept any format — one per line, comma-separated, or pasted from a spreadsheet. Then for each PV (or logical group), collect:

| Field | Question | Required? |
|-------|----------|-----------|
| PV name | The EPICS PV address (e.g., `SR:BPM:X:01`) | Yes |
| Description | What does this signal represent? | Yes |
| Units | What are the units? (mA, mm, degC, etc.) | Nice to have |
| Typical range | What values are normal? | Nice to have |
| Read/Write | Do you just read this, or also write to it? | Yes |

Group related PVs together (e.g., "these 5 are all BPM positions"). If they have many similar PVs (like BPM01 through BPM20), note the pattern and use a template entry in the channel database.

#### If PVs are NOT available yet

Collect conceptual information instead:
- What kinds of signals? (temperatures, positions, currents, pressures, counts, voltages, etc.)
- Roughly how many? (handful = <10, moderate = 10-50, many = 50+)
- What naming convention do their PVs follow? (e.g., `SYS:SUBSYS:DEV:SIGNAL`)
- Any example PV names they remember?

Note: generate a skeleton channel database with placeholder entries and clear comments showing where to fill in real PV names later.

#### If archiver is needed

Ask for the archiver URL if they know it (e.g., `https://archiver.facility.gov:port`). If they don't know it, note that mock archiver will be configured as a placeholder.

---

### Phase 3 — Safety & Write Access

**Q1 — Access level**: "Will the AI assistant need to change or write any values (like adjusting a setpoint), or is it purely for reading and monitoring?"
- Read-only — just monitoring and analysis (safest, recommended to start)
- Read and write — needs to adjust settings, with safety checks
- Mostly read, maybe write occasionally

Explain that OSPREY has multiple safety layers for writes:
- Every write requires explicit human approval (you click "approve" before anything changes)
- Optional limits checking (prevents writes outside safe ranges)
- Optional readback verification (confirms the value actually changed)

If write access is needed, ask:
- Which PVs specifically need write access?
- Are there hard limits that should never be exceeded? (collect per-PV min/max)
- Should writes be verified with a readback? (Recommended: yes)

---

### Phase 4 — Infrastructure & Provider

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

If they chose "Real EPICS connection", follow up for gateway details:
- Read-only gateway address and port
- Write gateway address and port (if write access is needed)
- Whether they use name servers (most people say no, if unsure default to no)

---

### Phase 5 — Additional Features

**Q1** (multi-select): "Would any of these extra features be useful for you?"
- Electronic logbook search — search past shift logs and operator notes using AI
- Channel finder — discover PV names by describing what you need in plain English (useful if you have hundreds of PVs)
- Web dashboard — browser-based terminal with built-in panels

Explain each briefly. For a simple detector application, suggest skipping logbook and channel finder — they add complexity. Data visualization is always included by default.

For each selected feature, ask relevant follow-ups (e.g., logbook source, number of PVs for channel finder).

---

### Phase 6 — Custom Web Panel Design

OSPREY's web terminal can host custom panels as tabs alongside the main terminal. A custom panel is a great way to give the user a dedicated monitoring view for their detector or experiment. Ask whether they'd find this useful.

**Q1**: "OSPREY has a web dashboard with a terminal and configurable panels. Would you like a dedicated panel tab for your detector workflow — for example, a live status display, trend plots, or an alarm overview?"
- Yes, I'd like a custom monitoring panel
- Maybe later — let's keep it simple for now
- I already have a monitoring tool I'd like to embed (e.g., Grafana, custom web app)

#### If they want a custom panel

Walk through these questions conversationally. The goal is to capture enough detail to generate either a panel specification (for future development) or a config entry (if they have an existing service).

**Display & Layout** — Ask: "Picture your ideal monitoring screen. What would you see at a glance?"

Offer these common building blocks as inspiration (multi-select or free-form):

| Component | Description | Example |
|-----------|-------------|---------|
| Live value readouts | Current PV values updating in real time | "Beam current: 503.2 mA" |
| Status indicators | Color-coded health/state badges | Green = OK, Yellow = Warning, Red = Alarm |
| Trend plots | Time-series line charts | Last 1 hour of temperature readings |
| Gauges / meters | Analog-style dial or bar indicators | Pressure gauge showing 0–100% |
| Alarm table | List of active alarms or out-of-range values | "Detector temp HIGH: 45.2°C (limit: 40°C)" |
| Data table | Tabular view of multiple PVs | All BPM positions in a sortable table |
| Summary cards | Key metrics in large-font cards | "Uptime: 23h", "Events: 1.2M" |

Follow up to understand:
- **Which PVs** should appear on the panel? (Cross-reference with the PV list from Phase 2. If they mention PVs not already captured, add them.)
- **Update frequency** — How often should values refresh? (every second, every 5 seconds, on-demand?)
- **Alarm thresholds** — Are there values that should turn red/yellow? What are the limits?
- **Grouping** — Should signals be organized into sections? (e.g., "Temperatures" section, "Beam" section, "Vacuum" section)

**Q2**: "What should the panel be called in the tab bar?"
- Let them suggest a short label (e.g., "DETECTOR", "XRD MONITOR", "LIVE STATUS")
- Default to their system name in uppercase if they're unsure

#### If they have an existing monitoring tool

Ask:
- What's the URL? (e.g., `http://grafana.facility.gov:3000`)
- Does it have a health check endpoint? (e.g., `/api/health` for Grafana)
- Does it need authentication or special headers?
- What label should appear in the tab bar?

This case is simple — just a config entry, no panel specification needed.

#### Panel Specification Output

If they want a custom panel (not an existing tool), generate a `build-profile/panel-spec.md` document that captures:

```markdown
# Custom Panel Specification: [PANEL LABEL]

## Purpose
[One-sentence description of what the panel shows]

## Layout
[List of components with their data sources]

### Section: [Group Name]
| Component | PV / Data Source | Display | Alarm Thresholds |
|-----------|-----------------|---------|-----------------|
| Live value | PV:NAME:HERE | "Label: {value} {units}" | Yellow: >X, Red: >Y |
| Trend plot | PV:NAME:HERE | Last 1 hour, 1s resolution | — |
| Status LED | PV:NAME:HERE | Green if <X, Red if >Y | — |

## Behavior
- Update frequency: [X seconds]
- Theme: Syncs with OSPREY web terminal (dark/light)
- Responsive: [Yes/No]

## Technical Notes
- Data source: OSPREY control system MCP server (reads PVs via connector)
- Panel service: FastAPI app on port [XXXX]
- Health endpoint: /health
```

Also add the panel to the config.yml in the build profile:

```yaml
web:
  panels:
    detector-monitor:  # or whatever ID they chose
      label: "DETECTOR"
      url: "http://127.0.0.1:8095"
      health_endpoint: "/health"
```

Note in the README that the panel specification describes what to build but the panel service itself would need to be developed separately (or could be a follow-up project).

---

### Phase 7 — Devil's Advocate Review

> **Note:** Phase numbering shifted — this was Phase 6 before the panel section was added.

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

PANEL DESIGN GAPS:
- Custom panel requested but no PVs specified for it
- Panel components reference PVs not in the channel list
- Alarm thresholds specified for panel but no corresponding channel limits
- Panel update frequency seems too fast for the number of PVs (performance concern)
- Panel requested but web dashboard not enabled in features
- Existing monitoring tool URL provided but no health endpoint specified

SCOPE CONCERNS:
- Scope seems too narrow for what they described wanting to do
- Scope seems too broad for a first project — might be overwhelming
- Features selected that add complexity without clear benefit for their use case

For each issue found, categorize it as:
- CRITICAL: Must resolve before generating profile (safety issues, blocking gaps)
- RECOMMENDED: Should resolve for a better profile (incomplete info, likely oversights)
- OPTIONAL: Nice to address but fine to skip (minor improvements)

Return your findings as a structured list with the category, the issue, and a suggested follow-up question to ask the user. Be specific — reference actual PV names, features, and answers from the interview.

If you find no issues, say so explicitly — don't invent problems.
```

After the review agent returns:

1. If there are **CRITICAL** findings: present them to the user using AskUserQuestion and resolve every one before proceeding.
2. If there are **RECOMMENDED** findings: present them and let the user decide which to address. Use AskUserQuestion with options like "Good catch, let me clarify" vs "That's fine, skip it."
3. **OPTIONAL** findings: mention them briefly but don't block on them.

After addressing findings, if any answers changed substantially, consider running the review once more. One pass is usually enough.

---

### Phase 8 — Generate Build Profile

Read `references/osprey-config-reference.md` now for the exact config.yml structure and channel database schema.

Create a `build-profile/` directory with the following files:

#### 1. `build-profile/README.md`

Write a friendly summary including:
- Project name and one-line description
- What features are enabled and why
- What PVs are configured (or placeholder note)
- Step-by-step setup instructions:
  1. Install OSPREY (`pip install osprey` or `uv pip install osprey`)
  2. Create project: `osprey init --name <name> --template control_assistant --provider <provider> --model <model>`
  3. Replace `config.yml` with the one from this build profile
  4. Copy `channels.json` to `data/channel_databases/in_context.json` (if provided)
  5. Copy `channel_limits.json` to `data/` (if provided)
  6. Regenerate Claude Code artifacts: `osprey claude regen`
  7. Test: `osprey health` to verify everything works
  8. Start using: `cd <project-name> && claude` or `osprey web`
- How to switch from mock to real EPICS when ready (change `control_system.type` and fill in gateway addresses)
- Where to get help

#### 2. `build-profile/config.yml`

A complete, valid OSPREY config.yml. Use the control_assistant template as the base but:
- Disable features they don't need (ARIEL, channel finder, etc.) via the `claude_code.servers` and `claude_code.agents` sections
- Set the control system type (mock or epics)
- Set writes_enabled based on their access level
- Configure limits and verification if write access is enabled
- Set the AI provider and model
- Include helpful comments explaining each section

#### 3. `build-profile/channels.json` (if PVs were collected)

Use the in_context flat format:
```json
[
  {
    "template": false,
    "channel": "DescriptiveName_ReadBack",
    "address": "ACTUAL:PV:NAME:HERE",
    "description": "Human-readable description of what this signal represents"
  }
]
```

For groups of similar PVs (BPM01-BPM20), use template entries:
```json
{
  "template": true,
  "base_name": "BPM_Position",
  "instances": [1, 20],
  "sub_channels": ["X", "Y"],
  "description": "Beam position monitors",
  "address_pattern": "SR:BPM:{instance:02d}:POS:{suffix}",
  "channel_descriptions": {
    "X": "horizontal position at BPM {instance}",
    "Y": "vertical position at BPM {instance}"
  }
}
```

If PVs were NOT collected, generate a skeleton with clear placeholder comments.

#### 4. `build-profile/channel_limits.json` (if write access)

```json
{
  "channels": {
    "ACTUAL:PV:NAME": {
      "low_limit": 0.0,
      "high_limit": 100.0,
      "units": "mA",
      "description": "Safe operating range for this channel"
    }
  }
}
```

#### 5. `build-profile/panel-spec.md` (if custom panel was designed)

Generate this only if the user requested a custom monitoring panel in Phase 6. Use the panel specification template from Phase 6. This document serves as a blueprint for building the panel — include all collected details about components, PVs, alarm thresholds, update frequency, and layout.

Also update the `config.yml` to include the panel entry under `web.panels` and set `web.panels` for any relevant built-in panels they enabled.

If they provided an existing monitoring tool URL instead, just add the config entry — no spec document needed.

#### After generating all files

Tell the user:
> "Your build profile is ready in the `build-profile/` directory. Here's what I created: [list files]. Follow the README.md for setup instructions. The whole setup should take about 5 minutes."

If they started with mock mode, remind them:
> "Everything is set to simulated/mock mode right now, which is perfect for trying things out. When you're ready to connect to real hardware, just change `control_system.type` from `mock` to `epics` and fill in your gateway addresses in config.yml."

---

## Guidelines

- **Be conversational, not interrogative.** Explain why each question matters. "I'm asking about limits because the AI assistant needs to know what values are safe to set — this prevents accidental damage to equipment."
- **Provide defaults for everything.** If they say "I'm not sure", pick the safe/simple option and move on. Note it in the README so they can revisit.
- **Don't overwhelm.** If they seem unsure about multiple things, suggest: "Let's start with a minimal setup — just reading your main signals. You can always add more features later."
- **Summarize after each phase.** A quick "OK so far I have: ..." keeps them oriented and catches misunderstandings early.
- **The devil's advocate is mandatory.** Always run it. It catches real issues.
- **Generate practical output.** The build profile should work as-is for mock mode. The user should be able to follow the README and have a working agent in 5 minutes.
