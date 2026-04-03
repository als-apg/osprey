---
name: session-report
description: Generate a polished HTML session report based on the current session's work
summary: Generates polished HTML session reports from conversation work
disable-model-invocation: true
---

# Session Report — Intent-Based Report Generator

Generate a polished, self-contained HTML report based on this session's work. The workflow adapts to what the operator needs rather than forcing a fixed section structure.

Follow these four phases in order. Do not skip phases.

---

## Phase 1 — Intent

Ask the operator what kind of report they need. Present ONE `AskUserQuestion` with a **single-select** of report intents:

| Intent | Description |
|--------|-------------|
| **Session Log** | Chronological record of what happened — channels read, values observed, tools used. Minimal interpretation. |
| **Analysis Report** | Technical deep-dive with charts, data tables, and narrative observations about patterns and anomalies. |
| **Executive Briefing** | High-level summary for management — KPIs, status cards, key observations in non-technical language. |
| *(Other)* | Operator describes a custom report structure. |

Do NOT ask follow-up questions after this unless the operator response is ambigious.

---

## Phase 2 — Inventory & Structure

### 2a. Call `session_summary`

Use the `session_summary` MCP tool to get a compact inventory of all data context entries and artifacts in the workspace. This tells you exactly what data is available.

### 2b. Draft dynamic structure

Based on the operator's intent and the inventory, draft a report structure. There is no fixed section list — the structure should be driven by what data exists and what the intent calls for.

**Guidance (not constraints) on block types that work well:**

- **KPI row** — Good for any intent when you have countable metrics (channels read, values observed, artifacts created)
- **Data table** — Natural for Session Log intent; useful whenever channel values were read
- **Chart.js line/scatter** — Natural for Analysis Report when timeseries data exists. Use `archiver_downsample` to get chart-ready data (never embed raw timeseries).
- **Narrative paragraphs** — Observations about what was seen. Essential for Analysis Report and Executive Briefing.
- **Card grid** — Good for Executive Briefing status overview; good for linking to artifacts
- **CSS Timeline** — Good for Session Log to show a simple chronological list of actions (timestamps + descriptions)
- **Mermaid event timeline** — Use when the session involved grouped or multi-phase event sequences. Mermaid's `timeline` diagram type renders events organized by time periods with grouping, making complex sequences easier to scan than a flat list. Prefer this over the CSS timeline when there are 5+ events or natural phase groupings.
- **Mermaid state diagram** — Use when the session revealed state transitions in a control system (e.g., beam states, interlock chains, operational modes). Mermaid's `stateDiagram-v2` renders state machines with transitions and guards. Only include when actual state changes were observed — never fabricate state models.
- **Collapsible details** — Good for Analysis Report when there's dense technical data to organize

**Keep it proportional**: A 10-minute session with 2 channel reads doesn't need 8 sections. A 2-hour investigation with archiver data, plots, and multiple analyses might warrant a rich structure. Match the report's complexity to the session's complexity.

### 2c. Gather chart data (if needed)

If the structure includes charts and the inventory shows timeseries data, call `archiver_downsample` for each relevant data context entry now. This gives you chart-ready payloads (labels + datasets) that fit inline.

---

## Phase 3 — Delegate

Spawn a **Task subagent** to generate the report. Pass it:

1. **Intent** — the operator's chosen intent
2. **Inventory** — the `session_summary` output
3. **Structure** — your drafted report structure
4. **Chart data** — any `archiver_downsample` results (if applicable)
5. **Conversation context** — key observations and findings from this session
6. **Safety rules** — the content safety rules below

The subagent should:
- Read the reference file at `.claude/skills/session-report/reference.md` for CSS/JS patterns
- Use workspace MCP tools (`artifact_save`, `session_log`, `archiver_downsample`) as needed
- Generate a single self-contained HTML file and save it via `artifact_save`
- Block access to Bash, Read, Write, Edit (subagent uses only MCP tools)

### Content safety rules (MUST be included in subagent prompt)

> **CRITICAL — Observation-only reporting**
>
> This report documents what was observed during the session. It must NEVER generate recommendations, prescriptive advice, or action items.
>
> - ALLOWED: "Beam current was observed at 302.1 mA" / "A downward trend was noted between 14:00–15:00" / "The value exceeded the nominal range"
> - FORBIDDEN: "Investigate the corrector magnets" / "Consider adjusting the RF frequency" / "It is recommended to..."
>
> Exception: If the operator explicitly stated an action item during the session (e.g., "I need to check the vacuum pump tomorrow"), it may be included as an **attributed quote**: "Operator noted: 'Need to check vacuum pump tomorrow'".

### HTML requirements (pass to subagent)

- Single `<!DOCTYPE html>` file — all CSS in `<style>`, all JS before `</body>`
- Responsive sidebar TOC (desktop) / horizontal bar (mobile) using the reference pattern
- Light AND dark theme via `prefers-color-scheme` media query
- Pick a font pairing from the rotation table in the reference — never Inter/Roboto/Arial
- Use depth tiers: hero for header, elevated for KPIs, default for content, recessed for code/details
- Staggered `fadeUp` and `fadeScale` animations with `prefers-reduced-motion` respect
- Overflow protection globals (min-width: 0, overflow-wrap: break-word)
- Google Fonts via CDN with `display=swap`
- Chart.js via CDN (only if charts are included)
- Footer with generation timestamp and "Generated by OSPREY Session Report"
- Data integrity: every number, channel name, and timestamp must come from actual session data. Never fabricate values.

---

## Phase 4 — Register & Open

After the subagent returns:

1. **Focus the artifact** using `artifact_focus` so it appears in the gallery
2. **Open in browser** using Bash: `open <artifact_path>`
3. **Confirm to operator**: Report generated, list key sections, provide file path

---

## Anti-Patterns

Do NOT:
- Ask multiple rounds of questions — Phase 1 is ONE single-select, then generate
- Generate recommendations or prescriptive advice (see safety rules above)
- Over-engineer simple sessions — match report complexity to session complexity
- Fabricate data — every value must come from the actual session
- Embed raw timeseries data — always use `archiver_downsample` for charts
- Use emoji anywhere in the report
- Use Inter, Roboto, Arial, or plain system-ui as the primary font
- Skip reading the reference file — patterns drift from memory
- Use a generic dark theme — both light and dark must look intentional
- Omit the `prefers-reduced-motion` media query
- Hardcode colors instead of using CSS custom properties
- Skip the overflow protection globals
