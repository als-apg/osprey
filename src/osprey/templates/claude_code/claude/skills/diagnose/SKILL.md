---
name: diagnose
description: >
  Investigate OSPREY tool, agent, or MCP server failures. Use when a tool call
  returned an error, a subagent failed, or the MCP infrastructure is not
  responding. NOT for diagnosing accelerator or machine operational issues.
summary: Investigate OSPREY infrastructure and agent failures
---

# Diagnose — OSPREY Infrastructure Failure Investigation

Systematically investigate an OSPREY infrastructure failure, gather evidence from all available sources, and produce a structured incident report for developer handoff.

Follow these three phases in order. Do not skip phases.

---

## Phase 1 — Scope

Determine the investigation target. Use `$ARGUMENTS` if provided, otherwise infer from conversation history.

Establish:
- **What failed** — which tool, agent, or workflow produced unexpected results
- **When** — approximate time or position in the conversation
- **Expected vs actual** — what should have happened vs what did happen

Do NOT ask the user questions in this phase. Use conversation context to fill in the scope. If the scope is ambiguous, state your best understanding and proceed — the report will make any ambiguity visible.

---

## Phase 2 — Investigate

Gather evidence from ALL sources below. **Never stop after one source.** Empty results from one source are findings, not dead ends.

### 2a. Conversation Evidence

Review the conversation history for:
- Tool calls that returned errors or unexpected results
- Error envelopes (`error_code`, `error_class`, `message` fields)
- Sequences of calls that suggest a pattern (e.g., repeated retries, cascading failures)
- Subagent delegations and their outcomes

Record each piece of evidence with its source (e.g., "conversation turn N" or "tool call to X").

### 2b. Session Log

Query the session log in this order:

1. **Agent overview**: `session_log(list_agents=True)` — get the full picture of all agents and their tool/error counts
2. **Targeted queries** based on what the overview reveals:
   - If a specific agent had errors: `session_log(agent_id="<id>")`
   - If errors are suspected: `session_log(errors_only=True, last_n=20)`
   - If timing matters: `session_log(since="<timestamp>")`

**If session_log returns empty results**, this IS evidence. Record it as a finding and note the possible causes:
- No OSPREY MCP tool calls were made (only built-in tools like Read/Write/Bash were used)
- The transcript hasn't been written yet (session still in progress, no flush)
- The agent used only non-MCP tools that don't appear in the session log
- The MCP server wasn't connected or wasn't running

Do NOT treat empty session_log results as "nothing to see." Document what was queried, that it was empty, and what that absence implies.

### 2c. Workspace State

1. Call `session_summary()` to get the current workspace inventory (artifacts, agents)
2. Call `data_list()` to see all data entries

Cross-reference against session_log findings:
- Are there data entries that should exist but don't? (tool succeeded but no data saved)
- Are there data entries from unexpected sources? (different agent, different tool)
- Does the artifact count match expected outputs?

### 2d. Direct Evidence

Extract concrete details from conversation-visible tool responses:
- Exact error messages and error codes
- Parameter values that were passed to failing tools
- Response payloads that were malformed or unexpected
- Timing information (if timestamps are visible in responses)

---

## Phase 3 — Report

Produce a structured incident report as a **self-contained HTML artifact** and save it to the gallery.

### 3a. Generate the HTML report

Build a single HTML string with inline CSS (no external dependencies). Use this structure:

```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Diagnostic Report</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,system-ui,sans-serif; background:#0f172a; color:#e2e8f0; padding:24px; line-height:1.5; }
  .header { border-bottom:2px solid #ef4444; padding-bottom:12px; margin-bottom:24px; }
  .header h1 { font-size:1.4rem; color:#f8fafc; }
  .header .timestamp { font-size:0.85rem; color:#94a3b8; margin-top:4px; }
  .severity { display:inline-block; padding:2px 10px; border-radius:4px; font-weight:600; font-size:0.8rem; text-transform:uppercase; }
  .severity-critical { background:#991b1b; color:#fecaca; }
  .severity-warning  { background:#92400e; color:#fde68a; }
  .severity-info     { background:#1e3a5f; color:#93c5fd; }
  section { margin-bottom:24px; }
  section h2 { font-size:1.1rem; color:#f8fafc; border-bottom:1px solid #334155; padding-bottom:6px; margin-bottom:12px; }
  .evidence-item { background:#1e293b; border-radius:6px; padding:12px; margin-bottom:8px; border-left:3px solid #3b82f6; }
  .evidence-source { font-size:0.75rem; color:#60a5fa; text-transform:uppercase; letter-spacing:0.05em; }
  .evidence-finding { margin-top:4px; }
  .evidence-significance { margin-top:4px; font-size:0.85rem; color:#94a3b8; font-style:italic; }
  .cause-item { background:#1e293b; border-radius:6px; padding:12px; margin-bottom:8px; }
  .cause-rank { display:inline-block; width:24px; height:24px; border-radius:50%; background:#3b82f6; color:#fff; text-align:center; line-height:24px; font-size:0.8rem; font-weight:600; margin-right:8px; }
  .gap-item { background:#1e293b; border-radius:6px; padding:10px; margin-bottom:6px; border-left:3px solid #f59e0b; }
  .next-step { background:#1e293b; border-radius:6px; padding:10px; margin-bottom:6px; border-left:3px solid #22c55e; }
  .timeline-event { display:flex; gap:12px; margin-bottom:8px; }
  .timeline-time { flex:0 0 100px; font-size:0.8rem; color:#60a5fa; text-align:right; padding-top:2px; }
  .timeline-dot { flex:0 0 12px; position:relative; }
  .timeline-dot::before { content:''; display:block; width:10px; height:10px; border-radius:50%; background:#3b82f6; margin-top:5px; }
  .timeline-dot::after { content:''; position:absolute; top:18px; left:4px; width:2px; height:calc(100% + 4px); background:#334155; }
  .timeline-event:last-child .timeline-dot::after { display:none; }
  .timeline-desc { flex:1; }
</style></head><body>
<!-- Fill in sections dynamically based on investigation findings -->
</body></html>
```

Populate the HTML with:

1. **Header** — report title, timestamp, overall severity badge (CRITICAL / WARNING / INFO)
2. **Failure Summary** — 1-3 sentences. Include the error class (Connection, Permission, Validation, Data, Execution, Internal) if one applies.
3. **Evidence** — one `.evidence-item` per finding, each with source badge, finding text, and significance.
4. **Timeline** *(only if multiple events over time)* — chronological `.timeline-event` entries.
5. **What Was NOT Found** — one `.gap-item` per evidence source that returned empty/inconclusive. Each must say what was queried, that it was empty, and what the absence could mean.
6. **Possible Causes** — rank-ordered `.cause-item` entries. Each must cite supporting evidence numbers. Do NOT speculate beyond what evidence supports.
7. **Suggested Next Steps** — one `.next-step` per handoff action for HUMANS (not the agent).

### 3b. Save to the gallery

Call the `artifact_save` MCP tool:

```
artifact_save(
    title="Diagnostic Report — <brief failure description>",
    description="Infrastructure failure investigation: <1-line summary>",
    content=<the HTML string>,
    content_type="html",
    category="diagnostic_report"
)
```

Then call `artifact_focus(artifact_id=<id from response>)` to open the report in the gallery.

### 3c. Conversation summary

After saving, present a brief inline summary: the failure summary and suggested next steps. Don't duplicate the full report — point the user to the gallery artifact for details.

---

## Anti-Patterns — NEVER Do These

- **NEVER attempt to fix the problem.** This skill produces a report, not a resolution.
- **NEVER read source code.** You don't have access to OSPREY internals and shouldn't try.
- **NEVER speculate beyond evidence.** If you don't have evidence for a cause, don't list it.
- **NEVER stop after the first empty result.** Every source in Phase 2 must be queried.
- **NEVER use the word "debug."** This is diagnosis and evidence gathering, not debugging.
- **NEVER suggest actions for yourself.** Next steps are for humans.
- **NEVER retry failed tools** as part of the investigation. Observe what happened; don't try to reproduce it.
