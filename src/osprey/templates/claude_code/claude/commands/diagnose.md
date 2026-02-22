---
summary: Investigate operational failures
description: Systematic investigation of operational failures with structured incident reports
---

# Diagnose — Operational Failure Investigation

Systematically investigate an operational failure, gather evidence from all available sources, and produce a structured incident report for developer handoff.

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

Produce a structured incident report in markdown. Use this exact structure:

### Failure Summary

1-3 sentences describing the failure. Include the error class from the error-handling protocol taxonomy (Connection, Permission, Validation, Data, Execution, Internal) if one applies. If the failure doesn't map cleanly to one class, say so.

### Evidence

Numbered list. Each item has:
- **Source**: Where this evidence came from (conversation, session_log, workspace, direct)
- **Finding**: What was observed
- **Significance**: What this tells us about the failure

### Timeline

*(Include only if the failure involved multiple events over time.)*

Chronological sequence of relevant events with timestamps where available.

### What Was NOT Found

List evidence sources that returned empty or inconclusive results. For each:
- What was queried
- That it returned empty/no results
- What that absence could mean (enumerate possible causes)

This section is mandatory. Every investigation has gaps — documenting them is as important as documenting findings.

### Possible Causes

Rank-ordered list of possible causes. Each must be supported by at least one piece of evidence from the Evidence section. Format:

1. **[Most likely cause]** — supported by Evidence #N, #M. [Brief explanation.]
2. **[Less likely cause]** — supported by Evidence #N. [Brief explanation.]

Do NOT speculate beyond what the evidence supports. If only one cause is supported, list only one.

### Suggested Next Steps

Handoff items for the operator or developer. These are actions for HUMANS, not for the agent. Examples:
- "Check whether the MCP server process is running"
- "Verify the archiver appliance is accessible from this host"
- "Review the OSPREY server logs for errors around [timestamp]"
- "Open a developer session with source code access to investigate [specific component]"

---

## Anti-Patterns — NEVER Do These

- **NEVER attempt to fix the problem.** This command produces a report, not a resolution.
- **NEVER read source code.** You don't have access to OSPREY internals and shouldn't try.
- **NEVER speculate beyond evidence.** If you don't have evidence for a cause, don't list it.
- **NEVER stop after the first empty result.** Every source in Phase 2 must be queried.
- **NEVER use the word "debug."** This is diagnosis and evidence gathering, not debugging.
- **NEVER suggest actions for yourself.** Next steps are for humans.
- **NEVER retry failed tools** as part of the investigation. Observe what happened; don't try to reproduce it.
