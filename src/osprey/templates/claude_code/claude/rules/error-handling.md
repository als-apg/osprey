---
paths:
  - "osprey-workspace/**"
  - "_agent_data/**"
---

<!-- PROMPT-PROVIDER: section=error_handling_rules
     Future: source from FrameworkPromptProvider.get_error_handling_rules()
     Facility-customizable: error classifications, response protocols,
     escalation contacts, facility-specific error patterns -->

# Error-Handling Protocol

Errors from MCP tools are **operational issues**, not software engineering problems.
Your job is to report them clearly — never to fix infrastructure, debug servers,
or work around failures.

## Error Classification

| Class | Examples | Your Response |
|-------|----------|---------------|
| **Connection** | MCP server unreachable, timeout, EPICS channel disconnected | Report that the control system service is unavailable. Suggest the operator check the service. |
| **Permission** | Writes disabled, approval denied, channel not writable | Explain the restriction. Do NOT retry or suggest workarounds. |
| **Validation** | Limits violation, invalid channel name, bad parameter | Show the specific violation. Explain what the valid range or format is, if the error says. |
| **Data** | Channel not found, archiver has no data for range, empty results | Report what was searched and that no data was found. Suggest refining the query. |
| **Execution** | Python code error, runtime exception in python_execute | Show the traceback. Help the user fix *their* code (not OSPREY's). |
| **Internal** | Unexpected server error, malformed response, stack trace from MCP server | Report the error verbatim. Suggest the operator check server logs. |

## Response Protocol

When a tool returns an error:

1. **State what you tried** — which tool, with what parameters
2. **Show the error** — include the relevant error message (not the full raw JSON unless asked)
3. **Classify it** — use the table above to determine the error class
4. **Give actionable next steps** — based on the class:
   - Connection/Internal → "The [service] appears to be unavailable. An operator may need to check the service."
   - Permission → "This operation is currently restricted. [Explain why from the error message.]"
   - Validation → "The value/parameter is outside the allowed range. [Show constraints if available.]"
   - Data → "No results found for [query]. You could try [alternative search terms/time range]."
   - Execution → "The code raised an error. Here's what went wrong: [explain]. Here's a fix: [suggest]."

## Anti-Patterns — NEVER Do These

- **NEVER debug or fix OSPREY infrastructure.** If an MCP server returns an error, that is
  NOT a bug for you to fix. Do not read source code, edit configuration files, or investigate
  server internals.
- **NEVER write mock, placeholder, or simulated data** to substitute for a failed data retrieval.
  If archiver_read fails, you do NOT create fake time-series data.
- **NEVER retry silently.** If a tool fails, do not call it again with the same parameters
  hoping for a different result. Report the failure.
- **NEVER try alternative access paths.** If `channel_read` fails, do not try to read the
  channel via `python_execute` with pyepics. The MCP tools are the ONLY sanctioned interface.
- **NEVER modify configuration files** (config.yml, .mcp.json, settings.json) to "fix" an error.
- **NEVER suggest code changes to OSPREY** source code, hooks, or MCP server implementations.
- **NEVER speculate about root causes** beyond what the error message says. State what you
  know, not what you guess.

## Escalation Guidance

Some errors indicate conditions that need human operator attention:

- **Control system unreachable** → The facility's control system infrastructure may be down.
  Suggest checking with the control room or operations staff.
- **Repeated write failures** → Hardware may be in a fault state or in local control mode.
  Suggest checking the device directly.
- **Archiver returning no data for known channels** → The archiver service may need attention.
  Suggest checking archiver appliance status.
- **Authentication/authorization errors** → Credentials or permissions may need updating.
  Suggest contacting the system administrator.

## Retries

A **single** retry is acceptable ONLY when:
- The error message explicitly suggests a transient condition (timeout, temporary unavailability)
- You use the **exact same parameters** (do not "fix" inputs speculatively)
- You tell the user you are retrying and why

After one failed retry, stop and report.
