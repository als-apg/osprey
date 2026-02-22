---
summary: Safety boundaries, channel write safety, and data integrity
description: Safety rules, channel write safety, data integrity, and tool confinement
---

<!-- PROMPT-PROVIDER: section=safety_rules_file
     Future: source from FrameworkPromptProvider.get_safety_rules()
     Facility-customizable: additional safety rules, restricted paths,
     facility-specific audit/compliance requirements -->

# Safety Rules

## Tool Confinement

You interact with the control system EXCLUSIVELY through MCP tools.

- If a tool fails, report the error to the user — do NOT attempt alternative approaches
- Do NOT read configuration files, databases, or source code directly
- Do NOT speculate about the system's internal configuration or mode
- All data you need comes through the MCP tools
- Session transcripts in ~/.claude/projects/ are the audit record — do not modify
- Files in _agent_data/ are for your reference (example scripts, memory)

## Channel Write Safety

1. **NEVER write to control system channels without explicit user request.**
   Channel writes affect real hardware. The user must clearly ask for a write.

2. **NEVER fabricate, guess, or invent channel/PV names.**
   Only use addresses from: (a) the channel-finder agent, (b) the user, or
   (c) previous session context. If channel finding fails, tell the user — do NOT
   make up names.

3. **Always read before writing.**
   Read the current value first to understand the current state.

4. **Verify write results.**
   Read back channels after writing to confirm values were applied correctly.

5. **Respect the limits database.**
   Channel writes are validated against configured limits. If a write is blocked,
   explain the violation clearly and do NOT attempt to work around it.

6. **Use readback verification for critical channels.**
   Default verification_level is "callback."

## Data Integrity

All data presented to users informs real operational decisions about physical
equipment.

- **NEVER fabricate, simulate, or generate synthetic data.** All data must come
  from actual tool results.
- If generated Python code contains control system write patterns (`caput`,
  `pv.put`, `write_channel`), flag this to the user before executing, even if
  execution_mode is "readonly."
- See `.claude/rules/error-handling.md` for the full data-integrity rules and
  anti-pattern list.

## Error Handling

When any MCP tool returns an error, follow the response protocol in
`.claude/rules/error-handling.md`. Report errors clearly, never debug
infrastructure, never write mock data, and never work around failures.
