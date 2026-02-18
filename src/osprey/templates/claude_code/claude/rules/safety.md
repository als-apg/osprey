---
paths:
  - "osprey-workspace/**"
  - "_agent_data/**"
---

<!-- PROMPT-PROVIDER: section=safety_rules_file
     Future: source from FrameworkPromptProvider.get_safety_rules()
     Facility-customizable: additional safety rules, restricted paths,
     facility-specific audit/compliance requirements -->
# Safety Rules

- You can ONLY interact with the control system through MCP tools
- Never attempt to read project configuration, source code, or system files
- If an MCP tool returns an error, follow the error-handling protocol
  in `.claude/rules/error-handling.md` — report clearly, never work around it
- Audit logs in osprey-workspace/audit/ must never be deleted or modified
- Files in _agent_data/ are for your reference (example scripts, memory)

## Data Integrity

All data presented to users informs real operational decisions about physical
equipment. Never fabricate, simulate, or generate synthetic data. Full rules
and anti-patterns are in `.claude/rules/error-handling.md`.
