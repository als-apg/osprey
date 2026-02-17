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
- If an MCP tool returns an error, report it to the user — do not work around it
- Audit logs in osprey-workspace/audit/ must never be deleted or modified
- Files in _agent_data/ are for your reference (example scripts, memory)

## Data Integrity — Never Fabricate Data

This is a safety-critical control system. All data presented to users informs
real operational decisions about physical equipment.

- **NEVER generate simulated, synthetic, placeholder, or "demonstration" data.**
  All numeric values, time series, and measurements must come from actual tool results.
- **NEVER create plots or visualizations with fabricated data.**
  If the data source is unavailable, report the error instead of plotting fake data.
- **NEVER fill in "example" or "typical" values** when real data retrieval fails.
  There is no safe default for control system measurements.
- **If data is unavailable, report it as an error.** Explain what went wrong
  (tool error, missing channel, archiver timeout, etc.) and suggest resolution.
- **Distinguish clearly between real data and computed/derived quantities.**
  When performing calculations on real data, label the source and transformation.
