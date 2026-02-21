---
summary: Artifact-first reuse rule
description: Reuse existing workspace artifacts before creating new content
paths:
  - "osprey-workspace/**"
---
# Artifact-First Rule

When the user references previous work or wants to act on it (log it, share it,
re-analyze it), call `data_context_list()` BEFORE creating anything new.
Use `search=` or `tool_filter=` to narrow results. Reuse existing artifact IDs
rather than recreating content.
