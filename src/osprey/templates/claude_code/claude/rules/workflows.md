---
summary: Task planning, agent delegation, and parallel execution
description: How to plan work, delegate to specialized agents, and run independent tasks in parallel
---

# Workflow Rules

## Always Plan Before Acting

For any request that involves more than a single tool call, create a task list
before starting work. This makes your plan visible to the operator and ensures
nothing is missed.

- **Simple queries** (single channel read, one search): act directly.
- **Multi-step work** (investigate + plot, find channels + read + analyze):
  create a numbered task list first, then work through it.

## Delegate to Specialized Agents

When a task falls within a specialized agent's scope, always delegate to that
agent. Do NOT attempt the work yourself — agents carry domain-specific prompts,
constrained tool sets, and tailored strategies that produce better, safer
results.

## Run Independent Tasks in Parallel

When your task list contains steps that do not depend on each other's results,
launch them as parallel sub-agent calls. This dramatically reduces wait time
for the operator.

**Parallel when independent:** If task B does not need the output of task A,
run them together. Examples: searching multiple sources for the same topic,
finding channels for different subsystems, creating multiple plots from
already-collected data.

**Sequential when dependent:** If task B needs the output of task A, run them
in order. Examples: find channels then read their values, collect data then
plot it, read a value then decide whether to write.

## Workflow Pattern

For complex requests, follow this pattern:

1. **Understand** — Parse the user's request. Clarify if ambiguous.
2. **Plan** — Create a task list. Identify which tasks can be parallelized.
3. **Delegate** — Launch agent calls (parallel where possible).
4. **Synthesize** — Combine results into a coherent answer.
5. **Present** — Show findings with artifact references and clear structure.
