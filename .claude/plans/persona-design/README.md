# Persona design — roadmap card P1

Worktree `.claude/worktrees/persona-design`, branch `feature/persona-design`, cut 2026-10-17 from
`origin/main` at `d639869cd`. Board: `~/projects/osprey/roadmap`, cards P1 and AT; rulings in
`decisions/2026-10-17-agent-tree-card.md` there.

`/.claude` is gitignored in osprey. Every file in this folder is committed with `git add -f` and
pushed, so the design never lives only on one Mac (the agent-tree plan was lost that way).
Before the PR goes ready, decide whether these files stay in main or are dropped from the branch.

## What is here

| File | What |
|---|---|
| `BRIEF.md` | The P1 design brief. Written 2026-10-01 against staging `a73b7f83f`; re-measured 2026-10-17 on this branch (`d026bb7f7`, = current main). Rows marked **Δ** changed since the first version. |
| `measure.py` / `MEASURE.md` | The measurement: every bundled preset through the real resolver and MCP registry, the two agent pairs and the access-level axis diffed, the profile-card omission. Re-run: `PYTHONPATH=src:packages/osprey-connectors/src ~/.venv/bin/python .claude/plans/persona-design/measure.py > .claude/plans/persona-design/MEASURE.md` |
| `RULINGS.md` | The owner's answers (step 2): the 8 questions, the two public words, the agent preset names, R5′/R12′. |
| `ref/agent-tree-2026-09-08/` | The agent-tree design (rulings R1–R12), its open questions and two research inventories. Reference for card AT; its PLAN.md is lost. |

## Order

1. ~~Re-measure `BRIEF.md` on current main~~ — done 2026-10-17 (`measure.py`, `MEASURE.md`,
   BRIEF §1–§2 updated, §4 reworded, §5 names, §6 R5/R12 options).
2. The owner answers the 8 questions, approves the public names for access level and agent kind,
   and rules on R5/R12.
3. P1 build (P1.1 split access level from agent kind · P1.2 one definition per agent · P1.3
   profile card lists the graph server) through `/feature --no-worktree` in this worktree.
4. Then AT: re-cut the agent-tree plan from `ref/agent-tree-2026-09-08/DESIGN.md` with R5/R12 as
   revisited; its build follows P1's because both rewrite the persona presets.
