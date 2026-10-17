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
| `BRIEF.md` | The persona-pairs brief, written 2026-10-01 against staging `a73b7f83f` (before C1 merged). Copied unchanged; needs re-measuring on current main. |
| `ref/agent-tree-2026-09-08/` | The agent-tree design (rulings R1–R12), its open questions and two research inventories. Reference for card AT; its PLAN.md is lost. |

## Order

1. Re-measure `BRIEF.md` on current main: drift table, file:line refs, the 8 questions in §4.
   Add how agent-tree R5 ("select, never re-render": a persona is exclude/additions on a base)
   and R12 (web editor writes persona shadows) interact with "one definition per agent".
2. The owner answers the 8 questions, approves the public names for access level and agent kind,
   and rules on R5/R12.
3. P1 build (P1.1 split access level from agent kind · P1.2 one definition per agent · P1.3
   profile card lists the graph server) through `/feature --no-worktree` in this worktree.
4. Then AT: re-cut the agent-tree plan from `ref/agent-tree-2026-09-08/DESIGN.md` with R5/R12 as
   revisited; its build follows P1's because both rewrite the persona presets.
