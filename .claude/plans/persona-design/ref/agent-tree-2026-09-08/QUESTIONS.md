# agent-tree — open owner questions

Raised during the planning cycle (DA round 2, skeptic miss). Not covered by DESIGN.md rulings R1–R11.
All questions ruled; see each entry.

## Q1 — Should the web terminal's live file editor write through to `agent/`?

**RESOLVED 2026-09-08 — ruled R12: option 2, write through.** Recorded in DESIGN.md (R12), PROPOSAL.md (In Scope, FR15, rulings table) and PLAN.md Tasks 7.7 editor-agent-write-target, 7.8 editor-list-append, 7.9 editor-write-through-tests.

**Context.** `src/osprey/interfaces/web_terminal/claude_code_files.py:166` ships an editor that writes and creates files in the render's `.claude/` (agents, commands, output-styles are deliberately writable). Today those edits die at the next build. After this feature they land in a copy of `agent/` and still die at the next build, with no ledger record, no list entry and no route back into `agent/`. R6 says "No tool ever writes into `agent/`"; the editor is a human editing surface, not an MCP tool, so the ruling does not settle it.

**Options.**
1. **Out of scope for this feature (default in the plan).** Editor behaviour unchanged; documented as ephemeral. Consequence: no regression versus today, but the one interactive edit surface still contradicts the feature's purpose until a follow-up.
2. **Write through to `agent/`.** The editor saves to `agent/<sub>/<path>` (or the persona shadow) and the next build copies it; a new file is deployment-authored (no ledger record) and the editor appends the list entry. Consequence: roughly 3 tasks more (route + list append + tests); the editor becomes a first-class edit route that shows up in `osprey agent list` as `edited`/`yours`.
3. **Make the editor read-only for agent paths.** Consequence: smallest change, removes the misleading affordance, loses in-browser editing until option 2 lands.

**Recommendation.** Option 1 now, option 2 as the follow-up — the feature is already large, and write-through needs its own design of the persona/shadow target and the list-append UX.
