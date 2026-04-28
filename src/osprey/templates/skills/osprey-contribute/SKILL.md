---
name: osprey-contribute
description: >
  Guides a contributor through the OSPREY GitHub Flow contribution journey from
  a working-tree change to a merged PR on main. Use when someone says "I want
  to contribute X to osprey", "help me commit/push this", "prep this branch
  for a PR", "open a PR", "my CI is failing on this branch", "rebase onto
  main", or wants help following the contributing workflow. Auto-detects
  whether they have push access to als-apg/osprey or are contributing from a
  fork. Composes with the osprey-pre-commit, commit-organize, and
  osprey-release skills — invoke this whenever someone is contributing code to
  OSPREY, even if they haven't named the workflow explicitly.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# OSPREY Contribution Workflow

This skill walks a contributor from "I have a change" to "merged on main." It
is the connective tissue of OSPREY's GitHub Flow — branch off `main`, work,
validate, PR, merge — wrapping the existing scripts and skills you already
have so the contributor doesn't have to remember every gate.

The skill enters at whichever phase fits the contributor's current state. Run
**Phase 0: Orient** first; the orient step decides where to pick up.

## Out of Scope

| For | Use |
| --- | --- |
| Splitting a messy working tree into clean, atomic commits | `commit-organize` |
| Just running pre-commit checks without the full journey | `osprey-pre-commit` |
| Cutting a release / tagging a version | `osprey-release` |
| Designing a new feature or capability | `feature-dev` / brainstorming |

## Two Contributor Modes

OSPREY uses GitHub Flow with a single long-lived `main` branch. There are two
modes; detect which one applies on first run by inspecting `git remote -v`:

- **Internal mode** — `origin` points to `als-apg/osprey`. Contributor has push
  access. Flow: `branch off main → push branch to origin → PR → self-merge
  after green CI`.
- **Fork mode** — `origin` is a personal fork; another remote (typically
  `upstream`) points to `als-apg/osprey`. Flow: `branch off latest
  upstream/main → push branch to origin (the fork) → PR upstream → maintainer
  merges`.

If neither remote points to `als-apg/osprey`, ask the contributor. If `origin`
appears to be a fork but no upstream is configured, offer:

```bash
git remote add upstream https://github.com/als-apg/osprey.git
```

Mode detection is cheap; just re-detect each session rather than persisting.

---

## Phase 0: Orient

Before changing anything, gather the lay of the land. Run these in parallel:

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git remote -v
git fetch origin main --quiet && \
  git rev-list --left-right --count HEAD...origin/main
```

Decide where to enter:

| Observed state | Enter at |
| --- | --- |
| Uncommitted changes on `main` | Phase 1 (branch first, then commit) |
| Uncommitted changes on a topic branch | Phase 3 (commit) |
| Clean tree, topic branch, commits not pushed | Phase 4 (push) |
| Branch pushed, no PR yet | Phase 5 (open PR) |
| PR open with failing or pending CI | Phase 6 (watch and iterate) |
| PR green, internal mode | Phase 7 (merge) |
| Detached HEAD, mid-rebase, or conflicts | Resolve before proceeding |

Tell the contributor what you observed and which phase you're entering.

---

## Phase 1: Branch

> **Hard block: refuse to commit while on `main`.** Branch protection rejects
> direct pushes anyway, but failing locally saves the round-trip and the
> "rejected" error after the contributor has already typed the commit message.

1. Sync local `main` against the canonical remote:

   ```bash
   # internal mode
   git checkout main && git pull --ff-only origin main

   # fork mode
   git fetch upstream && git checkout main && git merge --ff-only upstream/main
   ```

2. Suggest a branch name from the change description and validate against the
   prefix convention from `CONTRIBUTING.md`:

   - `feature/<short-kebab>` — new functionality
   - `fix/<short-kebab>` — bug fixes
   - `docs/<short-kebab>` — documentation only
   - `refactor/<short-kebab>` — internal restructuring
   - `test/<short-kebab>` — tests only

   If the proposed name doesn't match a prefix, **warn** and suggest an
   alternative. Accept the contributor's choice on insist — these are
   conventions, not invariants.

3. Create the branch:

   ```bash
   git checkout -b <name>
   ```

4. If pre-commit hooks aren't installed in this clone, offer to install them
   once. Idempotent and safe on every fresh clone:

   ```bash
   pre-commit install
   ```

## Phase 2: Work

The contributor edits code; the skill is mostly absent. Two reminders:

- **Update `CHANGELOG.md` as you go**, not at the end. Add a bullet under the
  relevant heading (`### Added`, `### Changed`, `### Fixed`) in the
  `## [Unreleased]` section. Doing this concurrently keeps it accurate and
  turns the per-commit CHANGELOG check into a non-event.
- Keep the change focused. If the working tree starts to span unrelated
  concerns (e.g., "fix bug X" plus an unrelated refactor), suggest invoking
  `commit-organize` to split it before committing.

## Phase 3: Commit

> **Hard block: run `./scripts/quick_check.sh` (~30 s) before the commit.**
> Catches formatting drift and broken imports before they enter history.
> Fixing a polluted commit later is much more expensive than failing fast.

1. Show `git status` and `git diff --stat` so the contributor sees exactly
   what's about to land.
2. Stage surgically — list specific paths rather than `git add -A`. Blanket
   adds occasionally sweep up `.env`, scratch files, or large binaries.
3. Run `./scripts/quick_check.sh`. On failure, surface the output, suggest the
   minimal fix, re-run. Don't commit until clean.
4. Compose a conventional commit message:

   ```
   type(scope): summary

   Optional body. Explain *why*, not *what* — the diff already shows what.
   ```

   Where `type ∈ {feat, fix, docs, refactor, test, chore, ci, build, perf}`.
   Subject ≤ 70 chars, imperative mood ("add", not "added").

   **Soft prompt**: if the contributor's preferred message doesn't match the
   conventional form, propose a rewrite once. Accept their version on insist.

5. **Soft prompt — CHANGELOG**: if `CHANGELOG.md` isn't in the staged set, ask
   whether this commit needs an entry. Some genuinely don't (pure ruff
   renames, internal refactors invisible to users). Don't block.

6. Commit, then `git log -1 --stat` so the contributor sees what was recorded.

## Phase 4: Push

> **Hard block: run `./scripts/ci_check.sh` (~2-3 min) before the push.** CI
> minutes are shared and PR dashboards get noisy when obviously-broken
> branches sit waiting for triage. Pushing only what would have passed CI
> respects everyone's time.

1. Confirm the push target. Topic branches always go to the contributor's
   personal remote, never to `upstream`:

   - Internal mode → `origin` (which is `als-apg/osprey`).
   - Fork mode → `origin` (which is the contributor's fork).

2. Run `./scripts/ci_check.sh`. On failure, surface output, suggest the
   minimal fix, ask the contributor to amend or add a follow-up commit.

3. Push with upstream tracking the first time. Confirm before — this is the
   first visible-to-others state-change in the journey:

   ```bash
   git push -u origin <branch>
   ```

## Phase 5: Open the PR

> **Hard block: run `./scripts/premerge_check.sh main` before
> `gh pr create`.** Catches drift from `main` that surfaces only at merge
> time. Cheaper to find now than after CI burns minutes on the PR.

1. Compose the PR title and body:

   - **Title** — short (≤70 chars), conventional-style summary of the branch
     theme. If there's only one commit, that subject line is usually right.
   - **Body** — pull from the commit messages and the CHANGELOG entries.
     Sections:
     - `## Summary` — 1-3 bullets, *why* this change.
     - `## Changes` — what landed.
     - `## Test plan` — bulleted checklist of how this was validated.
     - `## Related issues` — if any.

2. Show the contributor the draft. They can edit before it goes out.

3. Open against `als-apg/osprey:main`:

   ```bash
   # internal mode
   gh pr create --base main --head <branch>

   # fork mode (gh handles the cross-repo head spec)
   gh pr create --repo als-apg/osprey
   ```

4. Print the PR URL and remind the contributor that CI will start running.

## Phase 6: Watch CI and Iterate

```bash
gh pr checks --watch
```

When checks complete:

- **All green** → Phase 7.
- **Failures** — fetch the failed run's logs, summarize the root cause, suggest
  a minimal fix. The contributor edits, re-stage, re-commit (or `git commit
  --amend` if the broken commit is the tip and not yet pulled by anyone else).
  Push again. Loop.
- **Stale checks** because `main` moved — rebase:

  ```bash
  git fetch origin main && git rebase origin/main
  git push --force-with-lease
  ```

  Always `--force-with-lease`, never plain `--force` — the lease prevents
  clobbering work someone else might have pushed to your branch in the
  meantime. Walk through any conflicts manually; do not force-push past a
  conflict the contributor hasn't reviewed.

`pre-commit.ci` runs as a required check and may auto-fix style by pushing a
commit to your branch. If you see "pre-commit.ci - autofix" land, just
`git pull --rebase` before the next push.

## Phase 7: Merge

**Internal mode** — when CI is green:

```bash
gh pr merge --rebase --delete-branch
```

`--rebase` because branch protection requires linear history; `--merge` would
be rejected. `--delete-branch` because long-lived branches aren't the model.

**Fork mode** — comment on the PR and ping a maintainer; merging is theirs to
do. The maintainer will use the same `--rebase --delete-branch` from their
side.

After merge, sync local:

```bash
git checkout main && git pull --ff-only origin main
```

The journey ends here. The next change starts again at Phase 0.

---

## Confirmation Pattern

Before any state-changing operation — `git commit`, `git push`, `gh pr create`,
`gh pr merge` — show the contributor exactly what's about to happen and ask
them to proceed. Reads (`status`, `diff`, `fetch`, `log`) don't need
confirmation.

The point is: every visible-to-others action has a deliberate moment where the
contributor sees what's about to ship and can stop it.

## Branch-Protection Reality Check

A few facts about OSPREY's `main` branch protection that shape behaviour here:

- **All 8 required CI checks must pass** before merge. There is no admin
  bypass; `enforce_admins` is on. If a required check is genuinely wrong (a
  flaky runner, a broken matrix), fix it forward — don't try to bypass.
- **Linear history is required** — that's why merges always use `--rebase`
  and rebases use `--force-with-lease`.
- **Direct push to `main` is rejected.** This skill never tries; the hard
  block in Phase 1 mirrors the server-side rule so the contributor fails
  locally instead of after typing a commit message.
- **PR approvals are not required** (`required_approving_review_count: 0`),
  so internal-mode contributors can self-merge once CI is green. Fork-mode
  contributors still need a maintainer to merge.

If the contributor hits an unexpected branch-protection error, surface the
GitHub message verbatim; the skill should not pretend the rules are looser
than they are.
