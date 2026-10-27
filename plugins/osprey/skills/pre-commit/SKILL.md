---
name: pre-commit
description: >
  Validates code before committing using OSPREY's check command and its tiers.
  Runs linting, formatting, and tests to catch issues early. Use when ready
  to commit, before pushing, before opening a PR, or when the user asks to
  run checks, validate, or verify their changes. For the full contribution
  journey (branching, commits, push, PR, merge), use `/osprey:contribute`
  instead — this skill is the focused validation step.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# Pre-Commit Validation

This skill runs OSPREY's check command at the right moment in the journey. It
is a focused validation tool — not a full contribution workflow. For the
end-to-end journey (branching, committing, pushing, opening a PR, merging) use
the `/osprey:contribute` skill, which invokes these checks at each gate.

## The Tiers

Every check is declared once, in `scripts/check.py`. A laptop, an automated
gate and each CI step run it from there.

| Tier | Command | Use Before |
| --- | --- | --- |
| Quick | `uv run python scripts/check.py quick` | every commit |
| Full | `uv run python scripts/check.py full` | every push |
| Quick, scoped | `uv run python scripts/check.py quick --changed origin/main` | every PR |

Each tier expands the surface area of what is verified. Running them in
order saves time: catch the cheap things cheaply, escalate only when the
cheap checks pass.

To see what a tier runs:

```bash
uv run python scripts/check.py list --tier quick
```

### Quick (before commits)

Runs the checks a commit needs, with the fast unit tests. Catches most
accidents that would clutter `git log`.

```bash
uv run python scripts/check.py quick
```

Add `--fix` to apply ruff first: it runs `uv run ruff check . --fix` and
`uv run ruff format .`, then the checks.

### Full (before pushing)

Adds the slow tests, the serial suites and the builds. Pushing only what
would have passed CI respects shared CI minutes and keeps PR dashboards clean.
Needs `uv sync --extra dev --extra docs`.

```bash
uv run python scripts/check.py full
```

### Quick, scoped (before opening a PR)

Compares the branch against the PR target (`origin/main`). It runs the checks the
diff touches and the tests it maps to, adds the checks that judge a range (the
changelog fragment and the plugin version), and
names the CI lanes the diff touched.

```bash
uv run python scripts/check.py quick --changed origin/main
```

Add `--worktree` to judge uncommitted and untracked files as well.

### Reading the verdict

The last line is the verdict: `clean`, or `not clean` with the checks that
failed. It also names what did not run here. Exit status 2 is a refusal: the
command could not judge, and the message says what to install or fetch.

## When a check fails

Read the output, fix the smallest thing that would make it pass, and re-run
the same command. Don't escalate to a higher tier until the lower tier is
green — escalating early just hides the cheaper failure under a slower one.

If the failure is in `ruff`, `quick --fix` usually resolves it.
If the failure is in tests, the most useful next step is to run that single
test verbosely:

```bash
pytest tests/path/to/test_file.py::test_function_name -v
```

## Composition with other skills

- **`/osprey:contribute`** — the full journey. Calls these checks at each
  gate (commit, push, PR). If you find yourself running these checks while
  also juggling branching, committing, and PR creation, switch to
  `/osprey:contribute` instead.
- **`commit-organize`** (optional, external) — when a working tree spans
  unrelated concerns, split it into atomic commits before running the checks.
- **`/osprey:release`** — release-cutting flow that includes a clean-venv
  pre-release test pass on top of these tiers.
