---
name: osprey-pre-commit
description: >
  Validates code before committing using OSPREY's three-tier check scripts.
  Runs linting, formatting, and tests to catch issues early. Use when ready
  to commit, before pushing, before opening a PR, or when the user asks to
  run checks, validate, or verify their changes. For the full contribution
  journey (branching, commits, push, PR, merge), use `osprey-contribute`
  instead — this skill is the focused validation step.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# Pre-Commit Validation

This skill runs OSPREY's three-tier check scripts at the right moment in the
journey. It is a focused validation tool — not a full contribution workflow.
For the end-to-end journey (branching, committing, pushing, opening a PR,
merging) use the `osprey-contribute` skill, which invokes these checks at
each gate.

## The Three Tiers

| Tier | Script | Time | Use Before |
| --- | --- | --- | --- |
| Quick | `./scripts/quick_check.sh` | ~30 s | every commit |
| Full | `./scripts/ci_check.sh` | ~2-3 min | every push |
| Pre-merge | `./scripts/premerge_check.sh main` | ~1-2 min | every PR |

Each tier expands the surface area of what is verified. Running them in
order saves time: catch the cheap things cheaply, escalate only when the
cheap checks pass.

### Quick Check (before commits)

Runs formatting auto-fix and the fast unit tests. Catches most accidents
that would clutter `git log`.

```bash
./scripts/quick_check.sh
```

If you want to do this manually:

```bash
ruff check src/ tests/ --fix --quiet
ruff format src/ tests/ --quiet
pytest tests/ --ignore=tests/e2e -x --tb=line -q
```

### Full CI Check (before pushing)

Replicates the GitHub Actions CI matrix locally, including type checking and
the full unit-test suite. Pushing only what would have passed CI respects
shared CI minutes and keeps PR dashboards clean.

```bash
./scripts/ci_check.sh
```

### Pre-merge Check (before opening a PR)

Compares the branch against the PR target (`main`), catches drift that
surfaces only at merge time, and runs the full check suite once more
against the rebased state.

```bash
./scripts/premerge_check.sh main
```

## When a check fails

Read the output, fix the smallest thing that would make it pass, and re-run
the same script. Don't escalate to a higher tier until the lower tier is
green — escalating early just hides the cheaper failure under a slower one.

If the failure is in `ruff`, the auto-fix flag (`--fix`) usually resolves it.
If the failure is in tests, the most useful next step is to run that single
test verbosely:

```bash
pytest tests/path/to/test_file.py::test_function_name -v
```

## Composition with other skills

- **`osprey-contribute`** — the full journey. Calls these checks at each
  gate (commit, push, PR). If you find yourself running these checks while
  also juggling branching, committing, and PR creation, switch to
  `osprey-contribute` instead.
- **`commit-organize`** — when a working tree spans unrelated concerns,
  split it into atomic commits before running the checks.
- **`osprey-release`** — release-cutting flow that includes a clean-venv
  pre-release test pass on top of these tiers.
