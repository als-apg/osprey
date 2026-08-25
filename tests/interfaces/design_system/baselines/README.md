# Visual regression baselines

PNGs in this directory are the Linux-rendered reference screenshots
`test_visual.py` compares against. They are produced and committed by the
CI job (`.github/workflows/ci.yml`), not authored by hand:

- The job runs `test_visual.py` on `ubuntu-latest` with chromium installed.
- A missing baseline is written on the spot (bootstrap case — nothing to
  compare against yet).
- Any baseline file that changed or was added is committed back to the PR
  branch by the job's auto-commit step, and also uploaded as a build
  artifact for inspection.

Non-Linux runs (e.g. a contributor's macOS machine) never compare pixels —
anti-aliasing and subpixel rendering differ across platforms — they only
verify screenshot capture succeeds, and print an explicit notice that the
byte-compare was skipped. Don't hand-author or hand-edit PNGs here; if a
baseline looks wrong, regenerate it via CI (or `pytest
tests/interfaces/design_system/test_visual.py --regen-baselines` on Linux)
and review the diff in the PR.

## The `[skip ci]` / required-checks trap

The auto-commit carries a `[skip ci]` marker. That marker exists to break the
regenerate → commit → regenerate loop (the commit would otherwise retrigger
the very job that made it), but it has a second effect: **no checks run on
that sha at all**, including the ones branch protection requires. A PR whose
head is a baseline auto-commit can therefore never go green — the required
checks are not failing, they are simply absent, and GitHub reports the merge
as blocked on checks that "haven't run yet" forever.

The way out, once the baselines have settled: put a normal commit on top of
the auto-commit (or amend the marker out of its message) and push, so CI runs
on the new head. An empty commit is enough:

    git commit --allow-empty -m "chore: rerun CI on settled baselines"

Expect this whenever a PR's last push only changed screenshots — it is not a
CI outage, and re-running the workflow from the UI does not help (the skip is
evaluated per commit, not per run).
