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
