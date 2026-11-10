Pull requests run the unit and serial-suite lanes on Python 3.12 only; 3.11 and 3.14 run on main
pushes and the nightly revalidation, and on pull requests labelled `full-ci` or opened by
Dependabot.
ci.yml writes the setup steps it repeats across lanes once, as YAML anchors, and drops the comments
that repeated alongside them.
ci.yml's zero-skip gates and stranded-stack cleanups each run one shared script
(`scripts/ci/junit_no_skips.py`, `scripts/ci/cleanup_stacks.sh`) instead of a copy per lane.
The merge gate applies one rule to every needed job's result and no longer needs the main-only qmd
cache warm-up, whose skip failed every nightly revalidation of main.
