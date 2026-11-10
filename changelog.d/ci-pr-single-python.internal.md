Pull requests run the unit and serial-suite lanes on Python 3.12 only; 3.11 and 3.14 run on main
pushes and the nightly revalidation, and on pull requests labelled `full-ci` or opened by
Dependabot.
ci.yml writes the setup steps it repeats across lanes once, as YAML anchors, and drops the comments
that repeated alongside them.
ci.yml's zero-skip gates, stranded-stack cleanups and the merge gate's always-on checks each run
one shared script or shell function (`scripts/ci/junit_no_skips.py`, `scripts/ci/cleanup_stacks.sh`,
`require_success`) instead of a copy per lane.
