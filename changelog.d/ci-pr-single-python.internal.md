Pull requests run the unit and serial-suite lanes on Python 3.12 only; 3.11 and 3.14 run on main
pushes and the nightly revalidation, and on pull requests labelled `full-ci` or opened by
Dependabot.
ci.yml writes the setup steps it repeats across lanes once, as YAML anchors, and drops the comments
that repeated alongside them.
ci.yml's zero-skip gates and stranded-stack cleanups each run one shared script
(`scripts/ci/junit_no_skips.py`, `scripts/ci/cleanup_stacks.sh`) instead of a copy per lane.
The merge gate applies one rule to every needed job's result and no longer needs the main-only qmd
cache warm-up, whose skip failed every nightly revalidation of main.
CI lanes no longer delete preinstalled toolchains to free runner disk; the failure diagnostics log
`df -h /` instead.
On a pull request, a failed lint, docs build, package build, frontend JS or plugin-smoke lane
cancels the rest of the run.
The e2e lanes no longer wait for a changed-path classifier, so they start sooner and run on
docs-only pull requests too; the classifier and `scripts/ci/docs_only.py` are gone.
ci.yml sets Python through setup-uv, runs lint's and frontend-js's base-free checks in one call
each, and drops steps that did nothing: Docker Compose setup, a never-saving uv cache, checks the
unit sweep already runs, and duplicated two-shape pre-boot and podman-compose install steps.
ci.yml's stranded-resource cleanups, all but the two-shape lane's, run `scripts/ci/cleanup_stacks.sh` (now with
`ENGINE`, `--sweep` and `--image`), and the browser lane selects its suites with `-m browser` instead of a file list.
