CI workflows follow GitHub Actions hardening practice. `ci.yml` grants a read-only token at the
workflow level, and every checkout but the one in the baseline-pushing job leaves no token in the
git config. That job, `visual-baselines-push`, is now separate from the Playwright suites: it holds
the only `contents: write` token, runs no repository code, and commits the baselines the suites
uploaded as an artifact back to a same-repo pull request's branch. Every external action in
`.github/workflows/` and `.github/actions/` is pinned by commit SHA with a `# vX.Y.Z` comment, and
Dependabot now updates the composite actions too. No `run:` script pastes in a `${{ }}` expression
any more: the values arrive through `env:`, and the wiring tests hold both rules for every workflow
and action. Job timeouts that used to sit at a generic 60 minutes now sit at about three times the
90th-percentile duration of recent green runs on main. The two proxy-probe base images are pinned by
tag and digest, podman-compose's dependencies are pinned, and Dependabot pull requests are
recognised by their author rather than by whoever triggered the run.

`dockerfile-e2e` no longer holds `ALS_APG_API_KEY`. Its one model call, the agent turn through the
generated image, moved to `tests/e2e/test_dockerfile_agent_e2e.py` and runs in the new
`dockerfile-agent-e2e` lane, which is model-spending: it runs under the `full-ci` label or the
revalidation dispatch, the release gate and `full_ci_on_main.py` count it, and it fails on a skipped
test. `deploy-e2e` holds no secret either: it calls no model, and `osprey up` only refuses an empty
provider key, so it boots on a placeholder value and now runs on Dependabot pull requests too.

The longer programs that lived inline in `ci.yml` are now scripts under `scripts/ci/`, linted
(shellcheck, ruff, mypy) and runnable on a laptop, each with its own tests under `tests/scripts/`.
`podman_compose.py` holds the podman-compose pin once for both podman jobs (`pin`, with
`--optional` for `static-checks`), the two-shape lane's provider assertion (`assert-provider`) and
the merged-topology parse (`parse-check`). `proxied_build.sh` runs `dockerfile-e2e`'s proxied
build in five subcommands (`start`, `probe`, `build`, `assert`, `stop`), and `fetch_n10s_jar.sh`
fetches the neosemantics jar for both graph lanes. The workflow keeps each step's name, timeout
and a comment on what it proves; `canopy-contract` stays inline so its credential boundary reads in
one place. `visual-baselines-push` now skips its commit, with a notice, when the branch has moved
past the commit that rendered the baselines; the check is pure git and reads that sha through
`env:`.

The skip-gated lanes no longer run a separate "Fail the lane on any skipped test" step over junit
reports. Their pytest steps load `scripts/ci/pytest_plugins/no_skips.py` instead (`-p no_skips`,
with that directory on `PYTHONPATH`), which fails the session on any skipped test, xfail or module
skipped at collection, under xdist too, and says why in an `::error::` annotation; a session that
collects nothing still exits 5. `--no-skips-message` prints the lane's message when nothing
skipped, and `--no-skips-allow-partial` keeps `privilege-split-e2e`'s module-wide rule. The gate
now also holds `agentic-per-preset`, `dispatch-stack-e2e`, `nextcloud-talk-bridge-e2e` and every
`stack-e2e` cell, so every model-spending lane but `e2e-tests`, whose skips are by design (opt-in
provider matrices, backends no runner provisions), fails on a skip.
`scripts/ci/junit_no_skips.py` is gone. The capture-diagnostics action's `name` input now defaults
to the job id, so a plain job's capture step is one shared YAML anchor, and only matrix jobs, whose
cells need distinct artifact names, pass a name.
