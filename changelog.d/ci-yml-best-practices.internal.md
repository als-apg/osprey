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
