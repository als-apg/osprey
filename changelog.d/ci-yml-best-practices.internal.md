`ci.yml` follows GitHub Actions hardening practice, and a few things change for contributors.

- **Lane ids and renamed checks.** A lane in `LANES` (`scripts/repo_checks.py`) has an `id`
  beside its `job`: the job id, or the cell's `lane:` value where one job hosts several lanes.
  `spending()`, release.yml's `REQUIRED_LANES`, `require_green_lanes.py` and
  `full_ci_on_main.py` key by it. `auth-perimeter-e2e` and `terminal-auth-multiuser-e2e` now
  render as `Stack E2E — Auth Perimeter` / `— Terminal Auth Multi-User`; `host-record-e2e` and
  `scan-agentic-e2e` as `Dispatch Stack E2E — Host Record` / `— Scan Stack Agentic`. Their caps
  were 60, 70, 60 and 75 minutes on main and are now 35, 70, 25 and 75. The Dockerfile agent
  turn is the new spending lane `Dockerfile Agent E2E`, and `visual-baselines-push` alone holds
  `contents: write`.
- **No skipped tests on gated lanes.** Their pytest steps load `-p no_skips`
  (`scripts/ci/pytest_plugins/no_skips.py`), which fails the run on any skip; `e2e-tests` adds
  `--no-skips-allow-partial` and reds only when everything skipped. `junit_no_skips.py` is gone.
- **New e2e modules.** `e2e-tests` derives its `--ignore` and cleanup lists from `LANES`, so a
  new `tests/e2e` module runs there unless a lane claims it (`scope`, or `Lane.modules` where the
  scope names more). To give it its own lane, claim it in that lane's `LANES` entry;
  `test_every_e2e_module_runs_in_exactly_one_place` holds each module to one host.
- **Dependabot and `full-ci`.** Dependabot PRs are recognised by author, so the `full-ci` label
  no longer starts spending lanes on them; run `gh workflow run ci.yml --ref <branch> -f
  revalidate_secret_lanes=true` instead.
- **Moved scripts.** Inline programs live in `scripts/ci/`, most with tests in `tests/scripts/`:
  `podman_compose.py`, `proxied_build.sh`, `fetch_n10s_jar.sh`, `e2e_shared_lane.py`,
  `llm_preflight_probe.sh` and `configure_playwright_apt.sh`. `canopy-contract` stays inline.
- **Token and pins.** `ci.yml` and `docs.yml` default to a read-only token, and no checkout but
  the push job's keeps credentials. Pin each new external action by SHA with a `# vX.Y.Z` comment.
- **`last-green` history.** Until the first revalidation of main after this merges,
  `full_ci_on_main.py last-green` reports no green for the whole run, nor for
  `host-record-e2e`, `scan-agentic-e2e` or `dockerfile-agent-e2e`: older runs rendered other names.
