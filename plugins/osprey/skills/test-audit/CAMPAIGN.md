# Test-pruning campaign

Adapted from openclaw's test-audit skill; see [LICENSE.txt](LICENSE.txt).

Campaign mode prunes one subsystem's whole test surface in one PR: a
subsystem such as `tests/connectors` with its owners in `src/osprey/connectors`
and `packages/osprey-connectors`, or one core area. The value bar, retention
bar, candidate evidence, and validation in [SKILL.md](SKILL.md) apply to every
lane. This file adds the order of work and the lessons of a full campaign.
Each step ends on its completion criterion; do not start the next step early.
Keep ledgers, lane plans and notes out of tracked files.

## 1. Baseline

Record the subsystem's test and support line counts and every test file's
pass/fail state at a pinned `main` SHA, serially and under
`-n 4 --dist loadgroup`. Keep baseline failures in their own list, split into
real failures and parallel-only failures (triage the latter with "If a new
test flakes under `-n 4`" in tests/README.md §7): a baseline failure is a
product-bug suspect, not a stale test.

Done when every in-scope test file has a recorded baseline result.

## 2. Lanes and inventory

Split the surface into **lanes** along production owner boundaries, not file
prefixes. Include the subsystem's cases at shared core boundaries (e.g.
`tests/integration`, `tests/mcp_server`), its lane-local support modules, its
QA and live-proof tests (its tests/e2e files, including the simulation-engine
scenario tests and the local-only tests skipped in CI, per tests/e2e/README.md;
and any tests/manual cases), and tests that CI routes to dedicated jobs (the
unit-lane `--ignore` list, dependency-floor, the dedicated e2e jobs, `frontend-js` for
vitest files).

Example lanes, for `connectors`:

- `types`: connector base types, `WriteOutcome` vocabulary, limits posture;
- `epics`: EPICS/Channel Access connector, subscribe and write paths;
- `pva`: PVA transport and subscription lifecycle;
- `doocs` / `tango`: the DOOCS and Tango connectors;
- `mock` and dynamic: mock connector, dynamic registration, factory;
- `archiver`: the EPICS, MYA, MongoDB, DOOCS and mock archiver connectors and
  `test_archiver_*.py`, owned by `osprey_connectors/archiver`;
- `va`: the virtual-accelerator connector and its gateway port fill;
- `context/targets`: control context, owner context, identity ladder, target
  resolver and factory control-target selection;
- `ipc`: `tests/connectors/ipc`;
- `shims`: `sys.modules` alias shims into `osprey_connectors`;
- boundary: connector cases under `tests/mcp_server` write/read tools and
  `tests/integration` mock-pair suites;
- harness: `tests/connectors/conftest.py` and `_mock_dynamic_connector.py`.

Done when every test file and QA scenario the subsystem owns belongs to
exactly one lane.

## 3. Read-only ledger per lane

Give each lane to its own read-only reviewer (a separate agent where the
tooling supports one). The reviewer reads every assigned test
in full, including `parametrize` tables and fixtures. It also reads the
production owners and their entry points, callers, history, and CI routing.
Each test declaration goes into a written **ledger** with one mark. A
parametrized test (`parametrize`, or `it.each`/`test.each` in vitest) is one
declaration unless its rows need different marks;
then mark each row by its param id.

- `R`: retain, naming the contract and the bug it catches; a retained test that
  only moves to a better-named file stays `R` with the move noted;
- `F`: retain the contract but repair the assertion, such as a vacuous negative
  that passes when only one of several items is missing, or a bare
  `pytest.raises(Exception)`;
- `C`: consolidate, naming the owner that absorbs the assertion first: a sibling
  parametrize row, a stronger boundary suite, or the shared owner in another
  package;
- `D`: delete, naming the proof that remains, or why no contract exists.

Judge a test by its assertions, not its name. A test named "disabled by
default" can take its default from the mock it installs.

Done when every declaration in the lane has a mark and an evidence line.

## 4. Layer plan per lane

Treat the per-test ledger as input, not as the edit list. A second read-only
pass, starting from the ledger, looks for the redundant **layer**: the same
transport-agnostic early return tested once per connector, or a mocked
collaborator replayed around a stronger suite. Name the **keeper** suite for
each contract. Prefer the real boundary with a fake transport (mock connector,
served tree) over a mocked collaborator. Keep per-connector copies where each
connector overrides the method or where write-path safety must hold per
connector. Correct any ledger errors this pass finds.

Done when each lane plan names its retired files, its keeper per contract, the
assertions to carry into keepers, and the test-only production seams unlocked.

## 5. Cutover

Edit lane by lane. Serialize changes to shared harnesses and support files
(tests/conftest.py, area conftests, `tests/_*.py`) through one owner. With each
lane, remove the test-only production seams it unlocks: injection parameters,
getters, `reset_*` exports, `_clear()` hooks, and indirection layers; never a
sanctioned isolation seam without replacing the isolation. Register moved suites
in CI routing and inventories (ci.yml path lists, the import-time WHITELIST,
matrix_e2e_config.json, markers) and keep test_ci_workflow_wiring.py green.
Add a `changelog.d/<ref>.internal.md` fragment when src/ or packages/ changes.
There are no shrink-only line-cap baselines; refresh scripts/mypy_baseline.json
with `--update` only if deletions leave stale entries. Put durable
test-ownership rules in tests/README.md or a scoped tracked README, drawn from
mistakes this campaign actually found; untracked local instruction files do
not reach other contributors.

Done when every lane plan is applied and each lane's keepers pass serially and
in parallel.

## 6. Preservation review

Before claiming completion, have independent reviewers compare deleted
coverage against the keepers, one reviewer per boundary group. They look for
contracts that lost their only proof, above all in the hardware-write safety
chain. They also look for new assertions that cannot fail, such as a rejection
row the production code never reaches.

For each restored contract, make one deliberate **mutation** of the production
owner and confirm the keeper goes red. Then restore the source byte for byte
(`git diff` empty on that file).

Done when every reported gap is restored or rejected with source evidence, and
every restored contract has a caught mutation.

## 7. Product defects

A baseline failure that survives into a keeper is a bug report. Fix it test-first
at its owner as a separate commit, and prove it through the real user flow
(CLI verb, MCP tool, or one e2e file by path), with a **control** run that
reverts the fix and shows the old behavior. Record unrelated product
discrepancies you find as follow-ups instead of fixing them in the campaign.

Done when each repaired defect has a failing control and a passing candidate
on the same harness.

## 8. Reconcile and hand off

Campaigns outlive many `main` commits. Merge `main` rather than rebasing a
long, many-commit campaign. When `main` modified a file the campaign
deleted, keep the deletion. Port the new contract into the keeper instead, and
confirm every new regression `main` added still has a home. Rerun the whole
subsystem suite on the merged head, and the dedicated CI lanes it feeds; repeat
the live proof there by rerunning the subsystem's e2e files by path.

Expect review tooling to see a truncated file list on a diff this large.
Record maintainer decisions on retained compatibility shims in the PR evidence
rather than editing gates.

Hand off with the [SKILL.md](SKILL.md) report, plus:

- baseline and final test/support line counts, with production counted separately;
- lanes, retired layers, and keepers;
- preservation gaps found and their mutations;
- product defects with control and candidate proof.
