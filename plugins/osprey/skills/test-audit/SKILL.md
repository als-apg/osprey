---
name: test-audit
description: >
  Gates and audits OSPREY tests: the pytest suites under tests/ and the vitest
  `*.test.mjs`/`*.test.js` suites under tests/interfaces. Authoring mode is a
  four-question gate every new or changed test passes before it lands; audit
  mode sweeps for low-value, implementation-coupled, or duplicative tests and
  the test-only production seams (reset_*, _clear, "for tests" injection
  parameters) they keep alive; campaign mode prunes one subsystem's whole test
  surface. Use whenever someone writes, changes, reviews, or sweeps tests, or
  says "is this test worth keeping", "audit the tests", "prune the connector
  tests", "these tests are testing the mock", or "remove test-only seams".
license: MIT (adapted from openclaw's test-audit skill); see LICENSE.txt
---

# Test Audit

Adapted from the test-audit skill of
[openclaw/openclaw](https://github.com/openclaw/openclaw)
(`.agents/skills/test-audit`, commit 80930af4), MIT License, Copyright (c) 2026
OpenClaw Foundation. The upstream notice is in [LICENSE.txt](LICENSE.txt).

Three modes, one value bar. Authoring mode gates every new or changed test at
write time. Audit mode runs focused sweeps of tests that re-assert source,
duplicate stronger proof, couple behavior to implementation, or keep test-only
production seams alive. Continue broad audits as separate coherent follow-up
PRs; optimize for confidence, not deletion count. Campaign mode prunes one
whole subsystem's test surface (every test file one `src/osprey/<area>` or
`packages/` owner has); before starting one, read [CAMPAIGN.md](CAMPAIGN.md).

## Authoring gate

Before adding any test, answer four questions; a missing answer means do not
add it yet:

1. What observable behavior, invariant, or independent contract does it protect?
2. What credible regression makes it fail?
3. Why does existing coverage not already catch that failure? Each contract has
   one primary test owner at the strongest boundary; another layer needs its
   own distinct risk, such as a transport or lifecycle failure the owner cannot
   reach. Prefer a new `@pytest.mark.parametrize` row (`it.each`/`test.each`
   in vitest) or a shared fixture
   (`tests/_*.py`, the area `conftest.py`) over a near-duplicate test;
   consolidate duplicated setup in the same change.
4. Does it need a production seam (export, flag, wrapper, `reset_*`, injection
   parameter) that no production caller needs? If yes, move the test to the
   real boundary instead. Exception: a named reset seam next to new global
   state is sanctioned (the reset-seam list in tests/README.md §2, "Isolation
   fixtures"); add it, do not hand-roll one.

Then check the test against every [junk pattern](#junk-patterns); a match fails
the gate unless the [retention bar](#retention-bar) names the contract it
independently guards. A test that would break under behavior-preserving
refactoring is asserting implementation, not behavior; rewrite it at the
owning boundary before landing it. Also pass every item of the tests/README.md
§7 checklist ("Adding a batch of new tests"), including: unique basename;
reuse the shared helpers (`free_port()` in tests/fixtures/bench_ioc.py,
`patch_subprocess()` in tests/cli/_scoped_subprocess.py, `isolated_home` in
tests/cli/conftest.py, the named reset seams); new global state gets a public reset seam plus an autouse
fixture; no import-time side effects; no new `xdist_group`; almost no flaky
markers; assert on `result.stdout`; green serially and under
`-n 4 --dist loadgroup`.

Bug fixes are test-first: the regression test must fail on the pre-fix code for
the intended reason and pass after the owner-boundary repair. A regression test
that never demonstrably failed proves the mock, not the fix. One regression at
the owner boundary covers the bug; do not replay the same scenario at every
layer it crosses.

## Junk patterns

The shared checklist for both modes: the authoring gate rejects a new test that
matches one, and audits hunt for existing tests that do. Python shape in
parentheses.

- assertion-free coverage probes (no `assert`, `pytest.raises` or
  `mock.assert_*`; a `# should not raise` comment in place of a check);
- self-comparisons and identity copiers (`f(x) == f(x)` in one process,
  `helper(obj) is obj`, `Dataclass(a) == Dataclass(a)`);
- copied fixtures, inventories, manifests, or export lists (a module-level
  `EXPECTED_*` repeating a production tuple, often in several files; a frozen
  replica of old production code kept as the baseline);
- exact source, import, or string greps (`inspect.getsource(fn)` or
  `Path(mod.__file__).read_text()` with `in`, `not in`, or
  `source.index(a) < source.index(b)` to pin call order);
- private predicate or call-shape tests duplicated at real boundaries
  (`obj._attr = ...; obj._validate()` while a sibling drives the public entry);
- duplicate invocations of the same contract (one copy asserts only
  `mock.called` or "no exception", the other the real output);
- provider-local replays of shared helpers (one scenario re-tested per
  connector, bridge adapter or transport over a shared code path);
- tests whose only purpose is preserving test-only exports, globals, or wrappers;
- dead production code whose only callers are tests (`"""For tests."""`,
  `_reset_*`, `_clear()`, re-exports marked `# noqa: F401 (re-exported for tests)`);
- expected values produced by the helper or renderer under test
  (`expected = render(...)`, or `json.dumps` with production's own kwargs);
- mocks that implement the asserted behavior, or one identical mock standing in
  for different APIs (`get_config_value` as `MagicMock(return_value=X)` for
  every key; an unspec'd `MagicMock()`; a global `patch("importlib.import_module")`);
- fixtures that supply the receipt, admission, or callback ordering the owner
  should produce, or persistence asserted against a store the path never writes
  (a hand-written or renamed approval stamp, audit record or queue entry);
- capability tests that restate declared flags instead of exercising the
  delivery or acknowledgement the flag promises (reading back
  `permissions_allow`, `enabled`, or `mock_init.called`);
- negative controls that pass for an unrelated reason, such as a denial from a
  different guard or a rejection the production path never reaches
  (`pytest.raises(Exception)` where a typo's `AttributeError` also passes;
  `"msg" not in caplog.text`, which passes when the wording changes);
- names or fixtures that promise more than the input exercises, such as a
  "disabled by default" test whose mock supplies the default.

OSPREY-specific:

- stdlib-behavior tests (frozen-dataclass mutation, `StrEnum` equality) that
  re-test Python; keep only when immutability is a documented safety property,
  then pin `dataclasses.FrozenInstanceError`;
- `assert mock.called` as the only assertion, usually on an unspec'd mock;
- caplog-only tests where the log line is not the operator-facing remedy;
- CliRunner `--help` smokes checking only `exit_code == 0` and the verb name;
- private-attribute construction bypassing the real constructor or
  `connect()` when a cheap public path exists (fine as a hardware-transport
  fixture seam);
- `if __name__ == "__main__": pytest.main([__file__])` boilerplate;
- issue-number regression docstrings asserting only a mock-shaped matrix
  (check the docstring for a stated backstop rationale first).

## Value bar

Tests justify their maintenance cost by protecting behavior, a credible
regression, or an independently meaningful contract. In an audit, an existing
test that must change for behavior-preserving source reorganization is suspect,
not automatically deletable; the authoring gate still rejects new ones.

Before judging a candidate, read the complete test and production owner, its
entry point, callers, callees, sibling implementations, overlapping tests, CI
routing (.github/workflows/ci.yml), and relevant history (`git log -L`,
`git log -S`). Read tests/README.md and, for e2e, tests/e2e/README.md first,
plus any scoped README next to the test data. Read tests/conftest.py before
judging any test touching global state: its autouse guards may be the real
isolation. When the test claims dependency-backed behavior (pyepics, p4p,
click, FastMCP, litellm), inspect the dependency source in `.venv` directly.

## Discovery

Keep discovery read-only and report evidence before editing. Use
`uv run pytest --collect-only -q <path>` and `uv run pytest --markers`; never
the full suite, e2e or a container run. For broad scope, run parallel
discovery lanes when available:

- core and packages (`src/osprey/`, `packages/osprey-connectors/`);
- subsystems with their own suites (`tests/connectors`, `tests/mcp_server`,
  `tests/bridges`, `tests/services/<svc>`, `tests/hooks`, `tests/va`);
- CLI, interfaces (including the vitest suites under tests/interfaces),
  deployment, templates, scripts, and docs guards;
- a cross-cutting pattern sweep, including test-only seams:
  `grep -rnE 'for tests|test seam|test hook|tests only|test-only|re-exported for tests' src/osprey packages --include='*.py'`.

Outside campaign mode, prefer a few high-confidence candidates over a large
speculative inventory. Hunt for the [junk patterns](#junk-patterns). Before
presenting candidates, check `origin/main` and recently merged PRs for work
that already moved them.

## Retention bar

Keep a test when it independently enforces a public API, protocol, config,
migration, storage, security, platform, default, prompt-byte, package, release,
or architecture contract. In OSPREY that includes:

- the hardware-write safety chain: human-approval hooks, approval stamps and
  session attribution, `permissions_ask` lists, write limits, verification,
  fail-closed paths, write posture — per connector where each connector
  implements it;
- the connector contract (`ControlSystemConnector`, `WriteOutcome`
  vocabulary, `sys.modules` alias shims that production imports still use);
- MCP tool schemas and server definitions; the CLI surface (verbs, flags,
  stdout purity on `result.stdout`);
- config keys and their docs parity; hook and prompt bytes; template
  scaffolding and render output, including byte-exact goldens;
- CI routing and repo guards: tests/deployment/test_ci_workflow_wiring.py,
  tests/infrastructure/test_import_time_audit.py, tests/docs guards,
  identity-literal and marker-placement guards;
- sanctioned isolation seams (the reset-seam list in tests/README.md §2) and
  the mypy `_static_conformance` seams in `bridges/*/ops.py`.

Also keep:

- call ordering when order is observable behavior;
- regressions with a credible failure mode;
- determinism self-comparisons when output could depend on dict, set, walk or
  time order, or when two different inputs must agree;
- source inspection when it is the cheapest independent guard: it fails when
  the contract changes (the user-facing key, byte, or path) and survives an
  identifier-only refactor, or it is a labelled backstop behind behavioral pins;
- a retained test that fails on the baseline: treat it as a possible product
  bug, reproduce it, and repair the owner rather than deleting it.

Static or slow is not a deletion reason. A test that resembles implementation
may still be the independent contract; prove otherwise before removing it.
A flaky test is deflaked robustly, never weakened or deleted for flakiness;
unit flakes are presumed leaks (the flaky-marker item of the tests/README.md §7
checklist).

## Candidate evidence

Record every field below before editing. A missing field means the candidate is
not ready for deletion:

- exact test node id (`tests/<area>/test_x.py::TestCls::test_name`);
- what failure it can actually detect;
- non-test callers of the covered production or support seam (grep `src/`
  and `packages/`; a seam with any production caller is not test-only);
- stronger remaining owner-boundary proof, or why no proof is needed;
- relevant history and the reason the test or seam exists;
- production or test-support deletion unlocked;
- CI routing and inventories touched (ci.yml path lists, the `frontend-js`
  job for vitest files, WHITELIST,
  scripts/benchmark/matrix_e2e_config.json, markers);
- risk and the focused validation command.

## Edit shape

Choose one coherent owner-boundary batch. Delete obsolete test-only exports,
globals, wrappers, and dead production paths instead of preserving aliases.
Remove a reset hook only when its global state is gone or a sanctioned seam or
autouse fixture already covers the isolation; otherwise keep it, or promote it
to a public seam next to the state (tests/README.md §2). Never replace it
with test code monkeypatching the module global. Move retained regressions to
their canonical owners.
Consolidate repeated assertions into one parametrized contract. Tighten weak
negatives to the exact exception type instead of deleting them. Migrate any
unscoped `patch("mod.subprocess.run")` you touch to `patch_subprocess()`.

Prefer net-negative production LOC. Do not add replacement tests that restate
the same implementation, and do not convert uncertain candidates into cleanup
to increase deletion counts.

When a test file is moved, renamed or deleted, update every registry that names
it: the unit-lane `--ignore` line in ci.yml, the e2e-tests `--ignore` list and
the dedicated e2e jobs that run the ignored files, the WHITELIST in
tests/infrastructure/test_import_time_audit.py, scripts/benchmark/matrix_e2e_config.json,
and the pyproject.toml marker list. Check `find tests -name <basename>.py`
first: a basename collision silently drops tests under xdist.

## Validation

Never edit source or tests, and never commit, while pytest or a gate is running
in the checkout. Run gates in the foreground. Run e2e only by path
(`uv run pytest tests/e2e/<file>`), never `-m e2e` (tests/e2e/README.md).

1. Run the smallest owner and sibling tests twice, serially then parallel:
   `uv run pytest tests/<area>/test_x.py`, then, for a unit area,
   `uv run pytest tests/<area> -m "not pty" -n 4 --dist loadgroup` with the
   unit lane's `--ignore` list from ci.yml (the files CI runs in their own
   jobs, such as tests/va/test_record_factory.py; run those by path). Run
   `tests/pty -m pty` serially. For e2e, run one file by path, serially or
   with `-n 4 --dist loadfile`. Neither run is evidence
   for the other. For vitest files, run `npm run test:js -- <path>`, then
   `npm run typecheck` and `npm run lint` (the `frontend-js` CI job).
2. For removed source greps or plan assertions, run the command that owns the
   real contract (the CLI verb, the render, the MCP tool call, or the owning
   `scripts/` check).
3. Run `uv run ruff format <paths>` and `uv run ruff check <paths> --fix`,
   then `git diff --check`.
4. Run the gates the diff demands: `uv run pytest tests/deployment/test_ci_workflow_wiring.py`
   and tests/infrastructure/test_import_time_audit.py for routing or file
   moves; `uv run python scripts/benchmark/check_e2e_coverage.py --check-lanes`
   for e2e changes; `uv run python scripts/mypy_gate.py`; for src/ or
   packages/ changes a NEW `changelog.d/<ref>.internal.md` and
   `uv run python scripts/changelog_fragments.py check --base origin/main`;
   then `./scripts/premerge_check.sh main --worktree`.
5. Inspect `git diff --numstat`; report src/, packages/, scripts/ and .github/
   separately from tests/ and `tests/_*.py` support.
6. After final audit edits, have the diff reviewed, with a code-review skill
   if one is installed.

## Landing and continuation

Commit, push, open a PR, or land only when authorized. Branch `test/<topic>`
(CONTRIBUTING.md), commits `test(<scope>): ...`. Keep audit ledgers and working
notes out of tracked files; the durable record is the tests and their docs.
Ship with `/osprey:contribute`. Mind the CI budget of three concurrent PR runs;
prefer `gh run rerun <id> --failed` to a re-push. The model-spending lanes run
on a pull request only under the `full-ci` label. Land one coherent PR at a
time; after landing, refresh from current `main` and rerun read-only discovery
for the next high-confidence batch.

## Handoff

Report:

- root cause and removed low-value categories;
- production owner simplifications;
- retained false positives and why they remain valuable;
- focused and full proof actually run;
- production versus test LOC;
- PR and merge state;
- named follow-ups.
