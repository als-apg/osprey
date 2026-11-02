# Osprey Testing Scripts

This directory contains testing and validation scripts for the Osprey Framework development workflow.

## Quick Reference

| Script | Purpose | Duration | When to Use |
|--------|---------|----------|-------------|
| `check.py quick` | The checks a commit needs | Minutes | Before every commit |
| `check.py full` | Every check, with the serial suites and the builds | The longest run | Before pushing |
| `check.py quick --changed origin/main` | The quick tier scoped to the branch's diff, plus the checks that judge a range | Scales with the diff | Before creating a PR |
| `check_config_keys.py` | Config-key resurrection guard | 2-5s | After touching a `config.yml.j2`, a preset, or config-reading code |
| `changelog_fragments.py` | Changelog-fragment gate and release fold | < 1s | After touching a gated directory; when cutting a release |
| `plugin_version.py` | Show, bump and check the plugin version | < 1s | After touching the plugin tree |
| `cli_tool_inventory.py` | CLI tool inventory of the pinned build | ~2 s plus an npm download | After bumping `_DEFAULT_CLAUDE_CLI_VERSION` |
| `bpm_polarity_picture.py` | Redraw the example facility's bpm-polarity logbook picture from its model and scenario | ~5 s | After changing the example's SR deck or wiring, the bpm-polarity scenario, or the BPM noise seeds |
| `va/capture_nominals.py` | Capture the served nominals golden from the VA container over Channel Access; `--out` writes, `--check` compares | ~2 min with the container boot | After a change to the served demo view or the VA's served values |

## Scripts

### check.py

**Purpose**: Answer "is this change clean" with one command. Every check is
declared once, by the code that owns it, in a `repo_checks.py` module beside
that code (`scripts/repo_checks.py` for the framework and the repository as a
whole), built from the records in `scripts/check_kit.py`. `scripts/check.py`
collects every such module; a laptop, an automated gate and each CI step run a
check from there, so they cannot disagree about what a check is.

**Usage**:

| Command | When | What it adds |
|---------|------|--------------|
| `uv run python scripts/check.py quick` | Before every commit | `--fix` lets ruff rewrite files first |
| `uv run python scripts/check.py full` | Before pushing | The serial suites (real terminal, wall clock, live Channel Access), the documentation build and the package build |
| `uv run python scripts/check.py quick --changed origin/main` | Before creating a PR | Scopes the tier to `origin/main...HEAD`, runs the checks that judge a range (changelog fragment and plugin version) and names the CI lanes the diff touches; with `--worktree` the tier and the range checks judge the working tree |
| `uv run python scripts/check.py run ID...` | To run one check as CI does | Execution flags after `--` reach pytest |
| `uv run python scripts/check.py list` | To see what a tier runs | `--tier quick` or `--tier full` narrows it, `--json` prints the declaration |

`--changed BASE --worktree` judges the working tree instead of `HEAD`, for the
runners and every range check alike, so staged, unstaged and untracked files
count. Test files, pytest node ids (`file.py::test`) and test directories after
`--` are added to what the range selects. `--base BASE` arms the checks that
judge a range without scoping the tier, so `full --base origin/main` judges the
whole tree and the range in one verdict.

**What it runs**: `uv run python scripts/check.py list` prints every check, its
kind, its tiers and its command.

**Provisioning**: before any check runs, two needs put the tree's pins in
place. The `claude-pin` need of the test sweep runs `pinned_claude.py` and puts
the directory whose `claude` reports the pinned version first on `PATH`; when
none does, the sweep is refused with the install command. The `node` need runs
the `js_deps_current.py` compare and runs `npm ci` only when `node_modules`
differs from `package-lock.json`.

**Requirements**:
- `uv sync --extra dev --extra docs` and graphviz (`dot`) on PATH (the full
  tier refuses the docs build and the link check without them; the dev extra
  carries actionlint and the shellcheck it runs)
- `gitleaks` on PATH for the secret scan, which runs whenever a base is given
- network access for the full tier's link check
- `uv sync --extra virtual-accelerator` on Linux x86_64 (the full tier refuses
  there without it; elsewhere the live Channel Access gate does not run)
- Node and npm on PATH (the TypeScript, ESLint, Vitest and dock-engine checks
  refuse without the Node toolchain)

**Exit codes**:
- `0`: Clean
- `1`: A check failed
- `2`: Refused: the command could not judge (a missing toolchain, a base that
  does not resolve, an unknown check)

The last line is the verdict. It names every check that failed, with up to five
failing tests of each failed test check and a count of the rest, and everything
that did not run here: a check with no base to judge, a check this machine
cannot host, a pytest run whose named paths the marker expression wholly
deselects (with the CI lane that runs them), and the CI lanes the command
never runs. A refusal for a missing toolchain names every check it could not
judge, with what to install, before any check runs.

---

### check_config_keys.py

**Purpose**: Stop deleted config keys from coming back, and stop live keys from
quietly losing their reader.

**What it does**: Renders the framework `config.yml.j2` template over its branch
matrix, resolves each shipped preset's `config:` block, collects every dotted key
the two produce, and checks each one against
`src/osprey/profiles/config_key_manifest.yml` — which records either the code
fragment that reads the key or the structural reason it has none. It also
re-checks the keys that were deliberately deleted (they must not reappear in the
rendered union, a preset `config:` override, or the loader's synthesized
defaults), the code sites that went with them, cross-preset parity, and the
manifest's own internal consistency. The script's module docstring is the
authoritative list of failure modes.

The manifest lives inside the package rather than beside this script because
`osprey config --defaults` renders its `default:` column at run time, and a
wheel ships only `src/osprey`.

**Usage**:
```bash
# Plain run (~2.4s) — works anywhere, including a shallow clone
uv run python scripts/check_config_keys.py

# What CI runs (~4.6s). Same checks, plus the back-test: every orphan-site
# regex must match at the recorded baseline commit AND not match on this
# branch, which is what proves it can actually fail. Extracts the baseline
# tree with `git archive`, so it needs full history — in a shallow clone it
# errors out rather than skipping.
uv run python scripts/check_config_keys.py --back-test

# Same, against a specific baseline instead of the manifest's recorded one
uv run python scripts/check_config_keys.py --back-test <commit>
```

**When to use**: After editing any `config.yml.j2`, any preset under
`src/osprey/profiles/presets/`, or code that reads configuration. Use the plain
form for a quick local check; use `--back-test` to reproduce CI exactly. CI runs
the `--back-test` form as a step of the `lint` job, which checks out with
`fetch-depth: 0`. `tests/scripts/test_config_key_guard.py` exercises the guard
from the unit-test lane as well, back-test included — but those cases skip
themselves when the baseline is unreachable, so that coverage is revocable by a
checkout-depth or marker change. The step's own `!cancelled()` guard is what
pins it down: it runs even when the lint steps before it fail.

**Exit codes**:
- `0`: No findings
- `1`: One or more failure records (each is printed with its key and location),
  or a hard error — notably an unreachable `--back-test` baseline, which raises
  a traceback rather than skipping

---

### changelog_fragments.py

**Purpose**: Make sure a change under a gated directory (`GATED_PREFIXES` in
the script; every verdict line names them) ships a changelog fragment, and fold the fragments into `CHANGELOG.md` when a release is cut.

**What it does**: `check` validates every fragment in `changelog.d/` (`README.md` is skipped) — the filename
grammar `<name>.<type>.md` and the body rules — and then compares the branch
against its base: a change under a gated directory (`GATED_PREFIXES` in the
script; every verdict line names them) must add a fragment, and
`## [Unreleased]` in `CHANGELOG.md` must not be edited by hand. `apply` inserts
each fragment as a bullet under its `### <Type>` heading in `## [Unreleased]`
and deletes the fragment files.

**Usage**:
```bash
# What the `lint` CI job and `scripts/check.py`'s changelog check run;
# --worktree judges staged, unstaged and untracked files too
uv run python scripts/changelog_fragments.py check --base origin/main

# The release fold — run once while cutting a release
uv run python scripts/changelog_fragments.py apply
```

**When to use**: `check` after touching a gated directory, though CI and
`scripts/check.py quick --changed origin/main` already run it for you. `apply` only when cutting
a release.

**Exit codes**:
- `0`: No findings
- `1`: Something you can fix — a missing fragment, a malformed fragment, a
  hand-written entry in `## [Unreleased]`, or a deleted fragment
- `2`: Environment problem — the base ref was not found, or the clone is too
  shallow to find a common ancestor

See `changelog.d/README.md` for the fragment format and the list of types.

---

### plugin_version.py

**Purpose**: Keep the CalVer version shared by the two plugin manifests
(`plugins/osprey/.claude-plugin/plugin.json` and
`plugins/osprey/.codex-plugin/plugin.json`) in step, and advanced whenever the
plugin tree changes.

**What it does**: `show` prints the shared version and refuses when the two
manifests disagree. `bump` advances `YYYY.M.N` in both (`--set` writes a given
version). `check --base <ref>` compares the branch against what it merges into:
when `plugins/`, `.claude-plugin/` or `.agents/plugins/` changed since the merge
base, the head version must differ from every version the base tip carries. A
plain branch name such as `main` is read as `origin/main` first, as every
range check in this directory reads it.

**Usage**:
```bash
uv run python scripts/plugin_version.py show
uv run python scripts/plugin_version.py bump

# What the `lint` CI job and `scripts/check.py`'s plugin-version check run;
# --worktree judges staged, unstaged and untracked files too
uv run python scripts/plugin_version.py check --base origin/main
```

**When to use**: `bump` after touching the plugin tree; CI and
`scripts/check.py quick --changed origin/main` run `check` for you.

**Exit codes** (`check`):
- `0`: The plugin tree is untouched, new, or carries a new version
- `1`: The plugin tree changed without a version bump, or the manifests cannot
  be read or disagree
- `2`: Environment problem — the base ref was not found, or the clone is too
  shallow to find a common ancestor

---

### cli_tool_inventory.py

**Purpose**: Record the built-in tools the pinned Claude Code build has, so the
tool names OSPREY denies or passes through can be checked against the real build.

**What it does**: `--write` installs the pinned build
(`_DEFAULT_CLAUDE_CLI_VERSION`), starts it under a clean, deny-free environment
plus the environment OSPREY gives every Claude Code process it starts, and
records the tools it lists in its `system/init` message as
`tests/fixtures/cli_tool_inventory/<version>.json`, deleting any other version's
file. The build is probed twice: offline (`tools`), and with remote
configuration reachable (`remote_config_tools`), which adds flag-gated tools
such as `Monitor`. `--check` probes the same build and fails when it has no
file or a recorded name is no longer listed under the same probe. A missing
name can reject a real tool, but a recorded name is never one the build lacks.

**Usage**:
```bash
uv run python scripts/cli_tool_inventory.py --write   # after bumping the pin
uv run python scripts/cli_tool_inventory.py --check   # what the static-checks CI job runs
```

---

### bpm_polarity_picture.py

**Purpose**: Redraw the picture the example facility's bpm-polarity scenario
attaches to its logbook entry, so it stays in step with the model it shows.

---

**What it does**: Reads the example facility's simulator view: `scenarios.json`
(the scenario's polarity faults), `variables.json` (the model's deck, settings
and wiring), `seeds.json` (each monitor's absolute noise) and
`decks/<model>.json`. It draws one panel per monitor of the lattice segment
holding the reversed monitor, the reading against a kick of the corrector
upstream of it, and writes
`src/osprey/templates/facilities/example/scenarios/bpm-polarity/plots/corrector_bump_test.png`.
Without `--simulator` it renders the view in-process first. A fixed noise seed
makes a re-run write the same bytes on one host and the same picture on any.

---

**Usage**:
```bash
uv run python scripts/bpm_polarity_picture.py
uv run python scripts/bpm_polarity_picture.py --simulator build/data/simulator
```

---

### va/capture_nominals.py

**Purpose**: Write `tests/facility/golden/nominal_va.json`, the served nominals
`tests/va/e2e/test_substrate_parity.py` holds the live VA container to.

---

**What it does**: Stages the project the live-container suite serves, applies
the `still` scenario on top of the suite's scenarios and asserts no captured
address moves, boots the VA image (`osprey-va-full:latest`, built by
`scripts/va/build_and_boot_check.sh`, or the one `OSPREY_VA_E2E_IMAGE` names)
and reads every served address over Channel Access until two snapshots agree.
A bool or enum is written as its label, a string as its text. The suite's
synthetic string channel and every waveform are left out. `--check` recaptures
and prints every header key and address that differs, exiting 1 on any.
Needs Docker.

---

**Usage**:
```bash
uv run python scripts/va/capture_nominals.py --out tests/facility/golden/nominal_va.json
uv run python scripts/va/capture_nominals.py --check tests/facility/golden/nominal_va.json
```

---

## Development Workflow

### Recommended Testing Flow

```bash
# 1. Make changes
vim src/osprey/some_file.py

# 2. Quick check before commit
uv run python scripts/check.py quick

# 3. Commit if passed
git add .
git commit -m "feat: add new feature"

# 4. Full check before push
uv run python scripts/check.py full

# 5. Push if passed
git push origin feature/my-feature

# 6. The quick tier scoped to the branch before the PR
uv run python scripts/check.py quick --changed origin/main

# 7. Create PR if passed
gh pr create
```

### Pre-commit Hook Integration

For automatic validation on every commit, install pre-commit hooks:

```bash
# One-time setup
pre-commit install

# Now pre-commit runs automatically on git commit
# Manual trigger:
pre-commit run --all-files
```

## Troubleshooting

### "Module not found" errors

Install dependencies:
```bash
uv sync --extra dev --extra docs
# or
pip install -e ".[dev,docs]"
```

### Tests pass locally but fail in CI

Common causes:
- Different Python version (test on 3.11 and 3.12)
- Different OS (test on both Ubuntu and macOS if possible)
- Missing dependencies in `pyproject.toml`

### Documentation build fails

```bash
# Clean and rebuild
cd docs
make clean
make html

# Check for missing dependencies
uv sync --extra docs
# or
pip install -e ".[docs]"
```

## CI/CD Integration

The `lint`, `test`, `docs`, `package` and `frontend-js` jobs of
`.github/workflows/ci.yml` call `scripts/check.py run <id>`, so a local tier and
CI run each check from the same declaration:

- `.github/workflows/ci.yml`: Main CI pipeline
- `.github/workflows/release.yml`: Release automation
- `.pre-commit-config.yaml`: Pre-commit hooks

See `docs/source/contributing/development-setup.rst` for the CI and testing guide.

## Contributing

If you modify these scripts:

1. Test them thoroughly on both macOS and Linux
2. Update this README
3. Update `docs/source/contributing/development-setup.rst`
4. Ensure exit codes are correct (0 = success, 1 = failure)
5. Add helpful error messages

## See Also

- [Contribution Workflow](../docs/source/contributing/workflow.rst)
- [Development Setup](../docs/source/contributing/development-setup.rst)
