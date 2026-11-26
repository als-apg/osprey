---
name: release
description: >
  Guides a maintainer through cutting an OSPREY release on the GitHub Flow
  workflow: run the two advisory drift sweeps, test in a clean environment,
  refresh the doc screenshots, re-record and review every docs video, land the
  release-notes PR, tag the merge commit,
  push the tag, verify the automated PyPI publish. Use when someone says "create
  a release", "bump the version", "cut v2026.X.Y", "publish to PyPI", "tag a
  release", or asks about the release process. Composes with `/osprey:contribute`
  for the notes PR and with `/osprey:housekeeping` and `/osprey:doc-sync` for the
  sweeps. Versions follow CalVer (vYYYY.M.P), with PEP 440 pre-release tags
  (vYYYY.M.PbN / rcN) for public betas, and the source of truth is the git tag —
  Hatch derives the version from it, so there is no version literal to bump.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# OSPREY Release Workflow

This skill cuts a properly versioned OSPREY release. Releases are CalVer tags
(`vYYYY.M.P`) on `main`; the PyPI publish runs automatically when the tag is
pushed.

The shape is:

1. Verify the working state and decide on the version number.
2. Run the two **advisory drift sweeps**, housekeeping and doc-sync, and land
   their accepted fixes as ordinary PRs before the release PR.
3. Test in a clean environment, refresh the doc screenshots, and **re-record
   every docs video and watch each take**: a pre-release check, run by hand,
   on every release.
4. Open a **release-notes PR**, labeled `full-ci`, carrying the CHANGELOG fold
   and rotation, the RELEASE_NOTES title, the re-captured screenshots and the
   video manifest and posters (no
   direct push to `main`; branch protection rejects it).
5. Merge the PR to `main`.
6. Tag the merge commit and push the tag. **This is what sets the version.**
7. Verify the automated GitHub Actions workflow publishes successfully.

For the PR mechanics in step 4, defer to `/osprey:contribute`.

## Versioning: CalVer

OSPREY uses **CalVer**: `YYYY.M.P` where:

- `YYYY` — four-digit year of the release
- `M` — calendar month, no zero-padding (e.g., `5`, not `05`)
- `P` — patch counter within the month, starting at `0`

Examples: `2026.5.0`, `2026.5.1` (patch within May 2026), `2026.6.0` (next
month). When the year or month rolls over, `P` resets to `0`.

### Pre-releases: beta and RC tags

A public beta is an ordinary release whose tag carries a PEP 440 pre-release
segment on the patch component: `v2026.9.0b1`, then `b2` if needed, then
`rc1` for a feature freeze, then the plain `v2026.9.0` (or the next month's
`.0` if the final slips a month — CalVer tells the truth either way). The
segment is appended directly — never `-beta`, `-rc1`, or a dot — because the
segment must survive two independent normalizers unchanged: `hatch-vcs`
naming the wheel, and the release workflow comparing that wheel to the tag.
PEP 440 orders the chain correctly: `b1 < b2 < rc1 < final`.

Every step in this skill applies to a pre-release tag unchanged. The
differences, each load-bearing:

- **pip hides it by default.** Plain `pip install osprey-framework` keeps
  resolving to the last stable release; the beta is reached only by
  `--pre` or an exact `==` pin. That opt-in IS the beta channel — nothing
  else has to be built for it.
- **Each component floor names a stable release that is on PyPI and whose
  tag is reachable from `main`.** `osprey-connectors` and `osprey-facility`
  each have their own calendar stream (`osprey-connectors-vYYYY.M.P`,
  `osprey-facility-vYYYY.M.P`), both published by `release-package.yml` from
  the PyPI environments `pypi-connectors` and `pypi-facility`; the framework
  declares only `osprey-connectors>=FLOOR` and `osprey-facility>=FLOOR` and
  never a ceiling. (While the root `pyproject.toml` lists `osprey-facility`
  with no specifier there is no facility floor, and its floor lane stands
  down.) A floor above the last reachable tag of its stream breaks every
  paired wheel install and `osprey up --dev` (dev wheels build as
  `<last tag>.postN`), and a floor nothing stable satisfies makes pip *and*
  uv (default policy) silently take a pre-release — so the order is always:
  component release first, floor raise in a following PR that contains the
  release's tag commit. `scripts/ci/compat_matrix.py --package <connectors|facility>`
  refuses a floor whose tag is not reachable. A beta framework needs no
  component beta: `OSPREY_PIP_PRE` resolves the whole deps layer with
  `--prerelease=allow`, which takes the newest release of every stream — so
  publish a component pre-release only when every beta deployment should
  absorb it. Raise a floor only when the framework starts needing something a
  newer stable release carries, and run `uv lock --check` after that edit
  (see "Releasing a component package" below). `osprey-panels` is not published and
  has no floor: the framework does not depend on it.
- **The install command goes in the release body.** The installation page
  carries it under *Pre-releases* (`uv tool install --prerelease allow
  "osprey-framework==YYYY.M.PbN"`); repeat the exact command at the top of the
  GitHub Release body, because a reader who follows the stable instructions
  from the pre-release's own docs directory gets the last stable release and
  not the beta.
- **The GitHub Release is marked pre-release automatically.** `release.yml`
  classifies the tag (exactly `X.Y.Z` = stable, anything else = pre-release)
  and sets the flag, so the beta never shows as "Latest".
- **Docs publish into the tag's own directory.** `docs.yml` deploys a
  pre-release tag to `/vX.Y.Z{a|b|rc}N/` with its own switcher entry; the site
  root and the switcher's preferred entry stay on the last stable release.
  Nothing to do here — but it is why the release body must carry the install
  command: that directory is where beta testers land.
- **The version module keeps the segment.** A clean checkout of the tag
  reports `is_release()` true and pins `osprey-framework==YYYY.M.PbN`;
  `tests/test_version.py::TestPreReleaseChannel` pins this. `base_version`
  anywhere in new code is a bug factory here: it silently drops the segment
  and manufactures pins to versions that do not exist.

Say "public beta" in the `RELEASE_NOTES.md` tagline and the GitHub Release
body; the tooling marks the channel, the prose sets the expectation.

## The Source of Truth

**The git tag is the version.** There is no version literal anywhere in the
tree to edit for the framework: `hatch-vcs` derives the package version from
`git describe` at build time, and `osprey.version` resolves it at runtime.
Tagging `v2026.7.0` *is* the act of setting the version to `2026.7.0`.

Between releases the version reports its distance from the last tag
(`2026.6.2.post783+g83fda5e60`), which is how a development build is
distinguishable from the release it descends from.

### The component packages have their own version streams

Two workspace packages — `osprey-connectors` and `osprey-facility` — are
versioned the same way (CalVer, derived from a tag, no
literal) but each from its **own** tags, `<dist>-vYYYY.M.P` (`bN` / `rcN` for
pre-releases), cut when that package changes rather than with every framework
tag. Each describe glob (`<dist>-v2[0-9][0-9][0-9].*`) reads only its own
stream's calendar tags — the connectors one also skips the retired
`osprey-connectors-v0.1.0` — and the framework's `v[0-9]*` glob never reads a
component tag. `scripts/ci/release_streams.py` is the table of the two
streams: tag prefix, environment, changelog, required CI lanes.

Connectors and facility releases stay backward-compatible with every
supported framework release, enforced in CI by the `connectors-compat` /
`framework-at-connectors-floor` and `facility-compat` /
`framework-at-facility-floor` lanes, which is why the framework declares only
a floor on each.

`osprey-panels` is a workspace package that is **not published**: it has no
tags, no PyPI project, no changelog and no floor, and no release step in this
skill applies to it. The `osprey-panels-standalone` lane tests it on every
PR.

A checkout with no calendar tag of a stream reachable builds that package as
`0.0.postN+g…`, which the framework's floor refuses — `osprey up --dev` then
stops before staging and names the tag glob. Fetch tags (`git fetch --tags`)
before building; a fork or mirror must sync the `osprey-connectors-v*` and
`osprey-facility-v*` tags, not only the framework's `v*` ones.

The one version literal that does exist is the **plugin's**: the two manifests
under `plugins/osprey/` carry their own CalVer, kept in sync by
`scripts/plugin_version.py`, and CI refuses any PR that changes the plugin
tree without bumping it. By the time a release is cut, that bump has already
happened on whichever PR touched the plugin; the release only checks it (step 4).

| File | Purpose | Updated by |
| --- | --- | --- |
| `RELEASE_NOTES.md` | First-line title with the release version, then the theme | This skill |
| `CHANGELOG.md` | Fold `changelog.d/` fragments into `## [Unreleased]` (`changelog_fragments.py apply`), then rotate it to `## [YYYY.M.P] - YYYY-MM-DD` | This skill |
| `changelog.d/` | Fragments, one per PR; `apply` deletes them | This skill |
| `plugins/osprey/.claude-plugin/plugin.json`, `.codex-plugin/plugin.json` | Plugin CalVer | `scripts/plugin_version.py bump`, on the PR that changed the plugin |
| `pyproject.toml` | `[tool.hatch.version] source = "vcs"` | **Do not edit** |
| `src/osprey/_version.py` | Build-time stamp, gitignored | **Never commit** |

The release.yml verify step builds the package and compares the *built wheel's*
version to the pushed tag; if these disagree, the publish fails. They disagree
when the tag is not on the commit being built, or when the checkout is shallow
(which yields `0.0.postN+g…` rather than failing outright).

---

## Step 0: Read the CHANGELOG and decide the theme

Open `CHANGELOG.md` and read the `## [Unreleased]` section together with the
pending fragments in `changelog.d/`; both are this release's content. Then
answer three questions before doing anything else:

1. **What is this release about?** Pick a short theme (e.g., "plan
   authoring & branch-protection enforcement"). It goes into the release
   title, the `RELEASE_NOTES.md` tagline, and the GitHub Release body.
2. **What is the version number?** Apply the CalVer rules above. Patch bump
   for fixes, month bump for feature batches, year bump only at January.
3. **Are there breaking changes?** Check the `### Changed` and `### Removed`
   sections. If user-facing API changed, the release should call it out
   prominently and (if it would surprise users) include a migration note.

Also look for **phantom sections**: a `## [YYYY.M.P]` heading in the CHANGELOG
with no matching `v*` tag on `origin` (`git ls-remote --tags origin`). They
appear when a release was prepared and never tagged. Their entries are still
unreleased and belong in this release's section; note them now so the rotation
in Step 4 folds them in.

Confirm theme + version + breaking-changes status with the maintainer before
proceeding.

## Step 1: The advisory sweeps

A tag is the moment every stale sentence in the tree goes public. Two skills
exist to find those sentences first, and a release is the one time they must
run. Both are advice, not gates: each writes a report, the maintainer rules on
every finding, and the accepted fixes land as ordinary PRs **before** the
release-notes PR so that the release ships them.

1. **`/osprey:housekeeping`** covers what OSPREY says outside the doc pages:
   the generated project, shipped prompts, runtime messages, pinned versions.
2. **`/osprey:doc-sync`** covers the doc pages themselves, and applies the
   doc-side fixes the maintainer accepts into a worktree ready for
   `/osprey:contribute`.

Run housekeeping first; it hands doc-page items to doc-sync. Both run from the
main checkout, and both are cheap enough to run on any release, so do not skip
them because the last one was clean. A release cut without reading the two
reports is a release that ships known drift.

## Step 2: Pre-release testing in a clean environment

Your working environment may carry packages the project does not declare. A
fresh environment catches missing dependencies before users do — and on a
release it is synced with every extra the test tree reaches for, not just
`dev`. A module whose extra is absent skips itself, and a skipped module is a
green that proved nothing.

### Provision the host first

Do this before the first test runs, not after reading the first skip list. A
test whose prerequisite is absent skips itself, so an unprovisioned host
returns a green run that left hundreds of tests out, and each prerequisite
found afterwards costs another full run.

**Pick the host.** The run needs a container runtime, about 20 idle cores and
most of a day. A laptop that is also running agent sessions does not finish:
the unit sweep hits its 90-minute bound with no failure reported. Use a
workstation or a build host, in a fresh clone, and tell the maintainer which
host before starting. On a host other runs share, give every service the
release run starts a port and a name of its own.

**Then provision everything in this table, and prove each line**, before the
unit sweep:

| Prerequisite | Tests it unlocks | Provision | Prove |
| --- | --- | --- | --- |
| Vendored front-end files | browser tests, every stack capture in Step 3 | `uv run osprey vendor fetch` | `uv run osprey vendor verify` |
| Playwright browsers | browser tests, including the WebKit cases | `uv run playwright install chromium webkit`; on macOS WebKit needs a logged-in desktop session, so run those cases there or in a Linux container | the browser tests run instead of skipping |
| QMD sidecar images | QMD sidecar and models-mount tests | both builds of the `qmd-sidecar-e2e` job in `ci.yml` | `docker image inspect` on both tags |
| VA live containers | `tests/va/e2e` | the image builds and `OSPREY_VA_E2E_ENABLE=1` of the `va-live-e2e` job | the directory collects and runs |
| Ollama with the models the tests name | Ollama-backed unit tests, the proxy e2e modules | install Ollama, `ollama pull` each model a skip or failure names, `OSPREY_E2E_OLLAMA=1` for the e2e lane | `curl -s localhost:11434/api/tags` lists them |
| Facility lane stand-in | `tests/cli/test_als_lane.py` | `OSPREY_ALS_LANE_STAND_IN=tests/fixtures/mml/nsls2` | the module runs |
| MML checkout, `als-ontology` checkout | `tests/facility`, the TTL seeder tests | clone them; `OSPREY_MML_MACHINE_ROOT=<MML>/machine/<facility>` | the modules run |
| Open-file limit | the whole sweep: one worker holds about 500 descriptors once everything is provisioned | `ulimit -n 8192` in the shell that runs the tests (macOS gives 256) | `ulimit -n` |
| pvAccess client | the pvAccess connector tests | run those modules as an overlay, `uv run --frozen --with pvapy pytest ...`; installed into the venv it fails the shipped-library review test | the modules run |
| `psql` client | the role tests in `tests/e2e/test_ariel_search.py` | any PostgreSQL client on `PATH` | `psql --version` |
| ARIEL Postgres, seeded | the ARIEL scenario tests, the retrieval half of the logbook delegation test | a pgvector Postgres on a port of its own, `OSPREY_ARIEL_DB_URI`, then `osprey sim apply nominal --yes` in a demo project pointed at it. Never the default port on a shared host: the tests purge and reseed whatever they reach | `psql "$OSPREY_ARIEL_DB_URI" -c 'select count(*) from enhanced_entries'` |
| A Linux container | the Linux-only tests, on a macOS host | run their modules in a `python:3.13` container with the tree mounted | they run instead of skipping |
| JS toolchain check | `app-check-js` | `uv run python scripts/check.py run app-check-js` | it reports `ran` |
| MATLAB with the AT toolbox | MML export parity | a licensed MATLAB on `PATH` | `matlab -batch "disp(1)"` |
| Keys a docs video names in `needs_env` | the video takes in Step 3 | export each one on the host | `python -m docs.screenshots list`, then check each named key is set |

The CI lane named in a row is the reference: copy its build commands and
environment out of `.github/workflows/ci.yml`, never from memory. When a
release run meets a skip this table does not cover, add the row in the same
release.

**What may still skip.** Only a test that cannot run on the host at all (a
Linux-only syscall on macOS) or one that is opt-in by design (a manual spike,
a ratchet with nothing to check). Every such skip is listed for the
maintainer with the CI lane that runs it, or with "no lane runs this". A skip
for something that could have been installed is not an acceptable skip.

### Run the suites

```bash
export UV_PROJECT_ENVIRONMENT=.venv-release-test
uv sync --extra dev --extra teams --extra gchat --extra virtual-accelerator

# Pick the provider CI picks: one gateway, no fallbacks. A run that names no
# provider is refused before the first test, and a stray key from another
# provider sends tests somewhere CI never goes.
export OSPREY_E2E_PROVIDER=als-apg
export ALS_APG_API_KEY=... ALS_APG_BASE_URL=...
unset CBORG_API_KEY OPENAI_API_KEY GOOGLE_API_KEY ANTHROPIC_API_KEY

# Unit tests: the parallel sweep, then the suites that run alone. The sweep
# does not include the second line's suites.
uv run python scripts/check.py run pytest -- -n 8 -q
uv run python scripts/check.py run pytest-pty pytest-wallclock pytest-wallclock-unit

rm -rf .venv-release-test
```

`check.py` refuses `-rs`, so it cannot print skip reasons. To read them, take
the pytest command `check.py` echoes on its `check: pytest:` line and run it
once more with `-rs` appended. `canopy-contract` is refused where CANOPY is not
installed, and it takes any check named in the same command down with it; it
has its own CI lane, so leave it out of the local run.

The e2e half is the bulk lane, run exactly as CI runs it — same parallelism,
same exclusions. Copy the invocation out of the `e2e-tests` job rather than
retyping the `--ignore` list:

```bash
sed -n '/^  e2e-tests:/,/^  benchmarks:/p' .github/workflows/ci.yml \
  | grep -e 'pytest tests/e2e/' -e '--ignore='
```

It runs `pytest tests/e2e/ -v --tb=short --durations=30 -n 4 --dist loadfile`
with that ignore list.

**The ignored modules are not untested.** Each has a dedicated CI lane of its
own, because each binds a fixed host port, leaves a fixed-name container
behind, or needs an extra the bulk lane does not install; CI gives them a
runner each for exactly that reason. This local run does not reach them, and
is not meant to. The `full-ci` run below is their gate.

Read the skips, not just the exit code. A missing key, container runtime or
backend turns a lane green having run nothing, and `tests/e2e/README.md` lists
what each one needs. A skip for want of something installable sends you back
to the provisioning table, not to the release notes.

The bulk lane shares the model endpoint with every agentic CI lane in flight.
Start it when at most two PR runs are running
(`gh run list --workflow ci.yml --status in_progress`); under contention its
tests hang to their caps instead of failing.

Also trigger the on-demand benchmark lane against the release tree, which no
PR runs: `gh workflow run ci.yml -f run_benchmarks=true`.

Budget most of a day. The bulk lane alone is 20-25 minutes in CI at four-way
parallelism, and the Docker stacks around it are slower on a laptop than on a
runner.

Any failure, and any unexplained skip, stops the release. Fix forward, then
re-run.

### The full-ci run is a hard requirement

The CI lanes the LANES table in `scripts/repo_checks.py` marks `why="models"`,
which `release.yml` requires by name in `REQUIRED_LANES`, drive real agent sessions against a live model
endpoint, and they run only on a pull request carrying the `full-ci` label.
They are the only proof the agentic paths work on the exact tree being tagged:
`release.yml` runs no tests and never reads the tagged commit's CI status.

So the release-notes PR in Step 4 carries the label from the moment it opens:

```bash
gh pr edit <number> --add-label full-ci
```

The label persists, so every refold push re-runs them; the run that
counts is the one for the commit that actually merges. If the label ever comes
off, put it back and let the lanes run again.

`release.yml` refuses a tag it cannot find such a run for, so this is enforced
rather than remembered. Check it before tagging anyway — the same answer costs
two commands here and a deleted tag there:

```bash
gh run list --workflow ci.yml --commit <sha> --limit 1
gh run view <run-id> --json jobs --jq '.jobs[] | [.conclusion, .name] | @tsv'
```

Check the component floors the same way. `release.yml`'s
`verify-component-floors` job refuses a tag while the framework imports
connectors or facility code its floor does not carry, even behind a stand-in,
since every install already on the floor would run with that feature switched
off:

```bash
SCAN=packages/osprey-connectors/tests/osprey_connectors_tests/compat/scan_framework_imports.py
uv run --no-project \
  --with "osprey-connectors==$(python3 scripts/ci/compat_matrix.py --package connectors floor)" \
  python -I "$SCAN" --release --target osprey_connectors \
  src/osprey packages/pyaml-cs-osprey/src/pyaml_cs_osprey
# Only while the root pyproject.toml carries an osprey-facility>= floor
# (`floor` prints nothing and says why on stderr while it does not):
uv run --no-project \
  --with "osprey-facility==$(python3 scripts/ci/compat_matrix.py --package facility floor)" \
  python -I "$SCAN" --release --target osprey_facility src/osprey
```

Anything it lists means: release that package and raise its floor first (see
"Releasing a component package" below).

**Skipped is not passed.** `All CI Checks Passed` cannot tell a lane that ran
and won from a lane whose `if:` never fired, so an unlabeled release PR merges
fully green with every agentic lane skipped. The conclusion has to be read per
lane.

Any red stops the release. Fix forward; never deselect a test or mark one
flaky to get a tag out. This holds for a beta exactly as it holds for a final
release — a beta is what users install.

## Step 3: Refresh the doc screenshots

The published docs embed committed PNGs, and each caption names the OSPREY
version its image was captured with. Nothing refreshes them automatically:
there is no CI job and no release step, so they age quietly, and a release is
where that staleness becomes public.

Read `docs/source/_static/screenshots/manifest.json`: every entry carries the
version and timestamp of its last capture. Compare it against the UI work in
this release. If a screen shown in the docs changed, re-capture it now, so the
images and their captions ship with the version being released.

```bash
python -m docs.screenshots list      # every recipe, its kind, its output files
cd docs && make screenshots          # all container-free recipes — no containers, no agent
```

Two recipes are opt-in because they cost more:

- `ariel` and `control_target_popover` need a container runtime and free host
  ports 26000–26999, 27064 and 27075 (the capture stack's own block) —
  `make screenshots SCREENSHOTOPTS=--stack`.
- `web_terminal_hero` drives a live agent session on that stack —
  `python -m docs.screenshots --agentic --only web_terminal_hero`. It spends
  real subscription budget, so re-capture it when the Web Terminal's appearance
  has actually changed, not on every release.

### Pre-release check: re-record and review the docs videos

**Every release, run by hand.** The docs embed recorded web-terminal sessions,
the landing-page demo and one per feature page (`python -m docs.screenshots
list` names them all), and a UI change can make any of them wrong without
anyone noticing. So the maintainer re-records all of them before every release,
whether or not the UI looks changed, and watches every take. It is never a CI
job and never skipped: a release whose videos were not watched does not get
tagged. A beta is a release here too.

Each take builds a fresh demo stack and runs a live agent, so it
needs a container runtime, the provider keys the run uses (the
`OSPREY_DOCSHOTS_PROVIDER` override when the default provider's key is over
budget), any key a video names in `needs_env`,
and only one stack on the host at a time. It spends real budget: minutes per
take.

```bash
python -m docs.screenshots list                         # every registered video and its page
uv run make -C docs doc-videos                          # every video, both themes, a fresh stack per take
uv run python -m docs.screenshots upload-check          # takes vs manifest, length and size warnings
```

To redo one take: `uv run make -C docs doc-videos VIDEOOPTS="--video <name>
--theme dark"`. A video skipped for a missing key prints one line; record it
before going on, never ship it stale.

**Before recording, read each video's `layout`** in
`docs/screenshots/videos.py` and each recipe's surface. The operator chat is
what the docs show; a take or screenshot of the Claude Code terminal is wrong
however long its storyboard has asked for it. Fix the entry first, and agree a
changed layout with the maintainer before recording.

**Watch every take, both themes, start to finish** (`docs/demo-video/*.mp4`).
Check that the dialogs, captions and panels shown match this release's UI,
that nothing covers what the caption points at, and that no stray notice (a
CLI update banner, an error toast) is on screen. A take outside its length
window is a warning, not a failure: judge it by watching. A take that shows the
wrong thing is re-recorded; when the storyboard itself no longer fits the UI,
fix its entry in `docs/screenshots/videos.py` first.

Then publish to the version being released (each OSPREY version has its own
`docs-media-vYYYY.M.P` release):

```bash
uv run make -C docs upload-doc-videos VERSION=YYYY.M.P  # uploads the new takes, names the release in the manifest
```

The upload names the release in `docs/source/_static/demo/manifest.json`; that
manifest and the posters ride along in the release-notes PR below, and the
MP4s are never committed. A new video is agreed with the maintainer as a
written description before any storyboard code: the view, the active sidebar
entry and open tiles (its `layout`), the prompts, the beats and the captions.
Then it is one `DocVideo` entry in `docs/screenshots/videos.py` plus the embed
snippet, both documented in `docs/README.md`.

The framework itself (environments, provenance, and why it is capture-only and
never a CI gate) is documented in the contributing guide under "Refreshing
documentation screenshots". Whatever changed (the PNGs and the updated
`manifest.json`) rides along in the release-notes PR below.

## Step 4: Release-notes PR

Release-notes commits cannot be pushed directly to `main`; branch protection
rejects it. Open a PR instead.

```bash
git checkout main && git pull --ff-only origin main
git checkout -b release/vYYYY.M.P
```

First fold the fragments in, so the rotation below has the full section to
rotate:

```bash
uv run python scripts/changelog_fragments.py apply
```

This inserts each fragment under its `### <Type>` heading in `## [Unreleased]`
and deletes the fragment files. Show the maintainer the resulting
`CHANGELOG.md` diff before continuing.

Then repair the section by hand, because `apply` only appends:

- **Merge duplicate `### <Type>` headings.** Fragments folded at different
  times leave several `### Changed` (or `### Fixed`, `### Added`) headings
  under `[Unreleased]`; one heading per type, entries concatenated.
- **Leave phantom sections where they are.** The changelog gate recognizes
  the release rotation by the `## [` heading count *growing*; folding a
  phantom heading away shrinks it, and the gate then reads the whole PR as
  illegal hand-editing (found live on the v2026.9.0b1 release PR). Fold the
  phantoms found in Step 0 into the released section in a **follow-up
  CHANGELOG-only PR** after this one merges and before the tag: that shape
  passes as a correction because it never touches `[Unreleased]`.

Then **condense the folded section — the maintainer's editorial pass.** The
fragments were each written for their own PR's reviewer; the released section
is read months later by someone deciding what changed and whether to upgrade.
The raw fold is the wrong document for that reader:

- **Collapse supersession chains.** A surface added, then reworked, then
  renamed in the same span is one entry describing where it landed, not three
  describing states that never shipped.
- **Merge per-PR snippets about one surface** into a single entry.
- **Drop what the later reader cannot act on** — churn on internals that were
  also removed in the span, fixes to bugs introduced in the same span.
- **Never drop or soften** breaking changes, migration notes, or `Security`
  entries.

This is judgment, not mechanics: propose the condensed diff and have the
maintainer review it before the PR merges. Sequence it as the **second-to-last
step before the release** — after the last content PR has landed and the final
fold has run — or the next batch's refold redoes it. On a held release PR
(see "Holding the release open") that means: final refold first, then
condense, then merge.

There is **no version literal to edit**; the tag in Step 6 sets the version.
This PR carries only the human-facing notes. Show the maintainer each diff
before applying:

| File | Change |
| --- | --- |
| `RELEASE_NOTES.md` | First line: `# Osprey Framework - Latest Release (vYYYY.M.P)` followed by the theme tagline |
| `CHANGELOG.md` | After the fold and repair, convert `## [Unreleased]` to `## [YYYY.M.P] - YYYY-MM-DD`; insert a fresh empty `## [Unreleased]` above it |
| `changelog.d/` | Fragment files deleted by `apply`; only `README.md` remains |
| `docs/source/_static/screenshots/` | Any images re-captured in Step 3, plus the updated `manifest.json` |
| `docs/source/_static/demo/` | Every video's two posters and `manifest.json`, re-recorded in Step 3, naming the `docs-media-vYYYY.M.P` release the upload created |

Stage the fold and the rotation together: `git add -A changelog.d/ CHANGELOG.md`
(pathspec-scoped, so the fragment deletions are included).

Then run a consistency check; every line should mention the same version, no
fragment should be left behind, and the plugin's own version should be current:

```bash
echo "=== VERSION CONSISTENCY CHECK ==="
echo "RELEASE_NOTES:  $(head -1 RELEASE_NOTES.md)"
echo "CHANGELOG.md:   $(grep -m1 '^## \[' CHANGELOG.md)"
echo "changelog.d/:   $(ls changelog.d | grep -vc '^README.md$') fragment(s) on disk (must be 0)"
echo "staged:         $(git diff --cached --name-only -- changelog.d/ CHANGELOG.md | tr '\n' ' ')"
echo "plugin:         $(python scripts/plugin_version.py show)"
echo "plugin changed: $(git diff --name-only $(git describe --tags --abbrev=0 --match 'v*')..HEAD -- plugins/ | wc -l | tr -d ' ') file(s) since last tag"
```

If the plugin tree changed since the last tag, its version must be newer than
the one that shipped with that tag; CI enforces this per PR, so a mismatch here
means a PR slipped past the gate and needs `python scripts/plugin_version.py
bump` on this branch.

Now hand off to `/osprey:contribute` for the rest of the PR mechanics:
`scripts/check.py quick` → commit (`release: notes for vYYYY.M.P`) →
`scripts/check.py full` → push → `scripts/check.py quick --changed origin/main` →
`gh pr create`.

The PR title should be `release: vYYYY.M.P — <theme>`. The PR body should
include the CHANGELOG entries verbatim so reviewers see exactly what's being
released.

### Holding the release open (the staged release)

Prep and the go decision can be decoupled: everything up to here can be done
early, and the open, green release-notes PR then sits one switch away —
merge, tag, done, about an hour end to end (one CI cycle plus minutes of
`release.yml`).

While it holds, every batch that lands on `main` costs a refold, and skipping
it is the one way this pattern ships a broken CHANGELOG: required checks do
not go stale when `main` moves, so the merge button stays green while the
newly landed fragments sit outside the rotated section — the "released
section is missing entries" failure mode below. The refold is mechanical:

1. Rebase the release branch on `main`.
2. Run `apply` again — the new fragments fold into the fresh `## [Unreleased]`.
3. Move those bullets down into the rotated `## [YYYY.M.P]` section, under
   their type headings; `[Unreleased]` ends empty again.
4. On the *final* refold — no more batches coming — run the condensation
   pass from Step 4 and get the maintainer's review of it.
5. Push. This is one full CI cycle, and the push cancels the in-flight run,
   so batch the refold rather than chasing every merge.

Mind the repo's CI budget (three concurrent PR runs) when the held release
PR re-runs beside other work.

## Step 5: Merge the PR

Two status checks are required on `main`: `pre-commit.ci - pr` and
`All CI Checks Passed`, the aggregate job every CI lane feeds. When both are
green:

```bash
gh pr merge --merge --delete-branch
```

Linear history is not enforced, so `--merge`, `--rebase`, and `--squash` are
all accepted; `--merge` keeps the release-notes commit identifiable as its own
merge. After merge:

```bash
git checkout main && git pull --ff-only origin main
```

Verify the latest commit on `main` is the release-notes merge.

## Step 6: Tag and push

Before the first tag whose tree contains a new lockstep workspace member (one
that rides the framework's v-tag, as `pyaml-cs-osprey` and `osprey-optimizer`
do): register a PyPI pending trusted publisher for it (this repository,
workflow `release.yml`, environment `pypi`) and add its `uv build --package`
line to `release.yml` — without the publisher PyPI refuses that wheel and the
tag lands half-published.
`osprey-connectors` and `osprey-facility` are not: a framework tag never
builds them (see "Releasing a component package" below). Nor does it build
`osprey-panels`, which is not published.

Run the before-tagging check once per package with compatibility lanes:

```bash
python3 scripts/ci/compat_matrix.py --package connectors check --before-tagging vYYYY.M.P
python3 scripts/ci/compat_matrix.py --package facility check --before-tagging vYYYY.M.P
```

Every earlier framework release after that package's cutoff must already be
listed (or retired) in `[tool.osprey-connectors.compat]` or
`[tool.osprey-facility.compat]`, or the next PR after the tag goes red.

Tags can be pushed directly; branch protection covers branches, not tags:

```bash
git tag vYYYY.M.P
git push origin vYYYY.M.P
```

The tag must point at the merge commit on `main`. The `release.yml` workflow
triggers on `v*.*.*` and:

1. Verifies the tagged commit is on `main` and a full CI run covered its tree,
   before anything is built.
2. Builds the framework's and its lockstep members' wheels and sdists — not
   the component packages; `verify-pypi-resolvable` resolves osprey-connectors
   and osprey-facility from PyPI, proving each floor is already published,
   and `verify-component-floors` proves each floor carries every component
   name the framework imports.
3. Verifies the built version matches the tag. A checkout without full history
   builds `0.0.postN+g…` instead of the tagged version; this gate catches that.
4. Validates the install docs and that the dependencies resolve from PyPI.
5. Publishes to PyPI via trusted publishing (OIDC; no token needed).
6. Creates a GitHub Release using the CHANGELOG section as the body.

If step 1 or step 3 fails, the publish aborts before any PyPI write.

Step 1 is the `verify-full-ci` job, and it is the automated half of the rule
in Step 2. It finds the `ci.yml` runs for the tagged commit and asserts every
model-spending lane concluded `success` there — a skipped lane fails it, which
is the case a green `All CI Checks Passed` cannot distinguish. Because a
`--merge` merge commit has a different sha from the PR head whose run carried
the label, the job also accepts a run on a nearby commit with the *same tree*:
what the lanes proved is a tree, not a sha. If it fails it prints the lanes at
fault and the two ways to run them. Re-pushing the tag after that run is green
re-triggers the whole workflow.

## Step 7: Verify

```bash
gh run watch                                 # follow the release.yml run
gh release view vYYYY.M.P                    # confirm GitHub Release exists
uv pip install --upgrade osprey-framework    # in a fresh environment
python -c "import osprey; print(osprey.__version__)"
open https://als-apg.github.io/osprey/        # switcher button reads vYYYY.M.P
```

Four success signals:

- `release.yml` finished green. It runs no tests of its own, so this proves
  the publish plus the `verify-full-ci` check that found the CI run; what
  proves the tree is that CI run itself.
- `https://pypi.org/project/osprey-framework/YYYY.M.P/` exists.
- `https://github.com/als-apg/osprey/releases/tag/vYYYY.M.P` has the CHANGELOG
  entries as the body.
- The version switcher *button* on `https://als-apg.github.io/osprey/` reads
  `vYYYY.M.P`, not the previous release. (The dropdown lists the new tag
  either way; the button is what proves the root was rebuilt.) If it still
  reads the old release, check the docs runs first
  (`gh run list --workflow=docs.yml --limit 5`): if no run for the tag ever
  started, it was superseded while pending (GitHub keeps one pending run per
  concurrency group), so `gh workflow run docs.yml -f tag=vYYYY.M.P` and
  re-check.

If any fail, stop and investigate before announcing the release. An empty
answer (`gh run watch` finding no matching run, or a command returning
nothing on an API hiccup) is neither success nor failure: re-query with an
explicit run selector (`gh run watch <run-id>`) before treating anything as
green.

## After the release

- Once `https://pypi.org/project/osprey-framework/YYYY.M.P/` exists, open the
  listing PR: add the version to `[tool.osprey-connectors.compat] frameworks`
  in `packages/osprey-connectors/pyproject.toml` and to
  `[tool.osprey-facility.compat] frameworks` in
  `packages/osprey-facility/pyproject.toml` (move the beta it supersedes to
  `retired` in each). This is what makes each package's compatibility lane
  and its nightly run hold that package's HEAD to the release.
- Update any installed copy of the plugin: `claude plugin marketplace update
  osprey && claude plugin update osprey@osprey` (add `--scope project` inside a
  deployment repository).
- Prune local tags that are not releases. Checkpoint and backup tags
  accumulate; `hatch-vcs` ignores them, but they clutter `git describe` and
  `git tag`. List them with `git tag -l | grep -v '^v'` and delete the ones
  nobody needs. Never push them.

---

## Releasing a component package

`osprey-connectors` and `osprey-facility` each release on their own, from
their own tag, through the one component workflow
`.github/workflows/release-package.yml` — never with a framework tag, and the
framework's `release.yml` never builds them. Everything that differs between
the two streams is read from the stream table in
`scripts/ci/release_streams.py`:

| Stream | Tag | PyPI environment | Changelog and fragments | Required CI lanes | Floor in the framework |
| --- | --- | --- | --- | --- | --- |
| connectors | `osprey-connectors-vYYYY.M.P` | `pypi-connectors` | `packages/osprey-connectors/CHANGELOG.md`, `packages/osprey-connectors/changelog.d/` | `test`, `lint`, `package`, `connectors-standalone`, `connectors-compat`, `framework-at-connectors-floor` | `osprey-connectors>=` (and pyaml-cs-osprey's) |
| facility | `osprey-facility-vYYYY.M.P` | `pypi-facility` | `packages/osprey-facility/CHANGELOG.md`, `packages/osprey-facility/changelog.d/` | `test`, `lint`, `package`, `osprey-facility-standalone`, `facility-compat`, `framework-at-facility-floor`, `framework-at-connectors-floor` | `osprey-facility>=`, once the root lists one |

`python3 scripts/ci/release_streams.py lanes refs/tags/<tag>` prints the
lanes the workflow will require for a tag; read them there rather than from
this table when the two disagree.

**Tag format.** `<dist>-vYYYY.M.P` (`osprey-facility-v2026.12.0`), spelled in
canonical PEP 440: a pre-release segment is attached with no hyphen and no
dot before it — `osprey-facility-v2026.12.0rc1`, `…0b2`, `…0.dev1` (`.devN`
takes its dot). The workflow fires on `osprey-connectors-v2*` and
`osprey-facility-v2*` pushes and nothing else — there
is no dispatch — and its first job, `resolve`, refuses any spelling that is
not canonical: `osprey-connectors-v2026.12.0-rc1`, which the old connectors
workflow accepted, now fails at resolve before anything is built. Delete such
a tag and push the canonical one. A component release is never marked
"Latest" (that stays the framework's).

**Component releases are stable by default.** The tooling accepts a `bN` /
`rcN` / `.devN` tag (it becomes a GitHub pre-release), but do not cut one
without a reason: a beta framework is installed with pre-releases admitted
for the whole resolve (`--prerelease=allow`), so it would absorb any
connectors or facility beta on PyPI.

**One-time setup per stream**, before that stream's first tag: register the
PyPI trusted publisher on the stream's project (`osprey-connectors` or
`osprey-facility`; owner `als-apg`, repository `osprey`,
workflow `release-package.yml`, environment `pypi-<key>`), and create the
GitHub environment `pypi-<key>`. A new project with no release yet takes a
*pending* publisher. Without it the publish job is refused after the build
and the tag lands unpublished; re-push the tag once it is registered.

**Preconditions:**

- **The changelog is rotated in a PR first.** On a branch off `main`, run
  `uv run python scripts/changelog_fragments.py apply --root <package dir>`,
  rotate `## [Unreleased]` in the package's `CHANGELOG.md` to
  `## [YYYY.M.P] - YYYY-MM-DD` (pre-release: `## [YYYY.M.PrcN] - …`), and
  merge that PR. The GitHub Release body is that section. A component change
  carries its fragment in that package's own `changelog.d/`, not the root
  `changelog.d/`.
- **The tag is on `main`.** Tag the merge commit of that PR (or a later one).
  The `verify-ci` job refuses a commit not on `main`, then requires a green
  `ci.yml` run on the tagged tree in every lane the stream's table row names
  (a compat lane that stands down — no framework listed yet, a floor that does
  not yet apply — concludes success). These are always-on lanes — no
  `full-ci` label is involved — but a `main` run superseded by a later merge
  is `cancelled`, not green; then `gh workflow run ci.yml --ref <tag>` and
  re-push the tag.
- **The before-tagging check passes**:
  `python3 scripts/ci/compat_matrix.py --package <connectors|facility> check`.
  It holds the package's `[tool.<dist>.compat]` listing and its floor to the
  tags reachable from `main`. (Its `--before-tagging vYYYY.M.P` form is for a
  framework tag; see Step 6.)

**What the tag runs:** `resolve` (the stream, from the tag; canonical spelling
or refuse) → `verify-ci` (the tag is on `main`, then the shared
`scripts/ci/require_green_lanes.py` gate over the stream's lanes) → `build`
(`uv build --package <dist>`; the wheel and sdist must carry exactly the tag's
version; `twine check`) → `verify-install` (a bare venv, dependencies from
PyPI, the lean import walk `scripts/ci/lean_import_walk.py <key>`, and the
installed `__version__`) → `publish-to-pypi` (in the stream's environment) → the
GitHub Release `<dist> vYYYY.M.P`, never Latest. A re-pushed tag resumes a
half-finished publish (`skip-existing`).

```bash
git tag osprey-facility-vYYYY.M.P <main merge commit>
git push origin osprey-facility-vYYYY.M.P
gh run watch                     # follow the release-package.yml run
```

**Raising a framework floor** — only after the release is on PyPI, and only when the framework starts
needing what it carries:

1. Confirm `https://pypi.org/project/<dist>/YYYY.M.P/` exists.
2. Raise `<dist>>=` in the root `pyproject.toml` to that release (for
   connectors, pyaml-cs-osprey's floor too), run `uv lock`, and confirm
   `uv lock --check` passes.
3. Land it in a normal PR, with a root `changelog.d/` fragment, on a branch
   that **contains the release's tag commit** (branch from `main` after the
   tag, or merge `main` in): `compat_matrix.py` refuses a floor whose tag is
   not reachable, and the paired-wheel and Docker E2E installs fail on one.
   The next framework tag's `verify-pypi-resolvable` resolves the floor from
   PyPI and fails if it is not there, so the order cannot be inverted
   silently.

When a facility release needs a newer osprey-connectors than the framework's
connectors floor (its own `osprey-connectors>=` is higher), **raise the
connectors floor first**: a connectors floor-raise PR, then the facility
floor-raise PR. Installed together at their floors, the two would otherwise
not resolve; `release.yml`'s `verify-component-floors` plan step warns when it
finds a component requiring more connectors than the framework floors.

## Manual Publish Fallback (only if Actions is broken)

If `release.yml` or `release-package.yml` is broken and the release is
time-sensitive, do by hand what the workflow would have, in the same order. The
tag is pushed already; build from a full clone checked out at it (`git fetch
--tags` first — a missing tag builds `0.0.postN+g…`). `twine upload` needs a
PyPI API token for each project (`TWINE_USERNAME=__token__`,
`TWINE_PASSWORD=<token>`); trusted publishing works only from the workflow.

Framework tag `vYYYY.M.P` — lockstep members upload before the framework, so a
refused member upload never leaves the framework live with a dependency nothing
satisfies:

```bash
git checkout vYYYY.M.P && rm -rf dist/
uv build                                  # the framework
uv build --package osprey-optimizer       # every lockstep member, as release.yml builds them
uv build --package pyaml-cs-osprey
ls dist/                                  # every file must carry YYYY.M.P exactly
uvx twine check dist/*
uvx twine upload $(ls dist/* | grep -v osprey_framework-)   # every member first
uvx twine upload dist/osprey_framework-*                     # the framework last
python3 scripts/ci/release_notes.py CHANGELOG.md YYYY.M.P > /tmp/notes.md
gh release create vYYYY.M.P --title "Release vYYYY.M.P" --notes-file /tmp/notes.md
```

Component tag `<dist>-vYYYY.M.P` (`<dist>` is `osprey-connectors` or
`osprey-facility`; its wheel prefix replaces the hyphens
with underscores):

```bash
DIST=osprey-facility                      # or osprey-connectors
python3 scripts/ci/release_streams.py resolve "refs/tags/$DIST-vYYYY.M.P"   # refuses a non-canonical tag
git checkout "$DIST-vYYYY.M.P" && rm -rf dist/
uv build --package "$DIST"
ls dist/                                  # one wheel and one sdist, both YYYY.M.P
uvx twine check dist/*
uvx twine upload "dist/${DIST//-/_}-"*
python3 scripts/ci/release_notes.py "packages/$DIST/CHANGELOG.md" YYYY.M.P > /tmp/notes.md
gh release create "$DIST-vYYYY.M.P" --title "$DIST vYYYY.M.P" \
  --notes-file /tmp/notes.md --latest=false
```

Add `--prerelease` to `gh release create` for a `bN` / `rcN` tag.

This is a fallback. The default path is the automated workflow.

## Common Failure Modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `release.yml` "Verify built versions match tag" fails | The wheel built from the tagged commit carries a different version | The tag points at the wrong commit, or the checkout lacked the history `hatch-vcs` needs. Delete the tag locally and on origin, fix, retag |
| PyPI rejects the upload as a duplicate | This version was already published | CalVer means version numbers are unique; you cannot republish. Bump the patch counter and try again |
| `gh pr merge` fails with "not mergeable" | Stale checks because `main` moved | Merge or rebase `origin/main` into the release branch, push, wait for CI to re-run |
| CI "plugin tree changed without a version bump" | A PR touched `plugins/osprey/` without `plugin_version.py bump` | Run the bump on that branch and push |
| GitHub Release body is just "See CHANGELOG.md for details.", or wrong | `scripts/ci/release_notes.py` found no line starting with `## [<version>]` (matched literally, the tag's version without the `v` / `<dist>-v` prefix) in `CHANGELOG.md`, or in the package's own `CHANGELOG.md` for a component tag | Make sure the released section's heading is exactly `## [YYYY.M.P] - YYYY-MM-DD` (pre-release: `## [YYYY.M.PbN] - …`) in that tree's `CHANGELOG.md` |
| `release-package.yml` fails at `resolve` | The tag is outside the three stream globs, or its version is not canonical PEP 440 (`-rc1`, `.rc1`, a leading zero) | Delete the tag locally and on origin, push the canonical spelling (`<dist>-vYYYY.M.PrcN`) |
| `uv lock --check` fails on the release PR | A component floor (or any `pyproject.toml` edit) changed without regenerating the lockfile | Run `uv lock`, commit `uv.lock` in the same commit |
| `changelog_fragments.py apply` exits 1 | A fragment filename is malformed or carries an unrecognized type | Rename it `<name>.<type>.md` using one of added/changed/deprecated/removed/fixed/security/internal |
| Released section is missing entries, or fragments are still on `main` after the release | `apply` was not run before the rotation, or its deletions were not staged | Fold the leftover fragments into the released section by hand, delete them, and open a PR carrying just `CHANGELOG.md` and the fragment deletions (`git add -A changelog.d/ CHANGELOG.md`) |

## Out of Scope

- **Hotfix branches** — OSPREY uses GitHub Flow, no special hotfix branches.
  A hotfix is just a `fix/<short-kebab>` branch off `main`, PR'd back; then
  this skill cuts a follow-up release.
- Release candidates and beta tags are **in** scope — see "Pre-releases:
  beta and RC tags" above. (An older revision of this skill claimed the
  workflow ignores them; wrong twice — the `v*.*.*` glob matches `v2026.9.0b1`,
  and the pipeline now handles the pre-release correctly.)
- **Documentation builds** — `docs.yml` publishes the docs from the tag on
  its own: the site root shows the newest release and `main` publishes at
  `/latest/`. Nothing to run by hand unless the root did not pick up the new
  tag, in which case Step 7's re-dispatch applies.
