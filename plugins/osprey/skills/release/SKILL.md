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
- **The connectors floor names a stable osprey-connectors release that is on
  PyPI and whose tag is reachable from `main`.** `osprey-connectors` has its
  own calendar stream (`osprey-connectors-vYYYY.M.P`, `release-connectors.yml`,
  PyPI environment `pypi-connectors`); the framework declares only
  `osprey-connectors>=FLOOR` and never a ceiling. A floor above the last
  reachable connectors tag breaks every paired wheel install and
  `osprey up --dev` (dev wheels build as `<last tag>.postN`), and a floor
  nothing stable satisfies makes pip *and* uv (default policy) silently take a
  pre-release — so the order is always: connectors release first, floor raise
  in a following PR. `connectors_compat_matrix.py` refuses a floor whose tag is
  not reachable. A beta framework needs no connectors beta: `OSPREY_PIP_PRE`
  resolves the whole deps layer with `--prerelease=allow`, which takes the
  newest connectors — so publish a connectors pre-release only when every beta
  deployment should absorb it. Raise the floor only when the framework starts
  needing something a newer stable connectors release carries, and run
  `uv lock --check` after that edit (see "Releasing osprey-connectors" below).
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

### osprey-connectors has its own version stream

The `osprey-connectors` workspace package is versioned the same way — CalVer,
derived from a tag, no literal — but from its **own** tags,
`osprey-connectors-vYYYY.M.P` (`bN` / `rcN` for pre-releases), cut when the
package changes rather than with every framework tag. Its describe glob
(`osprey-connectors-v2[0-9][0-9][0-9].*`) skips both the framework's `v*` tags
and the retired `osprey-connectors-v0.1.0`; the framework's `v[0-9]*` glob
never reads a connectors tag. Each connectors release stays
backward-compatible with every supported framework release, enforced in CI by
the `connectors-compat` and `framework-at-connectors-floor` lanes, which is why
the framework declares only a floor on it.

A checkout with no connectors calendar tag reachable builds the package as
`0.0.postN+g…`, which the framework's floor refuses — `osprey up --dev` then
stops before staging and names the tag glob. Fetch tags (`git fetch --tags`)
before building; a fork or mirror must sync the `osprey-connectors-v*` tags,
not only the framework's `v*` ones.

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

```bash
export UV_PROJECT_ENVIRONMENT=.venv-release-test
uv sync --extra dev --extra teams --extra gchat --extra virtual-accelerator

# Pick the provider CI picks: one gateway, no fallbacks. A run that names no
# provider is refused before the first test, and a stray key from another
# provider sends tests somewhere CI never goes.
export OSPREY_E2E_PROVIDER=als-apg
export ALS_APG_API_KEY=... ALS_APG_BASE_URL=...
unset CBORG_API_KEY OPENAI_API_KEY GOOGLE_API_KEY ANTHROPIC_API_KEY

# Unit tests
uv run pytest tests/ --ignore=tests/e2e -m "not pty and not wallclock" -n 4 --dist loadgroup -q

rm -rf .venv-release-test
```

The e2e half is the bulk lane, run exactly as CI runs it — same parallelism,
same exclusions. Copy the invocation out of the `e2e-tests` job rather than
retyping thirty-three `--ignore` flags:

```bash
sed -n '/^  e2e-tests:/,/^  channel-finder-benchmarks:/p' .github/workflows/ci.yml \
  | grep -e 'pytest tests/e2e/' -e '--ignore='
```

It runs `pytest tests/e2e/ -v --tb=short -n 4 --dist loadfile` with that
ignore list. The path, not the marker — `-m e2e` leaks registry state between
tests.

**The ignored modules are not untested.** Each has a dedicated CI lane of its
own, because each binds a fixed host port, leaves a fixed-name container
behind, or needs an extra the bulk lane does not install; CI gives them a
runner each for exactly that reason. This local run does not reach them, and
is not meant to. The `full-ci` run below is their gate.

Read the skips, not just the exit code. A missing key, container runtime or
backend turns a lane green having run nothing, and `tests/e2e/README.md` lists
what each one needs. Anything still skipping for want of Postgres or Ollama
gets named in the release notes as untested rather than passed over.

Also trigger the on-demand benchmark lane against the release tree, which no
PR runs: `gh workflow run ci.yml -f run_benchmarks=true`.

Budget most of a day. The bulk lane alone is 20-25 minutes in CI at four-way
parallelism, and the Docker stacks around it are slower on a laptop than on a
runner.

Any failure, and any unexplained skip, stops the release. Fix forward, then
re-run.

### The full-ci run is a hard requirement

Eight CI lanes drive real agent sessions against a live model endpoint, and
they run only on a pull request carrying the `full-ci` label. They are the
only proof the agentic paths work on the exact tree being tagged: `release.yml`
runs no tests and never reads the tagged commit's CI status.

So the release-notes PR in Step 4 carries the label from the moment it opens:

```bash
gh pr edit <number> --add-label full-ci
```

The label persists, so every refold push re-runs the eight; the run that
counts is the one for the commit that actually merges. If the label ever comes
off, put it back and let the lanes run again.

`release.yml` refuses a tag it cannot find such a run for, so this is enforced
rather than remembered. Check it before tagging anyway — the same answer costs
two commands here and a deleted tag there:

```bash
gh run list --workflow ci.yml --commit <sha> --limit 1
gh run view <run-id> --json jobs --jq '.jobs[] | [.conclusion, .name] | @tsv'
```

By their `ci.yml` job ids, the eight are `agentic-per-preset` (one job per
preset), `e2e-tests`, `dispatch-deploy-e2e`, `dispatch-overlay-e2e`,
`scan-agentic-e2e`, `nextcloud-talk-bridge-e2e`, `gchat-bridge-e2e` and
`target-switch-agentic-e2e`.

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

- `ariel` needs a container runtime and a free port 10800 (the layout's postgres
  slot at the default base; `services.postgresql.port_host` if the deployment
  moved it) — `make screenshots SCREENSHOTOPTS=--stack`.
- `web_terminal_hero` drives a live agent session on that stack —
  `python -m docs.screenshots --agentic --only web_terminal_hero`. It spends
  real subscription budget, so re-capture it when the Web Terminal's appearance
  has actually changed, not on every release.

`channel_finder_*.png` has no recipe at all; it is hand-captured, so it can
only be redone by hand.

### Pre-release check: re-record and review the docs videos

**Every release, run by hand.** The docs embed recorded web-terminal sessions,
the landing-page demo and one per feature page (`python -m docs.screenshots
list` names them all), and a UI change can make any of them wrong without
anyone noticing. So the maintainer re-records all of them before every release,
whether or not the UI looks changed, and watches every take. It is never a CI
job and never skipped: a release whose videos were not watched does not get
tagged. A beta is a release here too.

Each take builds a fresh control-assistant stack and runs a live agent, so it
needs a container runtime, the provider keys the run uses (the
`OSPREY_DOCSHOTS_PROVIDER` override when the default provider's key is over
budget), any key a video names (the watchdog video needs `TYPESAFE_API_KEY`),
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
MP4s are never committed. Adding a video is one `DocVideo` entry in
`docs/screenshots/videos.py` plus the embed snippet; both are documented in
`docs/README.md`.

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
that rides the framework's v-tag, as `pyaml-cs-osprey` does): register a PyPI
pending trusted publisher for it (this repository, workflow `release.yml`,
environment `pypi`) and add its `uv build --package` line to `release.yml` —
without the publisher PyPI refuses that wheel and the tag lands half-published.
`osprey-connectors` is not one: a framework tag never builds it (see
"Releasing osprey-connectors" below).

Run `python3 scripts/ci/connectors_compat_matrix.py check --before-tagging vYYYY.M.P`:
every earlier split-era framework release must already be listed (or retired)
in `[tool.osprey-connectors.compat]`, or the next PR after the tag goes red.

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
   `osprey-connectors`, which `verify-pypi-resolvable` then resolves from PyPI,
   proving the framework's `osprey-connectors>=` floor is already published.
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
  in `packages/osprey-connectors/pyproject.toml` (move the beta it supersedes
  to `retired`). This is what makes the connectors compatibility lane and its
  nightly run hold connectors HEAD to the release.
- Update any installed copy of the plugin: `claude plugin marketplace update
  osprey && claude plugin update osprey@osprey` (add `--scope project` inside a
  deployment repository).
- Prune local tags that are not releases. Checkpoint and backup tags
  accumulate; `hatch-vcs` ignores them, but they clutter `git describe` and
  `git tag`. List them with `git tag -l | grep -v '^v'` and delete the ones
  nobody needs. Never push them.

---

## Releasing osprey-connectors

`osprey-connectors` releases on its own, from its own tag, through its own
workflow (`.github/workflows/release-connectors.yml`) — never with a framework
tag, and the framework's `release.yml` never builds it.

**Tag format:** `osprey-connectors-vYYYY.M.P` (`osprey-connectors-v2026.10.0`).
The workflow fires on `osprey-connectors-v2*` pushes and nothing else — there is
no dispatch — and a connectors release is never marked "Latest" (that stays the
framework's). **Connectors releases are stable by default.** The tooling would
accept a `bN` / `rcN` tag (it becomes a GitHub pre-release), but do not cut one
without a reason: a beta framework is installed with pre-releases admitted for
the whole resolve (`--prerelease=allow`), so it would absorb any connectors
beta on PyPI.

**Preconditions:**

- The tagged commit is on `main` and its `ci.yml` run is green in every lane
  the `verify-ci` job's `REQUIRED_LANES` names: `test`, `lint`, `package`,
  `connectors-standalone`, plus the backward-compatibility lanes
  `connectors-compat` and `framework-at-connectors-floor` (a lane that stands
  down — no framework listed yet, a floor that predates the independent
  releases — concludes success). These are always-on lanes — no `full-ci` label is involved — but a
  `main` run superseded by a later merge is `cancelled`, not green; then
  `gh workflow run ci.yml --ref osprey-connectors-vYYYY.M.P` and re-push the tag.
- The package's changelog is folded and rotated on `main`: run
  `python scripts/changelog_fragments.py apply --root packages/osprey-connectors`,
  rotate `## [Unreleased]` in `packages/osprey-connectors/CHANGELOG.md` to
  `## [YYYY.M.P] - YYYY-MM-DD`, and merge that PR first. The GitHub Release body
  is that section. Connectors changes carry their fragments in
  `packages/osprey-connectors/changelog.d/`, not the root `changelog.d/`.
- PyPI's trusted publisher for `osprey-connectors` names this repository,
  workflow `release-connectors.yml`, environment `pypi-connectors`, and the
  GitHub environment `pypi-connectors` exists.

**What the tag runs:** `verify-ci` (the tag is on `main`, then the shared
`scripts/ci/require_green_lanes.py` gate) → `build` (`uv build --package osprey-connectors`; the wheel and sdist must
carry exactly the tag's PEP 440-normalized number; `twine check`) →
`verify-install` (a bare venv, dependencies from PyPI, the same lean import-walk
the `package` CI job runs) → `publish-to-pypi` → the GitHub Release
`osprey-connectors vYYYY.M.P`. A re-pushed tag resumes a half-finished publish
(`skip-existing`).

```bash
git tag osprey-connectors-vYYYY.M.P <main merge commit>
git push origin osprey-connectors-vYYYY.M.P
gh run watch                     # follow the release-connectors.yml run
```

**Raising the framework's floor** — only after the connectors release is on
PyPI, and only when the framework starts needing what it carries:

1. Confirm `https://pypi.org/project/osprey-connectors/YYYY.M.P/` exists.
2. Raise `osprey-connectors>=` in the root `pyproject.toml` to that release, run
   `uv lock`, and confirm `uv lock --check` passes.
3. Land it in a normal PR (with a root `changelog.d/` fragment). The next
   framework tag's `verify-pypi-resolvable` resolves the floor from PyPI and
   fails if it is not there, so the order cannot be inverted silently.

> **Cut-over (October 2026) — one time.** The order in which the split lands:
>
> 0. Maintainer, before PR B (the version stream) merges: register the PyPI
>    trusted publisher on `osprey-connectors` (owner `als-apg`, repo `osprey`,
>    workflow `release-connectors.yml`, environment `pypi-connectors`), create
>    the GitHub environment `pypi-connectors`, and push the bootstrap tag
>    `osprey-connectors-v2026.9.0b3` on `0584524d4` (it fires nothing).
> 1. Merge #1810 (the connectors public API) before the first independent
>    connectors tag.
> 2. Merge the own-suite PR.
> 3. Merge the version-stream PR.
> 4. Merge the release-workflows PR.
> 5. Merge the compatibility-lanes PR, then remove `release.yml`'s publisher
>    from the `osprey-connectors` PyPI project.
> 6. Connectors notes PR (`uv run python scripts/changelog_fragments.py apply
>    --root packages/osprey-connectors`, rotate `## [2026.10.0]`), tag
>    `osprey-connectors-v2026.10.0`, confirm it on PyPI.
> 7. Floor-raise PR: `osprey-connectors>=2026.10.0` (+ `uv lock`, and
>    pyaml-cs-osprey's floor too) — only after PyPI shows the release. This
>    arms `framework-at-connectors-floor`.
> 8. Framework notes PR for 2026.10.0b1 (`full-ci`), run the before-tagging
>    check, tag `v2026.10.0b1`.
> 9. Listing PR: `frameworks = ["2026.10.0b1"]`.
>
> Framework 2026.9.0b1–b3 are superseded: they are never listed, `main`'s
> connectors already break them, and rebuilding such an image needs
> `osprey-connectors==2026.9.0b3` pinned.

## Manual Publish Fallback (only if Actions is broken)

If `release.yml` or `release-connectors.yml` is broken and the release is
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
uv build --package pyaml-cs-osprey        # every lockstep member, as release.yml builds them
ls dist/                                  # every file must carry YYYY.M.P exactly
uvx twine check dist/*
uvx twine upload $(ls dist/* | grep -v osprey_framework-)   # every member first
uvx twine upload dist/osprey_framework-*                     # the framework last
python3 scripts/ci/release_notes.py CHANGELOG.md YYYY.M.P > /tmp/notes.md
gh release create vYYYY.M.P --title "Release vYYYY.M.P" --notes-file /tmp/notes.md
```

Connectors tag `osprey-connectors-vYYYY.M.P`:

```bash
git checkout osprey-connectors-vYYYY.M.P && rm -rf dist/
uv build --package osprey-connectors
ls dist/                                  # one wheel and one sdist, both YYYY.M.P
uvx twine check dist/*
uvx twine upload dist/osprey_connectors-*
python3 scripts/ci/release_notes.py packages/osprey-connectors/CHANGELOG.md YYYY.M.P > /tmp/notes.md
gh release create osprey-connectors-vYYYY.M.P --title "osprey-connectors vYYYY.M.P" \
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
| GitHub Release body is just "See CHANGELOG.md for details.", or wrong | `scripts/ci/release_notes.py` found no line starting with `## [<version>]` (matched literally, the tag's version without the `v` / `osprey-connectors-v` prefix) in `CHANGELOG.md`, or in `packages/osprey-connectors/CHANGELOG.md` for a connectors tag | Make sure the released section's heading is exactly `## [YYYY.M.P] - YYYY-MM-DD` (pre-release: `## [YYYY.M.PbN] - …`) in that tree's `CHANGELOG.md` |
| `uv lock --check` fails on the release PR | The connectors floor (or any `pyproject.toml` edit) changed without regenerating the lockfile | Run `uv lock`, commit `uv.lock` in the same commit |
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
