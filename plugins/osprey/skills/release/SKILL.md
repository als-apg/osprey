---
name: release
description: >
  Guides a maintainer through cutting an OSPREY release on the GitHub Flow
  workflow: run the two advisory drift sweeps, test in a clean environment,
  refresh the doc screenshots, land the release-notes PR, tag the merge commit,
  push the tag, verify the automated PyPI publish. Use when someone says "create
  a release", "bump the version", "cut v2026.X.Y", "publish to PyPI", "tag a
  release", or asks about the release process. Composes with `/osprey:contribute`
  for the notes PR and with `/osprey:housekeeping` and `/osprey:doc-sync` for the
  sweeps. Versions follow CalVer (vYYYY.M.P) and the source of truth is the git
  tag — Hatch derives the version from it, so there is no version literal to bump.
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
3. Test in a clean environment and refresh the doc screenshots.
4. Open a **release-notes PR** carrying the CHANGELOG fold and rotation, the
   RELEASE_NOTES title, and any re-captured screenshots (no direct push to
   `main`; branch protection rejects it).
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

## The Source of Truth

**The git tag is the version.** There is no version literal anywhere in the
tree to edit for the framework: `hatch-vcs` derives the package version from
`git describe` at build time, and `osprey.version` resolves it at runtime.
Tagging `v2026.7.0` *is* the act of setting the version to `2026.7.0`.

Between releases the version reports its distance from the last tag
(`2026.6.2.post783+g83fda5e60`), which is how a development build is
distinguishable from the release it descends from.

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
(which yields `0.1.devN` rather than failing outright).

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
fresh environment catches missing dependencies before users do:

```bash
UV_PROJECT_ENVIRONMENT=.venv-release-test uv sync --extra dev

# Unit tests
UV_PROJECT_ENVIRONMENT=.venv-release-test uv run pytest tests/ --ignore=tests/e2e -m "not pty" -n 4 --dist loadgroup -q

# E2E tests — the path, not a marker; they build real images and need Docker
UV_PROJECT_ENVIRONMENT=.venv-release-test uv run pytest tests/e2e/ -v

rm -rf .venv-release-test
```

Any failures stop the release. Fix forward, then re-run.

## Step 3: Refresh the doc screenshots

The published docs embed committed PNGs, and each caption names the OSPREY
version its image was captured with. Nothing refreshes them automatically:
there is no CI job and no release step, so they age quietly, and a release is
where that staleness becomes public.

Read `docs/source/_static/screenshots/manifest.json`: every entry carries the
version and timestamp of its last capture. Compare that against the UI work in
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
- **Fold phantom sections** found in Step 0 into this release: move their
  entries under the matching `### <Type>` heading and delete the untagged
  `## [YYYY.M.P]` heading. Nothing is lost, and the CHANGELOG stops
  advertising a version that was never published.

There is **no version literal to edit**; the tag in Step 6 sets the version.
This PR carries only the human-facing notes. Show the maintainer each diff
before applying:

| File | Change |
| --- | --- |
| `RELEASE_NOTES.md` | First line: `# Osprey Framework - Latest Release (vYYYY.M.P)` followed by the theme tagline |
| `CHANGELOG.md` | After the fold and repair, convert `## [Unreleased]` to `## [YYYY.M.P] - YYYY-MM-DD`; insert a fresh empty `## [Unreleased]` above it |
| `changelog.d/` | Fragment files deleted by `apply`; only `README.md` remains |
| `docs/source/_static/screenshots/` | Any images re-captured in Step 3, plus the updated `manifest.json` |

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
`quick_check.sh` → commit (`release: notes for vYYYY.M.P`) →
`ci_check.sh` → push → `premerge_check.sh main` → `gh pr create`.

The PR title should be `release: vYYYY.M.P — <theme>`. The PR body should
include the CHANGELOG entries verbatim so reviewers see exactly what's being
released.

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

Tags can be pushed directly; branch protection covers branches, not tags:

```bash
git tag vYYYY.M.P
git push origin vYYYY.M.P
```

The tag must point at the merge commit on `main`. The `release.yml` workflow
triggers on `v*.*.*` and:

1. Builds the wheel and sdist.
2. Verifies the built version matches the tag. A checkout without full history
   builds `0.1.devN` instead of the tagged version; this gate catches that.
3. Validates the install docs and that the dependencies resolve from PyPI.
4. Publishes to PyPI via trusted publishing (OIDC; no token needed).
5. Creates a GitHub Release using the CHANGELOG section as the body.

If step 2 fails, the publish aborts before any PyPI write.

## Step 7: Verify

```bash
gh run watch                                 # follow the release.yml run
gh release view vYYYY.M.P                    # confirm GitHub Release exists
uv pip install --upgrade osprey-framework    # in a fresh environment
python -c "import osprey; print(osprey.__version__)"
open https://als-apg.github.io/osprey/        # switcher button reads vYYYY.M.P
```

Four success signals:

- `release.yml` finished green.
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

- Update any installed copy of the plugin: `claude plugin marketplace update
  osprey && claude plugin update osprey@osprey` (add `--scope project` inside a
  deployment repository).
- Prune local tags that are not releases. Checkpoint and backup tags
  accumulate; `hatch-vcs` ignores them, but they clutter `git describe` and
  `git tag`. List them with `git tag -l | grep -v '^v'` and delete the ones
  nobody needs. Never push them.

---

## Manual Publish Fallback (only if Actions is broken)

If `release.yml` is broken and the release is time-sensitive:

```bash
rm -rf dist/ build/ src/*.egg-info/
uv build
uvx twine check dist/*
uvx twine upload dist/*    # requires PyPI credentials in env
```

Then manually create the GitHub Release: `gh release create vYYYY.M.P
--notes-file <(awk '/^## \[YYYY.M.P\]/,/^## \[/' CHANGELOG.md | head -n -1)`.

This is a fallback. The default path is the automated workflow.

## Common Failure Modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `release.yml` "Verify built versions match tag" fails | The wheel built from the tagged commit carries a different version | The tag points at the wrong commit, or the checkout lacked the history `hatch-vcs` needs. Delete the tag locally and on origin, fix, retag |
| PyPI rejects the upload as a duplicate | This version was already published | CalVer means version numbers are unique; you cannot republish. Bump the patch counter and try again |
| `gh pr merge` fails with "not mergeable" | Stale checks because `main` moved | Merge or rebase `origin/main` into the release branch, push, wait for CI to re-run |
| CI "plugin tree changed without a version bump" | A PR touched `plugins/osprey/` without `plugin_version.py bump` | Run the bump on that branch and push |
| GitHub Release body is empty or wrong | CHANGELOG section heading didn't match the regex `release.yml` uses | Make sure the CHANGELOG heading is exactly `## [YYYY.M.P] - YYYY-MM-DD` |
| `changelog_fragments.py apply` exits 1 | A fragment filename is malformed or carries an unrecognized type | Rename it `<name>.<type>.md` using one of added/changed/deprecated/removed/fixed/security/internal |
| Released section is missing entries, or fragments are still on `main` after the release | `apply` was not run before the rotation, or its deletions were not staged | Fold the leftover fragments into the released section by hand, delete them, and open a PR carrying just `CHANGELOG.md` and the fragment deletions (`git add -A changelog.d/ CHANGELOG.md`) |

## Out of Scope

- **Hotfix branches** — OSPREY uses GitHub Flow, no special hotfix branches.
  A hotfix is just a `fix/<short-kebab>` branch off `main`, PR'd back; then
  this skill cuts a follow-up release.
- **Release candidates / beta tags** — not currently supported by
  `release.yml`, which triggers on `v*.*.*` only. If you need an RC channel,
  the workflow needs changes first.
- **Documentation builds** — `docs.yml` publishes the docs from the tag on
  its own: the site root shows the newest release and `main` publishes at
  `/latest/`. Nothing to run by hand unless the root did not pick up the new
  tag, in which case Step 7's re-dispatch applies.
