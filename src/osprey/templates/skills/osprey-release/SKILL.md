---
name: osprey-release
description: >
  Guides a maintainer through cutting an OSPREY release on the GitHub Flow
  workflow: land the release-notes PR, tag the merge commit, push the tag,
  verify the automated PyPI publish. Use when someone says "create a release",
  "bump the version", "cut v2026.X.Y", "publish to PyPI", "tag a release", or
  asks about the release process. Composes with `osprey-contribute` for the
  notes PR. Versions follow CalVer (vYYYY.M.P) and the source of truth is the
  git tag — Hatch derives the version from it, so there is no version literal
  to bump.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# OSPREY Release Workflow

This skill cuts a properly versioned OSPREY release. Releases are CalVer tags
(`vYYYY.M.P`) on `main`; the PyPI publish runs automatically when the tag is
pushed.

The shape is:

1. Verify the working state and decide on the version number.
2. Open a **release-notes PR** carrying the CHANGELOG, RELEASE_NOTES and README
   updates (no direct push to `main` — branch protection rejects it).
3. Merge the PR to `main`.
4. Tag the merge commit and push the tag. **This is what sets the version.**
5. Verify the automated GitHub Actions workflow publishes successfully.

For the PR mechanics in step 2, defer to the `osprey-contribute` skill.

## Versioning: CalVer

OSPREY uses **CalVer**: `YYYY.M.P` where:

- `YYYY` — four-digit year of the release
- `M` — calendar month, no zero-padding (e.g., `5`, not `05`)
- `P` — patch counter within the month, starting at `0`

Examples: `2026.5.0`, `2026.5.1` (patch within May 2026), `2026.6.0` (next
month). When the year or month rolls over, `P` resets to `0`.

## The Source of Truth

**The git tag is the version.** There is no version literal anywhere in the
tree to edit — `hatch-vcs` derives the package version from `git describe` at
build time, and `osprey.version` resolves it at runtime. Tagging `v2026.7.0`
*is* the act of setting the version to `2026.7.0`.

Between releases the version reports its distance from the last tag
(`2026.6.2.post783+g83fda5e60`), which is how a development build is
distinguishable from the release it descends from.

| File | Purpose | Updated by |
| --- | --- | --- |
| `RELEASE_NOTES.md` | First-line title with the release version | This skill |
| `CHANGELOG.md` | Fold `changelog.d/` fragments into `## [Unreleased]` (`changelog_fragments.py apply`), then rotate it to `## [YYYY.M.P] - YYYY-MM-DD` | This skill |
| `README.md` | "Latest Release" line with version + theme | This skill |
| `pyproject.toml` | `[tool.hatch.version] source = "vcs"` | **Do not edit** |
| `src/osprey/_version.py` | Build-time stamp, gitignored | **Never commit** |

The release.yml verify step builds the package and compares the *built wheel's*
version to the pushed tag — if these disagree, the publish fails. They disagree
when the tag is not on the commit being built, or when the checkout is shallow
(which yields `0.1.devN` rather than failing outright).

---

## Step 0: Read the CHANGELOG and decide the theme

Open `CHANGELOG.md` and read the `## [Unreleased]` section together with the
pending fragments in `changelog.d/` — both are this release's content. Then
answer three questions before doing anything else:

1. **What is this release about?** Pick a short theme (e.g., "plan
   authoring & branch-protection enforcement"). It goes into the release
   title, the README "Latest Release" line, and the GitHub Release body.
2. **What is the version number?** Apply the CalVer rules above. Patch bump
   for fixes, month bump for feature batches, year bump only at January.
3. **Are there breaking changes?** Check the `### Changed` and `### Removed`
   sections. If user-facing API changed, the release should call it out
   prominently and (if it would surprise users) include a migration note.

Confirm theme + version + breaking-changes status with the maintainer before
proceeding.

## Step 1: Pre-release testing in a clean venv

Your working venv may have packages that aren't declared in `pyproject.toml`.
A clean venv catches missing dependencies before users do:

```bash
python -m venv .venv-release-test
source .venv-release-test/bin/activate
pip install -e ".[dev]"

# Unit tests (fast, free)
pytest tests/ --ignore=tests/e2e -v

# E2E tests (~10-12 min, ~$1-2 in API calls — must use path, not marker)
pytest tests/e2e/ -v

deactivate && rm -rf .venv-release-test
```

Any failures stop the release. Fix forward, then re-run.

## Step 2: Refresh the doc screenshots

The published docs embed committed PNGs, and each caption names the OSPREY
version its image was captured with. Nothing refreshes them automatically —
there is no CI job and no release step — so they age quietly, and a release is
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

`channel_finder_*.png` has no recipe at all — it is hand-captured, so it can
only be redone by hand.

The framework itself — environments, provenance, and why it is capture-only and
never a CI gate — is documented in the contributing guide under "Refreshing
documentation screenshots". Whatever changed (the PNGs and the updated
`manifest.json`) rides along in the release-notes PR below.

## Step 3: Release-notes PR

Release-notes commits cannot be pushed directly to `main` — branch protection
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

There is **no version literal to edit** — the tag in Step 5 sets the version.
This PR carries only the human-facing notes. Show the maintainer each diff
before applying:

| File | Change |
| --- | --- |
| `RELEASE_NOTES.md` | First line: `# Osprey Framework - Latest Release (vYYYY.M.P)` followed by the theme tagline |
| `CHANGELOG.md` | After the fold, convert `## [Unreleased]` to `## [YYYY.M.P] - YYYY-MM-DD`; insert a fresh empty `## [Unreleased]` above it |
| `changelog.d/` | Fragment files deleted by `apply`; only `README.md` remains |
| `README.md` | Update the "Latest Release" line with version + theme |
| `docs/source/_static/screenshots/` | Any images re-captured in Step 2, plus the updated `manifest.json` |

Stage the fold and the rotation together — `git add -A changelog.d/ CHANGELOG.md`
(pathspec-scoped, so the fragment deletions are included).

Then run a consistency check — every line should mention the same version, and
no fragment should be left behind:

```bash
echo "=== VERSION CONSISTENCY CHECK ==="
echo "RELEASE_NOTES:  $(head -1 RELEASE_NOTES.md)"
echo "README.md:      $(grep 'Latest Release:' README.md)"
echo "CHANGELOG.md:   $(grep -m1 '^## \[' CHANGELOG.md)"
echo "changelog.d/:   $(ls changelog.d | grep -vc '^README.md$') fragment(s) on disk (must be 0)"
echo "staged:         $(git diff --cached --name-only -- changelog.d/ CHANGELOG.md | tr '\n' ' ')"
```

Now hand off to `osprey-contribute` for the rest of the PR mechanics:
`quick_check.sh` → commit (`release: notes for vYYYY.M.P`) →
`ci_check.sh` → push → `premerge_check.sh main` → `gh pr create`.

The PR title should be `release: vYYYY.M.P — <theme>`. The PR body should
include the CHANGELOG entries verbatim so reviewers see exactly what's being
released.

## Step 4: Merge the PR

After CI passes (all 8 required checks green):

```bash
gh pr merge --rebase --delete-branch
```

Linear history is required, so `--rebase`. After merge:

```bash
git checkout main && git pull --ff-only origin main
```

Verify the latest commit on `main` is the version bump.

## Step 5: Tag and push

Tags can be pushed directly — branch protection covers branches, not tags:

```bash
git tag vYYYY.M.P
git push origin vYYYY.M.P
```

The tag must point at the merge commit on `main`. The `release.yml` workflow
triggers on `v*.*.*` and:

1. Builds the wheel and sdist.
2. Verifies the built version matches the tag. A checkout without full history
   builds `0.1.devN` instead of the tagged version; this gate catches that.
3. Publishes to PyPI via trusted publishing (OIDC; no token needed).
4. Creates a GitHub Release using the CHANGELOG section as the body.

If step 2 fails, the publish aborts before any PyPI write — safe.

## Step 6: Verify

```bash
gh run watch                                 # follow the release.yml run
gh release view vYYYY.M.P                    # confirm GitHub Release exists
pip install --upgrade osprey-framework       # in a fresh shell
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
  started, it was superseded while pending — GitHub keeps one pending run per
  concurrency group — so `gh workflow run docs.yml -f tag=vYYYY.M.P` and
  re-check.

If any fail, stop and investigate before announcing the release. An empty
answer — `gh run watch` finding no matching run, or a command returning
nothing on an API hiccup — is neither success nor failure: re-query with an
explicit run selector (`gh run watch <run-id>`) before treating anything as
green.

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
| `release.yml` "Verify built version matches tag" fails | The wheel built from the tagged commit carries a different version | The tag points at the wrong commit, or the checkout lacked the history `hatch-vcs` needs. Delete the tag locally and on origin, fix, retag |
| PyPI rejects the upload as a duplicate | This version was already published | CalVer means version numbers are unique; you cannot republish. Bump the patch counter and try again |
| `gh pr merge --rebase` fails with "not mergeable" | Stale checks because `main` moved | `git rebase origin/main` on the release branch, force-push with lease, wait for CI to re-run |
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
  tag, in which case Step 5's re-dispatch applies.
