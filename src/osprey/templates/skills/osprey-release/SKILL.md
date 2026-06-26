---
name: osprey-release
description: >
  Guides a maintainer through cutting an OSPREY release on the GitHub Flow
  workflow: open a version-bump PR, merge it to main, tag the merge commit,
  push the tag, verify the automated PyPI publish. Use when someone says
  "create a release", "bump the version", "cut v2026.X.Y", "publish to PyPI",
  "tag a release", or asks about the release process. Composes with
  `osprey-contribute` for the bump PR. Versions follow CalVer (vYYYY.M.P) and
  the source of truth is `src/osprey/__init__.py` — Hatch derives the
  pyproject.toml version dynamically.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# OSPREY Release Workflow

This skill cuts a properly versioned OSPREY release. Releases are CalVer tags
(`vYYYY.M.P`) on `main`; the PyPI publish runs automatically when the tag is
pushed.

The shape is:

1. Verify the working state and decide on the version number.
2. Open a **version-bump PR** (no direct push to `main` — branch protection
   rejects it).
3. Merge the PR to `main`.
4. Tag the merge commit and push the tag.
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

`src/osprey/__init__.py` holds `__version__`. Everything else either reads
from there (Hatch, the GitHub Actions verify step) or is a doc string that
also needs updating.

| File | Purpose | Updated by |
| --- | --- | --- |
| `src/osprey/__init__.py` | Source of truth — Hatch reads this for the package version | This skill |
| `src/osprey/cli/main.py` | Fallback version printed by `osprey --version` when not installed | This skill |
| `RELEASE_NOTES.md` | First-line title with the release version | This skill |
| `CHANGELOG.md` | Add `## [vYYYY.M.P] - YYYY-MM-DD` heading; rotate `## [Unreleased]` content | This skill |
| `README.md` | "Latest Release" line with version + theme | This skill |
| `pyproject.toml` | Uses `dynamic = ["version"]`; Hatch reads from `__init__.py` | **Do not edit** |

The release.yml verify step greps `__version__ =` out of `src/osprey/__init__.py`
and compares it to the pushed tag — if these disagree, the publish fails.

---

## Step 0: Read the CHANGELOG and decide the theme

Open `CHANGELOG.md`, read the `## [Unreleased]` section, and answer three
questions before doing anything else:

1. **What is this release about?** Pick a short theme (e.g., "GitHub Flow
   migration & branch-protection enforcement"). It goes into the release
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

## Step 2: Version-bump PR

The version-bump commit cannot be pushed directly to `main` — branch
protection rejects it. Open a PR instead.

```bash
git checkout main && git pull --ff-only origin main
git checkout -b release/vYYYY.M.P
```

Update each file with the new version. Show the maintainer each diff before
applying:

| File | Change |
| --- | --- |
| `src/osprey/__init__.py` | `__version__ = "YYYY.M.P"` |
| `src/osprey/cli/main.py` | The fallback `__version__ = "YYYY.M.P"` line |
| `RELEASE_NOTES.md` | First line: `# Osprey Framework - Latest Release (vYYYY.M.P)` followed by the theme tagline |
| `CHANGELOG.md` | Convert `## [Unreleased]` to `## [YYYY.M.P] - YYYY-MM-DD`; insert a fresh empty `## [Unreleased]` above it |
| `README.md` | Update the "Latest Release" line with version + theme |

Then run a consistency check — every line should mention the same version:

```bash
echo "=== VERSION CONSISTENCY CHECK ==="
echo "__init__.py:    $(grep '__version__ = ' src/osprey/__init__.py)"
echo "cli/main.py:    $(grep '__version__ = ' src/osprey/cli/main.py)"
echo "RELEASE_NOTES:  $(head -1 RELEASE_NOTES.md)"
echo "README.md:      $(grep 'Latest Release:' README.md)"
echo "CHANGELOG.md:   $(grep -m1 '^## \[' CHANGELOG.md)"
```

Now hand off to `osprey-contribute` for the rest of the PR mechanics:
`quick_check.sh` → commit (`release: bump version to YYYY.M.P`) →
`ci_check.sh` → push → `premerge_check.sh main` → `gh pr create`.

The PR title should be `release: vYYYY.M.P — <theme>`. The PR body should
include the CHANGELOG entries verbatim so reviewers see exactly what's being
released.

## Step 3: Merge the PR

After CI passes (all 8 required checks green):

```bash
gh pr merge --rebase --delete-branch
```

Linear history is required, so `--rebase`. After merge:

```bash
git checkout main && git pull --ff-only origin main
```

Verify the latest commit on `main` is the version bump.

## Step 4: Tag and push

Tags can be pushed directly — branch protection covers branches, not tags:

```bash
git tag vYYYY.M.P
git push origin vYYYY.M.P
```

The tag must point at the merge commit on `main`. The `release.yml` workflow
triggers on `v*.*.*` and:

1. Verifies the tag matches `__version__` in `src/osprey/__init__.py`.
2. Builds the wheel and sdist.
3. Publishes to PyPI via trusted publishing (OIDC; no token needed).
4. Creates a GitHub Release using the CHANGELOG section as the body.

If step 1 fails, the publish aborts before any PyPI write — safe.

## Step 5: Verify

```bash
gh run watch                                 # follow the release.yml run
gh release view vYYYY.M.P                    # confirm GitHub Release exists
pip install --upgrade osprey-framework       # in a fresh shell
python -c "import osprey; print(osprey.__version__)"
```

Three success signals:

- `release.yml` finished green.
- `https://pypi.org/project/osprey-framework/YYYY.M.P/` exists.
- `https://github.com/als-apg/osprey/releases/tag/vYYYY.M.P` has the CHANGELOG
  entries as the body.

If any fail, stop and investigate before announcing the release.

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
| `release.yml` "Verify version matches tag" fails | `__version__` in `__init__.py` doesn't match the pushed tag | Tag the wrong commit, or the version-bump PR didn't actually update `__init__.py`. Delete the tag locally and on origin, fix, retag |
| PyPI rejects the upload as a duplicate | This version was already published | CalVer means version numbers are unique; you cannot republish. Bump the patch counter and try again |
| `gh pr merge --rebase` fails with "not mergeable" | Stale checks because `main` moved | `git rebase origin/main` on the release branch, force-push with lease, wait for CI to re-run |
| GitHub Release body is empty or wrong | CHANGELOG section heading didn't match the regex `release.yml` uses | Make sure the CHANGELOG heading is exactly `## [YYYY.M.P] - YYYY-MM-DD` |

## Out of Scope

- **Hotfix branches** — OSPREY uses GitHub Flow, no special hotfix branches.
  A hotfix is just a `fix/<short-kebab>` branch off `main`, PR'd back; then
  this skill cuts a follow-up release.
- **Release candidates / beta tags** — not currently supported by
  `release.yml`, which triggers on `v*.*.*` only. If you need an RC channel,
  the workflow needs changes first.
- **Documentation builds** — handled separately by `docs.yml`; no manual
  step needed in the release flow.
