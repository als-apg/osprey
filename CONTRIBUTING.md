# Contributing to Osprey Framework

Thank you for your interest in contributing to Osprey! 🎉

This document provides a quick start guide. For comprehensive contribution guidelines, please visit our **[full Contributing Guide in the documentation](https://als-apg.github.io/osprey/contributing/)**.

## Quick Start

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR-USERNAME/osprey.git
cd osprey
```

### 2. Set Up Development Environment

```bash
# Install all development dependencies (creates .venv automatically)
uv sync --extra dev --extra docs
```

**Optional — enable local pre-commit checks.** If you'd like formatting and
whitespace issues caught *before* you push (rather than auto-fixed by the bot
on your PR), run this once per machine:

```bash
uv run pre-commit install
```

After that, `git commit` will automatically run ruff and basic file-hygiene
checks on your staged files. You can skip this step entirely — `pre-commit.ci`
will auto-fix common issues on your PR either way. Recommended for frequent
contributors; safe to ignore otherwise.

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes and Test

```bash
# Run tests
pytest tests/ --ignore=tests/e2e -v

# Run linters
ruff check src/ tests/
ruff format --check src/ tests/
```

### 5. Submit Pull Request

- Push your branch to GitHub
- Open a Pull Request with a clear description
- Address review feedback

## Claude Code Workflow Skill (Optional but Recommended)

If you use [Claude Code](https://docs.claude.com/en/docs/claude-code), install
the bundled `osprey-contribute` skill to get guided help with this workflow:

```bash
uv run osprey skills install osprey-contribute
```

It walks you through branching, commits, push, PR, and CI iteration following
the conventions on this page — including the protected-branch reality on
`main` (no direct pushes; eight required CI checks; linear history). Once
installed, just open Claude Code in the repo and describe what you want to
contribute; the skill picks up wherever you are in the journey.

Other available skills (`osprey skills install --help` lists them all):
`osprey-build-interview`, `osprey-build-deploy`, `osprey-release`.

## Branch Strategy

Osprey uses **GitHub Flow**: `main` is the single long-lived branch and is always the PR target. Releases are CalVer tags (`vYYYY.M.P`) on `main` — there is no separate `develop`, `release/*`, or `next` branch. Hotfixes follow the same flow: branch from the tag or from `main`, open a PR back to `main`, and tag a follow-up release.

For details (CI gates, branch protection, release cuts) see the [full Contributing Guide](https://als-apg.github.io/osprey/contributing/).

## Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

## Code Standards

- Follow PEP 8 (100 character line length)
- Use Ruff for linting and formatting
- Add tests for new functionality
- Write Google-style docstrings
- Update documentation as needed

## Running Tests

```bash
# Unit tests (fast)
pytest tests/ --ignore=tests/e2e -v

# E2E tests (requires API keys)
pytest tests/e2e/ -v
```

## Front-End JavaScript Tooling

The browser interfaces ship a small dev/CI-only JS toolchain. It is **not**
needed to install or run OSPREY — only if you edit front-end JavaScript. It
requires Node LTS.

```bash
# One-time: install the dev-only toolchain
npm install

# Type-check the JS (tsc --noEmit)
npm run typecheck

# Lint the JS (eslint — house style + a per-file LOC cap)
npm run lint

# Run the JS unit tests (Vitest)
npm run test:js
```

All three run in CI (the `frontend-js` job); any failure blocks the merge.

**Type-checking is on by default.** `tsconfig.json` sets `checkJs: true`, so every
JS file under the interface and test globs is type-checked — there is no per-file
opt-in. (The `// @ts-check` headers still present on the already-clean files are now
redundant no-ops; they are harmless and left in place.)

**Three shrink-only ratchets** grandfather code that has not yet been retrofitted.
Each is the single source of truth for its debt and may **only shrink** — a later
hardening phase's exit criterion is deleting its interface's entries while keeping
CI green. **New or edited code gets no exemption:** a new `var`, an unused variable,
a `==`, an over-cap module, or a fresh type error fails CI immediately.

1. **`@ts-nocheck` type allowlist** — a file that is not yet type-clean carries a
   leading `// @ts-nocheck`. List the allowlist with:
   ```bash
   grep -rl '@ts-nocheck' src tests
   ```
2. **`max-lines` exemption** — the few over-cap (>450 code-line) modules awaiting a
   split, listed in the `max-lines` override block of `eslint.config.js`. Regenerate
   that list from eslint's own count with:
   ```bash
   npx eslint 'src/osprey/interfaces/**/*.js' \
     --rule '{"max-lines":["error",{"max":450,"skipComments":true,"skipBlankLines":true}]}' \
     -f json | python3 -c 'import sys,json,os; print("\n".join(sorted({os.path.relpath(f["filePath"]) for f in json.load(sys.stdin) for m in f["messages"] if m.get("ruleId")=="max-lines"})))'
   ```
3. **Legacy rule-exemption block** — files with grandfathered `no-unused-vars` /
   `eqeqeq` violations, set to `off` for those files only in `eslint.config.js`.

A localized one-off residual uses an inline `// eslint-disable-next-line <rule> --
<reason>` at the site instead of a file-wide exemption.

## Building Documentation

```bash
cd docs
make html
```

## Need Help?

- Read the [full Contributing Guide](https://als-apg.github.io/osprey/contributing/)
- Check [existing issues](https://github.com/als-apg/osprey/issues)
- Join [GitHub Discussions](https://github.com/als-apg/osprey/discussions)

## Code of Conduct

Be respectful, welcoming, and inclusive. Focus on what's best for the community.

---

For detailed guidelines please visit our **[complete Contributing documentation](https://als-apg.github.io/osprey/contributing/)**.
