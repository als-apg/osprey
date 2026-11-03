# osprey-connectors changelog fragments

Changelog entries for the `osprey-connectors` package live here, one small file
per change, and are folded into `packages/osprey-connectors/CHANGELOG.md` when
a connectors release is cut. The package releases on its own
`osprey-connectors-v*` tags, so its changes are recorded in its own changelog
rather than the framework's.

The rules are the repository's: see the root `changelog.d/README.md` for how a
fragment is named (`<name>.<type>.md`) and written. What differs is only which
directory a fragment goes in:

- a change to what ships in the connectors wheel — this package's `src/`,
  `pyproject.toml` or `README.md` — needs a fragment **here**;
- a change to the framework (`src/`, `plugins/`, any other `packages/` member)
  needs one in the root `changelog.d/`;
- a pull request that changes both brings one of each.

`uv run python scripts/changelog_fragments.py check --base origin/main` checks
both directories; the `lint` CI job runs the same command. The release fold for
this package is

```bash
uv run python scripts/changelog_fragments.py apply --root packages/osprey-connectors
```

This README is the only permanent file in this directory.
