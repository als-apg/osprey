# Osprey Framework - Latest Release (v2026.5.0)

**CalVer adoption, doc-executability CI gate, deploy skill & friction logging**

## What's New in v2026.5.0

### Highlights

- **CalVer**: switched from SemVer to `YYYY.MM.MICRO`. The first segment identifies the release window; the micro segment increments for hotfixes. Compatibility is documented in release notes, not encoded in the version string. See `CHANGELOG.md` preamble.
- **Doc-executability CI gate**: a new reusable workflow extracts bash blocks from `installation.rst` / `build-interview.rst`, builds a wheel from the current SHA, installs it on a clean Ubuntu runner, and runs the docs end-to-end. `release.yml` gates `publish-to-pypi` on this check, so doc-vs-wheel drift is caught before tag.
- **`osprey-build-deploy` skill**: a second installable skill alongside `build-interview`. The interview's new Phase 8 lands a project-local copy in your generated profile repo, so the deploy operator is wired up before you see the final summary.
- **Friction log capture**: `build-interview` passively logs colleague hesitation, mid-flow questions, and contradictions to `build-profile/.interview-notes.md`. Phase 9b folds notes into the feedback email, opt-in.
- **Vendor assets default to CDN**: web interfaces load `xterm.js`, Plotly, highlight.js, KaTeX, marked from `cdn.jsdelivr.net` / `cdn.plot.ly` out of the box. Set `OSPREY_OFFLINE=1` for firewalled deployments.

### Breaking changes

- **`osprey init` retired** — folded into `osprey build --preset`. Use `osprey build <name> --preset <hello-world|control-assistant|...>` for bundled scaffolding, or `osprey build <name> <profile.yml>` for a custom profile. Override files via `-O FILE` (repeatable, deep-merge); key-value overrides via `--set KEY.PATH=VALUE`.
- **`osprey migrate` removed** — legacy config-migration workflow retired. Run `osprey init` … wait, that's gone too — use `osprey build` fresh, or hand-edit.
- **`migrate-legacy` skill removed** — migration is now a path inside `build-interview` (Phase 1 Q3 → Migration Scan → Phase 5.5 → Phase 8).
- **`lattice_design` template removed** — `--template lattice_design` and `data_bundle: lattice_design` are no longer valid.
- **Manifest schema bumped to 1.2.0** — `.osprey-manifest.json` records `build_args` (renamed from `init_args`) with `source: "preset"|"profile"` discriminator. No backward-compat shim.

### Other changes

See `CHANGELOG.md` for the full list — connector batch writes, `osprey vendor fetch --insecure`, build-profile typo warnings, CBORG model pinning, and more.

---

## Installation

```bash
uv tool install --upgrade osprey-framework
```

`uv tool` shims the `osprey` CLI properly onto `PATH`. The previous `pip install` form is no longer the recommended path.

---

## What's Next?

Check out our [documentation](https://als-apg.github.io/osprey) for:
- The [installation guide](https://als-apg.github.io/osprey/getting-started/installation.html) — now CI-validated end-to-end against the released wheel
- The [build interview](https://als-apg.github.io/osprey/getting-started/build-interview.html) — guided 10–15 min setup for your facility
- ARIEL Quick Start, native capabilities, `osprey eject`, and the full tutorial series

## Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
