CI: a pull request that changes only `docs/` and `changelog.d/` now skips the
pull-request-scoped e2e and deploy lanes. The unit, lint, docs-build and other
always-on lanes still run, and the `full-ci` label restores the full run.
