# Changelog

All notable changes to `osprey-panels` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this package adheres to [Calendar Versioning](https://calver.org/).

The package has its own calendar stream, released from its own
`osprey-panels-vYYYY.M.P` tags by
`.github/workflows/release-package.yml`, independently of the framework's `v*` tags.
Changes from before this stream began are recorded in the repository's root
`CHANGELOG.md`, under the framework releases that shipped them.

Unreleased changes are written as fragments in `changelog.d/` beside this file
and folded into the `[Unreleased]` section below when a release is cut:

    python scripts/changelog_fragments.py apply --root packages/osprey-panels

## [Unreleased]
