`osprey-connectors` now releases on its own: an `osprey-connectors-vYYYY.M.P`
tag publishes it from `release-connectors.yml`, gated on a green CI run of the
tagged tree, with release notes from the package's own
`packages/osprey-connectors/CHANGELOG.md`. A framework `v*` tag no longer builds
or publishes `osprey-connectors`; the framework's `osprey-connectors>=` floor
must already be on PyPI, which the release now checks.
