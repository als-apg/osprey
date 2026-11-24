One release workflow, `release-package.yml`, now publishes every component
package (`osprey-connectors-v*`, `osprey-facility-v*`, `osprey-panels-v*`) from
a single stream table, and each stream gates on its own CI lanes. The
connectors compatibility lanes are generalised to osprey-facility, and the
framework release verifies every declared package floor together. A
hyphenated pre-release tag such as `-rc1` is refused; use `rc1`.
