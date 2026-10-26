`osprey-connectors` now versions from its own `osprey-connectors-vYYYY.M.P`
calendar tags instead of the framework's `v*` tags, so it releases when the
package changes rather than with every framework release. Each release stays
backward-compatible with every supported framework release, and the framework
keeps declaring only a floor on it. A checkout building the package needs those
tags fetched (a fork or mirror must sync the `osprey-connectors-v*` tags):
without one reachable it builds as `0.0.postN`, and `osprey up --dev` now
refuses such a build before staging it, naming the tag glob.
