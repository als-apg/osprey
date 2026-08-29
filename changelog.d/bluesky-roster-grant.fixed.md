The Bluesky panel no longer answers `401 — invalid credential` to every user
of a multi-user deployment. `osprey build` now renders the persona projects
before the services compose files and reads their entitlements from the tree
it is building, so the `bluesky_web` sidecar lists each entitled user's
secret instead of the previous build's (or none).
