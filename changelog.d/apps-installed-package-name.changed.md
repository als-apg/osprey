An installed application's package may not live under `osprey` or
`osprey_connectors`, as a deployment application's may not: the build and
`osprey app check` refuse it with `taken_package`, and the import boundary
reads every import under either as OSPREY's.
