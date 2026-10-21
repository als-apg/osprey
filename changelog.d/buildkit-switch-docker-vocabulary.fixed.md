The BuildKit pre-build check reads `DOCKER_BUILDKIT` the way Docker does (`false`, `true`, `F` …) and refuses, by name, a value Docker would refuse.
