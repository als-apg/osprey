The published `facility.json` JSON Schema now carries `$schema` and a public
`$id` (`https://als-apg.github.io/osprey/latest/schemas/facility/<N>.schema.json`).
The compatibility check treats both as identity, not as a breaking addition,
and the package's compatibility window against framework releases is now
enforced in CI as osprey-connectors' is.
