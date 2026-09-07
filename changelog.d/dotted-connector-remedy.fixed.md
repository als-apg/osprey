A write refused because writes are off now tells an operator with a custom
(dotted) connector type what to actually edit. It quotes the
`control_system.connector:` mapping with the connector type as one key under
it, instead of a dotted `control_system.connector.<type>.writes_enabled` key
that a build profile would split into a nest nothing reads.
