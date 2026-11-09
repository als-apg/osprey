A connector block can say its target has nothing to probe with an explicit
`gateways: []`. The derivation carries that as `declares_no_endpoints`, and
child verification names it in plain words; a missing table is not read as
that statement.
