The connectors package is type-checked under mypy's strict checks and the error
codes that catch dead or doubtful code. Every overriding method is marked
`@override`, and an explicit `Any` appears only as the `ContractAny` alias for
the values a public contract leaves untyped.
