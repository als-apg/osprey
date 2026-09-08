A `readwrite` run now checks writes against the channel limits database on
routes that previously went straight to the machine: `aioca`, pvaPy's
`Channel` setters, p4p's raw client under the flavour clients, and p4p values
written as a structure or as JSON, which are checked on the number they carry
rather than skipped for not looking like one.

Writes that carry nothing to bound are refused instead of passed: p4p `rpc`
and builder callbacks, pvaPy `parsePut`/`parsePutGet`, Tango commands, and
Tango group writes, which fan one value out to every device in the group. The
routes still unchecked in a `readwrite` run are listed in the architecture
docs.
