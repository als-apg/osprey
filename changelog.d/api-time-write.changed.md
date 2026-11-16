`osprey write-authorization list`, the web terminal's authorization chip and
`osprey status` read the connectors' new `datetime` instants; what they show is
unchanged. Grants, ledgers and requests written before this release are
refused as unreadable: approve a new authorization after upgrading. The
watchdog flags route serves `opened_at` with `+00:00` (it was `Z`), and `null`
for the flag of an unreadable hold record (it was an empty string).
