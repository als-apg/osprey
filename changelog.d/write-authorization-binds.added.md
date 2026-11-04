The deploy provisions the holders' write ledger, `var/write_ledger/` with its
`requests/` directory. Every web terminal binds `var/write_ledger/requests`
read-only (`OSPREY_WRITE_REQUESTS_DIR`) and the whole `var/write_ledger`
read-only for the programs' self-issued grants and their ledgers
(`OSPREY_WRITE_LEDGER_VIEW_DIR`); no terminal sees another login's grants. The
entrypoint joins both directories' group.
