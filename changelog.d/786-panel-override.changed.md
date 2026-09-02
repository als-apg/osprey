The ARIEL panel no longer rewrites an explicitly configured
`ariel.database.uri` to point at a container loopback address. An operator
who relied on that rewrite should either drop the URI so the address is
derived from `ARIEL_DATABASE_HOST`/`ARIEL_DATABASE_PORT`, or run the panel
in host network mode so the authored loopback address is already correct.
