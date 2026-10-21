The virtual accelerator's compose healthcheck now fails when its publishing
passes keep failing. The runner keeps a health record -- `serving`,
`degraded` or `failed`, with the latest pass and pass counters -- rewrites it
to `/run/osprey-va/health.json` after every pass, and the healthcheck reads
that file in place of a TCP connect to the Channel Access port. More than
three failed passes in a row mark the container unhealthy; a pass that
succeeds marks it healthy again. The model RPC's `status` reply carries the
same record as `health`.
