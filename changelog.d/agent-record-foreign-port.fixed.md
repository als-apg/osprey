A host session whose agent-record port is held by another project's record now
says so at once (it refused the host login), instead of waiting out the bound
and, on a loaded host, reporting a `ReadTimeout`. A later probe that times out
no longer replaces the answer an earlier probe had.
