An external graph store whose corpus lives in a database other than `neo4j` can
now be named with `services.graphdb.database`. Both halves — the seeder's
writes and the agent's reads — open every session against it, so they cannot
end up on different databases. The store OSPREY runs for itself serves one
database and is unaffected.
