The facility deployment guide has a new *What the perimeter needs* section
stating the four limits of the multi-user web tier: its own hostname or
host:port, one origin, the host network, and at most 100 users. The refusals
and warnings that enforce them name the limit and link that section, and a
roster past 100 users no longer suggests moving a port family's base port,
which never lifted the limit.
