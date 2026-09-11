The agent's SQL tool now queries the logbook through a SELECT-only Postgres
role (``<username>_ro``) rather than the role ingestion writes with, so a
statement naming a server-side function such as ``pg_read_file()`` is refused
by the database itself. The role is created when the ARIEL database volume is
first initialized, and ``osprey up`` mints its password as
``ARIEL_DB_READONLY_PASSWORD``. A database created before this release has no
such role and keeps working on the previous connection — the agent logs one
warning at start-up, and "Read-only role for the SQL tool" in the standalone
deployment how-to has a one-shot command for adopting it.
