"""Record backends — where the agent record is stored.

Deliberately empty of re-exports. Each backend lives in its own module
(``osprey.agent_record.backends.embedded`` for the one-file DuckDB backend,
``osprey.agent_record.backends.clickhouse`` for the ClickHouse one)
and is imported by name, so that importing the package never pulls in a
backend's driver for code that only needs the record contract.
"""
