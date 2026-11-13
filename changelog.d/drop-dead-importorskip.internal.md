Tests import core dependencies (pyAT, pyAML, bluesky, ophyd-async, DuckDB, Pillow,
psycopg, the agent SDK) directly instead of through `pytest.importorskip`, which
could never skip, and import at the top of the file, dropping 211 `# noqa: E402`.
