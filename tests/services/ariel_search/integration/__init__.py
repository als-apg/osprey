"""Integration tests for ARIEL Search.

These tests require Docker and use real PostgreSQL + pgvector.
They test actual SQL queries, migrations, and database operations.

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.

Run integration tests:
    pytest tests/services/ariel_search/integration/ -v

If Docker is not available, tests will skip with a warning message.
"""
