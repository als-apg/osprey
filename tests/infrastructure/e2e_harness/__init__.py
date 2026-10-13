"""Unit tests of the e2e harness itself: the judge, the SDK helpers, the MCP readiness barrier.

Nothing here runs an agent or reaches a model. The modules under test live in
``tests/e2e/`` because the e2e suite is what uses them; the tests live here so the
unit lane runs them on every push and the benchmark board never counts them.
"""
