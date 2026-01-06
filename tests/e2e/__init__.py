"""End-to-end workflow tests for Osprey framework.

These tests validate complete workflows from project creation to execution,
using LLM-based evaluation to assess success.

Tests in this module:
- Create fresh projects from templates
- Execute real queries through the framework
- Use LLM judges to evaluate results
- Require API keys and are marked as slow

Run with: pytest tests/e2e/ -v -s
"""
