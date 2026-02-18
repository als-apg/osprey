"""Shared fixtures for MATLAB MML MCP server tests.

Provides in-memory SQLite fixtures and sample function data for testing
tools and indexer in isolation.
"""

import json
import sqlite3

import pytest


def get_tool_fn(tool_or_fn):
    """Extract the raw async function from a FastMCP FunctionTool."""
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn


# Sample data matching the structure of matlab_dependencies.json
SAMPLE_DATA = {
    "nodes": [
        {
            "id": "getbpm",
            "file_path": "/mock/path/StorageRing/getbpm.m",
            "docstring": "GETBPM - Get BPM orbit data\n\nReads beam position monitor values.",
            "group": "StorageRing",
            "type": "defined",
            "in_degree": 25,
            "out_degree": 5,
        },
        {
            "id": "setsp",
            "file_path": "/mock/path/StorageRing/setsp.m",
            "docstring": "SETSP - Set setpoint for a family\n\nSets the setpoint value.",
            "group": "StorageRing",
            "type": "defined",
            "in_degree": 30,
            "out_degree": 8,
        },
        {
            "id": "getgolden",
            "file_path": "/mock/path/Common/getgolden.m",
            "docstring": "GETGOLDEN - Get golden orbit values\n\nReturns the golden orbit.",
            "group": "Common",
            "type": "defined",
            "in_degree": 15,
            "out_degree": 3,
        },
        {
            "id": "family2channel",
            "file_path": "/mock/path/MML/family2channel.m",
            "docstring": "FAMILY2CHANNEL - Convert family name to channel names\n\nMaps family to PV names.",
            "group": "MML",
            "type": "defined",
            "in_degree": 40,
            "out_degree": 2,
        },
        {
            "id": "bts_init",
            "file_path": "/mock/path/BTS/bts_init.m",
            "docstring": "BTS_INIT - Initialize BTS parameters\n\nSets up Booster-to-Storage ring.",
            "group": "BTS",
            "type": "script",
            "in_degree": 1,
            "out_degree": 12,
        },
        {
            "id": "orbitcorrection",
            "file_path": "/mock/path/StorageRing/orbitcorrection.m",
            "docstring": "ORBITCORRECTION - Perform orbit correction\n\nCalculates corrector strengths.",
            "group": "StorageRing",
            "type": "defined",
            "in_degree": 5,
            "out_degree": 10,
        },
        {
            "id": "gtb_config",
            "file_path": "/mock/path/GTB/gtb_config.m",
            "docstring": "GTB_CONFIG - Configure Gun-to-Booster parameters.",
            "group": "GTB",
            "type": "script",
            "in_degree": 2,
            "out_degree": 4,
        },
        {
            "id": "booster_ramp",
            "file_path": "/mock/path/Booster/booster_ramp.m",
            "docstring": "BOOSTER_RAMP - Control booster energy ramp\n\nManages ramping sequence.",
            "group": "Booster",
            "type": "defined",
            "in_degree": 3,
            "out_degree": 7,
        },
    ],
    "edges": [
        # orbitcorrection calls getbpm and setsp
        {"source": "orbitcorrection", "target": "getbpm"},
        {"source": "orbitcorrection", "target": "setsp"},
        {"source": "orbitcorrection", "target": "getgolden"},
        {"source": "orbitcorrection", "target": "family2channel"},
        # getbpm calls family2channel
        {"source": "getbpm", "target": "family2channel"},
        # setsp calls family2channel
        {"source": "setsp", "target": "family2channel"},
        # bts_init calls several
        {"source": "bts_init", "target": "getbpm"},
        {"source": "bts_init", "target": "setsp"},
        # booster_ramp calls setsp
        {"source": "booster_ramp", "target": "setsp"},
        # Self-loop (should be filtered during indexing)
        {"source": "getbpm", "target": "getbpm"},
    ],
    "function_definitions": [],
    "statistics": {},
}


@pytest.fixture
def sample_data_file(tmp_path):
    """Create a temporary matlab_dependencies.json file."""
    data_file = tmp_path / "matlab_dependencies.json"
    data_file.write_text(json.dumps(SAMPLE_DATA))
    return data_file


@pytest.fixture
def indexed_db(tmp_path, sample_data_file):
    """Build an indexed database from sample data and return its path."""
    from osprey.mcp_server.matlab.indexer import build_index

    db_path = tmp_path / "test_mml.db"
    build_index(data_file=sample_data_file, db_path=db_path)
    return db_path


@pytest.fixture
def db_connection(indexed_db):
    """Return a connection to the indexed test database."""
    conn = sqlite3.connect(str(indexed_db))
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture
def patch_db(indexed_db, monkeypatch):
    """Patch get_db_path to return the test database path."""
    monkeypatch.setattr(
        "osprey.mcp_server.matlab.db.get_db_path",
        lambda: indexed_db,
    )
    return indexed_db
