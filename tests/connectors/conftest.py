"""Pytest fixtures for connector tests.

Fixtures defined here are automatically available to all test files in
``tests/connectors/`` and any subdirectories.

The MongoDB fixtures seed the session's ``mongodb_container`` (defined in
``tests/conftest.py``, so the recorder and simulation lanes share it), whose
bring-up is the shared recipe in ``tests/_mongo_container.py``: the image pin,
the start, the readiness wait and the teardown all live there.

``ioc_prefix``, the record prefix every soft-IOC suite names its records
with, is the harness's (``osprey_connectors_tests._harness._guards``), shared with the
connectors suite.
"""

from datetime import datetime, timedelta

import pytest

from osprey_connectors_tests._harness._guards import (
    ioc_prefix,  # noqa: F401 - a fixture, used by name
)


@pytest.fixture(scope="function")
def mongodb_test_data(mongodb_container):
    """Seed the test collection with 3 days of hourly PV data.

    Cleans up after each test so the collection state stays predictable.
    """
    from pymongo import MongoClient

    client = MongoClient(
        host=mongodb_container["host"],
        port=mongodb_container["port"],
        username=mongodb_container["username"],
        password=mongodb_container["password"],
        authSource=mongodb_container["auth_db"],
    )

    collection = client[mongodb_container["db_name"]][mongodb_container["collection_name"]]
    collection.delete_many({})

    start_date = datetime(2024, 1, 1, 0, 0, 0)
    end_date = datetime(2024, 1, 4, 0, 0, 0)
    channels = ["BEAM:CURRENT", "BEAM:LIFETIME", "BEAM:ENERGY"]

    documents = []
    current_date = start_date
    hour_count = 0
    while current_date < end_date:
        doc = {
            "date": current_date,
            "BEAM:CURRENT": 100.0 + (hour_count % 100),
            "BEAM:LIFETIME": 10.0 + (hour_count % 40),
            "BEAM:ENERGY": 1.0 + (hour_count % 15) * 0.1,
        }
        documents.append(doc)
        current_date += timedelta(hours=1)
        hour_count += 1

    if documents:
        collection.insert_many(documents)
    collection.create_index("date")

    yield {
        "channels": channels,
        "start_date": start_date,
        "end_date": end_date,
        "document_count": len(documents),
    }

    collection.delete_many({})
    client.close()


@pytest.fixture
def mongodb_config(mongodb_container, monkeypatch):
    """Provide a connector-ready config dict and set the password env var.

    Tests that need seeded data should also depend on ``mongodb_test_data``;
    this fixture is intentionally not coupled to the data fixture so that
    error-path tests (missing config keys, etc.) don't pay seeding cost.
    The in-network address overrides are cleared: they beat the config, so a
    shell that sets them would send these tests to another store.
    """
    from osprey_connectors.archiver._mongo_client import HOST_OVERRIDE_ENV, PORT_OVERRIDE_ENV

    monkeypatch.delenv(HOST_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(PORT_OVERRIDE_ENV, raising=False)
    password_env = "MONGODB_TEST_PASSWORD"
    monkeypatch.setenv(password_env, mongodb_container["password"])

    config = {
        "host": mongodb_container["host"],
        "port": mongodb_container["port"],
        "name": mongodb_container["db_name"],
        "collection": mongodb_container["collection_name"],
        "auth": {
            "source": mongodb_container["auth_db"],
            "username": mongodb_container["username"],
            "password_env": password_env,
        },
        "timeout_s": 10,
    }

    yield config
