"""Pytest fixtures for connector tests.

Fixtures defined here are automatically available to all test files in
``tests/connectors/`` and any subdirectories.

The MongoDB fixtures spin up a real container via testcontainers and
skip cleanly when Docker is unavailable, matching the pattern used by
``tests/services/ariel_search/conftest.py``.
"""

from datetime import datetime, timedelta

import pytest

from tests._container_support import (
    is_docker_available,
    start_or_skip,
    stop_quietly,
    wait_until_ready,
)


@pytest.fixture(scope="session")
def mongodb_container():
    """Start a MongoDB container for the test session.

    Yields a dict with connection parameters. Skips the entire chain
    of dependent tests if Docker isn't available.

    The store is waited for rather than assumed: testcontainers' readiness
    check runs inside the container, so the first connection from the host on
    the freshly published port can be reset while the port forwarder settles.
    """
    if not is_docker_available():
        pytest.skip(
            "Docker not available — install Docker Desktop or Podman "
            "to run MongoDB archiver integration tests."
        )

    try:
        from testcontainers.mongodb import MongoDbContainer
    except ImportError:
        pytest.skip("testcontainers[mongodb] not installed")

    username = "testuser"
    password = "testpass123"
    db_name = "test_archiver_db"
    collection_name = "test_archiver_collection"
    auth_db = "admin"

    container = start_or_skip(
        lambda: MongoDbContainer("mongo:7", username=username, password=password),
        label="mongodb",
    )

    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(27017))

    def answers() -> None:
        """Ask the store for a pong, raising while it is not yet answering.

        The one-second selection window is what makes this a poll rather than
        one long wait: a client left on its own default would spend the whole
        retry budget inside a single attempt.
        """
        from pymongo import MongoClient

        client = MongoClient(
            host,
            port,
            username=username,
            password=password,
            authSource=auth_db,
            serverSelectionTimeoutMS=1000,
        )
        try:
            client.admin.command("ping")
        finally:
            client.close()

    wait_until_ready(answers, "mongodb")

    try:
        yield {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "auth_db": auth_db,
            "db_name": db_name,
            "collection_name": collection_name,
        }
    finally:
        stop_quietly(container)


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
    """
    password_env = "MONGODB_TEST_PASSWORD"
    monkeypatch.setenv(password_env, mongodb_container["password"])

    config = {
        "host": mongodb_container["host"],
        "port": mongodb_container["port"],
        "name": mongodb_container["db_name"],
        "collection": mongodb_container["collection_name"],
        "auth": mongodb_container["auth_db"],
        "username": mongodb_container["username"],
        "password_env": password_env,
        "timeout": 10,
    }

    yield config
