"""Pytest fixtures for connector tests.

This file is automatically discovered and loaded by pytest - no explicit import needed.

How it works:
- Pytest automatically searches for `conftest.py` files in test directories
- Fixtures defined here are automatically available to all test files in this directory
  (tests/connectors/) and any subdirectories
- Test functions can use these fixtures by simply including them as parameters
- Example: `def test_something(self, mongodb_config):` will automatically receive
  the `mongodb_config` fixture value
"""

import os
import subprocess
import time
from datetime import datetime, timedelta

import pytest
from pymongo import MongoClient

# Try to import pytest-docker, skip tests if not available
try:
    from pytest_docker import docker_compose
except ImportError:
    docker_compose = None


def _check_docker_available():
    """Check if Docker is available."""
    try:
        subprocess.run(["docker", "ps"], check=True, capture_output=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="module")
def mongodb_container(tmp_path_factory):
    """
    Start a MongoDB container for testing using pytest-docker.

    Returns a dictionary with connection information:
    - host: MongoDB host (usually 'localhost')
    - port: MongoDB port (dynamically allocated)
    - username: Test username
    - password: Test password
    - auth_db: Authentication database
    - db_name: Test database name
    - collection_name: Test collection name

    The container is automatically cleaned up after all tests in the module.
    """
    if docker_compose is None:
        pytest.skip("pytest-docker not installed, skipping MongoDB tests")

    if not _check_docker_available():
        pytest.skip("Docker not available, skipping MongoDB tests")

    # MongoDB configuration
    username = "testuser"
    password = "testpass123"
    auth_db = "admin"
    db_name = "test_archiver_db"
    collection_name = "test_archiver_collection"

    # Create temporary directory for docker-compose file
    temp_dir = tmp_path_factory.mktemp("mongodb_test")
    compose_file = temp_dir / "docker-compose.yml"

    # Write docker-compose.yml
    compose_content = f"""version: "3.8"
services:
  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: {username}
      MONGO_INITDB_ROOT_PASSWORD: {password}
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 2s
      timeout: 5s
      retries: 10
"""
    compose_file.write_text(compose_content)

    # Use pytest-docker's docker_compose as context manager
    with docker_compose(str(compose_file.parent), compose_file_name=compose_file.name):
        # Wait for MongoDB to be ready
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            try:
                client = MongoClient(
                    host="localhost",
                    port=27017,
                    username=username,
                    password=password,
                    authSource=auth_db,
                    serverSelectionTimeoutMS=2000,
                )
                client.admin.command("ping")
                client.close()
                break
            except Exception:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                time.sleep(1)

        yield {
            "host": "localhost",
            "port": 27017,
            "username": username,
            "password": password,
            "auth_db": auth_db,
            "db_name": db_name,
            "collection_name": collection_name,
        }


@pytest.fixture(scope="function")
def mongodb_test_data(mongodb_container):
    """
    Insert test PV data into MongoDB collection.

    Creates sample documents with structure:
    {date: ISODate(...), PV1: value1, PV2: value2, ...}

    The data spans multiple days with timestamps every hour.
    """
    # Connect to MongoDB
    client = MongoClient(
        host=mongodb_container["host"],
        port=mongodb_container["port"],
        username=mongodb_container["username"],
        password=mongodb_container["password"],
        authSource=mongodb_container["auth_db"],
    )

    db = client[mongodb_container["db_name"]]
    collection = db[mongodb_container["collection_name"]]

    # Clear any existing data
    collection.delete_many({})

    # Generate test data: 3 days of hourly data
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    end_date = datetime(2024, 1, 4, 0, 0, 0)
    current_date = start_date

    # PV names for testing
    pv_names = ["BEAM:CURRENT", "BEAM:LIFETIME", "BEAM:ENERGY"]

    documents = []
    hour_count = 0

    while current_date < end_date:
        doc = {"date": current_date}
        # Add PV values with some variation
        for pv in pv_names:
            if pv == "BEAM:CURRENT":
                # Current varies between 100-200 mA
                doc[pv] = 100.0 + (hour_count % 100)
            elif pv == "BEAM:LIFETIME":
                # Lifetime varies between 10-50 hours
                doc[pv] = 10.0 + (hour_count % 40)
            elif pv == "BEAM:ENERGY":
                # Energy varies between 1.0-2.5 GeV
                doc[pv] = 1.0 + (hour_count % 15) * 0.1

        documents.append(doc)
        current_date += timedelta(hours=1)
        hour_count += 1

    # Insert documents in batches
    if documents:
        collection.insert_many(documents)

    # Create index on date field for faster queries
    collection.create_index("date")

    yield {
        "pv_names": pv_names,
        "start_date": start_date,
        "end_date": end_date,
        "document_count": len(documents),
    }

    # Cleanup: remove test data
    collection.delete_many({})
    client.close()


@pytest.fixture
def mongodb_config(mongodb_container, mongodb_test_data):
    """
    Provide MongoDB connection config for connector tests.

    Sets up the password environment variable and returns config dict.
    """
    password_env = "MONGODB_TEST_PASSWORD"
    os.environ[password_env] = mongodb_container["password"]

    config = {
        "host": mongodb_container["host"],
        "port": mongodb_container["port"],
        "name": mongodb_container["db_name"],
        "collection": mongodb_container["collection_name"],
        "auth": mongodb_container["auth_db"],
        "username": mongodb_container["username"],
        "password_env": password_env,
        "timeout": 10,  # Shorter timeout for tests
    }

    yield config

    # Cleanup: remove password env var
    if password_env in os.environ:
        del os.environ[password_env]
