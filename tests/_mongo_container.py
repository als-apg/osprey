"""One recipe for a throwaway MongoDB.

Four suites need one — the connector fixtures, the world-contract fixtures and the
two seeder suites. They differ in the credentials they hand out, the server flags
they ask for and what they put in the mapping they yield, not in how the store is
brought up, so the bring-up lives here once.

**Why the wait cannot be left to the container library.** ``mongo`` boots twice
when it has a root user to create: a throwaway server for the init scripts, then
the real one. The readiness strategy the container library applies to this image
matches the log line ``waiting for connections``, and both servers emit it, so
``start()`` returns while the published port can still belong to the first one.
The recipe therefore waits on a connection the test itself could make, and hands
the container to the wait so a store that exited is reported as dead rather than
as slow.

The leading underscore keeps the module out of pytest collection: ``python_files``
matches ``test_*.py``/``*_test.py`` only, and a helper module that got collected
would report its imports as test failures.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import pytest

from tests._container_support import start_or_skip, stop_quietly, wait_until_ready

#: Server line every throwaway store in the suite runs. One pin, so a store one
#: suite seeds and another reads back is the same server.
MONGO_IMAGE = "mongo:7"

#: Port mongod listens on inside the container.
MONGO_PORT = 27017

#: Database a root user authenticates against.
MONGO_AUTH_DB = "admin"


@contextmanager
def started_mongo(
    label: str,
    *,
    username: str,
    password: str,
    command: Sequence[str] | None = None,
) -> Iterator[tuple[str, int]]:
    """A started MongoDB that has answered, stopped again when the caller is done.

    Args:
        label: Human-readable name for the store, used in skip and failure messages.
        username: Root user the container creates, and the user the probe
            authenticates as against :data:`MONGO_AUTH_DB`.
        password: That user's password.
        command: Server flags to run mongod with, for a suite that needs any.

    Yields:
        The host and the published port to reach the store on.

    Raises:
        Skipped: Via ``pytest.skip`` when the mongodb extra is absent, or when
            the container will not start.
        AssertionError: When the store started and never answered.
    """
    try:
        from testcontainers.community.mongodb import MongoDbContainer
    except ImportError:
        pytest.skip("testcontainers[mongodb] not installed")

    def build():
        container = MongoDbContainer(MONGO_IMAGE, username=username, password=password)
        return container.with_command(list(command)) if command else container

    container = start_or_skip(build, label=label)
    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(MONGO_PORT))

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
            authSource=MONGO_AUTH_DB,
            serverSelectionTimeoutMS=1000,
        )
        try:
            client.admin.command("ping")
        finally:
            client.close()

    wait_until_ready(answers, label, container=container)
    try:
        yield host, port
    finally:
        stop_quietly(container)
