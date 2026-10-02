"""Pin which HTTP client stack the suite's ASGI test client is built on."""

import subprocess
import sys

import httpx2
from starlette.testclient import TestClient


def test_starlette_test_client_is_built_on_httpx2():
    assert issubclass(TestClient, httpx2.Client)


def test_importing_the_test_client_warns_nothing():
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import httpx\nimport starlette.testclient"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
