"""Embedded Redis server management using redis-wheel."""

import atexit
import logging
import subprocess
import time

import redis

from osprey.services.optimizer.utils.redis_client import ping_url

logger = logging.getLogger(__name__)


class EmbeddedRedisServer:
    """Manages an embedded Redis server lifecycle."""

    def __init__(self, port: int = 6379, timeout: float = 10.0) -> None:
        """Initialize the embedded Redis server manager.

        Args:
            port: Port to run Redis on (default: 6379)
            timeout: Timeout in seconds waiting for Redis to start
        """
        self.port = port
        self.timeout = timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._started = False

    def start(self) -> None:
        """Start the embedded Redis server."""
        if self._started:
            return

        try:
            import redis_wheel
        except ImportError as e:
            raise ImportError(
                "redis-wheel is not installed. To use the embedded Redis server, "
                "install the 'local' optional dependency group: uv sync --extra local"
            ) from e

        redis_server_path = redis_wheel.REDIS_SERVER_PATH
        logger.info("Starting embedded Redis server on port %s", self.port)

        self._process = subprocess.Popen(
            [redis_server_path, "--port", str(self.port), "--save", ""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self._wait_for_ready()
        self._started = True

        atexit.register(self.stop)
        logger.info("Embedded Redis server started successfully")

    def _wait_for_ready(self) -> None:
        """Wait for Redis to accept connections."""
        url = f"redis://localhost:{self.port}"
        start_time = time.monotonic()

        while time.monotonic() - start_time < self.timeout:
            try:
                ping_url(url)
                return
            except redis.ConnectionError:
                time.sleep(0.1)

        self.stop()
        raise TimeoutError(f"Redis did not start within {self.timeout} seconds")

    def stop(self) -> None:
        """Stop the embedded Redis server."""
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None
        self._started = False

    def is_running(self) -> bool:
        """Check if the Redis server is running.

        Test seam: read only by the embedded-Redis tests; production manages the
        lifecycle via ``start_embedded_redis`` + the instance's atexit ``stop``.
        """
        return self._process is not None and self._process.poll() is None


_embedded_redis: EmbeddedRedisServer | None = None


def start_embedded_redis(port: int = 6379, timeout: float = 10.0) -> EmbeddedRedisServer:
    """Start the embedded Redis server (singleton).

    Args:
        port: Port to run Redis on
        timeout: Timeout waiting for Redis to start

    Returns:
        The EmbeddedRedisServer instance
    """
    global _embedded_redis
    if _embedded_redis is None:
        _embedded_redis = EmbeddedRedisServer(port=port, timeout=timeout)
    _embedded_redis.start()
    return _embedded_redis


def stop_embedded_redis() -> None:
    """Stop the embedded Redis server if running.

    Test-teardown counterpart of ``start_embedded_redis`` (production stops the
    singleton via atexit); called from the app_tests conftest.
    """
    global _embedded_redis
    if _embedded_redis is not None:
        _embedded_redis.stop()
        _embedded_redis = None
