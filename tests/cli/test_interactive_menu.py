"""Tests for interactive menu utilities."""

import socket

from osprey.cli.interactive_menu import _check_simulation_ioc_running


class TestCheckSimulationIOCRunning:
    """Tests for the _check_simulation_ioc_running helper function."""

    def test_port_closed_returns_false(self):
        """Test that port check returns False when no service is listening."""
        # Use a port that's very unlikely to be open
        assert _check_simulation_ioc_running("localhost", 59999) is False

    def test_invalid_host_returns_false(self):
        """Test that invalid hostname returns False gracefully."""
        assert _check_simulation_ioc_running("invalid.host.that.does.not.exist", 5064) is False

    def test_port_open_returns_true(self):
        """Test that port check returns True when a service is listening."""
        # Create a temporary server socket to test against
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("localhost", 0))  # Bind to random available port
            server.listen(1)
            port = server.getsockname()[1]

            # Now test that our function detects it
            assert _check_simulation_ioc_running("localhost", port) is True
        finally:
            server.close()

    def test_default_parameters(self):
        """Test default host and port parameters."""
        # Just verify it doesn't crash with defaults
        # Result depends on whether an IOC is actually running
        result = _check_simulation_ioc_running()
        assert isinstance(result, bool)

    def test_timeout_on_unresponsive_host(self):
        """Test that function times out gracefully on unresponsive hosts."""
        # 10.255.255.1 is a non-routable IP that should timeout
        # This test verifies the timeout works (should complete in ~1 second)
        import time

        start = time.time()
        result = _check_simulation_ioc_running("10.255.255.1", 5064)
        elapsed = time.time() - start

        assert result is False
        # Should timeout within reasonable time (1s timeout + overhead)
        assert elapsed < 3.0, f"Timeout took too long: {elapsed:.1f}s"
