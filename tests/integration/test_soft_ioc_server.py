"""Integration tests for soft IOC server with custom backends.

These tests verify the complete flow:
1. Generate soft IOC code with custom simulation backends
2. Run the IOC as a real caproto server in a subprocess
3. Connect with a caproto client
4. Perform PV read/write operations
5. Verify backend method calls (initialize, on_write, step)
6. Test chained backend behaviors (physics simulation, fault injection)
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from osprey.generators.soft_ioc_template import generate_soft_ioc

if TYPE_CHECKING:
    from caproto.threading.client import PV, Context


# Skip all tests if caproto is not available
pytest.importorskip("caproto", reason="caproto not installed")


# =============================================================================
# Tracking Backend Template
# =============================================================================

TRACKING_BACKEND_MODULE = textwrap.dedent('''
    """Tracking backend for integration tests.

    Logs all method calls to a JSON file for verification.
    """

    import json
    from pathlib import Path
    from typing import Any


    class TrackingBackend:
        """Test backend that logs all method calls to a file."""

        def __init__(self, log_file: str, scale: float = 1.0, offset: float = 0.0):
            """Initialize backend.

            Args:
                log_file: Path to JSON file for logging calls
                scale: Multiplier for on_write return values
                offset: Offset added to on_write return values
            """
            self.log_file = Path(log_file)
            self.scale = scale
            self.offset = offset
            self._calls: list[dict] = []
            self._state: dict[str, Any] = {}

        def _log(self, method: str, **kwargs) -> None:
            """Log a method call."""
            self._calls.append({"method": method, **kwargs})
            self.log_file.write_text(json.dumps(self._calls, indent=2))

        def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
            """Initialize PVs and return initial values."""
            self._log("initialize", pv_count=len(pv_definitions))

            initial_values = {}
            for pv in pv_definitions:
                name = pv["name"]
                # Skip heartbeat PV
                if "HEARTBEAT" in name:
                    continue
                # Set initial value based on name
                if ":SP" in name:
                    initial_values[name] = 10.0
                elif ":RB" in name:
                    initial_values[name] = 10.0
                else:
                    initial_values[name] = 0.0
                self._state[name] = initial_values[name]

            return initial_values

        def on_write(self, pv_name: str, value: Any) -> dict[str, Any]:
            """Handle setpoint write, return transformed value."""
            self._log("on_write", pv_name=pv_name, value=float(value))
            self._state[pv_name] = value

            # Transform value: result = value * scale + offset
            transformed = float(value) * self.scale + self.offset
            updates = {pv_name: transformed}

            # If this is a setpoint, also update the paired readback
            if ":SP" in pv_name:
                rb_name = pv_name.replace(":SP", ":RB")
                if rb_name in self._state or True:  # Always try to update RB
                    updates[rb_name] = transformed
                    self._state[rb_name] = transformed

            return updates

        def step(self, dt: float) -> dict[str, Any]:
            """Advance simulation - log call but no updates."""
            self._log("step", dt=round(dt, 4))
            return {}
''')


# =============================================================================
# IOC Server Wrapper Template
# =============================================================================

IOC_WRAPPER_TEMPLATE = textwrap.dedent('''
    """IOC server wrapper for integration tests.

    This wrapper:
    1. Imports the tracking backend
    2. Starts the caproto server
    3. Writes the actual port to a file for client discovery
    4. Handles graceful shutdown on SIGTERM
    """

    import signal
    import sys
    from pathlib import Path

    # Add backend module path to sys.path
    backend_path = Path("{backend_path}")
    sys.path.insert(0, str(backend_path.parent))

    from tracking_backend import TrackingBackend

    # Import generated IOC code
    exec(open("{ioc_path}").read())

    # Configuration from test
    LOG_FILE = "{log_file}"
    PORT_FILE = "{port_file}"
    SCALE = {scale}
    OFFSET = {offset}

    # Signal handler for graceful shutdown
    shutdown_requested = False

    def handle_signal(signum, frame):
        global shutdown_requested
        shutdown_requested = True
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if __name__ == "__main__":
        import os
        from caproto.server import run

        # Set EPICS CA environment variables for port
        port = {port}
        os.environ["EPICS_CA_SERVER_PORT"] = str(port)
        os.environ["EPICS_CAS_SERVER_PORT"] = str(port)
        os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
        os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

        # Create backend with tracking
        backend = TrackingBackend(
            log_file=LOG_FILE,
            scale=SCALE,
            offset=OFFSET,
        )

        # Create IOC instance with empty prefix (PV names are already fully qualified)
        ioc = {ioc_class}(prefix="", backend=backend)

        # Write port file to indicate server is starting
        Path(PORT_FILE).write_text(str(port))

        # Run server on localhost only to avoid network issues
        run(
            ioc.pvdb,
            interfaces=["127.0.0.1"],
        )
''')


# =============================================================================
# IOC Server Fixture
# =============================================================================


class IOCServer:
    """Manages IOC server lifecycle for tests."""

    def __init__(self, tmp_path: Path):
        """Initialize server manager.

        Args:
            tmp_path: Temporary directory for test files
        """
        self.tmp_path = tmp_path
        self.process: subprocess.Popen | None = None
        self.port: int | None = None
        self.log_file = tmp_path / "backend_calls.json"
        self.port_file = tmp_path / "server_port.txt"
        self.ioc_file = tmp_path / "generated_ioc.py"
        self.backend_file = tmp_path / "tracking_backend.py"
        self.wrapper_file = tmp_path / "ioc_wrapper.py"
        self._context: Context | None = None

    @staticmethod
    def _find_free_port() -> int:
        """Find a free port for the server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start(
        self,
        channels: list[dict],
        pairings: dict[str, str] | None = None,
        port: int | None = None,
        scale: float = 1.0,
        offset: float = 0.0,
        update_rate: float = 10.0,
    ) -> int:
        """Start the IOC server.

        Args:
            channels: List of PV definitions
            pairings: SP->RB pairings (optional)
            port: CA server port (default: dynamically allocated)
            scale: Scale factor for backend
            offset: Offset for backend
            update_rate: Simulation update rate in Hz

        Returns:
            Port number the server is running on
        """
        if pairings is None:
            pairings = {}

        # Use dynamic port allocation if no port specified
        if port is None:
            port = self._find_free_port()

        # Write tracking backend module
        self.backend_file.write_text(TRACKING_BACKEND_MODULE)

        # Generate IOC code
        config = {
            "ioc": {"name": "test_ioc", "port": port},
            "base": {
                "type": "passthrough",  # We use wrapper's backend
                "noise_level": 0.0,
                "update_rate": update_rate,
            },
            "overlays": [],
        }
        ioc_code = generate_soft_ioc(config, channels, pairings)

        # Remove the if __name__ == '__main__' block since wrapper handles startup
        # Find where the main block starts and truncate
        main_marker = "if __name__ == '__main__':"
        if main_marker in ioc_code:
            ioc_code = ioc_code[: ioc_code.index(main_marker)]

        self.ioc_file.write_text(ioc_code)

        # Get IOC class name from config
        ioc_class = "TestIoc"  # _to_class_name("test_ioc")

        # Write wrapper script
        wrapper_code = IOC_WRAPPER_TEMPLATE.format(
            backend_path=self.backend_file,
            ioc_path=self.ioc_file,
            log_file=self.log_file,
            port_file=self.port_file,
            scale=scale,
            offset=offset,
            ioc_class=ioc_class,
            port=port,
        )
        self.wrapper_file.write_text(wrapper_code)

        # Set up environment for subprocess
        env = os.environ.copy()
        # Ensure EPICS CA settings for localhost-only operation
        env["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
        env["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
        env["EPICS_CA_SERVER_PORT"] = str(port)

        # Start server subprocess
        self.process = subprocess.Popen(
            [sys.executable, str(self.wrapper_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for port file to appear (indicates server is starting)
        timeout = 10.0
        start_time = time.time()
        while not self.port_file.exists():
            if time.time() - start_time > timeout:
                # Get any error output
                if self.process.poll() is not None:
                    _, stderr = self.process.communicate()
                    raise TimeoutError(
                        f"IOC server failed to start within {timeout}s. "
                        f"Process exited with code {self.process.returncode}. "
                        f"stderr: {stderr.decode()}"
                    )
                raise TimeoutError(f"IOC server failed to start within {timeout}s")
            time.sleep(0.1)

        # Read port from file
        self.port = int(self.port_file.read_text().strip())

        # Wait for server to fully initialize (longer for CI environments)
        time.sleep(1.0)

        # Check if process is still running
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(
                f"IOC server exited unexpectedly with code {self.process.returncode}. "
                f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            )

        return self.port

    def stop(self) -> None:
        """Stop the IOC server gracefully."""
        if self.process is not None:
            # Try graceful shutdown first
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self.process.kill()
                self.process.wait()
            self.process = None

        # Clean up client context
        if self._context is not None:
            self._context = None

    def get_backend_calls(self) -> list[dict]:
        """Get all backend method calls from log file.

        Returns:
            List of call records, each with 'method' key and additional args
        """
        if not self.log_file.exists():
            return []
        content = self.log_file.read_text()
        if not content.strip():
            return []
        return json.loads(content)

    def get_context(self) -> Context:
        """Get or create a caproto client context.

        Creates a new context configured for this server's port.
        Note: caproto's SharedBroadcaster reads from os.environ at creation time,
        so we must set environment BEFORE creating the SharedBroadcaster.

        Returns:
            Caproto threading client Context
        """
        from caproto.threading.client import Context, SharedBroadcaster

        if self._context is None:
            # IMPORTANT: Set environment BEFORE creating SharedBroadcaster
            # SharedBroadcaster reads os.environ at __init__ time
            os.environ["EPICS_CA_ADDR_LIST"] = f"127.0.0.1:{self.port}"
            os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
            os.environ["EPICS_CA_SERVER_PORT"] = str(self.port)

            # Create a dedicated broadcaster for this server's port
            broadcaster = SharedBroadcaster()

            self._context = Context(broadcaster=broadcaster)
        return self._context

    def get_pv(self, pv_name: str, timeout: float = 10.0) -> PV:
        """Get a PV object for the given name.

        Args:
            pv_name: PV name to connect to
            timeout: Connection timeout in seconds (default: 10s for CI environments)

        Returns:
            Connected PV object
        """
        ctx = self.get_context()
        (pv,) = ctx.get_pvs(pv_name)

        # Wait for connection with exponential backoff
        start = time.time()
        sleep_time = 0.1
        while not pv.connected:
            if time.time() - start > timeout:
                raise TimeoutError(f"Failed to connect to PV {pv_name}")
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 1.5, 1.0)  # Cap at 1s

        return pv


@pytest.fixture
def ioc_server(tmp_path):
    """Fixture that manages IOC server lifecycle."""
    server = IOCServer(tmp_path)
    yield server
    server.stop()


# =============================================================================
# Test Fixtures - Channel Definitions
# =============================================================================


@pytest.fixture
def simple_channels():
    """Simple channel definitions for testing."""
    return [
        {
            "name": "TEST:SETPOINT:SP",
            "python_name": "TEST_SETPOINT_SP",
            "type": "float",
            "description": "Test setpoint",
            "read_only": False,
            "units": "V",
            "precision": 3,
            "high_alarm": 100.0,
            "low_alarm": 0.0,
        },
        {
            "name": "TEST:SETPOINT:RB",
            "python_name": "TEST_SETPOINT_RB",
            "type": "float",
            "description": "Test readback",
            "read_only": True,
            "units": "V",
            "precision": 3,
            "high_alarm": 100.0,
            "low_alarm": 0.0,
        },
    ]


@pytest.fixture
def simple_pairings():
    """Simple SP->RB pairings."""
    return {"TEST:SETPOINT:SP": "TEST:SETPOINT:RB"}


# =============================================================================
# Test Cases
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not shutil.which("python"),
    reason="Python interpreter not available",
)
class TestSoftIOCServer:
    """Integration tests for soft IOC server with custom backend."""

    def test_ioc_starts_with_custom_backend(
        self, ioc_server: IOCServer, simple_channels, simple_pairings
    ):
        """Test that IOC server starts successfully and calls initialize."""
        # Start server
        port = ioc_server.start(
            channels=simple_channels,
            pairings=simple_pairings,
        )

        # Verify server started (port was allocated)
        assert port > 0

        # Verify process is running
        assert ioc_server.process is not None
        assert ioc_server.process.poll() is None  # Still running

        # Wait for initialize to be logged (CI environments may be slower)
        calls = []
        max_wait = 5.0
        start = time.time()
        while time.time() - start < max_wait:
            calls = ioc_server.get_backend_calls()
            if len(calls) >= 1:
                break
            time.sleep(0.2)

        # Verify initialize was called
        assert len(calls) >= 1, f"Expected >= 1 backend call, got {len(calls)}"
        assert calls[0]["method"] == "initialize"
        assert calls[0]["pv_count"] > 0

    def test_pv_read_returns_initial_values(
        self, ioc_server: IOCServer, simple_channels, simple_pairings
    ):
        """Test that reading a PV returns the initial value from backend."""
        # Start server
        ioc_server.start(
            channels=simple_channels,
            pairings=simple_pairings,
        )

        # Connect and read setpoint PV
        pv = ioc_server.get_pv("TEST:SETPOINT:SP")
        reading = pv.read()

        # Backend sets initial SP values to 10.0
        assert reading.data[0] == pytest.approx(10.0, rel=0.01)

    def test_pv_write_triggers_on_write(
        self, ioc_server: IOCServer, simple_channels, simple_pairings
    ):
        """Test that writing to a PV triggers on_write callback."""
        # Start server
        ioc_server.start(
            channels=simple_channels,
            pairings=simple_pairings,
        )

        # Connect to setpoint PV
        pv = ioc_server.get_pv("TEST:SETPOINT:SP")

        # Write a value
        pv.write(42.0, wait=True)

        # Wait for on_write callback with retry (CI environments may be slower)
        sp_write_calls = []
        max_wait = 3.0
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(0.2)
            calls = ioc_server.get_backend_calls()
            on_write_calls = [c for c in calls if c["method"] == "on_write"]
            sp_write_calls = [
                c
                for c in on_write_calls
                if c["pv_name"] == "TEST:SETPOINT:SP" and c["value"] == pytest.approx(42.0)
            ]
            if len(sp_write_calls) >= 1:
                break

        assert len(sp_write_calls) >= 1, f"Expected SP write with value 42.0, got: {on_write_calls}"

    def test_backend_transforms_write_value(
        self, ioc_server: IOCServer, simple_channels, simple_pairings
    ):
        """Test that backend scale/offset transforms write values."""
        # Start server with scale=2.0, offset=5.0
        # Result should be: value * 2.0 + 5.0
        ioc_server.start(
            channels=simple_channels,
            pairings=simple_pairings,
            scale=2.0,
            offset=5.0,
        )

        # Write to setpoint
        sp_pv = ioc_server.get_pv("TEST:SETPOINT:SP")
        sp_pv.write(10.0, wait=True)

        # Wait for write and update to propagate
        time.sleep(0.3)

        # Read back the setpoint - should be transformed
        # 10.0 * 2.0 + 5.0 = 25.0
        reading = sp_pv.read()
        assert reading.data[0] == pytest.approx(25.0, rel=0.01)

    def test_readback_updated_on_setpoint_write(
        self, ioc_server: IOCServer, simple_channels, simple_pairings
    ):
        """Test that writing setpoint updates the paired readback."""
        # Start server with scale=2.0
        ioc_server.start(
            channels=simple_channels,
            pairings=simple_pairings,
            scale=2.0,
            offset=0.0,
        )

        # Write to setpoint
        sp_pv = ioc_server.get_pv("TEST:SETPOINT:SP")
        sp_pv.write(20.0, wait=True)

        # Wait for update to propagate
        time.sleep(0.3)

        # Read readback - should be 20.0 * 2.0 = 40.0
        rb_pv = ioc_server.get_pv("TEST:SETPOINT:RB")
        reading = rb_pv.read()
        assert reading.data[0] == pytest.approx(40.0, rel=0.01)

    def test_simulation_loop_calls_step(
        self, ioc_server: IOCServer, simple_channels, simple_pairings
    ):
        """Test that simulation loop calls step() periodically."""
        # Start server with fast update rate
        ioc_server.start(
            channels=simple_channels,
            pairings=simple_pairings,
            update_rate=10.0,  # 10 Hz = 0.1s period
        )

        # Wait for step calls with retry (CI environments may be slower)
        step_calls = []
        max_wait = 3.0  # Total max wait time
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(0.3)
            calls = ioc_server.get_backend_calls()
            step_calls = [c for c in calls if c["method"] == "step"]
            if len(step_calls) >= 3:
                break

        # Should have at least 3 step calls
        assert len(step_calls) >= 3, f"Expected >= 3 step calls, got {len(step_calls)}"

        # Verify dt values are approximately correct (~0.1s)
        for call in step_calls:
            assert call["dt"] == pytest.approx(0.1, abs=0.05)

    def test_heartbeat_pv_increments(self, ioc_server: IOCServer, simple_channels, simple_pairings):
        """Test that heartbeat PV increments over time."""
        # Start server
        ioc_server.start(
            channels=simple_channels,
            pairings=simple_pairings,
            update_rate=10.0,
        )

        # Read initial heartbeat
        hb_pv = ioc_server.get_pv("SIM:HEARTBEAT")
        initial = hb_pv.read().data[0]

        # Wait for updates
        time.sleep(0.5)

        # Read again - should have incremented
        final = hb_pv.read().data[0]
        assert final > initial

    def test_multiple_pvs(self, ioc_server: IOCServer):
        """Test IOC with multiple PV pairs."""
        channels = [
            {
                "name": "MAG:Q1:CURRENT:SP",
                "python_name": "MAG_Q1_CURRENT_SP",
                "type": "float",
                "description": "Q1 current setpoint",
                "read_only": False,
                "units": "A",
                "precision": 3,
                "high_alarm": 500.0,
                "low_alarm": -500.0,
            },
            {
                "name": "MAG:Q1:CURRENT:RB",
                "python_name": "MAG_Q1_CURRENT_RB",
                "type": "float",
                "description": "Q1 current readback",
                "read_only": True,
                "units": "A",
                "precision": 3,
                "high_alarm": 500.0,
                "low_alarm": -500.0,
            },
            {
                "name": "MAG:Q2:CURRENT:SP",
                "python_name": "MAG_Q2_CURRENT_SP",
                "type": "float",
                "description": "Q2 current setpoint",
                "read_only": False,
                "units": "A",
                "precision": 3,
                "high_alarm": 500.0,
                "low_alarm": -500.0,
            },
            {
                "name": "MAG:Q2:CURRENT:RB",
                "python_name": "MAG_Q2_CURRENT_RB",
                "type": "float",
                "description": "Q2 current readback",
                "read_only": True,
                "units": "A",
                "precision": 3,
                "high_alarm": 500.0,
                "low_alarm": -500.0,
            },
        ]
        pairings = {
            "MAG:Q1:CURRENT:SP": "MAG:Q1:CURRENT:RB",
            "MAG:Q2:CURRENT:SP": "MAG:Q2:CURRENT:RB",
        }

        # Start server with scale=1.5
        ioc_server.start(channels=channels, pairings=pairings, scale=1.5)

        # Write to both setpoints
        q1_sp = ioc_server.get_pv("MAG:Q1:CURRENT:SP")
        q2_sp = ioc_server.get_pv("MAG:Q2:CURRENT:SP")

        q1_sp.write(100.0, wait=True)
        q2_sp.write(200.0, wait=True)

        time.sleep(0.3)

        # Read back both
        q1_rb = ioc_server.get_pv("MAG:Q1:CURRENT:RB")
        q2_rb = ioc_server.get_pv("MAG:Q2:CURRENT:RB")

        # Values should be scaled by 1.5
        assert q1_rb.read().data[0] == pytest.approx(150.0, rel=0.01)
        assert q2_rb.read().data[0] == pytest.approx(300.0, rel=0.01)

    def test_server_graceful_shutdown(
        self, ioc_server: IOCServer, simple_channels, simple_pairings
    ):
        """Test that server shuts down gracefully on stop."""
        # Start server
        ioc_server.start(
            channels=simple_channels,
            pairings=simple_pairings,
        )

        # Verify it's running
        assert ioc_server.process.poll() is None

        # Stop server
        ioc_server.stop()

        # Verify process terminated
        assert ioc_server.process is None

    def test_backend_params_passed_correctly(self, ioc_server: IOCServer):
        """Test that backend params (scale, offset) affect behavior."""
        channels = [
            {
                "name": "CALC:INPUT:SP",
                "python_name": "CALC_INPUT_SP",
                "type": "float",
                "description": "Input value",
                "read_only": False,
                "units": "",
                "precision": 4,
                "high_alarm": 1000.0,
                "low_alarm": -1000.0,
            },
            {
                "name": "CALC:INPUT:RB",
                "python_name": "CALC_INPUT_RB",
                "type": "float",
                "description": "Output value",
                "read_only": True,
                "units": "",
                "precision": 4,
                "high_alarm": 1000.0,
                "low_alarm": -1000.0,
            },
        ]

        # Start with specific scale and offset
        # result = value * 10 + 5
        ioc_server.start(
            channels=channels,
            pairings={"CALC:INPUT:SP": "CALC:INPUT:RB"},
            scale=10.0,
            offset=5.0,
        )

        # Write 1.0, expect 1.0 * 10 + 5 = 15.0
        sp_pv = ioc_server.get_pv("CALC:INPUT:SP")
        sp_pv.write(1.0, wait=True)

        time.sleep(0.3)

        rb_pv = ioc_server.get_pv("CALC:INPUT:RB")
        assert rb_pv.read().data[0] == pytest.approx(15.0, rel=0.01)

        # Write 0.0, expect 0.0 * 10 + 5 = 5.0
        sp_pv.write(0.0, wait=True)
        time.sleep(0.3)
        assert rb_pv.read().data[0] == pytest.approx(5.0, rel=0.01)


# =============================================================================
# Verification Tests - Run after implementation
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestSoftIOCVerification:
    """Verification tests to ensure the complete flow works."""

    def test_full_workflow(self, ioc_server: IOCServer):
        """Test the complete workflow: generate -> start -> read/write -> verify."""
        # Define channels
        channels = [
            {
                "name": "WORKFLOW:VALUE:SP",
                "python_name": "WORKFLOW_VALUE_SP",
                "type": "float",
                "description": "Workflow test setpoint",
                "read_only": False,
                "units": "units",
                "precision": 2,
                "high_alarm": 1000.0,
                "low_alarm": 0.0,
            },
            {
                "name": "WORKFLOW:VALUE:RB",
                "python_name": "WORKFLOW_VALUE_RB",
                "type": "float",
                "description": "Workflow test readback",
                "read_only": True,
                "units": "units",
                "precision": 2,
                "high_alarm": 1000.0,
                "low_alarm": 0.0,
            },
        ]

        # Start server
        port = ioc_server.start(
            channels=channels,
            pairings={"WORKFLOW:VALUE:SP": "WORKFLOW:VALUE:RB"},
            scale=1.0,  # No transformation
        )

        # 1. Verify server started
        assert port > 0
        assert ioc_server.process.poll() is None

        # 2. Verify initialize was called (retry for slow CI)
        max_wait = 3.0
        start = time.time()
        initialized = False
        while time.time() - start < max_wait:
            calls = ioc_server.get_backend_calls()
            if any(c["method"] == "initialize" for c in calls):
                initialized = True
                break
            time.sleep(0.2)
        assert initialized, f"Expected 'initialize' call, got: {calls}"

        # 3. Read initial value
        sp_pv = ioc_server.get_pv("WORKFLOW:VALUE:SP")
        initial = sp_pv.read().data[0]
        assert initial == pytest.approx(10.0, rel=0.1)  # Backend sets SP to 10.0

        # 4. Write new value
        sp_pv.write(99.0, wait=True)

        # 5. Verify on_write was called for our write (retry for slow CI)
        sp_write_calls = []
        max_wait = 3.0
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(0.2)
            calls = ioc_server.get_backend_calls()
            on_write_calls = [c for c in calls if c["method"] == "on_write"]
            sp_write_calls = [
                c
                for c in on_write_calls
                if c["pv_name"] == "WORKFLOW:VALUE:SP" and c["value"] == pytest.approx(99.0)
            ]
            if len(sp_write_calls) >= 1:
                break
        assert len(sp_write_calls) >= 1, f"Expected SP write with value 99.0, got: {on_write_calls}"

        # 6. Verify readback updated (retry for slow CI)
        rb_pv = ioc_server.get_pv("WORKFLOW:VALUE:RB")
        rb_value = None
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(0.2)
            rb_value = rb_pv.read().data[0]
            if rb_value == pytest.approx(99.0, rel=0.01):
                break
        assert rb_value == pytest.approx(99.0, rel=0.01)

        # 7. Verify step is being called (retry for slow CI)
        step_calls = []
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(0.2)
            calls = ioc_server.get_backend_calls()
            step_calls = [c for c in calls if c["method"] == "step"]
            if len(step_calls) >= 1:
                break
        assert len(step_calls) >= 1

        # 8. Clean shutdown
        ioc_server.stop()
        assert ioc_server.process is None


# =============================================================================
# Documentation Backend Templates (from 05_soft-ioc-backends.rst)
# =============================================================================

FIRST_ORDER_BACKEND_MODULE = textwrap.dedent('''
    """First-order dynamics backend from documentation.

    RB approaches SP with exponential dynamics.
    """

    import math
    from typing import Any


    class FirstOrderBackend:
        """RB approaches SP with exponential dynamics."""

        def __init__(self, tau: float = 1.0):
            """tau: time constant in seconds"""
            self.tau = tau
            self._setpoints: dict[str, float] = {}
            self._readbacks: dict[str, float] = {}

        def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
            return {}  # Let base backend set initial values

        def on_write(self, pv_name: str, value: Any) -> dict[str, Any] | None:
            if not pv_name.endswith(':SP'):
                return None  # Delegate non-setpoints

            rb_name = pv_name.replace(':SP', ':RB')
            self._setpoints[pv_name] = float(value)
            if rb_name not in self._readbacks:
                self._readbacks[rb_name] = float(value)
            return {pv_name: value}

        def step(self, dt: float) -> dict[str, Any]:
            updates = {}
            for sp_name, sp_val in self._setpoints.items():
                rb_name = sp_name.replace(':SP', ':RB')
                rb = self._readbacks.get(rb_name, sp_val)
                # Exponential approach: RB += (SP - RB) * (1 - e^(-dt/tau))
                rb += (sp_val - rb) * (1 - math.exp(-dt / self.tau))
                self._readbacks[rb_name] = rb
                updates[rb_name] = rb
            return updates
''')


DRIFT_BACKEND_MODULE = textwrap.dedent('''
    """Drift backend from documentation.

    RB drifts independently of SP (broken feedback).
    """

    from typing import Any


    class DriftBackend:
        """RB drifts independently of SP (broken feedback)."""

        def __init__(self, target_pv: str, drift_rate: float = 0.1):
            """
            Args:
                target_pv: Base PV name (without :SP/:RB suffix)
                drift_rate: Drift in units/second
            """
            self.target_rb = f"{target_pv}:RB"
            self.target_sp = f"{target_pv}:SP"
            self.drift_rate = drift_rate
            self._rb_value = 0.0

        def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
            return {}  # Let base set initial

        def on_write(self, pv_name: str, value: Any) -> dict[str, Any] | None:
            if pv_name == self.target_sp:
                return {}  # Block normal SP->RB update
            return None  # Delegate everything else

        def step(self, dt: float) -> dict[str, Any]:
            self._rb_value += self.drift_rate * dt
            return {self.target_rb: self._rb_value}
''')


# =============================================================================
# IOC Server with Backends List Support
# =============================================================================


IOC_BACKENDS_WRAPPER_TEMPLATE = '''\
"""IOC server wrapper with backends list support for integration tests."""

import signal
import sys
from pathlib import Path

# Add backend module paths to sys.path
backend_paths = {backend_paths}
for p in backend_paths:
    sys.path.insert(0, str(Path(p).parent))

# Import custom backends
{backend_imports}

# Import built-in backends from ioc_backends module
from osprey.generators.ioc_backends import ChainedBackend, MockStyleBackend, PassthroughBackend

# Import generated IOC code (defines IOC class)
exec(open("{ioc_path}").read())

# Configuration
PORT_FILE = "{port_file}"

# Signal handler for graceful shutdown
def handle_signal(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

if __name__ == "__main__":
    import os
    from caproto.server import run

    port = {port}
    os.environ["EPICS_CA_SERVER_PORT"] = str(port)
    os.environ["EPICS_CAS_SERVER_PORT"] = str(port)
    os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

    # Create backends list
    backends = {backends_list}

    # Wrap with ChainedBackend
    backend = ChainedBackend(backends)

    # Create IOC
    ioc = {ioc_class}(prefix="", backend=backend)

    # Write port file
    Path(PORT_FILE).write_text(str(port))

    # Run server
    run(ioc.pvdb, interfaces=["127.0.0.1"])
'''


class IOCServerWithBackends(IOCServer):
    """IOC server manager with backends list support."""

    def start_with_backends(
        self,
        channels: list[dict],
        backends_config: list[dict],
        pairings: dict[str, str] | None = None,
        port: int | None = None,
        update_rate: float = 10.0,
    ) -> int:
        """Start the IOC server with a backends list.

        Args:
            channels: List of PV definitions
            backends_config: List of backend configs (module_code, class_name, params)
            pairings: SP->RB pairings (optional)
            port: CA server port (default: dynamically allocated)
            update_rate: Simulation update rate in Hz

        Returns:
            Port number the server is running on
        """
        if pairings is None:
            pairings = {}

        if port is None:
            port = self._find_free_port()

        # Write backend modules
        backend_paths = []
        backend_imports = []
        backends_instantiation = []

        for i, backend in enumerate(backends_config):
            if "module_code" in backend:
                # Custom backend with module code
                module_code = backend["module_code"]
                class_name = backend["class_name"]
                params = backend.get("params", {})

                # Write module file
                module_file = self.tmp_path / f"backend_{i}.py"
                module_file.write_text(module_code)
                backend_paths.append(str(module_file))

                # Generate import
                module_name = f"backend_{i}"
                backend_imports.append(f"from {module_name} import {class_name}")

                # Generate instantiation
                if params:
                    param_strs = [f"{k}={repr(v)}" for k, v in params.items()]
                    backends_instantiation.append(f"{class_name}({', '.join(param_strs)})")
                else:
                    backends_instantiation.append(f"{class_name}()")
            elif backend.get("type") == "mock_style":
                # Built-in mock_style backend
                noise = backend.get("noise_level", 0.01)
                backends_instantiation.append(f"MockStyleBackend(noise_level={noise})")

        # Build base + overlays config for generator
        # First backend becomes the base, rest become overlays
        base_config = None
        overlay_configs = []

        for i, backend in enumerate(backends_config):
            if "module_code" in backend:
                backend_spec = {
                    "module_path": f"backend_{i}",
                    "class_name": backend["class_name"],
                    "params": backend.get("params", {}),
                }
            elif backend.get("type") == "mock_style":
                backend_spec = {
                    "type": "mock_style",
                    "noise_level": backend.get("noise_level", 0.01),
                }
            else:
                backend_spec = backend.copy()

            if base_config is None:
                base_config = backend_spec
            else:
                overlay_configs.append(backend_spec)

        # Generate IOC code with base + overlays
        config = {
            "ioc": {"name": "test_ioc", "port": port},
            "base": base_config
            or {"type": "mock_style", "noise_level": 0.0, "update_rate": update_rate},
            "overlays": overlay_configs,
        }
        ioc_code = generate_soft_ioc(config, channels, pairings)

        # Remove the main block since wrapper handles startup
        main_marker = "if __name__ == '__main__':"
        if main_marker in ioc_code:
            ioc_code = ioc_code[: ioc_code.index(main_marker)]

        self.ioc_file.write_text(ioc_code)

        # Write wrapper script
        wrapper_code = IOC_BACKENDS_WRAPPER_TEMPLATE.format(
            backend_paths=repr(backend_paths),
            backend_imports="\n".join(backend_imports) if backend_imports else "pass",
            ioc_path=self.ioc_file,
            port_file=self.port_file,
            port=port,
            backends_list="[" + ", ".join(backends_instantiation) + "]",
            ioc_class="TestIoc",
        )
        self.wrapper_file.write_text(wrapper_code)

        # Set up environment for subprocess
        env = os.environ.copy()
        env["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
        env["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
        env["EPICS_CA_SERVER_PORT"] = str(port)

        # Start server subprocess
        self.process = subprocess.Popen(
            [sys.executable, str(self.wrapper_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for port file
        timeout = 10.0
        start_time = time.time()
        while not self.port_file.exists():
            if time.time() - start_time > timeout:
                if self.process.poll() is not None:
                    _, stderr = self.process.communicate()
                    raise TimeoutError(
                        f"IOC server with backends failed to start. stderr: {stderr.decode()}"
                    )
                raise TimeoutError(f"IOC server failed to start within {timeout}s")
            time.sleep(0.1)

        self.port = int(self.port_file.read_text().strip())
        time.sleep(1.0)  # Wait for server to fully initialize (longer for CI)

        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(
                f"IOC server exited unexpectedly. "
                f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            )

        return self.port


@pytest.fixture
def ioc_server_with_backends(tmp_path):
    """Fixture that manages IOC server with backends list support."""
    server = IOCServerWithBackends(tmp_path)
    yield server
    server.stop()


# =============================================================================
# Documentation Backend Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not shutil.which("python"),
    reason="Python interpreter not available",
)
class TestDocumentationBackends:
    """Integration tests for the backends documented in 05_soft-ioc-backends.rst.

    These tests verify that the example backends work exactly as documented.
    """

    @pytest.fixture
    def sp_rb_channels(self):
        """Channels for SP/RB pair testing."""
        return [
            {
                "name": "TEST:VALUE:SP",
                "python_name": "TEST_VALUE_SP",
                "type": "float",
                "description": "Test setpoint",
                "read_only": False,
                "units": "V",
                "precision": 3,
                "high_alarm": 1000.0,
                "low_alarm": -1000.0,
            },
            {
                "name": "TEST:VALUE:RB",
                "python_name": "TEST_VALUE_RB",
                "type": "float",
                "description": "Test readback",
                "read_only": True,
                "units": "V",
                "precision": 3,
                "high_alarm": 1000.0,
                "low_alarm": -1000.0,
            },
        ]

    def test_first_order_backend_rb_approaches_sp(
        self, ioc_server_with_backends: IOCServerWithBackends, sp_rb_channels
    ):
        """Test FirstOrderBackend: RB exponentially approaches SP.

        This is the physics example from the documentation.
        Verifies that when you write to SP, RB approaches it over time
        with first-order dynamics (exponential approach).
        """
        tau = 0.5  # 500ms time constant for faster test
        backends_config = [
            {"type": "mock_style", "noise_level": 0.0},  # Base
            {
                "module_code": FIRST_ORDER_BACKEND_MODULE,
                "class_name": "FirstOrderBackend",
                "params": {"tau": tau},
            },
        ]

        ioc_server_with_backends.start_with_backends(
            channels=sp_rb_channels,
            backends_config=backends_config,
            pairings={"TEST:VALUE:SP": "TEST:VALUE:RB"},
            update_rate=20.0,  # 20 Hz for smooth dynamics
        )

        # Get PVs
        sp_pv = ioc_server_with_backends.get_pv("TEST:VALUE:SP")
        rb_pv = ioc_server_with_backends.get_pv("TEST:VALUE:RB")

        # Write setpoint to 100
        sp_pv.write(100.0, wait=True)
        time.sleep(0.1)

        # Immediately after write, RB should be initialized to SP (or close)
        initial_rb = rb_pv.read().data[0]

        # Now change SP to 200
        sp_pv.write(200.0, wait=True)

        # Sample RB over time - should approach 200 exponentially
        samples = []
        for _ in range(10):
            time.sleep(0.1)
            samples.append(rb_pv.read().data[0])

        # Verify exponential approach behavior:
        # 1. Values should be monotonically increasing toward 200
        # 2. Later values should be closer to 200
        # 3. After ~3*tau (1.5s), should be >95% of the way there

        # Check monotonic increase
        for i in range(1, len(samples)):
            assert samples[i] >= samples[i - 1] - 0.1, (
                f"RB should monotonically approach SP: {samples}"
            )

        # Check final value is close to SP (after 1s with tau=0.5, should be ~86% there)
        # 1 - e^(-1/0.5) = 1 - e^(-2) ≈ 0.865
        final_rb = samples[-1]
        progress = (final_rb - initial_rb) / (200.0 - initial_rb) if initial_rb != 200.0 else 1.0
        assert progress > 0.5, (
            f"RB should have progressed >50% toward SP after 1s, got {progress * 100:.1f}%"
        )

    def test_first_order_backend_multiple_setpoint_changes(
        self, ioc_server_with_backends: IOCServerWithBackends, sp_rb_channels
    ):
        """Test FirstOrderBackend tracks multiple SP changes."""
        tau = 0.3  # Fast time constant
        backends_config = [
            {"type": "mock_style", "noise_level": 0.0},
            {
                "module_code": FIRST_ORDER_BACKEND_MODULE,
                "class_name": "FirstOrderBackend",
                "params": {"tau": tau},
            },
        ]

        ioc_server_with_backends.start_with_backends(
            channels=sp_rb_channels,
            backends_config=backends_config,
            pairings={"TEST:VALUE:SP": "TEST:VALUE:RB"},
            update_rate=20.0,
        )

        sp_pv = ioc_server_with_backends.get_pv("TEST:VALUE:SP")
        rb_pv = ioc_server_with_backends.get_pv("TEST:VALUE:RB")

        # Write SP = 50
        sp_pv.write(50.0, wait=True)
        time.sleep(0.8)  # Wait for ~2.5*tau
        rb1 = rb_pv.read().data[0]
        assert rb1 == pytest.approx(50.0, abs=5.0), f"RB should approach SP=50, got {rb1}"

        # Change SP = 150
        sp_pv.write(150.0, wait=True)
        time.sleep(0.8)
        rb2 = rb_pv.read().data[0]
        assert rb2 == pytest.approx(150.0, abs=15.0), f"RB should approach SP=150, got {rb2}"

        # Change SP = 0
        sp_pv.write(0.0, wait=True)
        time.sleep(0.8)
        rb3 = rb_pv.read().data[0]
        assert rb3 == pytest.approx(0.0, abs=15.0), f"RB should approach SP=0, got {rb3}"

    def test_drift_backend_blocks_sp_to_rb(
        self, ioc_server_with_backends: IOCServerWithBackends, sp_rb_channels
    ):
        """Test DriftBackend: SP writes don't affect RB.

        This is the fault example from the documentation.
        Verifies that when you write to SP, RB does NOT follow.
        """
        backends_config = [
            {"type": "mock_style", "noise_level": 0.0},  # Base
            {
                "module_code": DRIFT_BACKEND_MODULE,
                "class_name": "DriftBackend",
                "params": {
                    "target_pv": "TEST:VALUE",
                    "drift_rate": 0.0,  # No drift for this test
                },
            },
        ]

        ioc_server_with_backends.start_with_backends(
            channels=sp_rb_channels,
            backends_config=backends_config,
            pairings={"TEST:VALUE:SP": "TEST:VALUE:RB"},
            update_rate=10.0,
        )

        sp_pv = ioc_server_with_backends.get_pv("TEST:VALUE:SP")
        rb_pv = ioc_server_with_backends.get_pv("TEST:VALUE:RB")

        # Read initial RB (should be 0 from DriftBackend's _rb_value)
        initial_rb = rb_pv.read().data[0]

        # Write to SP - should NOT change RB
        sp_pv.write(500.0, wait=True)
        time.sleep(0.3)

        # RB should NOT have changed
        final_rb = rb_pv.read().data[0]
        assert final_rb == pytest.approx(initial_rb, abs=0.1), (
            f"DriftBackend should block SP->RB update. "
            f"Expected RB={initial_rb}, got RB={final_rb} after SP write"
        )

    def test_drift_backend_rb_drifts_over_time(
        self, ioc_server_with_backends: IOCServerWithBackends, sp_rb_channels
    ):
        """Test DriftBackend: RB drifts independently over time."""
        drift_rate = 10.0  # 10 units per second
        backends_config = [
            {"type": "mock_style", "noise_level": 0.0},
            {
                "module_code": DRIFT_BACKEND_MODULE,
                "class_name": "DriftBackend",
                "params": {
                    "target_pv": "TEST:VALUE",
                    "drift_rate": drift_rate,
                },
            },
        ]

        ioc_server_with_backends.start_with_backends(
            channels=sp_rb_channels,
            backends_config=backends_config,
            pairings={"TEST:VALUE:SP": "TEST:VALUE:RB"},
            update_rate=10.0,
        )

        rb_pv = ioc_server_with_backends.get_pv("TEST:VALUE:RB")

        # Read initial RB
        initial_rb = rb_pv.read().data[0]

        # Wait for drift
        time.sleep(0.5)

        # RB should have drifted
        final_rb = rb_pv.read().data[0]
        drift = final_rb - initial_rb

        # Expected drift: drift_rate * time = 10 * 0.5 = 5 units
        assert drift > 0, f"RB should drift upward, got drift={drift}"
        assert drift == pytest.approx(5.0, abs=2.0), (
            f"Expected ~5 units drift in 0.5s at rate {drift_rate}, got {drift}"
        )

    def test_drift_backend_continues_drifting(
        self, ioc_server_with_backends: IOCServerWithBackends, sp_rb_channels
    ):
        """Test that DriftBackend continues drifting regardless of SP writes."""
        drift_rate = 5.0
        backends_config = [
            {"type": "mock_style", "noise_level": 0.0},
            {
                "module_code": DRIFT_BACKEND_MODULE,
                "class_name": "DriftBackend",
                "params": {
                    "target_pv": "TEST:VALUE",
                    "drift_rate": drift_rate,
                },
            },
        ]

        ioc_server_with_backends.start_with_backends(
            channels=sp_rb_channels,
            backends_config=backends_config,
            pairings={"TEST:VALUE:SP": "TEST:VALUE:RB"},
            update_rate=10.0,
        )

        sp_pv = ioc_server_with_backends.get_pv("TEST:VALUE:SP")
        rb_pv = ioc_server_with_backends.get_pv("TEST:VALUE:RB")

        # Sample RB while writing to SP
        samples = []
        for i in range(5):
            samples.append(rb_pv.read().data[0])
            sp_pv.write(float(i * 100), wait=True)  # Write different SP values
            time.sleep(0.2)

        # RB should continue drifting despite SP changes
        # Each sample should be larger than the previous
        for i in range(1, len(samples)):
            assert samples[i] > samples[i - 1] - 0.5, (
                f"RB should continue drifting despite SP writes: {samples}"
            )

    def test_chained_backends_first_order_then_drift(
        self, ioc_server_with_backends: IOCServerWithBackends
    ):
        """Test chaining FirstOrderBackend with DriftBackend.

        Chain: mock_style -> FirstOrderBackend -> DriftBackend

        FirstOrderBackend handles all :SP PVs with physics.
        DriftBackend overrides just one specific PV to break it.
        """
        channels = [
            # Working PV pair (FirstOrderBackend handles it)
            {
                "name": "WORKING:VALUE:SP",
                "python_name": "WORKING_VALUE_SP",
                "type": "float",
                "description": "Working setpoint",
                "read_only": False,
                "units": "V",
                "precision": 3,
                "high_alarm": 1000.0,
                "low_alarm": -1000.0,
            },
            {
                "name": "WORKING:VALUE:RB",
                "python_name": "WORKING_VALUE_RB",
                "type": "float",
                "description": "Working readback",
                "read_only": True,
                "units": "V",
                "precision": 3,
                "high_alarm": 1000.0,
                "low_alarm": -1000.0,
            },
            # Broken PV pair (DriftBackend overrides it)
            {
                "name": "BROKEN:VALUE:SP",
                "python_name": "BROKEN_VALUE_SP",
                "type": "float",
                "description": "Broken setpoint",
                "read_only": False,
                "units": "V",
                "precision": 3,
                "high_alarm": 1000.0,
                "low_alarm": -1000.0,
            },
            {
                "name": "BROKEN:VALUE:RB",
                "python_name": "BROKEN_VALUE_RB",
                "type": "float",
                "description": "Broken readback",
                "read_only": True,
                "units": "V",
                "precision": 3,
                "high_alarm": 1000.0,
                "low_alarm": -1000.0,
            },
        ]

        backends_config = [
            {"type": "mock_style", "noise_level": 0.0},  # Base
            {
                "module_code": FIRST_ORDER_BACKEND_MODULE,
                "class_name": "FirstOrderBackend",
                "params": {"tau": 0.3},
            },
            {
                "module_code": DRIFT_BACKEND_MODULE,
                "class_name": "DriftBackend",
                "params": {
                    "target_pv": "BROKEN:VALUE",
                    "drift_rate": 5.0,
                },
            },
        ]

        ioc_server_with_backends.start_with_backends(
            channels=channels,
            backends_config=backends_config,
            pairings={
                "WORKING:VALUE:SP": "WORKING:VALUE:RB",
                "BROKEN:VALUE:SP": "BROKEN:VALUE:RB",
            },
            update_rate=20.0,
        )

        working_sp = ioc_server_with_backends.get_pv("WORKING:VALUE:SP")
        working_rb = ioc_server_with_backends.get_pv("WORKING:VALUE:RB")
        broken_sp = ioc_server_with_backends.get_pv("BROKEN:VALUE:SP")
        broken_rb = ioc_server_with_backends.get_pv("BROKEN:VALUE:RB")

        # Write to both setpoints
        working_sp.write(100.0, wait=True)
        broken_sp.write(100.0, wait=True)

        # Wait for physics to settle
        time.sleep(0.8)

        # Working RB should approach 100
        working_rb_val = working_rb.read().data[0]
        assert working_rb_val == pytest.approx(100.0, abs=15.0), (
            f"Working RB should approach SP=100, got {working_rb_val}"
        )

        # Broken RB should be drifting, NOT following SP
        broken_rb_val = broken_rb.read().data[0]
        # After 0.8s at 5.0/s drift, should be ~4 (plus initial ~0.5s startup)
        assert broken_rb_val != pytest.approx(100.0, abs=20.0), (
            f"Broken RB should NOT follow SP. Expected drift value, got {broken_rb_val}"
        )
