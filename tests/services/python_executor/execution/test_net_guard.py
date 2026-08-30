"""Tests for the emitted network guard (``net_guard.render_net_guard``).

The guard is *source code*, not a callable, so — same reasoning as the
filesystem-guard suite (``tests/services/python_executor/test_fs_guard.py``) —
every behavioural test writes the rendered guard plus a probe into a script and
runs it in a **real subprocess** against **real listening sockets** on
ephemeral loopback ports, then asserts on what the probe printed. That keeps
the ``socket.socket`` patching out of the test process, where a leaked rebind
would poison every network-touching test after it.

Three layers:

* subprocess behaviour — refusal on a denied port through every patched entry
  point, pass-through on allowed ports, the high-level ``urllib`` funnel, the
  ``multiprocessing`` regression the proposal's IA-2 pins, and the unguarded
  baseline;
* renderer contract — what ``render_net_guard`` promises about its output and
  its arguments, no subprocess needed;
* wrapper emission — ``ExecutionWrapper.create_wrapper`` splices the guard iff
  ports are denied, in both execution modes.
"""

import ast
import functools
import importlib.util
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.services.python_executor.execution.net_guard import (
    NET_GUARD_REFUSAL_PREFIX,
    NET_GUARDED_MODULES,
    render_net_guard,
)
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

pytestmark = pytest.mark.unit


@functools.cache
def _h5py_importable() -> bool:
    """Whether the child interpreter can import h5py — probed once, lazily.

    A subprocess probe (not ``find_spec``) because what matters is a real
    import in the same interpreter the guarded child runs, and lazily (not in
    a ``skipif`` expression) so collection never pays for the spawn.
    """
    return (
        subprocess.run(  # noqa: S603 - fixed argv
            [sys.executable, "-c", "import h5py"], capture_output=True
        ).returncode
        == 0
    )


def _run(tmp_path: Path, guard: str, probe: str) -> str:
    """Run *guard* + *probe* in a real subprocess and return its stdout.

    ``cwd`` is the test's own tmp dir so a probe may use short relative paths
    (an ``AF_UNIX`` socket path has a hard length cap that pytest's deeply
    nested absolute ``tmp_path`` would blow through).
    """
    script = tmp_path / "probe_script.py"
    script.write_text(guard + "\n" + textwrap.dedent(probe) + "\n", encoding="utf-8")
    proc = subprocess.run(  # noqa: S603 - fixed argv, test-authored script
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, (
        f"probe script failed rc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc.stdout


@pytest.fixture
def servers():
    """Two real listening sockets on ephemeral loopback ports.

    The first plays the deployment's web port (denied), the second an
    unrelated service on the same host (allowed) — so a refused connect is
    provably the guard's doing and not a dead port, and an allowed connect
    proves the guard refuses by port, not by host.
    """
    denied_srv = socket.create_server(("127.0.0.1", 0))
    allowed_srv = socket.create_server(("127.0.0.1", 0))
    try:
        yield denied_srv.getsockname()[1], allowed_srv.getsockname()[1]
    finally:
        denied_srv.close()
        allowed_srv.close()


# ---------------------------------------------------------------------------
# Subprocess behaviour — the guard live in a child interpreter
# ---------------------------------------------------------------------------
class TestGuardedChild:
    """What the emitted guard does once installed in a real child."""

    def test_connect_to_denied_port_is_refused_and_socket_stays_unconnected(
        self, tmp_path: Path, servers
    ):
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import socket
            sock = socket.socket()
            try:
                sock.connect(("127.0.0.1", {denied_port}))
                print("UNEXPECTED_CONNECTED")
            except PermissionError as exc:
                print("DENIED:", exc)
            try:
                sock.getpeername()
                print("PEER_PRESENT")
            except OSError:
                print("NOT_CONNECTED")
            print("STILL_RUNNING")
            """,
        )
        assert "UNEXPECTED_CONNECTED" not in out
        assert f"DENIED: {NET_GUARD_REFUSAL_PREFIX}" in out
        assert f"connect to port {denied_port} refused" in out
        # The message explains the perimeter and scopes the refusal.
        assert "open navigation perimeter" in out
        assert "unaffected" in out
        # The refused socket never connected, and the refusal is a catchable
        # exception inside the child, not a kill.
        assert "NOT_CONNECTED" in out
        assert "PEER_PRESENT" not in out
        assert "STILL_RUNNING" in out

    def test_connect_to_allowed_port_on_same_host_succeeds(self, tmp_path: Path, servers):
        denied_port, allowed_port = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import socket
            sock = socket.socket()
            sock.connect(("127.0.0.1", {allowed_port}))
            print("CONNECTED_TO:", sock.getpeername()[1])
            sock.close()
            """,
        )
        assert f"CONNECTED_TO: {allowed_port}" in out

    def test_create_connection_is_refused_on_denied_and_works_on_allowed(
        self, tmp_path: Path, servers
    ):
        denied_port, allowed_port = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import socket
            try:
                socket.create_connection(("127.0.0.1", {denied_port}), timeout=10)
                print("UNEXPECTED_CONNECTED")
            except PermissionError as exc:
                print("DENIED:", exc)
            sock = socket.create_connection(("127.0.0.1", {allowed_port}), timeout=10)
            print("CONNECTED_TO:", sock.getpeername()[1])
            sock.close()
            """,
        )
        assert "UNEXPECTED_CONNECTED" not in out
        assert f"DENIED: {NET_GUARD_REFUSAL_PREFIX}" in out
        assert f"CONNECTED_TO: {allowed_port}" in out

    def test_urllib_to_denied_port_is_refused(self, tmp_path: Path, servers):
        # The high-level HTTP stacks are not patched one by one; they funnel
        # through http.client into socket.create_connection. This pins that
        # the funnel actually carries the refusal up.
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import urllib.request
            try:
                urllib.request.urlopen("http://127.0.0.1:{denied_port}/", timeout=10)
                print("UNEXPECTED_FETCHED")
            except Exception as exc:
                print("DENIED:", exc)
            """,
        )
        assert "UNEXPECTED_FETCHED" not in out
        assert NET_GUARD_REFUSAL_PREFIX in out

    def test_connect_ex_raises_the_same_refusal_not_an_errno(self, tmp_path: Path, servers):
        # connect_ex normally *returns* an errno; the guard raises instead,
        # because callers read a nonzero return as a transient network failure
        # (a polling loop retries it silently, a scanner tallies "closed") and
        # the refusal's explanation would never surface.
        denied_port, allowed_port = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import socket
            sock = socket.socket()
            try:
                rc = sock.connect_ex(("127.0.0.1", {denied_port}))
                print("UNEXPECTED_ERRNO:", rc)
            except PermissionError as exc:
                print("DENIED:", exc)
            sock2 = socket.socket()
            print("ALLOWED_ERRNO:", sock2.connect_ex(("127.0.0.1", {allowed_port})))
            sock2.close()
            sock.close()
            """,
        )
        assert "UNEXPECTED_ERRNO" not in out
        assert f"DENIED: {NET_GUARD_REFUSAL_PREFIX}" in out
        assert "ALLOWED_ERRNO: 0" in out

    def test_numpy_integer_port_is_refused(self, tmp_path: Path, servers):
        # The C layer accepts any __index__ object wherever it accepts an int,
        # so a numpy.int64 port must hit the deny set exactly as a plain int
        # does — a bare isinstance(int) gate waved it straight past.
        if importlib.util.find_spec("numpy") is None:
            pytest.skip("numpy not importable in this environment")
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import numpy
            import socket
            sock = socket.socket()
            try:
                sock.connect(("127.0.0.1", numpy.int64({denied_port})))
                print("UNEXPECTED_CONNECTED")
            except PermissionError as exc:
                print("DENIED:", exc)
            sock.close()
            """,
        )
        assert "UNEXPECTED_CONNECTED" not in out
        assert f"DENIED: {NET_GUARD_REFUSAL_PREFIX}" in out

    def test_requests_to_denied_port_is_refused(self, tmp_path: Path, servers):
        # requests rides urllib3, whose OWN create_connection calls
        # sock.connect itself — this pins that the socket.socket.connect class
        # rebind carries the refusal up through that second funnel. urllib3
        # re-wraps the exception, so the assertion is on the message, not the
        # type.
        if importlib.util.find_spec("requests") is None:
            pytest.skip("requests not importable in this environment")
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import requests
            try:
                requests.get("http://127.0.0.1:{denied_port}/", timeout=10)
                print("UNEXPECTED_FETCHED")
            except Exception as exc:
                print("DENIED:", exc)
            """,
        )
        assert "UNEXPECTED_FETCHED" not in out
        assert NET_GUARD_REFUSAL_PREFIX in out

    def test_ssl_wrapped_socket_connect_is_refused(self, tmp_path: Path, servers):
        # SSLSocket overrides connect but reaches the wire via
        # super().connect(), which resolves through the class rebind — pinned
        # so a stdlib re-plumbing that stops doing so fails loudly.
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import socket
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            sock = ctx.wrap_socket(socket.socket())
            try:
                sock.connect(("127.0.0.1", {denied_port}))
                print("UNEXPECTED_CONNECTED")
            except PermissionError as exc:
                print("DENIED:", exc)
            sock.close()
            """,
        )
        assert "UNEXPECTED_CONNECTED" not in out
        assert f"DENIED: {NET_GUARD_REFUSAL_PREFIX}" in out

    def test_asyncio_open_connection_refused_on_denied_and_works_on_allowed(
        self, tmp_path: Path, servers
    ):
        denied_port, allowed_port = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import asyncio


            async def _probe():
                try:
                    await asyncio.open_connection("127.0.0.1", {denied_port})
                    print("UNEXPECTED_CONNECTED")
                except PermissionError as exc:
                    print("DENIED:", exc)
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", {allowed_port}
                )
                print("ASYNC_CONNECTED_TO:", writer.get_extra_info("peername")[1])
                writer.close()
                await writer.wait_closed()


            asyncio.run(_probe())
            """,
        )
        assert "UNEXPECTED_CONNECTED" not in out
        assert f"DENIED: {NET_GUARD_REFUSAL_PREFIX}" in out
        assert f"ASYNC_CONNECTED_TO: {allowed_port}" in out

    def test_server_side_round_trip_is_untouched(self, tmp_path: Path, servers):
        # bind/listen/accept are not patched: a probe under the guard can run
        # its own server on an undenied ephemeral port and complete a full
        # round trip through it.
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            """
            import socket
            server = socket.create_server(("127.0.0.1", 0))
            port = server.getsockname()[1]
            client = socket.create_connection(("127.0.0.1", port), timeout=10)
            conn, _addr = server.accept()
            client.sendall(b"ping")
            print("SERVER_GOT:", conn.recv(4).decode())
            conn.sendall(b"pong")
            print("CLIENT_GOT:", client.recv(4).decode())
            for closable in (conn, client, server):
                closable.close()
            """,
        )
        assert "SERVER_GOT: ping" in out
        assert "CLIENT_GOT: pong" in out

    @pytest.mark.skipif(not socket.has_ipv6, reason="host has no IPv6 support")
    def test_ipv6_sockaddr_carries_the_port_at_index_1_and_is_refused(
        self, tmp_path: Path, servers
    ):
        # No IPv6 server needed: the refusal fires before any network I/O.
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import socket
            sock = socket.socket(socket.AF_INET6)
            try:
                sock.connect(("::1", {denied_port}, 0, 0))
                print("UNEXPECTED_CONNECTED")
            except PermissionError as exc:
                print("DENIED:", exc)
            sock.close()
            """,
        )
        assert "UNEXPECTED_CONNECTED" not in out
        assert f"DENIED: {NET_GUARD_REFUSAL_PREFIX}" in out

    @pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX is POSIX-only")
    def test_af_unix_connect_passes_through_untouched(self, tmp_path: Path, servers):
        # A path-addressed family carries no port; the guard must not judge
        # it. Bound relative to the child's cwd to stay under the AF_UNIX
        # path-length cap.
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            """
            import socket
            server = socket.socket(socket.AF_UNIX)
            server.bind("probe.sock")
            server.listen(1)
            client = socket.socket(socket.AF_UNIX)
            client.connect("probe.sock")
            print("UNIX_CONNECTED")
            client.close()
            server.close()
            """,
        )
        assert "UNIX_CONNECTED" in out

    def test_multiprocessing_pool_works_under_the_guard(self, tmp_path: Path, servers):
        # IA-2 regression: the perimeter guard must leave multiprocessing
        # compute untouched in every mode — its parent/worker plumbing runs
        # over AF_UNIX sockets and pipes, none of which name a TCP port.
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            """
            import multiprocessing


            def _osprey_probe_square(value):
                return value * value


            if __name__ == "__main__":
                with multiprocessing.Pool(2) as pool:
                    print("POOL_RESULT:", pool.map(_osprey_probe_square, [1, 2, 3, 4]))
            """,
        )
        assert "POOL_RESULT: [1, 4, 9, 16]" in out

    def test_h5py_import_works_under_the_guard(self, tmp_path: Path, servers):
        if not _h5py_importable():
            pytest.skip("h5py not importable in this environment")
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            """
            import h5py
            print("H5PY_OK:", bool(h5py.__version__))
            """,
        )
        assert "H5PY_OK: True" in out

    def test_without_a_guard_both_ports_connect(self, tmp_path: Path, servers):
        # The unguarded baseline: proof the refusals above are the guard's
        # doing and nothing else's.
        denied_port, allowed_port = servers
        out = _run(
            tmp_path,
            "",
            f"""
            import socket
            for port in ({denied_port}, {allowed_port}):
                sock = socket.socket()
                sock.connect(("127.0.0.1", port))
                print("CONNECTED_TO:", sock.getpeername()[1])
                sock.close()
            """,
        )
        assert f"CONNECTED_TO: {denied_port}" in out
        assert f"CONNECTED_TO: {allowed_port}" in out

    def test_restore_handle_disarms_the_guard_today(self, tmp_path: Path, servers):
        # Characterization of the documented tamper limit, mirroring the
        # filesystem suite's TestTamperLimit: the guard is defense in depth,
        # and its own uninstall is in scope for the code it guards. If this
        # ever fails because the disarm became impossible, that is a change of
        # posture wanting its own task — not a quiet edit here.
        denied_port, _ = servers
        guard = render_net_guard(denied_ports=(denied_port,))
        out = _run(
            tmp_path,
            guard,
            f"""
            import socket
            try:
                socket.socket().connect(("127.0.0.1", {denied_port}))
                print("UNEXPECTED_CONNECTED")
            except PermissionError:
                print("REFUSED_WHILE_INSTALLED")
            _restore_net_patched_targets()
            sock = socket.socket()
            sock.connect(("127.0.0.1", {denied_port}))
            print("DISARMED_CONNECT_LANDED:", sock.getpeername()[1])
            sock.close()
            """,
        )
        assert "REFUSED_WHILE_INSTALLED" in out
        assert "UNEXPECTED_CONNECTED" not in out
        assert f"DISARMED_CONNECT_LANDED: {denied_port}" in out


# ---------------------------------------------------------------------------
# Renderer-level contract (no subprocess needed)
# ---------------------------------------------------------------------------
class TestRendererContract:
    """What the renderer itself promises about its output and its arguments."""

    def test_denied_ports_are_embedded_sorted_and_deduplicated(self):
        guard = render_net_guard(denied_ports=(8082, 8080, 8082, 8081))
        assert "_OSPREY_NET_DENIED = frozenset((8080, 8081, 8082))" in guard
        assert "_OSPREY_NET_PORTS_TEXT = '8080, 8081, 8082'" in guard

    def test_default_refusal_prefix_is_the_module_constant(self):
        guard = render_net_guard(denied_ports=(19680,))
        assert f"_OSPREY_NET_PREFIX = {NET_GUARD_REFUSAL_PREFIX!r}" in guard

    def test_refusal_prefix_is_overridable(self):
        guard = render_net_guard(denied_ports=(19680,), refusal_prefix="Blocked (perimeter):")
        assert "_OSPREY_NET_PREFIX = 'Blocked (perimeter):'" in guard

    def test_empty_refusal_prefix_is_rejected(self):
        with pytest.raises(ValueError, match="refusal_prefix"):
            render_net_guard(denied_ports=(19680,), refusal_prefix="  ")

    def test_empty_port_set_is_rejected(self):
        # The wrapper skips emission instead — see TestWrapperEmission.
        with pytest.raises(ValueError, match="non-empty"):
            render_net_guard(denied_ports=())

    @pytest.mark.parametrize("bad", [0, -1, 65536, "8080", True, None])
    def test_invalid_ports_are_rejected(self, bad):
        with pytest.raises(ValueError, match="denied_ports"):
            render_net_guard(denied_ports=(bad,))

    def test_emitted_source_is_valid_python_and_left_aligned(self):
        guard = render_net_guard(denied_ports=(19680, 19681))
        ast.parse(guard)
        assert not guard.startswith((" ", "\n"))

    def test_guarded_modules_names_socket_for_the_static_layer(self):
        # Exported for a static reload check to consume; deliberately NOT
        # merged into fs_guard.GUARDED_MODULES — see the module docstring.
        assert NET_GUARDED_MODULES == ("socket",)


# ---------------------------------------------------------------------------
# Wrapper emission — the guard is spliced iff ports are denied, in every mode
# ---------------------------------------------------------------------------
class TestWrapperEmission:
    """``create_wrapper`` output carries the guard exactly when it should."""

    @pytest.mark.parametrize("mode", ["readonly", "readwrite"])
    def test_guard_emitted_when_ports_denied_in_both_modes(self, mode: str):
        wrapped = ExecutionWrapper(
            execution_mode=mode, perimeter_denied_ports=(19680, 19681)
        ).create_wrapper("x = 1")
        assert "OSPREY network guard" in wrapped
        assert "_install_net_patched_targets" in wrapped
        assert "frozenset((19680, 19681))" in wrapped
        # Deterministic splice: after the filesystem guard, before user code.
        assert (
            wrapped.index("OSPREY filesystem guard")
            < wrapped.index("OSPREY network guard")
            < wrapped.index("x = 1")
        )
        # The shared cleanup tail takes the guard back off.
        assert "_restore_net_patched_targets" in wrapped

    @pytest.mark.parametrize("mode", ["readonly", "readwrite"])
    def test_guard_skipped_when_no_ports_denied_in_both_modes(self, mode: str):
        wrapped = ExecutionWrapper(execution_mode=mode).create_wrapper("x = 1")
        assert "OSPREY network guard" not in wrapped
        assert "_install_net_patched_targets" not in wrapped
        # The filesystem guard is unconditional either way.
        assert "OSPREY filesystem guard" in wrapped


# ---------------------------------------------------------------------------
# End-to-end arming path — executor -> wrapper -> emitted guard -> real child
# ---------------------------------------------------------------------------
class TestEndToEndArmingPath:
    """The full chain a deployment actually runs, not just its halves.

    ``tests/mcp_server/test_executor_env_scrub.py`` pins that the perimeter
    stamp (``OSPREY_WEB_PERIMETER`` / ``OSPREY_WEB_PERIMETER_DENY_PORTS``) is
    parsed and handed to a REAL ``ExecutionWrapper`` as a constructor
    argument (``test_local_wrapper_receives_the_denied_ports``, which wraps
    rather than mocks ``ExecutionWrapper``) — but every test that follows the
    chain past that point stubs ``asyncio.create_subprocess_exec``, so no
    existing test proves the wrapper's OUTPUT, once actually run, refuses a
    denied connect. This test closes that gap the same way
    ``test_executor_token_regression.py`` closes the analogous one for the
    env scrub: call ``_execute_via_local`` directly (mirroring the pattern in
    ``tests/mcp_server/test_python_execute_tool.py``'s
    ``save_artifact()`` regressions) with the perimeter env vars actually
    set, and read the refusal back out of the real child's stdout. No mock
    on the arming path itself — only the listening server the child connects
    to is test infrastructure.
    """

    async def test_denied_connect_is_refused_through_the_real_chain(self, tmp_path, monkeypatch):
        from osprey.mcp_server.python_executor.executor import _execute_via_local

        denied_srv = socket.create_server(("127.0.0.1", 0))
        try:
            denied_port = denied_srv.getsockname()[1]
            # A second, unopened port in the deny-list: its number appearing
            # in the live refusal (not just the one actually dialled) is what
            # pins refusal-message quality — the prefix constant AND the full
            # denied-port list are both child-visible, not just the port that
            # happened to be hit.
            second_denied_port = 19682
            # The real executor path writes an in-flight marker under
            # `target_state.state_dir()`, which with no stamp resolves through
            # `resolve_shared_data_root()` to the REPOSITORY — so this test
            # created `<repo>/var/agent_data/control_target/` and, because the
            # marker is unlinked on the way out, left only an empty gitignored
            # directory nothing would notice. Stamp it into tmp instead.
            monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(tmp_path / "agent_data"))
            monkeypatch.setenv("OSPREY_WEB_PERIMETER", "open")
            monkeypatch.setenv(
                "OSPREY_WEB_PERIMETER_DENY_PORTS", f"{denied_port},{second_denied_port}"
            )

            execution_folder = tmp_path / "exec"
            execution_folder.mkdir()
            (execution_folder / "figures").mkdir()

            result = await _execute_via_local(
                code=textwrap.dedent(
                    f"""
                    import socket
                    try:
                        socket.socket().connect(("127.0.0.1", {denied_port}))
                        print("UNEXPECTED_CONNECTED")
                    except PermissionError as exc:
                        print("DENIED:", exc)
                    """
                ),
                execution_mode="readonly",
                config={"timeout": 30, "python_env_path": None},
                execution_folder=execution_folder,
            )
        finally:
            denied_srv.close()

        assert result.success, (
            f"real end-to-end execution failed: {result.error_message}\n{result.stderr}"
        )
        assert "UNEXPECTED_CONNECTED" not in result.stdout
        assert f"DENIED: {NET_GUARD_REFUSAL_PREFIX}" in result.stdout
        # Message-quality assertion (checklist item 5): prefix constant AND
        # the full denied-port list are both present in the child-visible
        # refusal, not just the single port that was dialled.
        assert str(denied_port) in result.stdout
        assert str(second_denied_port) in result.stdout
