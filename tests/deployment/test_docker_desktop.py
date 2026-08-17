"""Reading Docker Desktop's host-networking setting, and refusing to guess.

The behaviour under test is a three-valued answer: enabled, disabled, or "cannot
tell". The third value is the one worth tests, because it is what keeps a
hedged-but-honest warning from being replaced by a confident wrong one on a host
this reader cannot interrogate.

The backend-API tests speak real HTTP over a real ``AF_UNIX`` socket rather than
stubbing the connection out. The adaptation being tested IS the socket dial, so a
test that stubbed it would assert nothing about the only part that can break.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from osprey.deployment import docker_desktop

pytestmark = pytest.mark.unit


#: A settings payload shaped like the real one, trimmed to the read path.
def _settings(host_networking: object) -> dict:
    return {
        "desktop": {"enhancedContainerIsolation": {"locked": False, "value": False}},
        "vm": {
            "network": {
                "defaultNetworkingMode": {"locked": False, "value": "ipv4only"},
                "hostNetworkingEnabled": host_networking,
            }
        },
    }


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    """A directory short enough to hold an ``AF_UNIX`` path.

    Not ``tmp_path``: macOS caps a Unix socket path near 104 bytes, and pytest's
    per-test temp directories under ``/var/folders`` are long enough that
    appending a filename can cross it. A ``/tmp`` child keeps the whole path
    comfortably short.
    """
    if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - POSIX-only test
        pytest.skip("AF_UNIX sockets are unavailable on this platform")
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="osprey-dd-") as tmp:
        yield Path(tmp)


def _serve_once(socket_path: Path, status: str, body: str) -> threading.Thread:
    """Answer exactly one HTTP request on ``socket_path``, then close.

    One request is all :func:`host_networking_enabled` makes, so a one-shot
    server needs no shutdown handshake: the thread is done when the read is.
    """
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)

    def serve() -> None:
        try:
            conn, _ = server.accept()
            with conn:
                conn.recv(65536)  # the request line and headers, which we ignore
                payload = body.encode("utf-8")
                conn.sendall(
                    f"HTTP/1.1 {status}\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\n"
                    "Connection: close\r\n\r\n".encode()
                    + payload
                )
        except OSError:  # pragma: no cover - only on a torn-down test socket
            pass
        finally:
            server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def no_settings_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the on-disk fallback at nothing, isolating the source under test.

    Without this the reader would fall through to the developer's own Docker
    Desktop settings and the test would assert on the machine it runs on.
    """
    monkeypatch.setattr(docker_desktop, "settings_store_paths", lambda: (tmp_path / "absent.json",))


@pytest.fixture
def no_backend_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the API source at nothing, so only the on-disk source can answer."""
    monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: tmp_path / "absent.sock")


class TestTheBackendApiSource:
    """The live setting, read from the running Docker Desktop."""

    @pytest.mark.parametrize("enabled", [False, True])
    def test_it_reports_what_docker_desktop_says(
        self, socket_dir, monkeypatch, no_settings_store, enabled
    ):
        """The load-bearing case: a disabled forwarder is reported as disabled,
        which is what lets a warning name the checkbox instead of hedging."""
        sock = socket_dir / "backend.sock"
        _serve_once(sock, "200 OK", json.dumps(_settings(enabled)))
        monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: sock)

        assert docker_desktop.host_networking_enabled() is enabled

    def test_it_reads_a_value_wrapped_with_its_lock_state(
        self, socket_dir, monkeypatch, no_settings_store
    ):
        """Docker Desktop wraps some settings as ``{"locked", "value"}`` and
        leaves others bare, and which is which has moved between versions."""
        sock = socket_dir / "backend.sock"
        _serve_once(sock, "200 OK", json.dumps(_settings({"locked": False, "value": True})))
        monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: sock)

        assert docker_desktop.host_networking_enabled() is True

    def test_an_api_that_does_not_answer_this_path_tells_us_nothing(
        self, socket_dir, monkeypatch, no_settings_store
    ):
        """A future Desktop that moves the endpoint must not read as disabled."""
        sock = socket_dir / "backend.sock"
        _serve_once(sock, "404 Not Found", "{}")
        monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: sock)

        assert docker_desktop.host_networking_enabled() is None

    def test_a_body_that_is_not_the_expected_json_tells_us_nothing(
        self, socket_dir, monkeypatch, no_settings_store
    ):
        sock = socket_dir / "backend.sock"
        _serve_once(sock, "200 OK", "not json at all")
        monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: sock)

        assert docker_desktop.host_networking_enabled() is None

    def test_a_payload_without_the_key_tells_us_nothing(
        self, socket_dir, monkeypatch, no_settings_store
    ):
        """Absent from the API is still absent: no key, no claim."""
        sock = socket_dir / "backend.sock"
        _serve_once(sock, "200 OK", json.dumps({"vm": {"network": {}}}))
        monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: sock)

        assert docker_desktop.host_networking_enabled() is None

    def test_a_socket_nobody_is_listening_on_tells_us_nothing(
        self, socket_dir, monkeypatch, no_settings_store
    ):
        """Docker Desktop stopped between the deploy starting and this read."""
        sock = socket_dir / "backend.sock"
        sock.write_bytes(b"")  # a plain file where a socket should be
        monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: sock)

        assert docker_desktop.host_networking_enabled() is None

    def test_no_socket_at_all_tells_us_nothing(self, monkeypatch, tmp_path, no_settings_store):
        monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: None)

        assert docker_desktop.host_networking_enabled() is None


class TestThePersistedStoreFallback:
    """The on-disk source, for the hosts whose backend cannot be dialled."""

    def test_an_explicit_setting_is_read(self, monkeypatch, tmp_path, no_backend_socket):
        store = tmp_path / "settings-store.json"
        store.write_text(json.dumps({"HostNetworkingEnabled": False}), encoding="utf-8")
        monkeypatch.setattr(docker_desktop, "settings_store_paths", lambda: (store,))

        assert docker_desktop.host_networking_enabled() is False

    def test_a_store_without_the_key_tells_us_nothing(
        self, monkeypatch, tmp_path, no_backend_socket
    ):
        """The case that makes this source the weaker one, and the reason it is
        second: the store persists only non-default settings, so a host that has
        never touched host networking has no key. This is what a fresh Docker
        Desktop install looks like, and reading it as "disabled" would blame
        this setting for every unrelated failure on such a host."""
        store = tmp_path / "settings-store.json"
        store.write_text(json.dumps({"AutoStart": False, "SettingsVersion": 44}), encoding="utf-8")
        monkeypatch.setattr(docker_desktop, "settings_store_paths", lambda: (store,))

        assert docker_desktop.host_networking_enabled() is None

    def test_the_older_filename_is_still_read(self, monkeypatch, tmp_path, no_backend_socket):
        """Docker Desktop renamed ``settings.json`` to ``settings-store.json``."""
        legacy = tmp_path / "settings.json"
        legacy.write_text(json.dumps({"HostNetworkingEnabled": True}), encoding="utf-8")
        monkeypatch.setattr(
            docker_desktop,
            "settings_store_paths",
            lambda: (tmp_path / "settings-store.json", legacy),
        )

        assert docker_desktop.host_networking_enabled() is True

    def test_unreadable_json_tells_us_nothing(self, monkeypatch, tmp_path, no_backend_socket):
        store = tmp_path / "settings-store.json"
        store.write_text("{ truncated", encoding="utf-8")
        monkeypatch.setattr(docker_desktop, "settings_store_paths", lambda: (store,))

        assert docker_desktop.host_networking_enabled() is None


class TestSourcePrecedence:
    """The live setting wins over whatever was last persisted."""

    def test_the_api_answer_beats_a_stale_store(self, socket_dir, monkeypatch, tmp_path):
        """A store written before the operator last toggled the setting must not
        override what the running Docker Desktop reports now."""
        sock = socket_dir / "backend.sock"
        _serve_once(sock, "200 OK", json.dumps(_settings(True)))
        monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: sock)
        store = tmp_path / "settings-store.json"
        store.write_text(json.dumps({"HostNetworkingEnabled": False}), encoding="utf-8")
        monkeypatch.setattr(docker_desktop, "settings_store_paths", lambda: (store,))

        assert docker_desktop.host_networking_enabled() is True

    def test_the_store_answers_when_the_api_cannot(self, monkeypatch, tmp_path):
        """The Windows path: no dialable socket, so the store is all there is."""
        monkeypatch.setattr(docker_desktop, "backend_socket_path", lambda: None)
        store = tmp_path / "settings-store.json"
        store.write_text(json.dumps({"HostNetworkingEnabled": False}), encoding="utf-8")
        monkeypatch.setattr(docker_desktop, "settings_store_paths", lambda: (store,))

        assert docker_desktop.host_networking_enabled() is False


class TestTheDockerDesktopPredicate:
    """Which hosts this whole module applies to."""

    def test_docker_on_macos_is_docker_desktop(self, monkeypatch):
        monkeypatch.setattr(docker_desktop.sys, "platform", "darwin")
        monkeypatch.setattr(docker_desktop, "get_runtime_command", lambda config: ["docker"])

        assert docker_desktop.on_docker_desktop({}) is True

    def test_a_linux_host_is_not(self, monkeypatch):
        """There ``network_mode: host`` binds on the machine itself, so an
        unreachable port means something else entirely."""
        monkeypatch.setattr(docker_desktop.sys, "platform", "linux")

        assert docker_desktop.on_docker_desktop({}) is False

    def test_the_runtime_is_not_resolved_on_a_linux_host(self, monkeypatch):
        """The platform is checked first so the cheap answer stays cheap."""
        monkeypatch.setattr(docker_desktop.sys, "platform", "linux")

        def fail(config: dict) -> list[str]:
            raise AssertionError("the runtime should not be resolved on Linux")

        monkeypatch.setattr(docker_desktop, "get_runtime_command", fail)

        assert docker_desktop.on_docker_desktop({}) is False

    def test_podman_on_macos_is_not_docker_desktop(self, monkeypatch):
        """Podman runs its own machine with its own port handling."""
        monkeypatch.setattr(docker_desktop.sys, "platform", "darwin")
        monkeypatch.setattr(
            docker_desktop, "get_runtime_command", lambda config: ["podman", "compose"]
        )

        assert docker_desktop.on_docker_desktop({}) is False
