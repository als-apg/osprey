"""What Docker Desktop's own settings say about this host's networking.

Docker Desktop runs containers inside a Linux VM, so ``network_mode: host``
binds a port on *that* VM and not on the machine the operator is sitting at. The
port reaches the real machine only through Docker Desktop's host-network
forwarder, which is off unless "Enable host networking" is turned on (Docker
Desktop 4.34 and later). With it off, a host-networked container starts, passes
every healthcheck it runs on itself, and is unreachable from any browser on the
host. OSPREY's web-terminal tier is exactly that shape, which is why this module
exists: it is the difference between telling an operator their deploy "is not
reachable" and telling them which checkbox to click.

The read is a question about Docker Desktop, not about OSPREY, so it is
deliberately advisory: every failure path returns ``None`` ("cannot tell") rather
than raising or guessing. A caller that gets ``None`` must keep whatever hedged
wording it had, because a confident wrong diagnosis is worse than an honest
vague one.

Two sources, most authoritative first.

1. Docker Desktop's backend API over its own Unix socket. This is the live
   setting, it always carries the key, and it agrees with what the Settings
   window shows. Preferred for that reason.
2. The persisted settings store on disk, as a fallback for the hosts where the
   socket is not reachable (Windows, where the backend listens on a named pipe
   rather than a Unix socket). This source is weaker in one specific way, and
   the weakness is the whole reason source 1 comes first: the store persists
   only settings that differ from their default, so a host that has never
   touched host networking has no key at all. Absent therefore means "cannot
   tell", NOT "disabled" -- reading it as "disabled" would blame this setting on
   every Windows host whose web tier is dark for some unrelated reason.
"""

from __future__ import annotations

import http.client
import json
import socket
import sys
from pathlib import Path
from typing import Any

from osprey.deployment.runtime_helper import get_runtime_command
from osprey.utils.logger import get_logger

logger = get_logger("deployment.docker_desktop")

#: What an operator does about a disabled forwarder, in the words the Docker
#: Desktop UI uses for each step. Shared so the preflight refusal and the
#: post-up warning cannot drift apart into two different sets of directions.
HOST_NETWORKING_REMEDY = (
    "in Docker Desktop, turn on Settings -> Resources -> Network -> "
    "'Enable host networking', then Apply & restart"
)

#: Seconds to wait on the backend socket. Generous enough for a busy Docker
#: Desktop and short enough that a wedged one cannot hold up a deploy: the
#: answer is advisory, so giving up is a valid outcome.
_SOCKET_TIMEOUT = 3.0

#: The settings endpoint on Docker Desktop's backend API.
_SETTINGS_PATH = "/app/settings"

#: The key in the persisted store, which spells it in PascalCase where the API
#: spells it in camelCase.
_STORE_KEY = "HostNetworkingEnabled"


class _UnixHTTPConnection(http.client.HTTPConnection):
    """``HTTPConnection`` that dials a Unix socket instead of a TCP port.

    The standard library has no HTTP-over-Unix-socket client, and Docker
    Desktop's backend speaks ordinary HTTP/1.1 over ``AF_UNIX``. Overriding
    :meth:`connect` is the whole adaptation: everything above the socket --
    request framing, chunked bodies, the response parser -- is unchanged.
    """

    def __init__(self, socket_path: str, timeout: float) -> None:
        # The Host header has to be *something* and is never read by a backend
        # that only ever hears from this machine.
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        """Open the ``AF_UNIX`` stream this connection will speak HTTP over."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def backend_socket_path() -> Path | None:
    """Docker Desktop's backend socket for this platform, or ``None``.

    ``None`` on Windows on purpose: the backend there listens on a named pipe,
    which this client cannot dial, so those hosts fall through to the persisted
    store. Returned rather than read from a constant so tests can point the
    reader at a socket they control.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library/Containers/com.docker.docker/Data/backend.sock"
    if sys.platform.startswith("linux"):
        return Path.home() / ".docker/desktop/backend.sock"
    return None


def settings_store_paths() -> tuple[Path, ...]:
    """The persisted settings files to try, in order, for this platform.

    Both spellings are listed because Docker Desktop renamed the file:
    ``settings.json`` up to 4.34, ``settings-store.json`` after it. Trying both
    costs one ``is_file`` call and keeps the fallback working across the
    versions an operator might actually be running.
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library/Group Containers/group.com.docker"
    elif sys.platform == "win32":
        appdata = Path.home() / "AppData/Roaming"
        base = appdata / "Docker"
    else:
        base = Path.home() / ".docker/desktop"
    return (base / "settings-store.json", base / "settings.json")


def _unwrap(value: Any) -> bool | None:
    """The boolean in ``value``, however Docker Desktop wrapped it.

    The settings API returns some fields bare (``false``) and others wrapped
    with their lock state (``{"locked": false, "value": false}``). Which
    treatment a given field gets has changed between versions, so read both
    shapes rather than pinning today's.
    """
    if isinstance(value, dict):
        value = value.get("value")
    return value if isinstance(value, bool) else None


def _from_backend_socket() -> bool | None:
    """Ask the running Docker Desktop, or ``None`` if it cannot be asked."""
    socket_path = backend_socket_path()
    if socket_path is None or not socket_path.exists():
        return None

    conn = _UnixHTTPConnection(str(socket_path), timeout=_SOCKET_TIMEOUT)
    try:
        conn.request("GET", _SETTINGS_PATH)
        response = conn.getresponse()
        if response.status != 200:
            logger.debug(f"Docker Desktop settings API answered {response.status}.")
            return None
        payload = json.loads(response.read())
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # Every way this can fail is a reason to stay quiet rather than to fail
        # a deploy: Desktop not running, socket moved, an API that no longer
        # answers this path, a body that is not the JSON this expects.
        logger.debug(f"Could not read the Docker Desktop settings API: {exc}")
        return None
    finally:
        conn.close()

    if not isinstance(payload, dict):
        return None
    network = (payload.get("vm") or {}).get("network") or {}
    if not isinstance(network, dict):
        return None
    return _unwrap(network.get("hostNetworkingEnabled"))


def _from_settings_store() -> bool | None:
    """Read the persisted store, or ``None`` when it does not say.

    Only an explicit key counts. See this module's docstring for why a missing
    key is "cannot tell" rather than "disabled".
    """
    for path in settings_store_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and _STORE_KEY in payload:
            return _unwrap(payload[_STORE_KEY])
    return None


def host_networking_enabled() -> bool | None:
    """Whether Docker Desktop will forward a host-networked port to this machine.

    :return: ``True`` or ``False`` when Docker Desktop says so, ``None`` when
        this host cannot be asked. Callers must treat ``None`` as "no new
        information" and keep their existing wording.
    """
    from_api = _from_backend_socket()
    if from_api is not None:
        return from_api
    return _from_settings_store()


def on_docker_desktop(config: dict) -> bool:
    """Whether this deployment's containers run under Docker Desktop.

    Docker Desktop is the macOS and Windows product, so the platform check is
    what distinguishes it from a Linux host running the engine directly -- where
    ``network_mode: host`` binds on the machine itself and none of this applies.
    Podman on macOS runs its own machine with its own port handling and is not
    Docker Desktop, hence the runtime check as well.

    The platform is read first so that a Linux host never invokes the runtime
    resolver at all, which is what keeps this cheap on the hosts where the
    answer is structurally ``False``.

    :param config: Raw deploy config, read only for which runtime it selects.
    """
    if sys.platform not in ("darwin", "win32"):
        return False
    return get_runtime_command(config)[0] == "docker"
