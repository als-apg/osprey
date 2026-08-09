"""The Google Cloud Pub/Sub emulator, in a container, for the Chat bridge lane.

The bridge ingests Chat events from a Pub/Sub subscription, so the one piece of
the hermetic e2e stack that cannot be a hand-written HTTP fake is Pub/Sub itself:
the product's subscriber speaks gRPC through ``google.cloud.pubsub_v1``, and the
emulator is what makes that real without a Google project. Everything else in
this fixture directory is a plain loopback server (see ``gchat_chat_fake`` and
``gchat_gcs_fake``) and needs no runtime at all.

----------------------------------------------------------------------------
CONTAINER-OPS SAFETY (every runtime-mutating call in this file honors this)
----------------------------------------------------------------------------
Exactly ONE container is ever created, exact-named off :data:`RESOURCE_PREFIX`:

  * container: ``osprey-e2e-gchat-pubsub``

Nothing here runs ``system prune``, ``volume prune``, ``container prune``,
``image prune``, ``-a``/``--all`` on a removal, or any wildcard/filter match. The
two removal sites — the pre-clean before ``run`` and the teardown in ``finally``
— each name that one container and nothing else, and both are best-effort
(``check=False``, so "no such container" is success). No volumes are created. No
image is ever removed. The host's unrelated stacks are untouched.

----------------------------------------------------------------------------
The image
----------------------------------------------------------------------------
:data:`DEFAULT_IMAGE` is Google's public cloud-sdk image with the emulator
component. Unlike the private Nextcloud fixture image, this one is public and
pullable, so a missing image is PULLED once (bounded, exact reference) rather
than treated as fatal — and if that pull fails, the emulator fixture SKIPS with a
message naming the reference. Override with :data:`IMAGE_ENV` to pin a version.

----------------------------------------------------------------------------
Skipping, and the vacuous-green hazard
----------------------------------------------------------------------------
:func:`pubsub_emulator` is the ONLY thing in this fixture directory allowed to
skip, and only for a genuinely absent container runtime or image. A suite whose
tests all skip is green while proving nothing, so the HTTP fakes deliberately
have no skip path: if they cannot boot, their tests fail.

----------------------------------------------------------------------------
Provisioning over REST
----------------------------------------------------------------------------
Topics, subscriptions and publishes go over the emulator's HTTP/JSON surface
(``PUT /v1/projects/{p}/topics/{t}`` and friends) with no credentials, so the
test side needs no Google client library even when the product side does. The
product's subscriber reaches the same emulator over gRPC via
``PUBSUB_EMULATOR_HOST`` — see :attr:`PubSubEmulator.env`.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

__all__ = [
    "PUBSUB_CONTAINER",
    "RESOURCE_PREFIX",
    "PubSubEmulator",
    "decode_pubsub_message",
    "image_available",
    "pubsub_emulator",
    "runtime_available",
]

# ---------------------------------------------------------------------------
# Container runtime selection — ``docker`` by default, ``podman`` opt-in via
# OSPREY_E2E_RUNTIME. Any other value fails clearly at import time rather than
# silently falling back to docker. Same contract as
# tests/e2e/test_nextcloud_talk_bridge_e2e.py, so one env var drives both lanes.
# ---------------------------------------------------------------------------
_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

# ---------------------------------------------------------------------------
# Fixture identity — the one runtime resource this module creates.
# See the CONTAINER-OPS SAFETY block in the module docstring.
# ---------------------------------------------------------------------------

RESOURCE_PREFIX = "osprey-e2e-gchat"
PUBSUB_CONTAINER = f"{RESOURCE_PREFIX}-pubsub"

IMAGE_ENV = "OSPREY_E2E_GCHAT_PUBSUB_IMAGE"
DEFAULT_IMAGE = "gcr.io/google.com/cloudsdktool/google-cloud-cli:emulators"
IMAGE = os.environ.get(IMAGE_ENV) or DEFAULT_IMAGE

EMULATOR_CONTAINER_PORT = 8085
"""Port inside the container. The host port is picked free, per run."""

DEFAULT_PROJECT = "osprey-e2e-gchat"

READY_TIMEOUT_SEC = 180.0
"""Container start plus the emulator's JVM boot, with slack for a cold machine."""

PULL_TIMEOUT_SEC = 900.0
"""One-time pull of a ~1 GB image on a slow link."""


def _runtime(*args: str, check: bool = True, timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Run one container-runtime command."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [RUNTIME, *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"{RUNTIME} {' '.join(args)} failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc


def _remove_container(name: str) -> None:
    """Remove ONE exact-named container. Best-effort: "no such container" is success."""
    _runtime("rm", "-f", name, check=False)


def runtime_available() -> bool:
    """Whether the chosen runtime's CLI exists AND its daemon actually answers."""
    if shutil.which(RUNTIME) is None:
        return False
    try:
        return _runtime("info", check=False, timeout=60.0).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def image_available() -> bool:
    """Whether :data:`IMAGE` is in the local store, pulling it once if it is not."""
    if _runtime("image", "inspect", IMAGE, check=False).returncode == 0:
        return True
    return _runtime("pull", IMAGE, check=False, timeout=PULL_TIMEOUT_SEC).returncode == 0


def _free_port() -> int:
    """Bind to :0, read the assigned port, release it (standard free-port trick)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _container_logs(name: str, tail: int = 60) -> str:
    proc = _runtime("logs", "--tail", str(tail), name, check=False)
    return f"--- {name} logs (last {tail}) ---\n{proc.stdout}\n{proc.stderr}"


@dataclass
class PubSubEmulator:
    """A running emulator: its address, its provisioning calls, its teardown.

    Build it with :meth:`start`; prefer the :func:`pubsub_emulator` fixture, which
    handles the runtime gate and the teardown.
    """

    port: int
    project: str

    @property
    def host(self) -> str:
        """``host:port`` — the value ``PUBSUB_EMULATOR_HOST`` takes."""
        return f"127.0.0.1:{self.port}"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}"

    @property
    def env(self) -> dict[str, str]:
        """Environment a Pub/Sub client needs to reach this emulator instead of Google."""
        return {"PUBSUB_EMULATOR_HOST": self.host, "PUBSUB_PROJECT_ID": self.project}

    # -- resource names -----------------------------------------------------

    def topic_path(self, topic_id: str) -> str:
        return f"projects/{self.project}/topics/{topic_id}"

    def subscription_path(self, subscription_id: str) -> str:
        return f"projects/{self.project}/subscriptions/{subscription_id}"

    # -- provisioning (REST, no credentials, no Google client library) ------

    def create_topic(self, topic_id: str) -> str:
        """Create a topic; returns its full resource name."""
        path = self.topic_path(topic_id)
        self._call("PUT", f"/v1/{path}", {})
        return path

    def create_subscription(
        self, subscription_id: str, topic_id: str, *, ack_deadline_seconds: int = 10
    ) -> str:
        """Create a pull subscription on ``topic_id``; returns its resource name."""
        path = self.subscription_path(subscription_id)
        self._call(
            "PUT",
            f"/v1/{path}",
            {"topic": self.topic_path(topic_id), "ackDeadlineSeconds": ack_deadline_seconds},
        )
        return path

    def publish(
        self, topic_id: str, data: bytes | str, attributes: Mapping[str, str] | None = None
    ) -> str:
        """Publish one message; returns the emulator's message id.

        ``data`` is sent verbatim (base64 on the wire, as the API requires), so a
        byte-identical republish is byte-identical to the subscriber — which is
        what the exactly-once assertions in the bridge e2e depend on.
        """
        payload = data.encode("utf-8") if isinstance(data, str) else data
        message: dict[str, object] = {"data": base64.b64encode(payload).decode("ascii")}
        if attributes:
            message["attributes"] = dict(attributes)
        body = self._call(
            "POST", f"/v1/{self.topic_path(topic_id)}:publish", {"messages": [message]}
        )
        ids = body.get("messageIds")
        if not isinstance(ids, list) or not ids:
            raise AssertionError(f"emulator accepted the publish but returned no id: {body!r}")
        return str(ids[0])

    def pull(self, subscription_id: str, *, max_messages: int = 10) -> list[dict[str, Any]]:
        """Pull whatever is immediately available; returns ``receivedMessages``.

        The test side's own read path — for checking what was published. The
        product's subscriber pulls over gRPC and is never driven from here.
        """
        body = self._call(
            "POST",
            f"/v1/{self.subscription_path(subscription_id)}:pull",
            {"maxMessages": max_messages, "returnImmediately": True},
        )
        received = body.get("receivedMessages")
        if not isinstance(received, list):
            return []
        return [message for message in received if isinstance(message, dict)]

    def _call(self, method: str, path: str, body: dict[str, object]) -> dict[str, object]:
        response = httpx.request(
            method, f"{self.base_url}{path}", json=body, timeout=30.0, trust_env=False
        )
        if response.status_code >= 400:
            raise AssertionError(
                f"emulator {method} {path} failed: HTTP {response.status_code} {response.text[:400]}"
            )
        try:
            parsed = response.json()
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def start(cls, *, project: str = DEFAULT_PROJECT) -> PubSubEmulator:
        """Start the emulator container and wait until it answers.

        Pre-cleans the one exact-named container first, so a previous run that
        died without teardown does not block this one.
        """
        _remove_container(PUBSUB_CONTAINER)
        port = _free_port()
        _runtime(
            "run",
            "-d",
            "--name",
            PUBSUB_CONTAINER,
            "-p",
            f"127.0.0.1:{port}:{EMULATOR_CONTAINER_PORT}",
            IMAGE,
            "gcloud",
            "beta",
            "emulators",
            "pubsub",
            "start",
            f"--host-port=0.0.0.0:{EMULATOR_CONTAINER_PORT}",
            f"--project={project}",
        )
        emulator = cls(port=port, project=project)
        try:
            emulator._wait_until_ready()
        except BaseException:
            emulator.stop()
            raise
        return emulator

    def _wait_until_ready(self) -> None:
        """Poll the emulator's root endpoint until it answers over HTTP.

        The emulator serves HTTP/JSON and gRPC on the same port; an HTTP answer of
        any kind means the listener is up and the container's port publish works,
        which is exactly the readiness the tests need.
        """
        deadline = time.monotonic() + READY_TIMEOUT_SEC
        last = "(not probed yet)"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{self.base_url}/", timeout=5.0, trust_env=False)
            except httpx.HTTPError as exc:
                last = f"transport error: {exc}"
            else:
                if response.status_code < 500:
                    return
                last = f"HTTP {response.status_code}: {response.text[:200]!r}"
            time.sleep(1.0)
        raise AssertionError(
            f"{PUBSUB_CONTAINER} did not answer on {self.base_url} within "
            f"{READY_TIMEOUT_SEC:.0f}s.\n  last probe: {last}\n{_container_logs(PUBSUB_CONTAINER)}"
        )

    def stop(self) -> None:
        """Remove the one exact-named container. Never a prune."""
        _remove_container(PUBSUB_CONTAINER)


@pytest.fixture(scope="module")
def pubsub_emulator() -> Iterator[PubSubEmulator]:
    """Module-scoped running emulator, or a clean SKIP when the runtime is absent.

    The skip is the one in this fixture directory (see the module docstring): it
    fires only when the container runtime or the image genuinely is not there,
    never to paper over a broken emulator — a started container that does not
    answer FAILS with its logs attached.
    """
    if not runtime_available():
        pytest.skip(
            f"{RUNTIME} is unavailable (CLI missing or daemon down); "
            f"set OSPREY_E2E_RUNTIME to choose a runtime"
        )
    if not image_available():
        pytest.skip(f"Pub/Sub emulator image {IMAGE!r} is unavailable (not present, pull failed)")

    emulator = PubSubEmulator.start()
    try:
        yield emulator
    finally:
        emulator.stop()


def decode_pubsub_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Decode one emulator ``receivedMessages[].message`` payload to a dict.

    A convenience for tests that pull from the emulator to check what they
    published; the product's own decoder is the thing under test elsewhere.
    """
    raw = base64.b64decode(str(message.get("data") or ""))
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}
