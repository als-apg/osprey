"""Real-container e2e for the Nextcloud Talk bridge against a REAL Talk server.

Every other test of ``osprey.bridges.nextcloud_talk`` drives the adapter over an
``httpx.MockTransport``, so all of them assert OSPREY's half of a two-party contract
against OSPREY's own idea of Talk's wire format. This module is the instrument for the
other half: a genuine, fully provisioned Nextcloud 33 with Talk (spreed) 23 in a
container, a real ``osprey.dispatch`` dispatcher/worker pair as subprocesses, and the
bridge itself booted through its own entrypoint seams. What it proves is that the
adapter's room-type rules, its mention filter, its poll cursor and its restart recovery
behave the way the mocked tests claim when the server is the real one.

The four tests here are **deterministic**: none of their assertions reads model output,
so they pass with no provider key configured at all. A dispatch is observed as "the
dedup entry now carries a ``run_id``" — the engine persists that the moment the
dispatcher handshake yields one, long before the agent finishes — so a run that errors
out for want of an API key proves exactly as much as one that answers. The reply half
(what the agent actually said) is the agentic lane's, and lives below the marked section
at the bottom of this file.

----------------------------------------------------------------------------
CONTAINER-OPS SAFETY (every runtime-mutating call in this file honors this)
----------------------------------------------------------------------------
Every container/volume this test creates is exact-named off :data:`RESOURCE_PREFIX`:

  * container: ``osprey-e2e-nctalk-server`` (the Nextcloud + Talk fixture).
  * volume: ``osprey-e2e-nctalk-html`` (its ``/var/www/html``).

Nothing here ever runs ``system prune``, ``volume prune``, ``container prune``,
``image prune``, ``-a``/``--all`` on a removal, or a wildcard/glob container-name
match. Every removal names one of the two exact resources above and nothing else, and
the fixture IMAGE is never removed, never rebuilt and never pulled (see below). The
pre-clean at the start of the ``nextcloud`` fixture and the teardown in its ``finally``
are the only two removal sites, both best-effort (``check=False``, so "already gone" is
success) — a safety net, never a prune. The host's unrelated stacks are untouched.

----------------------------------------------------------------------------
The fixture image is NEVER built or pulled here
----------------------------------------------------------------------------
:data:`NEXTCLOUD_FIXTURE_IMAGE` is expected to be present in the local image store
already — published by ``.github/workflows/build-nextcloud-fixture.yml``, or built once
by hand from ``tests/e2e/fixtures/Dockerfile.nextcloud_talk`` (a ~1 hour build). If it
is absent this module SKIPS with a message naming that workflow; it does not pull (the
ghcr package may not exist for the current checkout) and it does not build. ``docker
run`` is passed ``--pull=never`` so even an accidental fallback cannot reach a registry.
Override the reference with the :data:`FIXTURE_IMAGE_ENV` environment variable to test
against a locally built tag.

----------------------------------------------------------------------------
Three failure modes this module is shaped around
----------------------------------------------------------------------------
**1. Trusted domains are a health-INVISIBLE 400 wall.** ``NEXTCLOUD_TRUSTED_DOMAINS`` is
applied by the entrypoint at *install* time only, and the image bakes exactly
``localhost 127.0.0.1 host.docker.internal nextcloud nextcloud-talk-fixture``. The
container's HEALTHCHECK probes ``127.0.0.1``, which is always trusted, so a container
addressed under any other hostname reports **healthy** while every OCS call returns
``HTTP 400 "Trusted domain error"``. Hence :data:`NEXTCLOUD_BASE_URL` addresses the
fixture as ``http://localhost:<port>`` — one of the five baked names. A port is fine;
Nextcloud strips it before the comparison. :func:`_ocs` spells this out in the failure
message it raises on a 400, because that is the only place it would ever surface.

**2. Reusing a ``/var/www/html`` volume can silently produce a spreed-less server.** The
official entrypoint rsyncs ``/usr/src/nextcloud`` into the served tree only when the
image version is *greater* than the installed one, so a volume previously populated by a
plain ``nextcloud:33.0.7-apache`` leaves no spreed in the served tree and hard-fails
provisioning at ``app:enable``. The volume is therefore purged at **both** ends — before
the container starts and again in teardown. That is a correctness requirement, not
hygiene.

**3. Docker's own ``unhealthy`` verdict takes ~640 s** (240 s start-period + 40 retries x
10 s), which is far longer than this module's own
:data:`NEXTCLOUD_READY_TIMEOUT_SEC`. So the readiness wait below is the ONLY signal a
broken fixture will ever produce, and its failure message is correspondingly
diagnostic — it dumps the container's recent logs, the runtime's health verdict, and the
last result of each readiness half. That wait polls **both** halves of the HEALTHCHECK
(the marker file AND ``status.php`` reporting ``"installed":true``) and is deliberately
not simplified to one: on a ``docker restart`` the marker survives in the writable
layer, so the marker alone is a false-ready path.

----------------------------------------------------------------------------
Host port
----------------------------------------------------------------------------
:data:`NEXTCLOUD_PORT` is pinned to 18107, free alongside the ports its sibling e2e
modules pin (5064, 15080, 18090, 18095, 18099, 18101-18106, 19081, 25080, 25432). The
dispatcher/worker pair deliberately takes **free** ports instead (:func:`_free_port`):
they are per-run subprocesses with no need to be predictable, and a pin would collide
with a developer's own running stack.

Gating: needs a container runtime whose daemon is actually running (not just the CLI
installed) and the fixture image in the local store. Runtime is ``docker`` by default;
set ``OSPREY_E2E_RUNTIME=podman`` to run against podman instead (any other value fails
at collection time with a clear error), matching ``tests/e2e/test_deploy_lifecycle.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from osprey.bridges.core import PNG_MAGIC, TERMINAL_STATUSES
from osprey.bridges.nextcloud_talk.__main__ import Wiring, build_wiring, config_from_env, run
from osprey.bridges.nextcloud_talk.client import (
    CHAT_API,
    CONNECT_TIMEOUT,
    DAV_FILES_ROOT,
    REQUEST_TIMEOUT,
    ROOM_API,
    SHARE_TYPE_ROOM,
    SHARES_API,
    upload_dir,
)
from osprey.bridges.nextcloud_talk.events import FILE_PARAM_TYPE
from osprey.bridges.nextcloud_talk.ops import ACK_TEXT, EMPTY_ANSWER_TEXT, ERROR_TEXT

# ---------------------------------------------------------------------------
# Container runtime selection — ``docker`` by default, ``podman`` opt-in via
# OSPREY_E2E_RUNTIME. Any other value fails clearly at collection time rather
# than silently falling back to docker. Same contract as
# tests/e2e/test_deploy_lifecycle.py, so one env var drives both modules.
# ---------------------------------------------------------------------------
_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.dockerbuild,
    pytest.mark.skipif(
        shutil.which(RUNTIME) is None,
        reason=f"{RUNTIME} CLI not installed (set OSPREY_E2E_RUNTIME to choose a runtime)",
    ),
]


# ---------------------------------------------------------------------------
# The fixture image
# ---------------------------------------------------------------------------

NEXTCLOUD_FIXTURE_IMAGE = "ghcr.io/als-apg/nextcloud-talk-fixture:33.0.7-spreed23.0.9"
"""The pinned Nextcloud + Talk fixture image this lane runs against.

Both halves of the tag are load-bearing and move together: Talk 23.x declares
``min-version == max-version == 33``, so the pair is not independently selectable. The
authoritative pins live as greppable tokens in
``tests/e2e/fixtures/Dockerfile.nextcloud_talk`` (``NEXTCLOUD_PIN`` / ``SPREED_PIN``) and
a wiring test asserts THIS constant against them — so it must stay a plain literal, not
an assembled or env-derived string, and it must change in lockstep with the Dockerfile,
the fixture build workflow, and the ghcr tag."""

FIXTURE_IMAGE_ENV = "OSPREY_E2E_NEXTCLOUD_IMAGE"
"""Environment variable overriding :data:`NEXTCLOUD_FIXTURE_IMAGE` for this run.

For testing against a locally built tag (e.g. ``nextcloud-talk-fixture:local-verify``)
without touching the pin the drift guard checks. Never used by CI."""

FIXTURE_IMAGE = os.environ.get(FIXTURE_IMAGE_ENV) or NEXTCLOUD_FIXTURE_IMAGE
"""The image reference actually run — the pin unless overridden."""


# ---------------------------------------------------------------------------
# Fixture identity — every runtime resource this module creates is exact-named
# off this prefix. See the CONTAINER-OPS SAFETY block in the module docstring.
# ---------------------------------------------------------------------------

RESOURCE_PREFIX = "osprey-e2e-nctalk"
NEXTCLOUD_CONTAINER = f"{RESOURCE_PREFIX}-server"
NEXTCLOUD_VOLUME = f"{RESOURCE_PREFIX}-html"

NEXTCLOUD_PORT = 18107
NEXTCLOUD_BASE_URL = f"http://localhost:{NEXTCLOUD_PORT}"
"""``localhost`` is NOT interchangeable here — it is one of the five hostnames the image
bakes into ``trusted_domains`` at install time. See failure mode 1 in the module
docstring."""

READY_MARKER = "/var/lib/talk-fixture/ready"
"""Provisioning marker the image's HEALTHCHECK tests for. Lives outside the
``/var/www/html`` volume, so it is per-container: a fresh container is never born
ready."""

# Throwaway fixture credentials, part of the Dockerfile's contract (its ENV block
# documents that changing either requires changing them here).
ADMIN_ACCOUNT = "admin"
ADMIN_PASSWORD = "talk-fixture-admin-9F3k"

# Accounts this module provisions. The bot authenticates with its plain account
# password: a real deployment issues an app password instead, but the bridge only ever
# sees "the Basic-auth secret for NEXTCLOUD_BOT_ACCOUNT", so the distinction is invisible
# to the code under test and an app-password round trip would only add a failure mode.
BOT_ACCOUNT = "osprey-bot"
BOT_PASSWORD = "talk-fixture-bot-7Qv2"
HUMAN_ACCOUNT = "operator"
HUMAN_PASSWORD = "talk-fixture-human-5Rk8"

ROOM_TYPE_GROUP = 2
ROOM_TYPE_DIRECT = 1

# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

NEXTCLOUD_READY_TIMEOUT_SEC = 300.0
"""How long to wait for the fixture to finish its first-boot install and provision Talk.

This is the ONLY signal a broken fixture produces — the runtime's own ``unhealthy``
verdict needs ~640 s, far past this. See failure mode 3 in the module docstring."""

HEALTH_TIMEOUT_SEC = 45.0
"""Per-subprocess wait for the dispatcher's / worker's ``/health``."""

ANCHOR_TIMEOUT_SEC = 90.0
"""Wait for a room's first persisted poll offset — one room-type fetch plus one long-poll
hold (30 s) plus slack."""

OFFSET_TIMEOUT_SEC = 90.0
"""Wait for the persisted offset to move past a specific message id."""

RUN_ID_TIMEOUT_SEC = 150.0
"""Wait for a claimed message's dedup entry to carry a ``run_id`` — one webhook POST plus
the dispatcher's accept handshake, well short of any agent work."""

BRIDGE_JOIN_TIMEOUT_SEC = 90.0
"""Wait for the in-process bridge's threads after signalling shutdown.

Generous, and it is not politeness. The room threads are daemons that cannot block the
interpreter, but a poller sitting in a 30 s long poll only notices the stop event when
that poll returns — and ``serve_rooms`` gives each poller a bounded join and then returns
regardless. A poller that outlives the bridge object still holds a live dedup store on
the same files, so it can claim a message the test believes arrived while the bridge was
DOWN. That is precisely the vacuity the restart proof exists to rule out, so shutdown
waits for the room threads themselves rather than for ``run`` to return."""

DISPATCH_TOKEN = "nextcloud-talk-e2e-token"
"""Shared dispatcher<->worker bearer for this run. Both halves are local subprocesses."""

TRIGGER_NAME = "nextcloud-talk-e2e"
"""Name of the deterministic trigger :func:`_write_triggers` generates."""

WORKER_RUN_CAP_SEC = 90
"""``DISPATCH_TIMEOUT_SEC`` for the worker AND the bridge's ``poll_budget`` floor.

Bounds how long a wedged run can hold a room thread. The two must agree: the bridge's
``CoreConfig`` refuses to build when ``POLL_BUDGET < DISPATCH_TIMEOUT_SEC``."""


# ---------------------------------------------------------------------------
# Low-level runtime helpers. Removals are best-effort and EXACT-NAMED only.
# ---------------------------------------------------------------------------


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


def _remove_volume(name: str) -> None:
    """Remove ONE exact-named volume. Best-effort: "no such volume" is success.

    Never a prune, and never ``-a``: the host runs unrelated stacks whose volumes must
    survive a failed run of this module.
    """
    _runtime("volume", "rm", name, check=False)


def _daemon_available() -> bool:
    """Whether the chosen runtime's daemon actually answers (not just the CLI existing)."""
    try:
        return _runtime("info", check=False, timeout=60.0).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _image_present(reference: str) -> bool:
    """Whether ``reference`` is already in the local image store. Never pulls."""
    return _runtime("image", "inspect", reference, check=False).returncode == 0


def _container_logs(name: str, tail: int = 60) -> str:
    proc = _runtime("logs", "--tail", str(tail), name, check=False)
    return f"--- {name} logs (last {tail}) ---\n{proc.stdout}\n{proc.stderr}"


def _health_verdict(name: str) -> str:
    proc = _runtime(
        "inspect", "--format", "{{.State.Status}}/{{.State.Health.Status}}", name, check=False
    )
    return (proc.stdout or proc.stderr).strip() or "(no verdict)"


def _occ(*args: str, env: Mapping[str, str] | None = None) -> str:
    """Run ``occ`` inside the fixture container as ``www-data``.

    The official image's entrypoint hooks already run as ``www-data``; an ``occ`` invoked
    as root refuses to run, hence the explicit ``-u``.
    """
    cmd = [RUNTIME, "exec", "-u", "www-data"]
    for name, value in (env or {}).items():
        cmd += ["-e", f"{name}={value}"]
    cmd += [NEXTCLOUD_CONTAINER, "php", "occ", *args]
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        cmd, capture_output=True, text=True, timeout=180.0, check=False
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"occ {' '.join(args)} failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout


def _marker_present() -> bool:
    """Half one of the HEALTHCHECK: Talk provisioned and the marker written."""
    return (
        _runtime("exec", NEXTCLOUD_CONTAINER, "test", "-f", READY_MARKER, check=False).returncode
        == 0
    )


def _status_installed(client: httpx.Client) -> tuple[bool, str]:
    """Half two of the HEALTHCHECK, probed from the host over the mapped port.

    Returns ``(ready, description)``; the description is what the readiness failure
    message reports as the last observed state.
    """
    try:
        resp = client.get(f"{NEXTCLOUD_BASE_URL}/status.php", timeout=5.0)
    except httpx.HTTPError as exc:
        return False, f"transport error: {exc}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]!r}"
    if '"installed":true' not in resp.text.replace(" ", ""):
        return False, f"status.php does not report installed: {resp.text[:200]!r}"
    return True, "installed"


def _wait_until_ready(client: httpx.Client) -> None:
    """Poll BOTH halves of the image's HEALTHCHECK until the fixture is usable.

    Both halves, never one. The marker lives in the container's writable layer and
    survives a restart, so "marker present" alone is a false-ready path; ``status.php``
    alone would pass before Talk was provisioned. The failure message is the only
    diagnostic a broken fixture will produce (see failure mode 3), so it carries the
    container's recent logs, the runtime's own verdict, and the last state of each half.
    """
    deadline = time.monotonic() + NEXTCLOUD_READY_TIMEOUT_SEC
    marker = False
    status = "(not probed yet)"
    while time.monotonic() < deadline:
        marker = _marker_present()
        ready, status = _status_installed(client)
        if marker and ready:
            return
        time.sleep(2.0)
    raise AssertionError(
        f"{NEXTCLOUD_CONTAINER} was not ready within {NEXTCLOUD_READY_TIMEOUT_SEC:.0f}s.\n"
        f"  marker {READY_MARKER}: {'present' if marker else 'ABSENT'}\n"
        f"  status.php at {NEXTCLOUD_BASE_URL}: {status}\n"
        f"  runtime verdict (state/health): {_health_verdict(NEXTCLOUD_CONTAINER)}\n"
        f"{_container_logs(NEXTCLOUD_CONTAINER)}"
    )


# ---------------------------------------------------------------------------
# Talk OCS surface used to PROVISION and OBSERVE (never the code under test)
# ---------------------------------------------------------------------------

_OCS_HEADERS = {"OCS-APIRequest": "true", "Accept": "application/json"}

_TRUSTED_DOMAIN_HINT = (
    "HTTP 400 from an OCS call is how Nextcloud reports a trusted-domain mismatch, and "
    "the container stays HEALTHY while it happens. Address the fixture as one of the "
    "names its image bakes in at install time: localhost, 127.0.0.1, "
    "host.docker.internal, nextcloud, nextcloud-talk-fixture."
)


@dataclass(frozen=True)
class TalkFixture:
    """The provisioned Nextcloud used by the tests, plus the calls they make on it.

    Deliberately separate from :class:`~osprey.bridges.nextcloud_talk.client.TalkClient`:
    this is the *test's* view of the server (create rooms, post as a human, read what
    the room shows), so a bug in the product client cannot make an assertion pass.
    """

    base_url: str
    client: httpx.Client

    # --- raw OCS ----------------------------------------------------------

    def _ocs(
        self,
        method: str,
        path: str,
        *,
        auth: tuple[str, str],
        data: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        allow_not_modified: bool = False,
    ) -> Any:
        url = f"{self.base_url}/ocs/v2.php/{path}"
        resp = self.client.request(
            method, url, headers=_OCS_HEADERS, auth=auth, data=data, params=params
        )
        if allow_not_modified and resp.status_code == httpx.codes.NOT_MODIFIED:
            return []
        if resp.status_code >= httpx.codes.BAD_REQUEST:
            hint = f"\n{_TRUSTED_DOMAIN_HINT}" if resp.status_code == 400 else ""
            raise AssertionError(
                f"{method} {url} returned {resp.status_code}: {resp.text[:400]!r}{hint}"
            )
        return resp.json()["ocs"]["data"]

    # --- provisioning -----------------------------------------------------

    def create_group_room(self, name: str) -> str:
        """Create a group conversation owned by the human and add the bot to it."""
        data = self._ocs(
            "POST",
            ROOM_API,
            auth=(HUMAN_ACCOUNT, HUMAN_PASSWORD),
            data={"roomType": ROOM_TYPE_GROUP, "roomName": name},
        )
        token = str(data["token"])
        self._ocs(
            "POST",
            f"{ROOM_API}/{token}/participants",
            auth=(HUMAN_ACCOUNT, HUMAN_PASSWORD),
            data={"newParticipant": BOT_ACCOUNT, "source": "users"},
        )
        return token

    def create_direct_room(self) -> str:
        """Create the human<->bot one-to-one conversation and make it visible to the bot.

        The priming post is NOT optional. Talk materializes a one-to-one conversation for
        the *invitee* only once the first message is sent: before that, the bot's own
        ``GET /room/{token}`` answers ``404``, the room never resolves, and every message
        posted into it is deferred forever by the poller-side room-type hold. Priming it
        here also gives the room a tail for the bridge's first-ever-start skip-history
        anchor to land on.
        """
        data = self._ocs(
            "POST",
            ROOM_API,
            auth=(HUMAN_ACCOUNT, HUMAN_PASSWORD),
            data={"roomType": ROOM_TYPE_DIRECT, "invite": BOT_ACCOUNT},
        )
        token = str(data["token"])
        self.post(token, "conversation primed by the e2e fixture")
        return token

    # --- acting as the human ----------------------------------------------

    def post(self, room: str, text: str) -> int:
        """Post ``text`` into ``room`` as the human operator; return its message id."""
        data = self._ocs(
            "POST",
            f"{CHAT_API}/{room}",
            auth=(HUMAN_ACCOUNT, HUMAN_PASSWORD),
            data={"message": text},
        )
        return int(data["id"])

    def post_mentioning_bot(self, room: str, text: str) -> int:
        """Post a message that addresses the bot, and return its message id.

        The mention is written as plain ``@account`` text: Talk parses it server-side into
        a ``messageParameters`` entry of type ``user``, which is the exact shape the
        adapter's group-room filter matches on. Nothing is hand-assembled — the server
        produces the wire form.
        """
        return self.post(room, f"@{BOT_ACCOUNT} {text}")

    # --- observing --------------------------------------------------------

    def messages(self, room: str) -> list[dict[str, Any]]:
        """Every message the human can see in ``room``, oldest first.

        Read as the human on purpose: "did a reply appear?" is a question about what the
        person who asked sees.
        """
        data = self._ocs(
            "GET",
            f"{CHAT_API}/{room}",
            auth=(HUMAN_ACCOUNT, HUMAN_PASSWORD),
            params={
                "lookIntoFuture": 1,
                "lastKnownMessageId": 0,
                "includeLastKnown": 0,
                "timeout": 0,
                "limit": 200,
            },
            # An empty room answers 304, which is "nothing new", not a failure.
            allow_not_modified=True,
        )
        return list(data) if isinstance(data, list) else []

    def bot_messages(self, room: str) -> list[str]:
        """The texts the BOT has posted into ``room``, oldest first."""
        return [
            str(message.get("message", ""))
            for message in self.messages(room)
            if str(message.get("actorId", "")) == BOT_ACCOUNT
        ]


# ---------------------------------------------------------------------------
# Nextcloud fixture: start, provision, tear down
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nextcloud() -> Iterator[TalkFixture]:
    """A running, provisioned Nextcloud + Talk with a bot and a human account.

    Module-scoped: one first-boot install serves every test. Task 5.4's agentic tests
    should reuse this fixture rather than starting a second server.
    """
    if not _daemon_available():
        pytest.skip(f"{RUNTIME} daemon is not running")
    if not _image_present(FIXTURE_IMAGE):
        pytest.skip(
            f"fixture image {FIXTURE_IMAGE} is not in the local {RUNTIME} image store. "
            "It is never pulled or built here — publish/load it via "
            ".github/workflows/build-nextcloud-fixture.yml (or build "
            "tests/e2e/fixtures/Dockerfile.nextcloud_talk once by hand, ~1h), or point "
            f"{FIXTURE_IMAGE_ENV} at a local tag."
        )

    # Pre-clean, exact-named. The volume purge is a CORRECTNESS step, not hygiene: a
    # /var/www/html left behind by a plain nextcloud:33.0.7-apache would suppress the
    # entrypoint's rsync and yield a server with no spreed in its served tree.
    _remove_container(NEXTCLOUD_CONTAINER)
    _remove_volume(NEXTCLOUD_VOLUME)

    client = httpx.Client(trust_env=False, timeout=30.0)
    try:
        _runtime(
            "run",
            "-d",
            "--name",
            NEXTCLOUD_CONTAINER,
            # Loopback-only publish: the fixture holds throwaway credentials and has no
            # business being reachable from the network.
            "-p",
            f"127.0.0.1:{NEXTCLOUD_PORT}:80",
            "-v",
            f"{NEXTCLOUD_VOLUME}:/var/www/html",
            # Belt for the "never pull" rule: even if the presence check above were
            # bypassed, this cannot reach a registry.
            "--pull=never",
            FIXTURE_IMAGE,
        )
        _wait_until_ready(client)

        _occ(
            "user:add",
            "--password-from-env",
            "--display-name=OSPREY Bot",
            BOT_ACCOUNT,
            env={"OC_PASS": BOT_PASSWORD},
        )
        _occ(
            "user:add",
            "--password-from-env",
            "--display-name=Test Operator",
            HUMAN_ACCOUNT,
            env={"OC_PASS": HUMAN_PASSWORD},
        )
        yield TalkFixture(base_url=NEXTCLOUD_BASE_URL, client=client)
    finally:
        client.close()
        # Both exact-named, both best-effort. The volume is purged here as well as
        # before the start, so the next run cannot inherit a half-populated tree.
        _remove_container(NEXTCLOUD_CONTAINER)
        _remove_volume(NEXTCLOUD_VOLUME)


AGENTIC_ROOMS = (
    "agentic_mention",
    "agentic_artifact",
    "agentic_input",
    "agentic_restart",
    "agentic_busy",
    "agentic_second",
)
"""Group conversations belonging to the agentic tests below the divider.

Each of the four deterministic rooms is spoken for, and ``direct`` is the only
one-to-one conversation Talk permits between this account pair, so the agentic lane
takes group rooms of its own rather than reusing any of them. A room accumulates the
bot's posts for the life of the module, and the agentic assertions read exactly that —
"what has the bot said in this room" — so sharing one across two tests would let one
test's answer satisfy another's assertion."""


@dataclass(frozen=True)
class Rooms:
    """One room per test, so no test can observe another's traffic.

    ``direct`` is the only one-to-one conversation possible between this human and this
    bot (Talk allows exactly one per pair), which is why the restart and transport proofs
    use group rooms with an explicit mention instead.

    The four fields the deterministic proofs use are spelled out; the agentic lane's are
    generated from :data:`AGENTIC_ROOMS` and reached through :meth:`agentic`.
    """

    group: str
    direct: str
    restart: str
    probe: str
    agentic_mention: str
    agentic_artifact: str
    agentic_input: str
    agentic_restart: str
    agentic_busy: str
    agentic_second: str

    def agentic(self, name: str) -> str:
        """The token of the agentic room called ``name`` (a :data:`AGENTIC_ROOMS` entry)."""
        return getattr(self, name)


@pytest.fixture(scope="module")
def rooms(nextcloud: TalkFixture) -> Rooms:
    """The provisioned conversations, each primed with a tail message."""
    group = nextcloud.create_group_room("osprey-e2e-group")
    nextcloud.post(group, "conversation primed by the e2e fixture")
    restart = nextcloud.create_group_room("osprey-e2e-restart")
    nextcloud.post(restart, "conversation primed by the e2e fixture")
    probe = nextcloud.create_group_room("osprey-e2e-probe")
    nextcloud.post(probe, "conversation primed by the e2e fixture")
    agentic: dict[str, str] = {}
    for name in AGENTIC_ROOMS:
        token = nextcloud.create_group_room(f"osprey-e2e-{name.replace('_', '-')}")
        nextcloud.post(token, "conversation primed by the e2e fixture")
        agentic[name] = token
    return Rooms(
        group=group,
        direct=nextcloud.create_direct_room(),
        restart=restart,
        probe=probe,
        **agentic,
    )


# ---------------------------------------------------------------------------
# Real dispatcher + worker as subprocesses (harness shape shared with
# tests/e2e/test_dispatch_tutorial.py)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Bind to :0, read the assigned port, release it (standard free-port trick)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _drain_output(proc: subprocess.Popen) -> str:
    """Best-effort grab of a subprocess's combined output for failure messages."""
    if proc.stdout is None:
        return "(no captured output)"
    try:
        data = proc.stdout.read1(16384) if hasattr(proc.stdout, "read1") else b""
    except Exception:
        data = b""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    return f"--- subprocess output (partial) ---\n{text}" if text else "(no captured output)"


def _wait_for_health(url: str, timeout: float, proc: subprocess.Popen) -> None:
    """Poll ``url`` until it returns HTTP 200, or fail with the captured output."""
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"subprocess for {url} exited early (rc={proc.returncode}).\n{_drain_output(proc)}"
            )
        try:
            req = urllib.request.Request(url, method="GET")  # noqa: S310 - localhost only
            with urllib.request.urlopen(req, timeout=3.0) as resp:  # noqa: S310
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(0.5)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {url} (last error: {last_err}).\n"
        f"{_drain_output(proc)}"
    )


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Ignored SIGKILL within the grace window; teardown is best-effort, so leave
            # it for the OS to reap rather than hang the test.
            pass


def _find_osprey_console_script() -> Path:
    """Locate the ``osprey`` console script for the ACTIVE interpreter.

    Deliberately interpreter-relative first: a bare ``osprey`` on PATH may belong to a
    different checkout entirely, and in a worktree it usually does.
    """
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError(
        "Could not locate the 'osprey' console script. "
        f"Tried {Path(sys.executable).parent / 'osprey'} and PATH."
    )


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Init + build a real control-assistant deployment repo once per module.

    Two steps because the surface has two: ``init`` writes the repo's source zone
    from the preset, ``build`` renders ``build/`` from it. ``--skip-deps`` keeps it
    fast (no project venv); the worker and dispatcher run with this repo's
    interpreter, so the project venv is not needed. The provider choice never
    reaches an assertion here — see the module docstring on why these tests are
    model-independent.
    """
    base = tmp_path_factory.mktemp("nextcloud_talk_build")
    repo = base / "proj"
    osprey_bin = _find_osprey_console_script()

    def _osprey(argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(osprey_bin), *argv],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env={**os.environ, "CLAUDECODE": ""},
        )

    init = _osprey(
        [
            "init",
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
        ]
    )
    if init.returncode != 0:
        pytest.fail(
            f"osprey init failed (rc={init.returncode}):\n"
            f"--- stdout ---\n{init.stdout}\n--- stderr ---\n{init.stderr}"
        )

    build = _osprey(["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )
    if not (repo / "build" / "config.yml").is_file():
        pytest.fail(f"build succeeded but build/config.yml missing under {repo}")
    return repo


def _write_triggers(dst: Path, worker_port: int) -> None:
    """Write the triggers document the dispatcher runs.

    :data:`TRIGGER_NAME` is the deterministic proofs' trigger, and is deliberately not one
    of the shipped tutorial triggers: those exist to demonstrate agent behaviour, and that
    lane must assert nothing about what a model says. It asks for a fixed word and no
    tools, so a run either completes or fails for want of a provider key — and both
    outcomes carry a ``run_id``, which is the only thing those tests read.

    :func:`_artifact_trigger` appends the agentic lane's second trigger rather than
    changing that one, so the deterministic lane keeps its model-light, tool-free
    definition. See the handoff comment at the foot of this file.
    """
    doc = {
        "dispatcher": {
            "dispatch_target": f"http://127.0.0.1:{worker_port}",
            "max_concurrent_runs": 2,
            "max_queue_depth": 50,
        },
        "triggers": [
            {
                "name": TRIGGER_NAME,
                "source": "webhook",
                "action": {
                    "prompt": (
                        "A Nextcloud Talk end-to-end test fired this event. Reply with "
                        "the single word ACKNOWLEDGED and nothing else. Do not use any "
                        "tools."
                    ),
                    "allowed_tools": [],
                },
            },
            _artifact_trigger(dst.parent),
        ],
    }
    dst.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


@pytest.fixture(scope="module")
def dispatch_stack(built_repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """A real worker + dispatcher pair as subprocesses on free ports.

    Module-scoped: the bridge is what these tests exercise, and the pair carries no state
    any assertion reads (every observation is made through the bridge's own dedup/offset
    stores), so one pair serves the whole module. Task 5.4 should reuse it too.
    """
    worker_port = _free_port()
    dispatcher_port = _free_port()
    triggers_path = tmp_path_factory.mktemp("nextcloud_talk_triggers") / "triggers.yml"
    _write_triggers(triggers_path, worker_port)

    worker_proc: subprocess.Popen | None = None
    dispatcher_proc: subprocess.Popen | None = None
    try:
        worker_env = {
            **os.environ,
            "DISPATCH_WORKER_PORT": str(worker_port),
            "DISPATCH_WORKER_TOKEN": DISPATCH_TOKEN,
            # Repo root + the render's config one level down, exactly as the
            # dispatch_worker compose template wires the deployed worker.
            "OSPREY_PROJECT_DIR": str(built_repo),
            "CONFIG_FILE": str(built_repo / "build" / "config.yml"),
            # The worker's own per-run wall-clock cap. Must match the bridge's
            # DISPATCH_TIMEOUT_SEC or its poll_budget floor is validated against the
            # wrong number; set from one constant so they cannot drift.
            "DISPATCH_TIMEOUT_SEC": str(WORKER_RUN_CAP_SEC),
            "CLAUDECODE": "",
        }
        worker_proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "osprey.mcp_server.dispatch_worker"],
            cwd=str(built_repo),
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _wait_for_health(f"http://127.0.0.1:{worker_port}/health", HEALTH_TIMEOUT_SEC, worker_proc)

        dispatcher_env = {
            **os.environ,
            "TRIGGERS_YML": str(triggers_path),
            "EVENT_DISPATCHER_TOKEN": DISPATCH_TOKEN,
            "DISPATCH_WORKER_TOKEN": DISPATCH_TOKEN,
            "FASTMCP_TRANSPORT": "http",
            "FASTMCP_PORT": str(dispatcher_port),
            "FASTMCP_HOST": "127.0.0.1",
            "MCP_TRANSPORT": "http",
            "MCP_PORT": str(dispatcher_port),
            "CLAUDECODE": "",
        }
        dispatcher_proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "osprey.dispatch"],
            cwd=str(built_repo),
            env=dispatcher_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _wait_for_health(
            f"http://127.0.0.1:{dispatcher_port}/health", HEALTH_TIMEOUT_SEC, dispatcher_proc
        )

        yield {
            "dispatcher_url": f"http://127.0.0.1:{dispatcher_port}",
            "worker_url": f"http://127.0.0.1:{worker_port}",
            "repo": built_repo,
        }
    finally:
        _terminate(dispatcher_proc)
        _terminate(worker_proc)


# ---------------------------------------------------------------------------
# The bridge under test, booted IN-PROCESS through its own entrypoint seams
# ---------------------------------------------------------------------------


class RecordingTransport(httpx.BaseTransport):
    """An ``httpx`` transport that records every request and forwards it for real.

    The instrument for the "no capability or health probe against Nextcloud" proof: it
    observes the adapter's ACTUAL Talk traffic rather than inspecting its source, so a
    probe added anywhere behind ``TalkClient`` — including inside a dependency — shows up.

    Thread-safe: one bridge process shares its Talk client across every room thread and
    the drain thread.
    """

    def __init__(self, inner: httpx.BaseTransport | None = None) -> None:
        self._inner = inner if inner is not None else httpx.HTTPTransport(trust_env=False)
        self._lock = threading.Lock()
        self._seen: list[tuple[str, str]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self._seen.append((request.method, request.url.path))
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()

    @property
    def records(self) -> list[tuple[str, str]]:
        """``(method, path)`` for every request made so far, in order."""
        with self._lock:
            return list(self._seen)


def _talk_client(transport: httpx.BaseTransport) -> httpx.Client:
    """A Talk HTTP client over ``transport``, timed like the one ``TalkClient`` builds.

    The timeouts are copied from the product client's own constants because a long poll
    passes its own per-request override anyway; what matters is that an ordinary call
    behaves the same way here as in the container.
    """
    return httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=REQUEST_TIMEOUT,
            write=REQUEST_TIMEOUT,
            pool=CONNECT_TIMEOUT,
        ),
    )


def _set_bridge_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rooms: tuple[str, ...],
    state_dir: Path,
    dispatch: Mapping[str, Any],
) -> None:
    """Put a complete bridge environment in place, exactly as compose would.

    Set through ``monkeypatch`` and read back by
    :func:`~osprey.bridges.nextcloud_talk.__main__.config_from_env`, so the test boots the
    bridge the way the container does — including its startup validation — rather than
    hand-building a config object that could satisfy the types and skip the checks.

    The three store paths MUST be overridden: they default to ``/data/*.json``, which
    exists only inside the bridge container.
    """
    env = {
        "NEXTCLOUD_BASE_URL": NEXTCLOUD_BASE_URL,
        "NEXTCLOUD_BOT_ACCOUNT": BOT_ACCOUNT,
        "NEXTCLOUD_APP_PASSWORD": BOT_PASSWORD,
        "NEXTCLOUD_ROOMS": ",".join(rooms),
        "DISPATCH_TRIGGER": TRIGGER_NAME,
        "EVENT_DISPATCHER_TOKEN": DISPATCH_TOKEN,
        "DISPATCH_WORKER_TOKEN": DISPATCH_TOKEN,
        # Present-but-EMPTY counts as missing for these two, deliberately: an unset bare
        # ${VAR} renders as "" under compose and no code default would ever replace it.
        "DISPATCHER_URL": dispatch["dispatcher_url"],
        "WORKER_URL": dispatch["worker_url"],
        "OFFSETS_PATH": str(state_dir / "offsets.json"),
        "DEDUP_PATH": str(state_dir / "dedup.json"),
        "HISTORY_PATH": str(state_dir / "history.json"),
        # poll_budget must stay >= worker_timeout or CoreConfig refuses to build.
        "POLL_BUDGET": str(WORKER_RUN_CAP_SEC),
        "DISPATCH_TIMEOUT_SEC": str(WORKER_RUN_CAP_SEC),
        "POLL_INTERVAL": "1",
        "DRAIN_INTERVAL": "5",
        "BRIDGE_TRUST_ENV": "0",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@dataclass
class RunningBridge:
    """A bridge running in a daemon thread, plus whatever killed it."""

    wiring: Wiring
    thread: threading.Thread
    state_dir: Path
    failure: list[BaseException] = field(default_factory=list)

    @property
    def offsets(self) -> dict[str, int]:
        return _read_json(self.state_dir / "offsets.json")

    @property
    def dedup(self) -> dict[str, dict[str, Any]]:
        return _read_json(self.state_dir / "dedup.json")


@contextlib.contextmanager
def _running_bridge(
    state_dir: Path, *, talk: httpx.Client | None = None
) -> Iterator[RunningBridge]:
    """Boot the bridge in-process and stop it on the way out.

    In-process rather than ``python -m osprey.bridges.nextcloud_talk`` on purpose: it
    gives the test the ``Wiring`` (hence the stop event and an injectable Talk client for
    the transport proof) with no signal handling in the way. ``run`` re-validates the
    config, so this cannot boot a half-configured bridge that the subprocess path would
    have refused.
    """
    cfg = config_from_env()
    wiring = build_wiring(cfg, talk=talk)
    failure: list[BaseException] = []

    def _serve() -> None:
        try:
            run(wiring)
        except BaseException as exc:  # noqa: BLE001 - re-raised from the test thread
            failure.append(exc)

    thread = threading.Thread(target=_serve, name="nc-talk-bridge-e2e", daemon=True)
    thread.start()
    bridge = RunningBridge(wiring=wiring, thread=thread, state_dir=state_dir, failure=failure)
    completed = False
    try:
        yield bridge
        completed = True
    finally:
        # One set() ends the room threads, the drain, and any backoff wait in progress.
        wiring.stop.set()
        thread.join(BRIDGE_JOIN_TIMEOUT_SEC)
        stragglers = _join_bridge_threads(cfg.rooms, BRIDGE_JOIN_TIMEOUT_SEC)
        wiring.client.close()
        # Only when the body itself succeeded — otherwise this would mask the real
        # failure with a secondary one.
        if completed and failure:
            raise AssertionError(f"bridge thread crashed: {failure[0]!r}") from failure[0]
        if completed and stragglers:
            raise AssertionError(
                "bridge threads outlived their stop event and are still holding the "
                f"state stores: {stragglers}"
            )


def _bridge_thread_names(rooms: tuple[str, ...]) -> set[str]:
    """Names of every thread one bridge process owns besides the caller's own.

    ``RoomPoller`` names its thread ``talk-poll-<room>`` and the engine names its single
    drain ``bridge-drain``; both are daemons, so neither shows up as a stuck process — it
    shows up as a store being written by a bridge the test thinks has stopped.
    """
    return {f"talk-poll-{room}" for room in rooms} | {"bridge-drain"}


def _join_bridge_threads(rooms: tuple[str, ...], timeout: float) -> list[str]:
    """Wait for a stopped bridge's room and drain threads to actually exit.

    ``serve_rooms`` joins each poller with its own bounded timeout and then returns
    regardless, so ``run`` returning does NOT mean the pollers are done — one sitting in a
    30 s long poll notices the stop only when that poll comes back, and until then it can
    still fetch a batch and claim messages into the shared dedup file.

    Returns:
        The names of any threads still alive after ``timeout``; empty on a clean stop.
    """
    names = _bridge_thread_names(rooms)
    deadline = time.monotonic() + timeout
    while True:
        alive = sorted(t.name for t in threading.enumerate() if t.name in names and t.is_alive())
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(0.5)


def _read_json(path: Path) -> Any:
    """Load one of the bridge's JSON stores; ``{}`` while it does not exist yet.

    Safe to read while the bridge runs: every store write is a tmp-file-plus-rename, so a
    partially written file is never observable.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _await(bridge: RunningBridge, what: str, probe: Callable[[], Any], timeout: float) -> Any:
    """Poll ``probe`` until it returns non-``None``, or fail informatively.

    ``None`` and only ``None`` means "not yet": a first-ever-start anchor is legitimately
    ``0`` for an empty room, and a truthiness test would wait that one out to a timeout.

    Checks the bridge thread on every pass: a crashed or exited bridge is reported as
    such immediately instead of surfacing as an opaque timeout minutes later.
    """
    deadline = time.monotonic() + timeout
    while True:
        if bridge.failure:
            raise AssertionError(f"bridge thread died before {what}: {bridge.failure[0]!r}")
        if not bridge.thread.is_alive():
            raise AssertionError(f"bridge thread exited before {what}")
        value = probe()
        if value is not None:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"timed out after {timeout:.0f}s waiting for {what}\n"
                f"  offsets: {bridge.offsets}\n"
                f"  dedup:   {json.dumps(bridge.dedup, indent=2, default=str)[:2000]}"
            )
        time.sleep(1.0)


def _wait_for_anchor(bridge: RunningBridge, room: str) -> int:
    """Wait until ``room`` has a persisted poll offset, and return it.

    This is the barrier every test needs before posting: a message posted BEFORE the
    bridge resolves the room lands at or below the first-ever-start skip-history anchor
    and is deliberately never dispatched — which would make an "it was ignored" assertion
    pass for entirely the wrong reason. The returned anchor is what the tests assert their
    message id is strictly greater than, closing that hole.
    """
    return _await(
        bridge,
        f"room {room} to persist its first poll offset",
        lambda: bridge.offsets.get(room),
        ANCHOR_TIMEOUT_SEC,
    )


def _wait_for_offset_past(bridge: RunningBridge, room: str, message_id: int) -> None:
    """Wait until ``room``'s persisted offset has moved to or past ``message_id``.

    The evidence that the bridge FETCHED a message and finished with it: the poller
    advances the persisted offset only after ``handle_event`` has returned for every
    message in the batch. That makes it the right barrier for a negative assertion — it
    proves the message was seen, so "nothing happened" cannot be confused with "nothing
    has happened yet".
    """
    _await(
        bridge,
        f"room {room} offset to reach message {message_id}",
        lambda: True if (bridge.offsets.get(room) or 0) >= message_id else None,
        OFFSET_TIMEOUT_SEC,
    )


def _wait_for_dispatch(bridge: RunningBridge, room: str, message_id: int) -> dict[str, Any]:
    """Wait until ``room:message_id`` is claimed AND carries a ``run_id``; return it.

    ``run_id`` is persisted the moment the dispatcher handshake yields one, before the
    agent does any work, so this is the model-independent observation of "a dispatch was
    fired for this message". Nothing here reads the run's outcome.
    """
    key = f"{room}:{message_id}"

    def probe() -> dict[str, Any] | None:
        entry = bridge.dedup.get(key)
        if isinstance(entry, dict) and entry.get("run_id"):
            return entry
        return None

    return _await(bridge, f"dedup entry {key} to carry a run_id", probe, RUN_ID_TIMEOUT_SEC)


@pytest.fixture
def bridge_state(tmp_path: Path) -> Path:
    """Per-test directory for the bridge's offsets/dedup/history stores.

    One test's stores are another's noise, so this is function-scoped — and the restart
    proof relies on the SAME directory surviving across two bridge lifetimes within one
    test, which is exactly what a per-test path gives it.
    """
    state = tmp_path / "bridge-state"
    state.mkdir()
    return state


# ---------------------------------------------------------------------------
# The four deterministic proofs
# ---------------------------------------------------------------------------


def test_group_room_ignores_unmentioned_message(
    nextcloud: TalkFixture,
    rooms: Rooms,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a group room, a message that does not mention the bot is ignored entirely.

    The negative half is the point — no dispatch, no reply — but on its own it would also
    pass against a bridge that was simply dead, so the same room then gets a message that
    DOES mention the bot and that one must dispatch. The pair is what makes this a proof
    that the mention filter discriminates.
    """
    _set_bridge_env(
        monkeypatch, rooms=(rooms.group,), state_dir=bridge_state, dispatch=dispatch_stack
    )
    with _running_bridge(bridge_state) as bridge:
        anchor = _wait_for_anchor(bridge, rooms.group)

        plain_id = nextcloud.post(rooms.group, "no mention here, just chatter between humans")
        assert plain_id > anchor, (
            f"message {plain_id} is at or below the skip-history anchor {anchor}; the "
            "bridge would never fetch it and this test would pass vacuously"
        )
        _wait_for_offset_past(bridge, rooms.group, plain_id)

        # The offset moved past it, so handle_event has returned for it. Anything the
        # bridge was going to do, it has already done.
        assert f"{rooms.group}:{plain_id}" not in bridge.dedup, (
            "an unmentioned group-room message was claimed for dispatch: "
            f"{bridge.dedup.get(f'{rooms.group}:{plain_id}')!r}"
        )
        assert nextcloud.bot_messages(rooms.group) == [], (
            "the bot replied to a group-room message that did not mention it: "
            f"{nextcloud.bot_messages(rooms.group)!r}"
        )

        # Positive control, same room, same bridge: a mention IS dispatched.
        mentioned_id = nextcloud.post_mentioning_bot(rooms.group, "what is the beam current?")
        entry = _wait_for_dispatch(bridge, rooms.group, mentioned_id)
        assert entry["nc_room"] == rooms.group
        assert entry["nc_message_id"] == mentioned_id


def test_one_to_one_message_dispatches_without_a_mention(
    nextcloud: TalkFixture,
    rooms: Rooms,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a one-to-one conversation, a message with no mention still dispatches.

    The room TYPE is the whole difference: the identical text is ignored in a group room
    (the test above) and dispatched here, and the type comes from Talk's own room payload
    rather than from anything the adapter guessed.
    """
    _set_bridge_env(
        monkeypatch, rooms=(rooms.direct,), state_dir=bridge_state, dispatch=dispatch_stack
    )
    with _running_bridge(bridge_state) as bridge:
        anchor = _wait_for_anchor(bridge, rooms.direct)

        message_id = nextcloud.post(rooms.direct, "no mention here, just chatter between humans")
        assert message_id > anchor
        entry = _wait_for_dispatch(bridge, rooms.direct, message_id)

        assert entry["nc_room"] == rooms.direct
        assert entry["nc_actor_id"] == HUMAN_ACCOUNT
        assert entry["history_key"] == f"nextcloud:{rooms.direct}"


def test_message_posted_while_stopped_is_dispatched_after_restart(
    nextcloud: TalkFixture,
    rooms: Rooms,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message that arrives while the bridge is DOWN is claimed once it comes back.

    Two bridge lifetimes over one state directory. The first exists only to establish the
    room's persisted offset — without it the second boot would be a first-ever start and
    would skip the message on purpose. The assertion is that the message is claimed and
    dispatched; what the agent then said is the agentic lane's business.
    """
    _set_bridge_env(
        monkeypatch, rooms=(rooms.restart,), state_dir=bridge_state, dispatch=dispatch_stack
    )
    with _running_bridge(bridge_state) as first:
        anchor = _wait_for_anchor(first, rooms.restart)

    assert not first.thread.is_alive(), "the first bridge did not stop on its stop event"

    message_id = nextcloud.post_mentioning_bot(rooms.restart, "did you survive the restart?")
    assert message_id > anchor
    assert f"{rooms.restart}:{message_id}" not in first.dedup, (
        "the message was claimed while the bridge was supposed to be stopped"
    )

    with _running_bridge(bridge_state) as second:
        entry = _wait_for_dispatch(second, rooms.restart, message_id)

    assert entry["nc_room"] == rooms.restart
    assert entry["nc_message_id"] == message_id
    assert entry["text"] == "did you survive the restart?", (
        "the claim did not persist the question with the bot mention stripped: "
        f"{entry.get('text')!r}"
    )


# Paths the adapter is allowed to request on the Nextcloud instance, derived from the
# product's own route constants so the allowlist cannot drift from the client.
_ALLOWED_TALK_PATHS = (
    f"/ocs/v2.php/{CHAT_API}/",
    f"/ocs/v2.php/{ROOM_API}/",
    f"/ocs/v2.php/{SHARES_API}",
    f"/{DAV_FILES_ROOT}/",
)

_PROBE_PATH = re.compile(r"status\.php|/capabilities|/health|heartbeat|/ping(?:$|/)", re.IGNORECASE)
"""Shapes a capability or liveness probe against a Nextcloud would take.

``status.php`` and ``/ocs/v*/cloud/capabilities`` are the two Nextcloud actually serves;
the rest are the generic spellings a well-meaning "check the server first" change would
reach for."""


def test_adapter_never_probes_nextcloud_for_capabilities_or_health(
    nextcloud: TalkFixture,
    rooms: Rooms,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter makes no capability or health request against the Talk server.

    Observed, not inferred: the bridge's Talk client is built over a
    :class:`RecordingTransport` wrapping a real ``httpx.HTTPTransport``, so this asserts
    the traffic that actually crossed the wire during a full dispatch cycle. A source
    grep would miss a probe introduced inside a dependency; this does not.

    This is about the NEXTCLOUD transport only. The engine's dispatcher/worker ``/health``
    gate is a different client against a different pair and is entirely legitimate — it
    is running during this very test, and nothing here constrains it.
    """
    _set_bridge_env(
        monkeypatch, rooms=(rooms.probe,), state_dir=bridge_state, dispatch=dispatch_stack
    )
    recorder = RecordingTransport()
    with _running_bridge(bridge_state, talk=_talk_client(recorder)) as bridge:
        anchor = _wait_for_anchor(bridge, rooms.probe)
        message_id = nextcloud.post_mentioning_bot(rooms.probe, "record my traffic please")
        assert message_id > anchor
        _wait_for_dispatch(bridge, rooms.probe, message_id)

    records = recorder.records
    paths = [path for _, path in records]

    # Non-vacuous: a recorder that saw nothing would satisfy every assertion below.
    assert any(path.startswith(f"/ocs/v2.php/{ROOM_API}/") for path in paths), (
        f"no room-info request was recorded, so the recorder proves nothing: {records!r}"
    )
    assert any(path.startswith(f"/ocs/v2.php/{CHAT_API}/") for path in paths), (
        f"no chat request was recorded, so the recorder proves nothing: {records!r}"
    )

    probes = [(method, path) for method, path in records if _PROBE_PATH.search(path)]
    assert probes == [], f"the adapter probed Nextcloud for capabilities or health: {probes!r}"

    unexpected = [
        (method, path) for method, path in records if not path.startswith(_ALLOWED_TALK_PATHS)
    ]
    assert unexpected == [], (
        "the adapter requested a Nextcloud path outside the Talk chat/room, share and DAV "
        f"routes it is supposed to use: {unexpected!r}"
    )


# ---------------------------------------------------------------------------
# LLM-gated (agentic) tests — task 5.4 — go BELOW this line
# ---------------------------------------------------------------------------
#
# Everything above is model-independent and must stay that way: it runs with no provider
# key at all, and an assertion that reads model output does not belong in it.
#
# The agentic half reuses this module's fixtures as-is:
#
#   * ``nextcloud`` (module) — the running, provisioned server plus the ``TalkFixture``
#     helpers (``post``, ``post_mentioning_bot``, ``messages``, ``bot_messages``).
#   * ``rooms`` (module) — four provisioned conversations. Each of the four above is
#     already spoken for, and ``direct`` is the ONLY one-to-one conversation Talk permits
#     between this pair, so an agentic test wanting a clean room should add a field to
#     ``Rooms`` and create it in the ``rooms`` fixture rather than reuse one.
#   * ``dispatch_stack`` (module) — the real dispatcher/worker pair. Its trigger
#     (``TRIGGER_NAME``, from ``_write_triggers``) asks for a fixed word and no tools; a
#     test that needs the agent to do something real should add a SECOND trigger to that
#     document and point its own bridge env at it via ``DISPATCH_TRIGGER`` rather than
#     changing this one.
#   * ``bridge_state`` (function) + ``_running_bridge`` / ``_set_bridge_env`` — the boot
#     harness. ``_wait_for_dispatch`` gets you as far as "a run exists"; poll
#     ``nextcloud.bot_messages(room)`` from there for the reply.
#
# Name them so they match ``-k agentic`` (the deterministic gate is run as
# ``-k "not agentic"``), and gate them on the provider key the same way the sibling
# agentic e2e modules do.
#
# What the five proofs below assert is deliberately SHAPE only: that *a* reply exists,
# that *a* ``.png`` share exists, that a list is non-empty. Not one of them reads a word
# the model chose. The fixed strings they do compare against — ``ACK_TEXT``,
# ``ERROR_TEXT``, ``EMPTY_ANSWER_TEXT`` — are the adapter's OWN constants, imported from
# the product rather than re-spelled, and they are used only to tell an answer apart from
# a notice. Each test is ``flaky(reruns=2)`` on ``AssertionError`` alone, so a genuine
# infrastructure fault still fails immediately instead of being retried into a timeout.

ARTIFACT_TRIGGER_NAME = "nextcloud-talk-e2e-artifact"
"""Name of the agentic lane's own trigger — the one that makes a run produce an image.

Separate from :data:`TRIGGER_NAME` on purpose: that trigger asks for a fixed word and no
tools so the deterministic proofs need no provider key at all, and giving it a tool would
cost them that property. A test that wants this one points its bridge at it by overriding
``DISPATCH_TRIGGER`` after :func:`_set_bridge_env`."""

ARTIFACT_SOURCE_FILENAME = "osprey-e2e-artifact-source.png"
"""Basename of the PNG :func:`_artifact_trigger` writes for the agent to register.

The agent is asked to register a file that already exists rather than to *draw*
something: what this lane proves is the bridge's delivery path (worker byte route ->
WebDAV upload -> room share), and a plotting step would put a matplotlib/sandbox
dependency between the test and the thing it is actually asserting."""

ANSWER_TIMEOUT_SEC = 240.0
"""Wait for the bot's answer to appear in a room after a dispatch was observed.

Well past :data:`WORKER_RUN_CAP_SEC` (the worker's own per-run cap) plus the bridge's
poll interval and the post itself, so a run that is merely slow is never mistaken for one
that produced nothing."""

RUN_TERMINAL_TIMEOUT_SEC = 240.0
"""Wait for the worker's run record to reach a terminal status."""

SHARE_TIMEOUT_SEC = 120.0
"""Wait for a shared file to surface as a message in a room.

Covers both directions: the human's share of an inbound image, and the bot's share of a
run artifact (which happens after the answer text has already landed)."""

_NOTICE_TEXTS = frozenset({ACK_TEXT, ERROR_TEXT, EMPTY_ANSWER_TEXT})
"""The adapter's fixed room-facing notices, which are never an answer.

``ACK_TEXT`` is posted before the dispatch, and the other two are what lands when a run
fails or completes silently — so a test that accepted any of them as "the bot replied"
would pass against a bridge that never got an answer at all."""


# ``FILE_PARAM_TYPE`` — the ``messageParameters`` type Talk gives a shared file — is
# imported from the adapter's own wire-format module rather than re-spelled, the same way
# the deterministic lane derives its path allowlist from the product's route constants.
# It is used to read the ROOM's view of a share, never the bridge's.


# ---------------------------------------------------------------------------
# The agentic lane's own helpers
# ---------------------------------------------------------------------------


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    """A minimal, valid 8-bit greyscale PNG.

    Built here rather than committed as a binary fixture or a base64 blob: the delivery
    path guards on PNG magic bytes at two points (the worker byte route's fetch guard and
    this module's own WebDAV check), so the test needs real PNG structure, and generating
    it keeps the bytes readable to someone auditing what the test feeds the pipeline.
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    # One filter byte (0 = none) per scanline, then the row's samples.
    raw = b"".join(b"\x00" + bytes([0x80] * width) for _ in range(height))
    return (
        PNG_MAGIC
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _artifact_trigger(directory: Path) -> dict[str, Any]:
    """The agentic lane's second trigger, plus the PNG its prompt names.

    Called from :func:`_write_triggers` so both triggers land in the one document the
    module-scoped dispatcher is started with. The source PNG is written next to that
    document, in the same module-scoped temp directory, so the worker subprocess (same
    user, same host) can read the path the prompt gives it.

    Args:
        directory: Directory holding the triggers document.

    Returns:
        The trigger definition to append to the document's ``triggers`` list.
    """
    source = directory / ARTIFACT_SOURCE_FILENAME
    source.write_bytes(_png_bytes())
    return {
        "name": ARTIFACT_TRIGGER_NAME,
        "source": "webhook",
        "action": {
            "prompt": (
                "A Nextcloud Talk end-to-end test needs exactly one image artifact "
                "registered. Call the tool mcp__osprey_workspace__artifact_save exactly "
                f"once, with file_path set to {source} and title set to 'Talk bridge "
                "e2e image'. Use no other tool and register no other file. Then reply "
                "with the single word SAVED and nothing else."
            ),
            "allowed_tools": ["mcp__osprey_workspace__artifact_save"],
        },
    }


def _bot_answers(nextcloud: TalkFixture, room: str) -> list[str]:
    """The bot's ANSWER texts in ``room``, oldest first.

    Excludes the fixed notices (see :data:`_NOTICE_TEXTS`) and any message that carries a
    shared file — an artifact share renders as a placeholder token, not as prose, and
    counting it as an answer would let the artifact proof satisfy the reply proof.
    """
    answers: list[str] = []
    for message in nextcloud.messages(room):
        if str(message.get("actorId", "")) != BOT_ACCOUNT:
            continue
        if _shared_files(message):
            continue
        text = str(message.get("message", "")).strip()
        if text and text not in _NOTICE_TEXTS:
            answers.append(text)
    return answers


def _shared_files(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The shared-file references on one chat message; ``[]`` when it carries none."""
    params = message.get("messageParameters")
    if not isinstance(params, Mapping):
        return []
    return [
        dict(param)
        for param in params.values()
        if isinstance(param, Mapping) and param.get("type") == FILE_PARAM_TYPE
    ]


def _wait_for_answer(bridge: RunningBridge, nextcloud: TalkFixture, room: str) -> list[str]:
    """Wait until the bot has posted at least one answer into ``room``; return them all."""
    try:
        return _await(
            bridge,
            f"an answer from the bot in room {room}",
            lambda: _bot_answers(nextcloud, room) or None,
            ANSWER_TIMEOUT_SEC,
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{exc}\n  everything the bot posted in {room}: {nextcloud.bot_messages(room)!r}"
        ) from exc


def _worker_run(dispatch: Mapping[str, Any], run_id: str) -> dict[str, Any] | None:
    """One-shot read of the worker's record for ``run_id``; ``None`` while it has none.

    Read straight from the worker rather than through the bridge: ``input_artifacts`` is
    the WORKER's statement about what it ingested, so taking it from anywhere else would
    be asserting the bridge's opinion of its own payload.
    """
    url = f"{dispatch['worker_url']}/dispatch/{urllib.parse.quote(run_id)}"
    req = urllib.request.Request(  # noqa: S310 - localhost only
        url, method="GET", headers={"Authorization": f"Bearer {DISPATCH_TOKEN}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            if resp.status != 200:
                return None
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def _wait_for_terminal_run(
    bridge: RunningBridge, dispatch: Mapping[str, Any], run_id: str
) -> dict[str, Any]:
    """Wait until the worker's record for ``run_id`` is terminal; return it."""

    def probe() -> dict[str, Any] | None:
        body = _worker_run(dispatch, run_id)
        if body is not None and body.get("status") in TERMINAL_STATUSES:
            return body
        return None

    return _await(
        bridge, f"worker run {run_id} to reach a terminal status", probe, RUN_TERMINAL_TIMEOUT_SEC
    )


def _dav_url(user: str, path: str) -> str:
    """URL of ``path`` inside ``user``'s WebDAV tree, each segment percent-encoded.

    Segment-wise on purpose: the bridge uploads under a collection whose name contains a
    space, so interpolating the whole path into a URL would not survive.
    """
    encoded = "/".join(urllib.parse.quote(segment) for segment in path.strip("/").split("/"))
    return f"{NEXTCLOUD_BASE_URL}/{DAV_FILES_ROOT}/{urllib.parse.quote(user)}/{encoded}"


def _dav(method: str, user: str, password: str, path: str, **kwargs: Any) -> httpx.Response:
    """One WebDAV request as ``user``, on a client of its OWN.

    **Never the ``nextcloud`` fixture's shared client, and this is not tidiness.** That
    client has already authenticated as the admin and the human, so it carries their
    Nextcloud *session* cookies — and Nextcloud resolves a WebDAV request under the
    session identity, ignoring the ``Authorization`` header that names somebody else. A
    bot-authenticated GET of the bot's own file on that client answers ``404`` (not
    ``401``: the session is valid, it simply cannot see another account's tree) and does
    so indefinitely, which reads exactly like "the bridge never uploaded anything". A
    per-call client has no session to inherit, so Basic auth is the only identity in play.

    The product is not exposed to this — :class:`TalkClient` owns a client that only ever
    authenticates as the bot — so it is a hazard of the test harness alone.
    """
    with httpx.Client(trust_env=False, timeout=30.0) as client:
        return client.request(method, _dav_url(user, path), auth=(user, password), **kwargs)


def _dav_put(user: str, password: str, path: str, data: bytes) -> None:
    """Upload ``data`` into ``user``'s WebDAV tree at ``path``."""
    resp = _dav("PUT", user, password, path, content=data)
    assert resp.status_code < httpx.codes.MULTIPLE_CHOICES, (
        f"PUT {path} as {user} returned {resp.status_code}: {resp.text[:300]!r}"
    )


def _dav_get(user: str, password: str, path: str) -> httpx.Response:
    """Fetch ``path`` from ``user``'s WebDAV tree, returning the raw response."""
    return _dav("GET", user, password, path)


def _share_file_to_room(nextcloud: TalkFixture, path: str, room: str) -> None:
    """Share a file from the HUMAN's tree into ``room``, as the human would in the UI.

    Uses the fixture's own OCS caller (never the product client) so the inbound share is
    produced by the server from a genuine user action, exactly like the outbound one the
    bridge makes — and so a 400 still carries the trusted-domain hint.
    """
    nextcloud._ocs(  # noqa: SLF001 - the fixture's own raw OCS caller, same module
        "POST",
        SHARES_API,
        auth=(HUMAN_ACCOUNT, HUMAN_PASSWORD),
        data={"path": path, "shareType": SHARE_TYPE_ROOM, "shareWith": room},
    )


def _reply_mentioning_bot(nextcloud: TalkFixture, room: str, text: str, reply_to: int) -> int:
    """Post a reply to ``reply_to`` that mentions the bot; return the new message id.

    The quote is what carries the shared file into the exchange: Talk ships the quoted
    message inline as ``parent``, so the file the human pointed at rides along with the
    question that mentions the bot.
    """
    data = nextcloud._ocs(  # noqa: SLF001 - the fixture's own raw OCS caller, same module
        "POST",
        f"{CHAT_API}/{room}",
        auth=(HUMAN_ACCOUNT, HUMAN_PASSWORD),
        data={"message": f"@{BOT_ACCOUNT} {text}", "replyTo": reply_to},
    )
    return int(data["id"])


def _wait_for_shared_file(
    bridge: RunningBridge, nextcloud: TalkFixture, room: str, *, actor: str, suffix: str = ""
) -> tuple[int, dict[str, Any]]:
    """Wait for a message from ``actor`` in ``room`` carrying a shared file.

    Args:
        bridge: The running bridge, so a crash is reported instead of a timeout.
        nextcloud: The fixture, read as the human.
        room: Room to watch.
        actor: Account whose share to wait for.
        suffix: Optional filename suffix the share must end with (case-insensitive).

    Returns:
        ``(message_id, file_reference)`` for the first matching share.
    """

    def probe() -> tuple[int, dict[str, Any]] | None:
        for message in nextcloud.messages(room):
            if str(message.get("actorId", "")) != actor:
                continue
            for ref in _shared_files(message):
                name = str(ref.get("name", ""))
                if not suffix or name.lower().endswith(suffix.lower()):
                    return int(message["id"]), ref
        return None

    return _await(
        bridge,
        f"a shared file{f' ending in {suffix}' if suffix else ''} from {actor} in room {room}",
        probe,
        SHARE_TIMEOUT_SEC,
    )


# ---------------------------------------------------------------------------
# The five agentic proofs
# ---------------------------------------------------------------------------


@pytest.mark.requires_als_apg
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_agentic_group_mention_is_answered_in_the_room(
    nextcloud: TalkFixture,
    rooms: Rooms,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mention in a group room comes back as a real answer the room can see.

    The deterministic lane stops at "a run exists"; this is the other half of the same
    round trip — the answer text actually reaches Talk. The assertion is shape only: at
    least one bot message that is neither the ack nor one of the failure notices.
    """
    room = rooms.agentic("agentic_mention")
    _set_bridge_env(monkeypatch, rooms=(room,), state_dir=bridge_state, dispatch=dispatch_stack)
    with _running_bridge(bridge_state) as bridge:
        anchor = _wait_for_anchor(bridge, room)

        message_id = nextcloud.post_mentioning_bot(room, "are you receiving questions?")
        assert message_id > anchor, (
            f"message {message_id} is at or below the skip-history anchor {anchor}; the "
            "bridge would never fetch it"
        )
        _wait_for_dispatch(bridge, room, message_id)
        answers = _wait_for_answer(bridge, nextcloud, room)

    assert answers, f"the bot posted no answer into room {room}"


@pytest.mark.requires_als_apg
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_agentic_run_artifact_is_shared_into_the_room(
    nextcloud: TalkFixture,
    rooms: Rooms,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An artifact a run produces is delivered into the room as a ``.png`` share.

    Drives :data:`ARTIFACT_TRIGGER_NAME` (the second trigger) so the run reliably
    registers one image, then asserts the whole outbound delivery path landed: the room
    shows a file share from the bot whose name ends in ``.png``, and that file really
    exists in the bot's WebDAV tree under the layout ``upload_dir`` defines, with PNG
    magic bytes. Nothing about the image's CONTENT is asserted.
    """
    room = rooms.agentic("agentic_artifact")
    _set_bridge_env(monkeypatch, rooms=(room,), state_dir=bridge_state, dispatch=dispatch_stack)
    # The one env var this test departs from the shared harness on: its run has to use a
    # tool, and the deterministic trigger deliberately has none.
    monkeypatch.setenv("DISPATCH_TRIGGER", ARTIFACT_TRIGGER_NAME)

    with _running_bridge(bridge_state) as bridge:
        anchor = _wait_for_anchor(bridge, room)

        message_id = nextcloud.post_mentioning_bot(room, "please register the test image")
        assert message_id > anchor
        entry = _wait_for_dispatch(bridge, room, message_id)
        run_id = str(entry["run_id"])

        run = _wait_for_terminal_run(bridge, dispatch_stack, run_id)
        assert run.get("status") == "completed", (
            f"the artifact run did not complete: status={run.get('status')!r} "
            f"error={run.get('error')!r}"
        )
        assert run.get("artifacts"), (
            "the run completed without registering any artifact, so there is nothing for "
            f"the bridge to deliver: {run!r}"
        )
        _, shared = _wait_for_shared_file(bridge, nextcloud, room, actor=BOT_ACCOUNT, suffix=".png")

    name = str(shared.get("name", ""))
    assert name.lower().endswith(".png"), f"the bot shared a non-PNG file: {shared!r}"

    # Real bytes in the bot's own tree, at the layout the product defines — not merely a
    # message that names a file. ``upload_dir`` is imported rather than re-spelled, so the
    # test cannot drift from the layout the adapter actually writes.
    stored = f"{upload_dir(room, run_id)}/{name}"
    resp = _dav_get(BOT_ACCOUNT, BOT_PASSWORD, stored)
    assert resp.status_code == httpx.codes.OK, (
        f"the shared artifact is not readable over WebDAV at {stored!r}: "
        f"HTTP {resp.status_code}\n  shared file reference: {shared!r}"
    )
    assert resp.content.startswith(PNG_MAGIC), (
        f"the delivered artifact is not a PNG (first bytes: {resp.content[:8]!r})"
    )

    # And the room member who asked can open it too — an artifact only the bot can read
    # would be delivered in name only.
    seen = _dav_get(HUMAN_ACCOUNT, HUMAN_PASSWORD, str(shared.get("path", "")))
    assert seen.status_code == httpx.codes.OK, (
        f"the human cannot fetch the shared artifact at {shared.get('path')!r}: "
        f"HTTP {seen.status_code}"
    )


@pytest.mark.requires_als_apg
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_agentic_shared_image_reaches_the_run_as_an_input(
    nextcloud: TalkFixture,
    rooms: Rooms,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image shared into the room reaches the run as an ingested input file.

    The inbound counterpart of the artifact proof, and the ``input_files`` round trip end
    to end: the human shares a real PNG into the room and then asks the bot about it by
    quoting that share and mentioning the bot. The assertion reads the WORKER's own record
    — ``input_artifacts`` non-empty — because that is the only place that says the bytes
    were actually ingested rather than merely referenced.

    A group room cannot carry both a file and a mention on one message, and ``direct`` is
    the module's single one-to-one conversation, so the quote is what joins the two.
    """
    room = rooms.agentic("agentic_input")
    # Unique per attempt: a rerun must not re-share a path this room already holds.
    filename = f"osprey-e2e-input-{int(time.time() * 1000)}.png"
    _set_bridge_env(monkeypatch, rooms=(room,), state_dir=bridge_state, dispatch=dispatch_stack)

    with _running_bridge(bridge_state) as bridge:
        anchor = _wait_for_anchor(bridge, room)

        _dav_put(HUMAN_ACCOUNT, HUMAN_PASSWORD, filename, _png_bytes())
        _share_file_to_room(nextcloud, f"/{filename}", room)
        share_id, _ = _wait_for_shared_file(bridge, nextcloud, room, actor=HUMAN_ACCOUNT)
        assert share_id > anchor

        message_id = _reply_mentioning_bot(nextcloud, room, "what does this image show?", share_id)
        entry = _wait_for_dispatch(bridge, room, message_id)
        run = _wait_for_terminal_run(bridge, dispatch_stack, str(entry["run_id"]))

    assert run.get("input_artifacts"), (
        "the shared image never reached the run as an input file; the worker ingested "
        f"nothing for run {entry['run_id']}: {run!r}"
    )


@pytest.mark.requires_als_apg
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_agentic_message_posted_while_stopped_is_answered_after_restart(
    nextcloud: TalkFixture,
    rooms: Rooms,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A question asked while the bridge was DOWN is answered once it comes back.

    The agentic half of the restart criterion. That the message is claimed and dispatched
    after a restart is proven deterministically above; what is added here is that the
    recovery goes all the way to an answer in the room, so a restart that recovers the
    claim but loses the reply cannot pass.
    """
    room = rooms.agentic("agentic_restart")
    _set_bridge_env(monkeypatch, rooms=(room,), state_dir=bridge_state, dispatch=dispatch_stack)

    with _running_bridge(bridge_state) as first:
        anchor = _wait_for_anchor(first, room)

    assert not first.thread.is_alive(), "the first bridge did not stop on its stop event"

    message_id = nextcloud.post_mentioning_bot(room, "did you survive the restart?")
    assert message_id > anchor
    assert _bot_answers(nextcloud, room) == [], (
        f"the bot answered while it was supposed to be stopped: {_bot_answers(nextcloud, room)!r}"
    )

    with _running_bridge(bridge_state) as second:
        _wait_for_dispatch(second, room, message_id)
        answers = _wait_for_answer(second, nextcloud, room)

    assert answers, f"the restarted bridge never answered in room {room}"


@pytest.mark.requires_als_apg
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_agentic_second_room_dispatches_while_the_first_is_in_flight(
    nextcloud: TalkFixture,
    rooms: Rooms,
    dispatch_stack: dict,
    bridge_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One room's run in flight does not hold up another room's.

    Each room gets its own poller thread, so a long agent run in room A must not stall
    room B. "In flight" is read from the room itself rather than from a clock: A has been
    acked but has no answer yet. B is then asked, and its dispatch must be observed while
    that is still true — with a different run id, so the two are genuinely separate runs.
    """
    busy = rooms.agentic("agentic_busy")
    second = rooms.agentic("agentic_second")
    _set_bridge_env(
        monkeypatch, rooms=(busy, second), state_dir=bridge_state, dispatch=dispatch_stack
    )

    with _running_bridge(bridge_state) as bridge:
        busy_anchor = _wait_for_anchor(bridge, busy)
        second_anchor = _wait_for_anchor(bridge, second)

        busy_id = nextcloud.post_mentioning_bot(busy, "take your time with this one")
        assert busy_id > busy_anchor
        busy_entry = _wait_for_dispatch(bridge, busy, busy_id)

        # The claim is taken and the ack has landed, but no answer has — room A's run is
        # in flight right now.
        assert _bot_answers(nextcloud, busy) == [], (
            f"room {busy} was already answered before room {second} was even asked, so "
            "this test would prove nothing about concurrency"
        )

        second_id = nextcloud.post_mentioning_bot(second, "and answer me too")
        assert second_id > second_anchor
        second_entry = _wait_for_dispatch(bridge, second, second_id)

        # Re-read A the moment B's dispatch was observed: still unanswered, so B was
        # dispatched WHILE A was running rather than after it finished.
        assert _bot_answers(nextcloud, busy) == [], (
            f"room {busy} finished before room {second} dispatched; the two runs did not "
            "overlap and the concurrency claim is unproven"
        )
        assert second_entry["run_id"] != busy_entry["run_id"], (
            "both rooms were attributed to the same run: "
            f"{second_entry['run_id']!r} == {busy_entry['run_id']!r}"
        )

        # Both conversations get their own answer in the end.
        assert _wait_for_answer(bridge, nextcloud, second)
        assert _wait_for_answer(bridge, nextcloud, busy)
