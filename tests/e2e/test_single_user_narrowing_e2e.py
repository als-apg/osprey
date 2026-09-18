"""E2E: a SINGLE-USER deployment's chip reaches a dispatch job's first write.

The multi-user half of this property is proved next door, in
``test_dispatch_deploy.py``: there a roster user fires a job through their own
terminal's panel proxy and the job's first control-system write is refused by
that user's narrowing. This module is the other shape, and it exists because
multi-user and single-user parity is a property of the product rather than a
detail of a render — a mechanism that holds only where a roster names the
identity would leave every single-user deployment silently un-narrowed.

**What is different about single-user, and why it needs its own stack.** With
no roster there is no ``web-<user>`` container, no nginx and no per-user
volume. The identity whose chip the lanes read is the account that ran the
build — ``acting_identity()`` on the host, named by nothing in the config — and
the build path is what provisions ``control_target/<build account>/`` for it.
Nothing in the multi-user stack exercises that: its every identity comes from
the roster. So the shape has to be deployed, not simulated, and the module
brings up its own stack with ``modules.web_terminals.enabled: false``.

**The cross-uid read is the mechanism.** The record is written on the HOST, by
the build account, and read inside the container by a process that has dropped
to uid 1000. Those are different uids on every host, so the read succeeds
through the shared group and through nothing else — which is why the record's
mode is asserted here as a first-class row rather than assumed. At 0600 the
record is unreadable across the drop and every owned write fails closed on
``control_context_unavailable``; under a CI umask of 022 a 0600 write would
still read as 0640 and the defect would never appear. ``write_json_atomic``
therefore fchmods the descriptor explicitly, and this module pins the result on
a real deployment.

**How the owner reaches the job.** A single-user deployment has no terminal
container to mint ``X-Osprey-Owner``, so the fire carries it directly to the
dispatcher's bearer-gated MCP transport — the same door, and the same header,
the multi-user proxy hop injects on the user's behalf. That is the documented
trust rule rather than a shortcut: the dispatcher's bearer is the gate, and a
caller holding it names the human the fire belongs to. The MINT itself is not
this module's subject and is pinned where it lives (the gate's middleware rows
and the proxy/gate drift test); what is proved here is everything downstream of
it — that the name travels to the worker, that the worker resolves it against
the tree the build provisioned, and that the record it finds there refuses the
write.

**Host precondition: a Linux host with 1:1 uid mapping.** There the group read
succeeds and an owned write is refused with the store's narrowing verdict. On
Docker Desktop (macOS) bind-mount ownership is remapped, the container's view of
the tree may not carry the host's group at all, and the correct answer becomes
``control_context_unavailable`` — a failure that is closed, not open, and so
still a pass for the safety property. Which one is right is a fact about the
HOST and not about this deployment, so the refusal row reads the entrypoint's
own access-probe line out of the worker's log and asserts whichever the probe
reports, exactly as the multi-user module does. Neither arm is told from the
other by the word "narrowing": the ``control_context_unavailable`` message
contains that word too ("This is not a narrowing anybody set"), so both arms are
matched on the clause each one CLOSES on.

**Cost.** One full deploy, paid to prove the single-user shape rather than to
re-read the multi-user one. The stack publishes this suite's usual virtual
accelerator ports, so it cannot coexist with another deploy e2e on one host and
must run sequentially (the fixture tears its own stack down in a ``finally``).
Every resource it names is exact-named and obviously throwaway.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import (
    CONTROL_TREE_MARKER_NAME,
    control_target_identity_dir,
    resolve_project_name,
)
from osprey.dispatch import DISPATCHER_MCP_PATH
from osprey.port_layout import PORT_BASE_CONFIG_KEY, default_port
from osprey.utils.identity import acting_identity
from osprey_connectors.control_context import RECORD_FILENAME, ControlContext, write_record
from osprey_connectors.posture_store import POSTURE_SANDBOX
from osprey_connectors.types import CONTROL_TARGETS, TARGET_VA
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.profile_edits import set_pairs

#: This deploy's own thousand-port block. Chosen clear of every other deploy e2e
#: in this suite (dispatch 20700, overlay 21100, sandbox-escape 21400, web-deploy
#: 21700, bump 21900, queue 22000, preflight 22100, full-chain 23000, jupyter
#: 24000, SDK helpers 25000) so a reader checking for a collision reads one list.
PORT_BASE = 20800

#: Outside the block by design, in the same 1506x series the other deploy e2es
#: claim one from each (queue 15064, ORM 15065, substrate 15066, archiver 15067,
#: dispatch 15068): the Channel Access port cannot be derived from PORT_BASE.
VA_CA_PORT = 15069

#: The deployment repo's directory name IS the deployment's name, and compose
#: renders every container as ``<project>-<service>``, so the container targets
#: below are derived from it rather than spelled host-globally.
PROJECT_NAME = "suproj"
DISPATCHER_CONTAINER = f"{PROJECT_NAME}-event-dispatcher"
WORKER_CONTAINER = f"{PROJECT_NAME}-dispatch-worker-1"
WORKER_AGENT_DATA = f"/app/{PROJECT_NAME}/var/agent_data"

DISPATCHER_URL = f"http://localhost:{default_port('dispatcher', base=PORT_BASE)}"
DISPATCHER_MCP_URL = f"{DISPATCHER_URL}{DISPATCHER_MCP_PATH}"
TOKEN = "dev-token"  # matches the .env tokens the fixture writes

DEPLOY_UP_TIMEOUT_SEC = 1800
HEALTH_TIMEOUT_SEC = 180.0
RUN_TIMEOUT_SEC = 300.0

#: The probe trigger, declared twice so the owned fire and the owner-less
#: control are two SEPARATE runs in the dispatcher feed rather than two readings
#: of one latest-wins entry.
WRITE_PROBE_OWNED_TRIGGER = "single-user-write-owned"
WRITE_PROBE_UNOWNED_TRIGGER = "single-user-write-unowned"

#: A WRITABLE setpoint the preset's virtual accelerator publishes, and the value
#: written to it, comfortably inside the limits its database gives it
#: (``SR:MAG:HCM:01:CURRENT:SP``: min -12.0, max 12.0, writable by the
#: ``defaults`` block, in the deployment's own ``data/channel_limits.json``).
#:
#: Which channel is chosen decides whether these rows can see anything at all,
#: and an earlier draft got it wrong: the limits database is enforced by a
#: PreToolUse HOOK that runs BEFORE the tool, so a channel it marks read-only —
#: any ``:RB`` readback — is refused there with "CHANNEL LIMITS VIOLATION" and
#: the call never reaches the connector where the store verdict is read. The
#: refusal rows would then pass or fail on a fact about the address rather than
#: on the narrowing under test. A writable setpoint inside its limits clears
#: that gate, so the only refusal a narrowed owner can meet is the store's.
PROBE_CHANNEL = "SR:MAG:HCM:01:CURRENT:SP"
PROBE_VALUE = "1.0"

#: The limits hook's refusal headline. Named so the rows below can say "this
#: probe never reached the mechanism" instead of failing with a wall of text
#: about an address — the exact trap that cost a run.
LIMITS_REFUSAL = "CHANNEL LIMITS VIOLATION"

#: The closing clause of each store arm of the connector's refusal. The
#: unavailable arm's prose CONTAINS the word "narrowing" ("This is not a
#: narrowing anybody set"), so the arms are told apart by the clause each one
#: CLOSES on and never by the bare word. Spelled as in ``test_dispatch_deploy``,
#: because a drift between the two modules' spellings would let one of them pass
#: on a message the other no longer recognises.
NARROWING_CLAUSE = "The store answered narrowing."
UNAVAILABLE_CLAUSE = "the store answered control_context_unavailable."

_EVIDENCE_CAP = 6000

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    pytest.mark.flaky(reruns=1),
]


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


def _docker_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return ((proc.stdout or "") + (proc.stderr or "")).strip() or "(no output)"


_WRITE_PROBE_ACTION = f"""    source: webhook
    action:
      prompt: >-
        Call the control-system channel-write tool exactly once, writing the
        value {PROBE_VALUE} to the channel {PROBE_CHANNEL}, and then report the
        tool's response verbatim. Do not retry and do not call any other tool.
      allowed_tools:
        - mcp__controls__channel_write
"""

_PROBE_TRIGGERS = (
    "\n  # -- e2e single-user narrowing probes "
    "(appended by tests/e2e/test_single_user_narrowing_e2e.py) --\n"
    "  #\n"
    "  # One action under two names: the owned fire and the owner-less control\n"
    "  # must produce two separate runs in the dispatcher feed.\n"
    f"  - name: {WRITE_PROBE_OWNED_TRIGGER}\n{_WRITE_PROBE_ACTION}"
    f"  - name: {WRITE_PROBE_UNOWNED_TRIGGER}\n{_WRITE_PROBE_ACTION}"
)


def _repo_triggers_file(repo: Path) -> Path:
    """The deployment's own ``triggers.yml`` — the SOURCE copy, never the render.

    ``osprey init`` writes the preset's trigger set into the repo's source zone
    and ``osprey build`` renders it; appending to the render would be overwritten
    by the next build and never reach the dispatcher. Located by name rather than
    by a spelled path, and asserted to be exactly one file so a second copy
    cannot be edited silently.
    """
    found = [path for path in repo.rglob("triggers.yml") if "build" not in path.parts]
    assert len(found) == 1, f"expected exactly one source triggers.yml under {repo}, found: {found}"
    return found[0]


def _append_probe_triggers(repo: Path) -> None:
    triggers_file = _repo_triggers_file(repo)
    text = triggers_file.read_text(encoding="utf-8")
    assert "hello-dispatch" in text, (
        f"{triggers_file} does not look like the preset's trigger set:\n{text[:500]}"
    )
    with triggers_file.open("a", encoding="utf-8") as handle:
        handle.write(_PROBE_TRIGGERS)


def _build_account() -> str:
    """The identity a single-user deployment's lanes read the chip of.

    The account that ran the build, resolved through the one ladder every
    surface shares. Named by nothing in the config — which is the whole point of
    this module — so it is read here the same way the build path reads it when
    it provisions the directory.
    """
    return acting_identity()


def _host_record(repo: Path) -> Path:
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    return control_target_identity_dir(config, repo, _build_account()) / RECORD_FILENAME


def _narrow_the_build_account(repo: Path) -> Path:
    """Narrow the build account to read-only on every target, on the HOST tree.

    The chip's effect without a browser: in a single-user deployment the chip is
    flipped by this same account, through the same writer, into the same
    directory the build provisioned for it. Every target is narrowed rather than
    the one this deployment happens to point at — the assertion is about whose
    narrowing a job obeys, and pinning a target name would make the row fail for
    a deployment whose connector resolved a different one.

    The tree marker is ASSERTED rather than created. A reader that finds no
    marker answers ``control_context_unavailable`` for every owner, which would
    make the refusal row below pass for entirely the wrong reason; and the build
    is what provisions the tree, so a missing marker is a defect in the thing
    under test rather than a state to paper over.
    """
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    identity_dir = control_target_identity_dir(config, repo, _build_account())
    marker = identity_dir.parent / CONTROL_TREE_MARKER_NAME
    assert marker.is_file(), (
        f"osprey build did not provision the control-context tree at "
        f"{identity_dir.parent} (no {CONTROL_TREE_MARKER_NAME}); in a single-user "
        f"deployment the build is the ONLY thing that provisions "
        f"{_build_account()!r}'s own directory, and a narrowing seeded into an "
        "unprovisioned tree reads as unavailable for every owner"
    )
    assert identity_dir.is_dir(), (
        f"osprey build provisioned the control-context tree but not the build "
        f"account's own directory at {identity_dir} — in single-user that is the "
        "directory whose chip every lane reads, and no roster names it"
    )
    record = ControlContext(
        target=TARGET_VA,
        generation=1,
        owner=None,
        posture=dict.fromkeys(CONTROL_TARGETS, POSTURE_SANDBOX),
    )
    return write_record(record, path=identity_dir / RECORD_FILENAME)


@pytest.fixture(scope="module")
def single_user_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Init + build + ``osprey up`` a SINGLE-USER control-assistant stack."""
    if not os.environ.get("ALS_APG_API_KEY"):
        pytest.skip("ALS_APG_API_KEY not set")

    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("single_user_narrowing_build")
    repo = base / PROJECT_NAME

    # The single-user shape, stated as one leaf: no roster, so no nginx, no
    # per-user container, and no identity but the build account's own. Dotted
    # LEAF keys on purpose — each edit states only its own leaf and leaves its
    # subtree's siblings intact.
    edits = {
        "config": {
            "modules.web_terminals.enabled": False,
            PORT_BASE_CONFIG_KEY: PORT_BASE,
        }
    }

    init = _run(
        [
            str(osprey_bin),
            "init",
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            *set_pairs(edits),
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
            "--set",
            f"virtual_accelerator.port={VA_CA_PORT}",
        ],
        cwd=base,
        timeout=300,
    )
    if init.returncode != 0:
        pytest.fail(
            f"osprey init failed (rc={init.returncode}):\n"
            f"--- stdout ---\n{init.stdout}\n--- stderr ---\n{init.stderr}"
        )

    # Before the build, so the render the dispatcher loads carries them.
    _append_probe_triggers(repo)

    build = _run(
        [str(osprey_bin), "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        cwd=base,
        timeout=300,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    # APPENDED, never rewritten: `osprey init` seeded this file and `osprey
    # build` appended the keys the virtual-accelerator containers boot from.
    # Rewriting it would drop those and the VA would refuse an unnamed manifest.
    base_url_line = (
        f"ALS_APG_BASE_URL={os.environ['ALS_APG_BASE_URL']}\n"
        if os.environ.get("ALS_APG_BASE_URL")
        else ""
    )
    with (repo / ".env").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n# ── e2e fixture ──\n"
            "EVENT_DISPATCHER_TOKEN=dev-token\n"
            "DISPATCH_WORKER_TOKEN=dev-token\n"
            f"ALS_APG_API_KEY={os.environ['ALS_APG_API_KEY']}\n" + base_url_line
        )

    # Narrow before anything is deployed: the dispatch worker binds the tree
    # read-only at start, so the record has to exist as a file the container's
    # group can read by the time an owned job attempts its first write.
    _narrow_the_build_account(repo)

    # Force a fresh image build so the deployed services run CURRENT source.
    # `osprey up` does not pass --build to compose, so it would otherwise reuse
    # existing images and silently test stale code.
    project = resolve_project_name({"project_name": PROJECT_NAME})
    for image in (f"{project}-dispatch:local", f"{project}:local"):
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)

    try:
        up = _run([str(osprey_bin), "up", "-d", "--dev"], cwd=repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        if up.returncode != 0:
            pytest.fail(
                f"osprey up failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{DISPATCHER_URL}/health", HEALTH_TIMEOUT_SEC)
        # A healthy dispatcher does not mean a healthy worker: the worker runs
        # the heavier project image and boots later. Gate on its proxied feed so
        # no row races a still-warming worker, and a worker that never comes up
        # fails here with container logs rather than as a bare 502 mid-test.
        _wait_for_worker_feed(HEALTH_TIMEOUT_SEC)
        yield repo
    finally:
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        # Volumes are deliberately kept by `down`; remove this project's own via
        # the shared label-scoped sweep. Exact-named only, never a prune.
        remove_project_volumes(resolve_project_name({"project_name": PROJECT_NAME}))


def _wait_for_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:  # noqa: S310 - localhost
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {url} (last: {last_err})")


def _dump_stack_diagnostics(context: str) -> str:
    """Container state + recent service logs, for failure output.

    A full-Docker e2e that fails mid-run is nearly undiagnosable from the pytest
    traceback alone, because the interesting state is inside the containers.
    """
    lines = [f"=== stack diagnostics ({context}) ==="]
    ps = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={PROJECT_NAME}-",
            "--format",
            "{{.Names}}\t{{.Status}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines.append("containers:\n" + (ps.stdout.strip() or ps.stderr.strip() or "(none)"))
    for name in (WORKER_CONTAINER, DISPATCHER_CONTAINER):
        lines.append(
            f"--- {name} (last 40 log lines) ---\n"
            + _docker_capture(["docker", "logs", "--tail", "40", name])
        )
        # A tail alone cannot reach a fire that failed five minutes earlier: both
        # services log a poll line every couple of seconds, so forty lines is
        # under a minute of history and the interesting line has scrolled past.
        # The whole log, with the routine access lines dropped, keeps the
        # warnings (a rejected dispatch, a dropped one, a traceback) at any age.
        whole = _docker_capture(["docker", "logs", name])
        noteworthy = [
            line
            for line in whole.splitlines()
            if "GET /dashboard/runs" not in line and "GET /health" not in line
        ]
        lines.append(
            f"--- {name} (whole log minus routine polling) ---\n"
            + ("\n".join(noteworthy)[-_EVIDENCE_CAP:] or "(nothing but polling)")
        )
    return "\n".join(lines)


def _wait_for_worker_feed(timeout: float) -> None:
    """Wait until the dispatcher can proxy the worker run feed (HTTP 200)."""
    deadline = time.monotonic() + timeout
    last = "(no response yet)"
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{DISPATCHER_URL}/dashboard/runs",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
                if resp.status == 200:
                    return
                last = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(2.0)
    raise AssertionError(
        f"worker feed at {DISPATCHER_URL}/dashboard/runs not ready after "
        f"{timeout:.0f}s (last: {last})\n{_dump_stack_diagnostics('worker feed wait timeout')}"
    )


def _sse_payloads(body: str) -> list[dict]:
    """Every JSON object an SSE body's ``data:`` lines carry.

    The dispatcher's MCP transport answers in ``text/event-stream`` frames even
    for a single response, so a caller reading the result unwraps the frame
    rather than json-loading the body. A line that is not JSON is skipped rather
    than raised on: the caller asserts on what it found, and a parse error here
    would replace that assertion with a traceback about framing.
    """
    payloads: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            payloads.append(json.loads(line[len("data:") :].strip()))
        except json.JSONDecodeError:
            continue
    return payloads


def _mcp_post(body: dict, *, owner: str | None, session: str | None = None) -> tuple:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {TOKEN}",
    }
    if owner is not None:
        headers["X-Osprey-Owner"] = owner
    if session:
        headers["Mcp-Session-Id"] = session
    req = urllib.request.Request(  # noqa: S310 - localhost only
        DISPATCHER_MCP_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, raw
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            {k.lower(): v for k, v in exc.headers.items()},
            exc.read().decode("utf-8", "replace"),
        )


def _manual_fire(trigger: str, *, owner: str | None) -> dict:
    """Call ``manual_fire`` on the dispatcher's own MCP transport.

    Streamable HTTP takes three requests — initialize, the initialized
    notification, then the call — and answers in SSE frames, so the raw bodies
    come back for the caller to unwrap.

    ``owner`` is the header a terminal's proxy would inject on a user's behalf.
    ``None`` sends none at all, which is the cron-shaped door: the control this
    module's refusal row needs, because a wire that stamped every run with one
    name would satisfy the refusal exactly as the real one does.
    """
    out: dict = {"trigger": trigger, "owner": owner}
    status, headers, body = _mcp_post(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "single-user-narrowing-e2e", "version": "1"},
            },
        },
        owner=owner,
    )
    out["initialize"] = {"status": status, "body": body[:2000]}
    session = headers.get("mcp-session-id")
    if status != 200 or not session:
        return out
    _mcp_post(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}, owner=owner, session=session
    )
    status, _headers, body = _mcp_post(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "manual_fire", "arguments": {"name": trigger, "payload": {}}},
        },
        owner=owner,
        session=session,
    )
    out["call"] = {"status": status, "body": body[:4000]}
    return out


def _tool_result(payloads: list[dict]) -> dict:
    """The tool's own answer, parsed out of the JSON-RPC response.

    Three layers have to come off, and each one is a place an earlier draft of
    this helper got it wrong. The SSE frame carries a JSON-RPC envelope; the
    envelope's ``result`` carries MCP content parts; and the text part is
    ITSELF a JSON document, serialised as a string. Searching the envelope for a
    substring cannot work: re-serialising it escapes the inner document's quotes
    (``\\"dispatched\\": true``), so the match fails on a call that succeeded.

    Read from ``structuredContent.result`` when FastMCP wrapped it there and
    from the first text content part otherwise, because which one appears is a
    property of the server's wrapping rather than of the answer.
    """
    envelope = next((item for item in payloads if "result" in item), None)
    assert envelope is not None, f"no JSON-RPC result among the SSE payloads: {payloads}"
    result = envelope["result"]
    assert not result.get("isError"), f"the dispatcher reported a tool error: {result}"

    raw = (result.get("structuredContent") or {}).get("result")
    if raw is None:
        parts = result.get("content") or []
        text_parts = [part.get("text") for part in parts if part.get("type") == "text"]
        assert text_parts, f"the tool answered with no text content: {result}"
        raw = text_parts[0]
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def _fire_and_assert_dispatched(trigger: str, *, owner: str | None) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SEC
    fired = _manual_fire(trigger, owner=owner)
    while not fired.get("initialize", {}).get("status") and time.monotonic() < deadline:
        # Status 0 is a transport failure, which at this point means the
        # dispatcher's MCP transport is still coming up behind a healthy /health.
        time.sleep(3.0)
        fired = _manual_fire(trigger, owner=owner)
    assert fired.get("initialize", {}).get("status") == 200, (
        f"the MCP handshake with the dispatcher failed: {fired}"
    )
    call = fired.get("call") or {}
    assert call.get("status") == 200, f"manual_fire for {trigger} failed: {fired}"
    payloads = _sse_payloads(call.get("body") or "")
    assert payloads, f"manual_fire returned no SSE payload: {call.get('body')!r}"
    answer = _tool_result(payloads)
    assert answer.get("dispatched") is True, f"manual_fire did not dispatch {trigger}: {answer}"
    # The trigger name is asserted too, so a fire that dispatched the OTHER probe
    # cannot satisfy this: the owned and owner-less rows read one run each out of
    # a shared feed, and a mixed-up name would give both rows the same run.
    assert answer.get("trigger") == trigger, (
        f"manual_fire dispatched {answer.get('trigger')!r} rather than {trigger!r}: {answer}"
    )


def _bearer_get(path: str, timeout: float) -> object | None:
    """GET a bearer-gated dispatcher endpoint, or None when it is not answering.

    Every read endpoint this module uses is gated by the same
    ``EVENT_DISPATCHER_TOKEN``, and a momentarily unreachable worker upstream
    surfaces as a 502/503/504 rather than as an error worth aborting a poll loop
    over. Returning None for "nothing to read yet" keeps that decision with the
    caller.
    """
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{DISPATCHER_URL}{path}",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
        # urllib.error.HTTPError is a URLError, so a 502 from a mid-restart
        # worker upstream lands here too and reads as "no feed yet".
        return None


def _runs_by_trigger() -> dict:
    """Snapshot the dispatcher's run feed keyed by trigger name, latest wins.

    The key is ``trigger_name`` and it comes from the DISPATCHER, not from the
    worker: the worker's own feed names a run by ``run_id`` alone and carries no
    trigger at all (``dispatch_api.dashboard_runs``), and the dispatcher adds
    ``trigger_name``/``dispatch_id`` by reversing its pool results onto the
    proxied rows (``server.dashboard_runs``). Keying on anything else silently
    matches nothing, which reads exactly like a job that never ran.

    Latest wins by ``created_at``, explicitly. The feed arrives newest-first, so
    a plain overwrite would keep the OLDEST run per trigger — and on a rerun,
    where the module fixture is not torn down, the poll loop would re-read the
    previous attempt's run instead of the one this attempt just fired.
    """
    feed = _bearer_get("/dashboard/runs", 10.0)
    runs = feed if isinstance(feed, list) else None
    by_trigger: dict = {}
    for run in runs or []:
        name = run.get("trigger_name")
        if not name:
            continue
        held = by_trigger.get(name)
        if held is None or (run.get("created_at") or 0) > (held.get("created_at") or 0):
            by_trigger[name] = run
    return by_trigger


def _fire_timeline(trigger: str) -> str:
    """What the dispatcher itself recorded for every fire of *trigger*.

    A run reaches the feed only once the worker accepted the job and answered
    with a run_id. When no run appears, the question is whether the fire was
    dispatched, refused by the worker, or dropped after an error — and that
    answer lives in the dispatcher's own trigger history, which
    ``/dashboard/state`` compacts to one status per fire (``dispatched``,
    ``rejected: <code>``, ``error: ...``, ``queue_full``). Read on failure only,
    so a timeout says which of those happened rather than leaving a reader to
    guess from container logs.
    """
    state = _bearer_get("/dashboard/state", 15.0)
    if not isinstance(state, dict):
        return "(the dispatcher did not answer /dashboard/state)"
    timeline = (state.get("timeline") or {}).get(trigger)
    pool = state.get("pool")
    worker_error = state.get("worker_error")
    lines = [f"dispatcher timeline for {trigger}: {json.dumps(timeline)}"]
    lines.append(f"dispatcher pool: {json.dumps(pool)}")
    if worker_error:
        lines.append(f"dispatcher could not reach the worker: {worker_error}")
    return "\n".join(lines)


def _wait_for_terminal_run(trigger: str, timeout: float = RUN_TIMEOUT_SEC) -> dict:
    """Poll the dispatcher feed until *trigger* has a terminal run; return it."""
    deadline = time.monotonic() + timeout
    run: dict = {}
    while time.monotonic() < deadline:
        run = _runs_by_trigger().get(trigger, {})
        if run.get("status") in ("completed", "error"):
            return run
        time.sleep(3.0)
    feed = _bearer_get("/dashboard/runs", 10.0)
    raise AssertionError(
        f"{trigger}: no terminal run within {timeout:.0f}s "
        f"(last: {run or 'no run in the feed'})\n"
        f"{_fire_timeline(trigger)}\n"
        f"whole feed: {json.dumps(feed)[:_EVIDENCE_CAP]}\n"
        f"{_dump_stack_diagnostics(f'{trigger} did not converge')}"
    )


def _persisted_run_record(run_id: str) -> dict:
    """The worker's own record of a run — the one artifact carrying tool results.

    The dispatcher feed reports status and a tool COUNT; what a tool ANSWERED
    lives only in ``<agent_data>/dispatch/<run_id>.json``. These rows assert on
    the tool's answer, so they read it there rather than trusting the agent's
    prose summary of it.
    """
    record = subprocess.run(
        ["docker", "exec", WORKER_CONTAINER, "cat", f"{WORKER_AGENT_DATA}/dispatch/{run_id}.json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert record.returncode == 0, (
        f"could not read the persisted record for run {run_id}: "
        f"{(record.stderr or record.stdout).strip()}"
    )
    return json.loads(record.stdout)


def _tool_answers(record: dict, tool_suffix: str) -> list[str]:
    """Every result text this run's calls to ``*<tool_suffix>`` came back with."""
    return [
        str(call.get("result") or "")
        for call in (record.get("tool_calls") or [])
        if str(call.get("name") or "").endswith(tool_suffix)
    ]


def _run_evidence(trigger: str, run: dict, record: dict | None = None) -> str:
    """Everything a failing row needs, gathered before the stack goes away."""
    feed = {key: run.get(key) for key in ("run_id", "status", "tool_count", "error")}
    lines = [f"=== {trigger} run evidence ===", f"feed: {feed}"]
    text = (run.get("text_output") or "").strip()
    if text:
        lines.append(f"agent text_output:\n{text[:_EVIDENCE_CAP]}")
    if record is not None:
        lines.append(
            f"persisted tool_calls:\n{json.dumps(record.get('tool_calls'), indent=2)[:_EVIDENCE_CAP]}"
        )
    lines.append(
        f"--- {WORKER_CONTAINER} (last 200 log lines) ---\n"
        + _docker_capture(["docker", "logs", "--tail", "200", WORKER_CONTAINER])
    )
    return "\n".join(lines)


def _control_tree_is_readable_in_the_worker() -> bool:
    """Whether the worker's entrypoint could read the control-context tree.

    The tree reaches the container through a shared group, and on Docker Desktop
    the bind's host ownership is remapped — so whether an owned write is answered
    with its owner's narrowing or with ``control_context_unavailable`` is decided
    by the access probe, not by the deployment. The entrypoint runs that probe at
    start and its log says which it got, so the row asserts whichever it reports
    rather than pinning the Linux answer on every host.
    """
    logs = _docker_capture(["docker", "logs", WORKER_CONTAINER])
    return "cannot read/traverse OSPREY_CONTROL_CONTEXT_TREE" not in logs


# ---------------------------------------------------------------------------
# The rows.
# ---------------------------------------------------------------------------


def test_the_build_accounts_record_is_group_readable_on_a_deployed_stack(
    single_user_stack: Path,
) -> None:
    """S1: the single-user record is 0640 — readable across the privilege drop.

    The record is written on the host by the build account and read inside the
    container by a process that has dropped to uid 1000. Those are different
    uids on every host, so the read succeeds through the shared group and
    through nothing else.

    The mode is asserted rather than assumed because a suite's own umask hides
    the defect: ``write_json_atomic`` fchmods the descriptor explicitly, and if
    it did not, a CI runner at umask 022 would still produce 0640 while a
    developer host at 077 produced 0600 — where every owned lane write fails
    closed on ``control_context_unavailable`` and narrowing silently stops
    holding. Group-readability is named as its own assertion, so a failure says
    which property broke rather than printing two octal numbers.

    Ownership is deliberately not asserted. Under a runtime that remaps bind
    ownership the uid on the host is a fact about the id map; the mode bits
    survive it, and they are what the group read depends on.
    """
    record_path = _host_record(single_user_stack)
    assert record_path.is_file(), (
        f"no control-context record at {record_path} for the build account "
        f"{_build_account()!r} after the fixture narrowed it"
    )
    mode = stat.S_IMODE(record_path.stat().st_mode)
    assert mode & stat.S_IRGRP, (
        f"{record_path} is {oct(mode)} and the group cannot read it — the dispatch "
        "worker drops to uid 1000 and reaches this record through the shared group "
        "alone, so every owned write would fail closed on control_context_unavailable"
    )
    assert mode == 0o640, (
        f"{record_path} is {oct(mode)}, not 0640. The mode is set on the descriptor "
        "by write_json_atomic rather than left to the umask, precisely so a host at "
        "umask 077 does not quietly produce a 0600 record that no lane can read"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert all(record["posture"][target] == POSTURE_SANDBOX for target in CONTROL_TARGETS), (
        f"the fixture's narrowing did not reach every target: {record}"
    )


def test_an_owned_jobs_first_write_is_refused_by_the_build_accounts_narrowing(
    single_user_stack: Path,
) -> None:
    """S2: in a single-user deployment the chip gates a dispatch job's writes.

    The property this module exists for. The build account is narrowed to
    read-only on the host tree, a job is fired carrying that account as its
    owner, and its first control-system write is refused by the reference
    monitor before the control system is asked — with no roster anywhere in the
    deployment, and with the identity resolved through the same ladder the build
    provisioned the directory under.

    Which refusal is correct is a fact about the HOST rather than about this
    deployment: on Docker Desktop the bind's ownership is remapped and the tree
    is unreadable, which fails closed on ``control_context_unavailable`` instead.
    The entrypoint's own access probe says which, and this row asserts that one —
    by the clause each arm CLOSES on, never by the word "narrowing", which the
    unavailable arm's prose also carries.
    """
    _fire_and_assert_dispatched(WRITE_PROBE_OWNED_TRIGGER, owner=_build_account())

    run = _wait_for_terminal_run(WRITE_PROBE_OWNED_TRIGGER)
    record = _persisted_run_record(run["run_id"])
    answers = _tool_answers(record, "__channel_write")
    evidence = _run_evidence(WRITE_PROBE_OWNED_TRIGGER, run, record)
    assert answers, (
        "the owned write probe never called the channel-write tool, so no write was "
        f"ever judged.\n{evidence}"
    )
    joined_answers = "\n".join(answers)
    assert LIMITS_REFUSAL not in joined_answers, (
        f"the write to {PROBE_CHANNEL} was refused by the LIMITS hook, which runs "
        "before the tool and therefore before the store verdict is ever read — so "
        "this row saw a fact about the address rather than the narrowing it exists "
        "to prove. Pick a channel this deployment's data/channel_limits.json marks "
        f"writable and a value inside its range.\n{joined_answers[:_EVIDENCE_CAP]}\n{evidence}"
    )
    readable = _control_tree_is_readable_in_the_worker()
    expected = NARROWING_CLAUSE if readable else UNAVAILABLE_CLAUSE
    why = (
        f"the build account {_build_account()!r} is narrowed to read-only, so the "
        "write must be refused with the store's narrowing verdict"
        if readable
        else "the worker's entrypoint reported it cannot read the control-context tree, "
        "so every owned write must fail closed on control_context_unavailable"
    )
    joined = "\n".join(answers)
    assert any(expected in answer for answer in answers), (
        f"{why}. The channel-write tool answered:\n{joined[:_EVIDENCE_CAP]}\n{evidence}"
    )


def test_an_owner_less_job_is_held_to_the_deployment_ceiling(
    single_user_stack: Path,
) -> None:
    """S3: the control for S2 — an owner-less job reads nobody's narrowing.

    The build account is narrowed; a cron-shaped fire of the same probe must not
    inherit that narrowing, because the run belongs to nobody and the deployment
    ceiling is what governs it. Without this row a wire that resolved the tree's
    only identity for every run — which is exactly the mistake a single-user
    deployment invites, since there IS only one — would satisfy S2 precisely as
    the real one does.

    Asserted two ways, because absence alone is weak. Neither store clause may
    appear — an owner-less run reads no record at all, so neither verdict can be
    its reason for refusing — AND the write must actually succeed, so that a
    probe refused for some unrelated reason cannot pass this row having judged
    nothing. The deployment's own ceiling permits it: this preset renders
    ``control_system.writes_enabled: true`` and lists the probe channel as
    writable.
    """
    _fire_and_assert_dispatched(WRITE_PROBE_UNOWNED_TRIGGER, owner=None)

    run = _wait_for_terminal_run(WRITE_PROBE_UNOWNED_TRIGGER)
    record = _persisted_run_record(run["run_id"])
    answers = _tool_answers(record, "__channel_write")
    evidence = _run_evidence(WRITE_PROBE_UNOWNED_TRIGGER, run, record)
    assert answers, (
        "the owner-less write probe never called the channel-write tool, so no write "
        f"was ever judged.\n{evidence}"
    )
    joined_answers = "\n".join(answers)
    for clause in (NARROWING_CLAUSE, UNAVAILABLE_CLAUSE):
        assert clause not in joined_answers, (
            "an owner-less job's write was judged against a recorded write state — "
            "it must be held to the deployment ceiling alone. The channel-write "
            f"tool answered:\n{joined_answers[:_EVIDENCE_CAP]}\n{evidence}"
        )
    # Absence of both clauses is not enough on its own, and a run proved why: a
    # probe refused for some unrelated reason — a read-only address, a limits
    # violation, a transport error — carries neither clause and would let this
    # row pass having judged nothing. The control is only a control if the write
    # the narrowed identity was refused actually SUCCEEDS when nobody owns it,
    # so the tool's own success envelope is asserted (channel_write answers
    # {"status": "success", "description": "Wrote N channel(s)", ...}).
    assert LIMITS_REFUSAL not in joined_answers, (
        f"the write to {PROBE_CHANNEL} was refused by the limits hook before the "
        "tool ran, so this row judged nothing. Pick a channel this deployment's "
        "data/channel_limits.json marks writable and a value inside its range.\n"
        f"{joined_answers[:_EVIDENCE_CAP]}\n{evidence}"
    )
    assert any('"status": "success"' in answer for answer in answers), (
        f"the owner-less write to {PROBE_CHANNEL} did not succeed. An owner-less run "
        "belongs to nobody, so no recorded write state governs it and the deployment "
        "ceiling alone decides — which here permits the write. A refusal of ANY kind "
        "means either the ceiling is not what this row assumes or the probe never "
        f"reached the machine.\n{joined_answers[:_EVIDENCE_CAP]}\n{evidence}"
    )
