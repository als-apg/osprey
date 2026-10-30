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
``record_unavailable``; under a CI umask of 022 a 0600 write would
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
succeeds, the narrowed record reads as the build account's switch being off, and
an owned write is refused as ``switch_off``. On Docker Desktop (macOS) bind-mount
ownership is remapped, the container's view of the tree may not carry the host's
group at all, and the correct answer becomes ``record_unavailable`` — a failure
that is closed, not open, and so still a pass for the safety property. Which one
is right is a fact about the HOST and not about this deployment, so the refusal
row reads the entrypoint's own access-probe line out of the worker's log and
asserts whichever the probe reports, exactly as the multi-user module does. Each
arm is matched by its refusals-contract phrase.

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
from osprey.deployment.web_terminals.roster import dispatch_tier_name
from osprey.dispatch import DISPATCHER_MCP_PATH
from osprey.port_layout import PORT_BASE_CONFIG_KEY, default_port
from osprey_connectors.control_context import RECORD_FILENAME, ControlContext, write_record
from osprey_connectors.identity import acting_identity
from osprey_connectors.posture_store import POSTURE_SANDBOX
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.types import CONTROL_TARGETS, TARGET_VA
from tests.e2e._deploy_support import (
    find_osprey_console_script,
    format_failure,
    run_command,
    wait_for_health,
)
from tests.e2e._facility_select import bounded_setpoint, facility_view, repo_view
from tests.e2e._mcp_sse import any_answer_succeeded, sse_payloads, tool_result
from tests.e2e._orm_stack import VA_PVA_PORT
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.profile_edits import set_pairs

#: This deploy's own thousand-port block. Chosen clear of every other deploy e2e
#: in this suite (dispatch 20700, overlay 21100, sandbox-escape 21400, web-deploy
#: 21700, bump 21900, queue 22000, preflight 22100, full-chain 23000, jupyter
#: 24000, SDK helpers 25000) so a reader checking for a collision reads one list.
#:
#: One entry in that list is not clear of this one: the dispatch module's block
#: starts at 20700, and its web-terminal slot lands on 20800 — the base itself.
#: The two modules therefore cannot deploy on one host whatever else moves, and
#: they do not have to: each has its own CI lane. Stated so a reader who finds
#: the overlap knows it is accounted for rather than missed.
PORT_BASE = 20800

#: Outside the block by design: the Channel Access port cannot be derived from
#: PORT_BASE, so it is claimed separately. The deploy e2es that pin a CA port
#: take theirs from this same 1506x series, and the ones whose stack can accept
#: any port reserve a free one at import instead — so the series carries no
#: fixed roster to copy, and ``grep -rn VA_CA_PORT tests/e2e`` is what a new
#: claim is checked against. 15069 is this module's and is claimed nowhere else.
VA_CA_PORT = 15069

#: The deployment repo's directory name IS the deployment's name, and compose
#: renders every container as ``<project>-<service>``, so the container targets
#: below are derived from it rather than spelled host-globally.
PROJECT_NAME = "suproj"
DISPATCHER_CONTAINER = f"{PROJECT_NAME}-event-dispatcher"
WORKER_CONTAINER = f"{PROJECT_NAME}-dispatch-worker-1"
#: The tier this module runs the dispatch workers at (``dispatch.tier``). The
#: preset ships ``readonly``, under which a worker has no write tool to call;
#: the rows here judge an owned and an owner-less write, so their worker is one
#: that may write. A tiered worker runs that tier's image, rooted at
#: ``/app/<project_name>-<tier>`` — with no web terminal beside it.
WORKER_TIER = "readwrite"

# The single-user shape, stated as one leaf: no roster, so no nginx, no
# per-user container, and no identity but the build account's own. Dotted LEAF
# keys on purpose — each edit states only its own leaf and leaves its
# subtree's siblings intact. Module-level so
# tests/cli/test_e2e_preset_login_tiers.py renders exactly this set-list.
#: ``osprey init <repo>`` arguments this module's stack is built from.
INIT_PRESET_ARGS = [
    "--preset",
    "control-assistant",
    "--no-git",
    *set_pairs(
        {
            "config": {
                "modules.web_terminals.enabled": False,
                PORT_BASE_CONFIG_KEY: PORT_BASE,
            }
        }
    ),
    "--set",
    "provider=als-apg",
    "--set",
    "model=claude-haiku-5-5",
    "--set",
    f"virtual_accelerator.port={VA_CA_PORT}",
    "--set",
    f"virtual_accelerator.pva_port={VA_PVA_PORT}",
    "--set",
    f"dispatch.tier={WORKER_TIER}",
]

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

#: The limits hook's refusal headline. Named so the rows below can say "this
#: probe never reached the mechanism" instead of failing with a wall of text
#: about an address — the exact trap that cost a run.
LIMITS_REFUSAL = "CHANNEL LIMITS VIOLATION"

#: The refusal phrases a judged write can come back with: the owner's switch is
#: off for the machine (a record narrowed to read-only reads as off), the owner's
#: record cannot be read, and a run that belongs to no login. Each is the
#: refusals contract's phrase without its machine-kind suffix, so a row matches
#: whichever kind the deployment's machine is. Spelled as in
#: ``test_dispatch_deploy``, because a drift between the two modules' spellings
#: would let one of them pass on a message the other no longer recognises.
SWITCH_OFF_REFUSAL = refusal_phrase("switch_off")
UNAVAILABLE_REFUSAL = refusal_phrase("record_unavailable")
OWNER_UNSCHEDULED_REFUSAL = refusal_phrase("owner_unscheduled")

_EVIDENCE_CAP = 6000

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    pytest.mark.flaky(reruns=1),
]


def _docker_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return ((proc.stdout or "") + (proc.stderr or "")).strip() or "(no output)"


def _write_probe_action(channel: str, value: float) -> str:
    """The write probe: one write of ``value`` to ``channel``."""
    return f"""    source: webhook
    action:
      prompt: >-
        Call the control-system channel-write tool exactly once, writing the
        value {value} to the channel {channel}, and then report the
        tool's response verbatim. Do not retry and do not call any other tool.
      allowed_tools:
        - mcp__controls__channel_write
"""


def _probe_triggers(channel: str, value: float) -> str:
    """The probe's trigger entries, writing ``value`` to ``channel``."""
    action = _write_probe_action(channel, value)
    return (
        "\n  # -- e2e single-user narrowing probes "
        "(appended by tests/e2e/test_single_user_narrowing_e2e.py) --\n"
        "  #\n"
        "  # One action under two names: the owned fire and the owner-less control\n"
        "  # must produce two separate runs in the dispatcher feed.\n"
        f"  - name: {WRITE_PROBE_OWNED_TRIGGER}\n{action}"
        f"  - name: {WRITE_PROBE_UNOWNED_TRIGGER}\n{action}"
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


def _append_probe_triggers(repo: Path, scratch: Path) -> tuple[str, float]:
    """Add the write probe to the deployment's trigger set, before the build.

    Which channel is chosen decides whether these rows can see anything at
    all: the limits database is enforced by a PreToolUse HOOK that runs BEFORE
    the tool, so a channel it marks read-only is refused there with "CHANNEL
    LIMITS VIOLATION" and the call never reaches the connector where the store
    verdict is read. The refusal rows would then pass or fail on a fact about
    the address rather than on the narrowing under test. A bounded setpoint at
    its band's midpoint clears that gate, so the only refusal a narrowed owner
    can meet is the store's. No build exists yet, so the pick is read from the
    repo's own facility tree rendered into ``scratch``; returns
    ``(channel, value)``.
    """
    channel, value = bounded_setpoint(facility_view(repo / "data" / "facility", scratch))
    triggers_file = _repo_triggers_file(repo)
    text = triggers_file.read_text(encoding="utf-8")
    assert "hello-dispatch" in text, (
        f"{triggers_file} does not look like the preset's trigger set:\n{text[:500]}"
    )
    with triggers_file.open("a", encoding="utf-8") as handle:
        handle.write(_probe_triggers(channel, value))
    return channel, value


def _assert_the_build_serves_the_probe(repo: Path, probe: tuple[str, float]) -> None:
    """The built view answers the pick made before the build, writable and in range."""
    channel, value = probe
    view = repo_view(repo)
    assert bounded_setpoint(view) == probe, (
        f"the build picks {bounded_setpoint(view)} as the bounded setpoint, but the "
        f"triggers were written for {probe} from the repo's facility tree"
    )
    served = view.channel(channel)
    assert served.writable and served.value_range is not None, (
        f"the build serves {channel} without a writable band"
    )
    low, high = served.value_range
    assert low <= value <= high, f"{value} lies outside {channel}'s band {served.value_range}"


@pytest.fixture(scope="module")
def probe_channel(single_user_stack: Path) -> str:
    """The channel the write probe writes, as the deployed build serves it."""
    return bounded_setpoint(repo_view(single_user_stack))[0]


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
    marker answers ``record_unavailable`` for every owner, which would
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

    osprey_bin = find_osprey_console_script()
    base = tmp_path_factory.mktemp("single_user_narrowing_build")
    repo = base / PROJECT_NAME

    init = run_command(
        [str(osprey_bin), "init", str(repo), *INIT_PRESET_ARGS],
        cwd=base,
        timeout=300,
    )
    if init.returncode != 0:
        pytest.fail(format_failure("osprey init", init))

    # Before the build, so the render the dispatcher loads carries them.
    probe = _append_probe_triggers(repo, base / "probe_view")

    build = run_command(
        [str(osprey_bin), "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        cwd=base,
        timeout=300,
    )
    if build.returncode != 0:
        pytest.fail(format_failure("osprey build", build))
    rendered = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    assert dispatch_tier_name(rendered) == WORKER_TIER, (
        f"the build runs the dispatch workers at {dispatch_tier_name(rendered)!r}, not "
        f"{WORKER_TIER!r}: every worker-side path below is rooted at the tier's project"
    )
    _assert_the_build_serves_the_probe(repo, probe)

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
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True, timeout=120)

    try:
        up = run_command(
            [str(osprey_bin), "up", "-d", "--dev"], cwd=repo, timeout=DEPLOY_UP_TIMEOUT_SEC
        )
        if up.returncode != 0:
            pytest.fail(format_failure("osprey up", up))
        wait_for_health(f"{DISPATCHER_URL}/health", HEALTH_TIMEOUT_SEC)
        # A healthy dispatcher does not mean a healthy worker: the worker runs
        # the heavier project image and boots later. Gate on its proxied feed so
        # no row races a still-warming worker, and a worker that never comes up
        # fails here with container logs rather than as a bare 502 mid-test.
        _wait_for_worker_feed(HEALTH_TIMEOUT_SEC)
        yield repo
    finally:
        down = run_command([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(  # surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        # Volumes are deliberately kept by `down`; remove this project's own via
        # the shared label-scoped sweep. Exact-named only, never a prune.
        remove_project_volumes(resolve_project_name({"project_name": PROJECT_NAME}))


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
    req = urllib.request.Request(  # localhost only
        f"{DISPATCHER_URL}/dashboard/runs",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
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
    req = urllib.request.Request(  # localhost only
        DISPATCHER_MCP_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
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
    payloads = sse_payloads(call.get("body") or "")
    assert payloads, f"manual_fire returned no SSE payload: {call.get('body')!r}"
    answer = tool_result(payloads)
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
    req = urllib.request.Request(  # localhost only
        f"{DISPATCHER_URL}{path}",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


#: Run inside the worker container, under the worker's own record login: the
#: run's timeline from the agent record, as ``{name, input, result}`` per tool
#: call. Retries while the read is incomplete (the transcript still ingesting).
_RECORDED_RUN_CLIENT = r"""
import json, os, sys, time
from osprey.agent_record.client import RecordClient

client = RecordClient.from_env(os.environ)
deadline = time.monotonic() + 60
while True:
    answer = client.run_timeline(sys.argv[1])
    if answer.complete or time.monotonic() > deadline:
        break
    time.sleep(2)


def text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return "" if content is None else json.dumps(content)


calls = []
for node in answer.data or []:
    if node.get("kind") != "tool_call":
        continue
    body = node.get("body") or {}
    result = (node.get("result") or {}).get("body") or {}
    calls.append(
        {"name": body.get("name"), "input": body.get("input"), "result": text(result.get("content"))}
    )
print(json.dumps({
    "complete": answer.complete,
    "unavailable": answer.unavailable,
    "reason": answer.reason,
    "tool_calls": calls,
}))
"""


def _persisted_run_record(run_id: str) -> dict:
    """The run as the agent record holds it — the one place carrying tool results.

    The dispatcher feed reports status and a tool COUNT; what a tool ANSWERED is
    in the run's timeline in the agent record, read here from inside the worker
    under its own record login, as ``tool_calls`` of ``{name, input, result}``.
    These rows assert on the tool's answer, so they read it there rather than
    trusting the agent's prose summary of it.
    """
    proc = subprocess.run(
        ["docker", "exec", WORKER_CONTAINER, "python3", "-c", _RECORDED_RUN_CLIENT, run_id],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"could not read run {run_id} from the agent record: {(proc.stderr or proc.stdout).strip()}"
    )
    recorded = json.loads(proc.stdout)
    assert recorded["complete"], (
        f"the record's timeline of run {run_id} is not complete: {recorded}"
    )
    return recorded


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
            f"recorded tool_calls:\n{json.dumps(record.get('tool_calls'), indent=2)[:_EVIDENCE_CAP]}"
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
    as its owner's switch being off or as ``record_unavailable`` is decided
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
    closed on ``record_unavailable`` and narrowing silently stops
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
        "alone, so every owned write would fail closed on record_unavailable"
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
    probe_channel: str,
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
    is unreadable, which fails closed on ``record_unavailable`` instead. The
    entrypoint's own access probe says which, and this row asserts that one by its
    refusal phrase; the narrowed record reads as the switch being off.
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
        f"the write to {probe_channel} was refused by the LIMITS hook, which runs "
        "before the tool and therefore before the write verdict is ever read — so "
        "this row saw a fact about the address rather than the narrowing it exists "
        f"to prove.\n{joined_answers[:_EVIDENCE_CAP]}\n{evidence}"
    )
    readable = _control_tree_is_readable_in_the_worker()
    expected = SWITCH_OFF_REFUSAL if readable else UNAVAILABLE_REFUSAL
    why = (
        f"the build account {_build_account()!r} is narrowed to read-only, so the "
        "write must be refused as switch_off"
        if readable
        else "the worker's entrypoint reported it cannot read the control-context tree, "
        "so every owned write must fail closed on record_unavailable"
    )
    joined = "\n".join(answers)
    assert any(expected in answer for answer in answers), (
        f"{why}. The channel-write tool answered:\n{joined[:_EVIDENCE_CAP]}\n{evidence}"
    )


def test_an_owner_less_job_is_refused_as_unscheduled(probe_channel: str) -> None:
    """S3: the control for S2 — an owner-less job's write is refused for its own reason.

    The build account is narrowed; a cron-shaped fire of the same probe must not
    inherit that narrowing, because the run belongs to nobody. Without this row a
    wire that resolved the tree's only identity for every run — which is exactly
    the mistake a single-user deployment invites, since there IS only one — would
    satisfy S2 precisely as the real one does.

    Nobody's switch covers an owner-less run either, so its write is refused as
    ``owner_unscheduled`` — never as the build account's ``switch_off`` and never
    as an unreadable record, which would mean the run was judged against somebody's
    record. The write must not succeed, and the expected phrase must be present, so
    a probe refused for some unrelated reason cannot pass this row.
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
    # A probe refused before the tool ran — a limits violation — judged nothing
    # about the owner, so it must not read as the refusal this row expects.
    assert LIMITS_REFUSAL not in joined_answers, (
        f"the write to {probe_channel} was refused by the limits hook before the "
        "tool ran, so this row judged nothing.\n"
        f"{joined_answers[:_EVIDENCE_CAP]}\n{evidence}"
    )
    for phrase in (SWITCH_OFF_REFUSAL, UNAVAILABLE_REFUSAL):
        assert phrase not in joined_answers, (
            "an owner-less job's write was judged against a recorded switch — it belongs "
            "to no login, so no record governs it. The channel-write tool "
            f"answered:\n{joined_answers[:_EVIDENCE_CAP]}\n{evidence}"
        )
    assert not any_answer_succeeded(answers), (
        f"the owner-less write to {probe_channel} succeeded. A cron-shaped fire runs "
        "for no login, so no switch covers it and the write must be refused.\n"
        f"{joined_answers[:_EVIDENCE_CAP]}\n{evidence}"
    )
    assert any(OWNER_UNSCHEDULED_REFUSAL in answer for answer in answers), (
        "the owner-less write must be refused as owner_unscheduled. The channel-write "
        f"tool answered:\n{joined_answers[:_EVIDENCE_CAP]}\n{evidence}"
    )
