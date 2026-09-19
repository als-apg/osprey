"""Full-stack Docker e2e for event dispatch (L2) — highest fidelity.

Unlike the subprocess sweep (``test_dispatch_tutorial.py``), this exercises the
REAL shipped artifacts: the compose templates, the bundled Dockerfile (which
installs Node + the Claude Code CLI the worker needs), the worker ``env_file``
wiring that carries provider auth, and the in-network ``dispatch-worker-1``
routing baked into the shipped ``tutorial_triggers.yml`` — none of which the
subprocess path touches.

It inits + builds a control-assistant deployment repo, deploys the stack with
``osprey up -d --dev``, fires all four tutorial webhooks at the dispatcher
(host-published on this deployment's dispatcher slot), and asserts:

  * hello-dispatch / triage-event / save-report -> a run completes
  * denied-tool-demo -> rejected by the worker denylist (no completed run)

The same deploy also stands up the preset's multi-user web tier (local image
mode): the build renders one persona project per ``personas/`` delta into
``build/<repo>-<persona>/``, ``up`` builds one ``:local`` image per persona and
brings up nginx + one web-terminal container per roster user. Four more things
are asserted on top:

  T1 topology:      nginx + both per-user containers come up healthy and the
                    landing page lists both roster users.
  T2 readonly tier:  the rendered READ-ONLY persona project pins
                     ``control_system.writes_enabled: false`` and its rendered
                     ``settings.json`` denies the channel-write tool.
  T3 readwrite tier: the rendered READ-WRITE persona project arms writes and
                     keeps the tool on the ask (human-approval) path — the
                     positive control that T2 is a real posture difference.
  T4 same surface:   both persona projects declare the IDENTICAL ``.mcp.json``
                     server set — the tier boundary is enforcement, never a
                     quietly different tool surface.

That topology — a dispatcher, a worker and a real terminal container per roster
user — is also the only place the dispatcher's owner wire can be driven end to
end, so four more rows ride the same deploy. They add two probe triggers to the
deployment's own trigger set and narrow the firing user on the host tree:

  O1 owner on the wire:  a job fired by ``manual_fire`` through a terminal's
                         ``/panel/events/mcp`` hop runs with that terminal's user
                         in ``OSPREY_CONTROL_OWNER``.
  O2 the owner's chip:   that job's first control-system write is refused,
                         because its owner is narrowed to read-only.
  O3 the cron door:      the same two probes fired at the webhook carry no owner
                         and are held to the deployment ceiling — the control
                         without which O1 and O2 pass on a wire that stamps
                         everyone.
  O4 the gate:           an uncredentialed call to ``/panel/events/mcp`` is
                         refused at the terminal and never reaches the dispatcher.

The preset also deploys a Bluesky bridge beside the dispatcher, which makes
this the one stack in the suite where a plan an agent QUEUES can be read back
off the queue. That is the owner's other carrier — the MCP server stamps the
enqueue, the bridge lifts the name onto the item, and the item outlives the run
— so two more rows ride the same deploy:

  Q1 the queued plan:    a plan queued by a job fired through a terminal's panel
                         hop is attributed to that terminal's user on the queue.
  Q2 nobody's plan:      the same probe fired at the webhook queues a plan that
                         names nobody — the control without which Q1 passes on a
                         wire that stamps every enqueue with one name.

COEXISTENCE: every host-published web port, the virtual accelerator's Channel
Access port (the one ``port_base`` does not move), the web-container name
prefix (``facility.prefix``) and the compose PROJECT NAME are e2e-unique,
so this deploy can run beside a real control-assistant stack on the same host
without colliding on ports or container names. The project name is part of that
promise and not a cosmetic choice: every container is named ``<project>-<service>``
and teardown addresses the stack BY PROJECT NAME, so a module sharing another's
project name would tear down that module's running containers as well as its
own. The persona ``:local`` image
tags are namespaced by the repo's own name (``osprey init`` writes each catalog
entry's ``project`` as ``<repo>-<persona>``), which keeps them off a real
deployment's tags for free. Teardown names exact resources only — never a
prune or wildcard.

The worker receives its provider key via ``env_file: ./.env`` (see the
dispatch_worker compose template) — the compose CLI reads the repo's ``.env``
on the HOST (as its owner, even when it's 0600) and injects the vars directly
into the container environment, so the non-root worker never has to open the
file itself. Without that wiring the agent run cannot authenticate, so this
test also guards it.

Gating: needs Docker and ``ALS_APG_API_KEY``. We do NOT gate on a host
``claude`` binary — the CLI lives inside the built image, not on the runner.
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
from typing import Any

import pytest
import yaml

from osprey import bluesky_tool_names
from osprey.deployment.compose_generator import (
    CONTROL_TREE_MARKER_NAME,
    control_target_identity_dir,
    ensure_control_target_dir,
    resolve_project_name,
)
from osprey.port_layout import PORT_BASE_CONFIG_KEY, default_port
from osprey_connectors.control_context import RECORD_FILENAME, ControlContext, write_record
from osprey_connectors.posture_store import POSTURE_SANDBOX
from osprey_connectors.types import CONTROL_TARGETS, TARGET_VA
from tests.e2e import _orm_stack, _queue_drive
from tests.e2e._mcp_sse import any_answer_succeeded, sse_payloads, tool_result
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.profile_edits import set_pairs

#: This deploy's own thousand-port block, chosen so the whole stack — the
#: dispatcher, nginx and every panel family — coexists with a live
#: control-assistant stack and with every other deploy e2e on the same host
#: (test_deploy_lifecycle.py tops out ~20600). One base moves all of them.
PORT_BASE = 20700

DISPATCHER_URL = f"http://localhost:{default_port('dispatcher', base=PORT_BASE)}"
TOKEN = "dev-token"  # matches the .env tokens written below

# Container image builds (Node + Claude CLI install; project + dispatch + two
# persona images) are slow on a cold cache. Four images on a cold cache have been
# measured past 30 minutes on a developer machine, where `up` then died on its own
# budget rather than on anything wrong with the deployment — and the module's
# teardown removes both persona tags, so every run pays that cold cache. The
# budget is a harness ceiling, not an assertion: too low turns a slow host into a
# red suite, while too high only delays a genuinely wedged deploy, which the
# per-service health waits below report on their own much sooner.
DEPLOY_UP_TIMEOUT_SEC = 3600
HEALTH_TIMEOUT_SEC = 180.0
RUN_TIMEOUT_SEC = 300.0
CONTAINER_HEALTH_TIMEOUT_SEC = 180.0

# The deployment repo's directory name IS the deployment's name; compose renders
# each container_name as ``<project>-<service>`` (services/*/docker-compose.yml.j2),
# so derive the docker targets below rather than hardcode host-global names that
# break the moment the templates are namespaced per-project.
#
# E2E-unique, and part of the coexistence promise above rather than a detail:
# teardown is BY PROJECT NAME, so two modules sharing one project name do not
# merely collide on container names when they deploy together — whichever tears
# down first deletes the other's running stack. ``proj`` is the suite's most
# shared spelling and is deliberately not used here.
PROJECT_NAME = "osprey-e2e-dispatch"
DISPATCHER_CONTAINER = f"{PROJECT_NAME}-event-dispatcher"
WORKER_CONTAINER = f"{PROJECT_NAME}-dispatch-worker-1"
# Where the worker's agent-data volume mounts inside the project image: the
# repo's durable state zone, named by ``agent_data.base_dir`` in the config the
# compose generator renders the mount from.
WORKER_AGENT_DATA = f"/app/{PROJECT_NAME}/var/agent_data"

# ---------------------------------------------------------------------------
# Multi-user web tier (T1-T4). Prefix and ports are remapped off the preset
# defaults to e2e-unique values (see COEXISTENCE in the module docstring); the
# roster itself (alice→readwrite, bob→readonly) is the preset's own and is
# asserted, not configured, here. The persona project names are the repo's, not
# this test's: `osprey init` writes every catalog entry as
# ``<repo>-<persona>`` at ``build/<repo>-<persona>``, and `project` must equal
# `project_path`'s basename for the render to land where the deploy mounts it.
# ---------------------------------------------------------------------------
WEB_PREFIX = "dde"  # facility.prefix override: container names dde-nginx, dde-web-<user>
# Which roster user holds which tier is the preset's decision, not this
# module's: ``modules.web_terminals.users`` binds each name to a persona, and
# the persona is what decides the tier. Pinned against the render by
# :func:`_assert_roster_personas` in the fixture, because the two names are
# otherwise interchangeable everywhere they are used in pairs — an inverted
# label costs nothing until a row reaches for a surface only one tier has, and
# then it fails an hour into a deploy as a 404 on that surface rather than as a
# wrong name here.
READONLY_USER = "bob"
READWRITE_USER = "alice"
READONLY_PROJECT = f"{PROJECT_NAME}-readonly"
READWRITE_PROJECT = f"{PROJECT_NAME}-readwrite"
READONLY_IMAGE = f"{READONLY_PROJECT}:local"
READWRITE_IMAGE = f"{READWRITE_PROJECT}:local"

# The write tool the readonly tier's rendered settings.json must deny and the
# readwrite tier's must not (write tools land in permissions.deny when
# writes_enabled is false — see osprey.cli.templates.claude_code).
CHANNEL_WRITE_TOOL = "mcp__controls__channel_write"

# EVERY published web port follows :data:`PORT_BASE` rather than the preset
# defaults, so this deploy coexists with a live control-assistant stack on the
# same host. The test derives them the same way the render does, so a slot
# that moves in the layout moves here without an edit.
WEB_PORTS = {
    slot: default_port(slot, base=PORT_BASE)
    for slot in (
        "nginx",
        "web",
        "artifact",
        "ariel",
        "lattice",
        "channel_finder",
        "okf",
        "system_health",
    )
}

# Probed from INSIDE the nginx container (docker exec + curl, which the
# container's own healthcheck already relies on), never from the host: with
# `network_mode: host` on Docker Desktop (macOS/Windows) the web stack binds
# inside the Docker Linux VM, so a host-side probe fails on any machine
# without the opt-in host-networking setting — while the in-container probe
# exercises the same nginx routing everywhere.
LANDING_URL = f"http://127.0.0.1:{WEB_PORTS['nginx']}/"

#: This deploy's Channel Access port — the ONE host port :data:`PORT_BASE` does
#: not move. Virtual-accelerator instance 1 serves EPICS on 5064 by default so
#: that clients configured for a real facility reach it unchanged, and a second
#: deployment on the same host has to name its own (``osprey up``'s port
#: preflight refuses otherwise, and the whole deploy aborts before any container
#: is touched). Without this the module's COEXISTENCE promise held for every
#: published web port and failed on the one port it never mentioned.
#:
#: A distinct literal rather than an offset off PORT_BASE, because this port is
#: outside the block by design: the other deploy e2es in this suite each claim
#: one in the same 1506x series (queue 15064, ORM 15065, substrate 15066,
#: archiver 15067), and a reader checking for a collision reads one list.
VA_CA_PORT = 15068


def _web_container(user: str) -> str:
    return f"{WEB_PREFIX}-web-{user}"


NGINX_CONTAINER = f"{WEB_PREFIX}-nginx"

# hello-dispatch / triage-event / save-report should complete; denied-tool-demo
# must be rejected by the server-side denylist.
_COMPLETING_TRIGGERS = ("hello-dispatch", "triage-event", "save-report")
_DENIED_TRIGGER = "denied-tool-demo"

_DEMO_PAYLOAD = {
    "signal": "demo:vacuum:pressure",
    "value": 4.2,
    "threshold": 3.0,
    "severity": "warning",
}

# ---------------------------------------------------------------------------
# Owner attribution on the dispatcher wire (O1-O4).
#
# The firing user is the READ-WRITE roster user, not the read-only one. Two
# things follow the persona's EVENTS panel declaration and nothing else: the
# `web.panels.events` entry the terminal resolves a `/panel/events/*` hop
# against (projected into a persona render only for a profile that selects the
# panel — osprey.deployment.reach.project_attached_overrides), and the
# EVENT_DISPATCHER_TOKEN its container receives (web_terminals/render.py). That
# split IS the tier boundary: a read-only persona must not hold a credential
# that can fire triggers, and its terminal answers the hop 404 because it
# carries no events panel to resolve. So these rows drive the hop in the
# read-write user's container, and only there.
# ---------------------------------------------------------------------------
#: The roster user whose terminal fires the jobs below, and whose narrowing the
#: fired jobs' writes are judged against.
FIRING_USER = READWRITE_USER

#: The ``.mcp.json`` key of the dispatcher wire — the framework server entry a
#: persona's render carries exactly when that persona selects the EVENTS panel.
DISPATCHER_MCP_SERVER = "event_dispatcher"

#: Triggers this module appends to the deployment's own ``triggers.yml`` before
#: the build. Two probes, each fired through BOTH doors, so the owner-carrying
#: door and the owner-less one are separate runs in the feed rather than two
#: readings of one latest-wins entry.
ENV_PROBE_PANEL_TRIGGER = "owner-env-panel"
ENV_PROBE_CRON_TRIGGER = "owner-env-cron"
WRITE_PROBE_PANEL_TRIGGER = "owner-write-panel"
WRITE_PROBE_CRON_TRIGGER = "owner-write-cron"
QUEUE_PROBE_PANEL_TRIGGER = "owner-queue-panel"
QUEUE_PROBE_CRON_TRIGGER = "owner-queue-cron"

#: The preset deploys a Bluesky bridge beside the dispatcher, on this
#: deployment's own port block — so the queue an agent enqueues onto is
#: readable from the host without a second deploy. Derived rather than pinned:
#: the slot moves with :data:`PORT_BASE` exactly as every other published port
#: of this stack does.
BRIDGE_URL = f"http://localhost:{default_port('bluesky', base=PORT_BASE)}"

#: The draft client name this module edits the shared plan draft under. The
#: bridge records it on the draft, so a stray edit from a panel or another
#: client is visible as somebody else's name rather than as this module's.
DRAFT_CLIENT_ID = "dispatch-deploy-e2e"

#: The queue tool the queue probes are held to. Imported rather than spelled,
#: so a tool rename detaches the trigger's allowlist from the tool loudly.
QUEUE_ADD_TOOL = bluesky_tool_names.matcher(bluesky_tool_names.QUEUE_ADD)

#: A channel the preset's virtual accelerator publishes, and it must be a
#: WRITABLE setpoint inside its limits. The limits check is a PreToolUse hook
#: that runs before the tool, so a channel the limits database marks read-only
#: (any ``:RB`` readback) is refused there with "CHANNEL LIMITS VIOLATION" and
#: the call never reaches the connector where the store verdict is read; the
#: refusal rows would then judge a fact about the address rather than the
#: narrowing under test. A setpoint well inside its range clears that gate, so
#: the only refusal a narrowed owner can meet is the store's.
PROBE_CHANNEL = "SR:MAG:HCM:01:CURRENT:SP"

#: The limits hook's refusal headline. Named so a row can say "this probe never
#: reached the mechanism" rather than fail with a wall of text about an address.
LIMITS_REFUSAL = "CHANNEL LIMITS VIOLATION"

#: What the agent prints for a run whose environment names no owner. Chosen so
#: the owner-less reading is a positive assertion rather than the absence of a
#: name — a run that failed to print anything must not read as "no owner".
NO_OWNER_MARKER = "<none>"

#: The closing clause of each store arm of the connector's refusal
#: (``_writes_disabled_result``). The unavailable arm's prose CONTAINS the word
#: "narrowing" ("This is not a narrowing anybody set"), so the arms are told
#: apart by the clause each one CLOSES on and never by the bare word.
NARROWING_CLAUSE = "The store answered narrowing."
UNAVAILABLE_CLAUSE = "the store answered control_context_unavailable."

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


#: The two probe actions, each declared under one panel-fired and one
#: webhook-fired name. Appended to the deployment's OWN triggers file rather than
#: shipped in the preset: they exist to read one run's environment and to attempt
#: one write, which is a test's business and not a tutorial's.
#:
#: Each prompt names exactly one tool and exactly what to do with it. A trigger's
#: ``allowed_tools`` is the enforcement (the worker holds the run to it), so the
#: prompt only has to make the single call the enforcement already bounds.
#: The python program the environment probe runs. One line, because the folded
#: YAML scalar it is embedded in joins its lines with spaces.
_ENV_PROBE_PROGRAM = (
    f'import os; print("OWNER=" + (os.environ.get("OSPREY_CONTROL_OWNER") or "{NO_OWNER_MARKER}"))'
)

_ENV_PROBE_ACTION = f"""    source: webhook
    action:
      prompt: >-
        Call the python executor exactly once with this program and nothing
        else, then report its output verbatim: {_ENV_PROBE_PROGRAM}
      allowed_tools:
        - mcp__python__execute
"""

_WRITE_PROBE_ACTION = f"""    source: webhook
    action:
      prompt: >-
        Call the control-system channel-write tool exactly once, writing the
        value 1.0 to the channel {PROBE_CHANNEL}, and then report the tool's
        response verbatim. Do not retry and do not call any other tool.
      allowed_tools:
        - mcp__controls__channel_write
"""

#: The queue probe. Unlike the other two it needs a value the trigger file
#: cannot know — which draft revision to queue — so the fire carries it in the
#: event payload, which the dispatcher folds into the prompt.
_QUEUE_PROBE_ACTION = f"""    source: webhook
    action:
      prompt: >-
        Call the Bluesky queue-add tool exactly once, queuing the draft
        revision named by draft_revision in the event payload, and then report
        the tool's response verbatim. Do not retry and do not call any other
        tool.
      allowed_tools:
        - {QUEUE_ADD_TOOL}
"""

_OWNER_PROBE_TRIGGERS = (
    "\n  # -- e2e owner probes (appended by tests/e2e/test_dispatch_deploy.py) --\n"
    "  #\n"
    "  # Each probe is declared twice, under one panel-fired and one webhook-fired\n"
    "  # name, so the two doors a job can be fired through — the terminal's panel\n"
    "  # proxy, which mints an owner, and the webhook, which is the cron-shaped\n"
    "  # owner-less door — produce two SEPARATE runs in the dispatcher feed rather\n"
    "  # than two readings of one latest-wins entry.\n"
    f"  - name: {ENV_PROBE_PANEL_TRIGGER}\n{_ENV_PROBE_ACTION}"
    f"  - name: {ENV_PROBE_CRON_TRIGGER}\n{_ENV_PROBE_ACTION}"
    f"  - name: {WRITE_PROBE_PANEL_TRIGGER}\n{_WRITE_PROBE_ACTION}"
    f"  - name: {WRITE_PROBE_CRON_TRIGGER}\n{_WRITE_PROBE_ACTION}"
    f"  - name: {QUEUE_PROBE_PANEL_TRIGGER}\n{_QUEUE_PROBE_ACTION}"
    f"  - name: {QUEUE_PROBE_CRON_TRIGGER}\n{_QUEUE_PROBE_ACTION}"
)


def _repo_triggers_file(repo: Path) -> Path:
    """The deployment's own ``triggers.yml`` — the SOURCE copy, never the render.

    ``osprey init`` writes the preset's trigger set into the repo's source zone
    and ``osprey build`` renders it; appending to the render would be overwritten
    by the next build and never reach the dispatcher. Located by name rather than
    by a spelled path so a move of the source zone moves this with it, and
    asserted to be exactly one file so a second copy cannot be edited silently.
    """
    found = [path for path in repo.rglob("triggers.yml") if "build" not in path.parts]
    assert len(found) == 1, f"expected exactly one source triggers.yml under {repo}, found: {found}"
    return found[0]


def _append_owner_probe_triggers(repo: Path) -> None:
    """Add the owner probes to the deployment's trigger set, before the build."""
    triggers_file = _repo_triggers_file(repo)
    text = triggers_file.read_text(encoding="utf-8")
    assert "hello-dispatch" in text, (
        f"{triggers_file} does not look like the preset's trigger set:\n{text[:500]}"
    )
    with triggers_file.open("a", encoding="utf-8") as handle:
        handle.write(_OWNER_PROBE_TRIGGERS)


def _narrow_on_the_host(repo: Path, identity: str) -> Path:
    """Narrow *identity* to read-only on every control target, on the HOST tree.

    This is the chip's effect without the browser: the record the terminal's
    header chip writes is the record this writes, in the directory the build
    provisions for that identity and through the writer that owns the file mode
    (``write_record``'s explicit 0640 on a 2770 directory). The dispatch worker
    binds this tree read-only and reads exactly this file for the owner its job
    carries.

    Every target is narrowed rather than the one this deployment happens to point
    at: the assertion is about whose narrowing a job obeys, and pinning it to a
    target name would make the row fail for a deployment whose connector resolved
    a different one.
    """
    record_path = _host_record_path(repo, identity)
    identity_dir = record_path.parent
    # The build is what provisions the tree and plants its marker, and a reader
    # that finds no marker answers ``control_context_unavailable`` for every
    # owner — which would make the refusal rows below pass for the wrong reason.
    # Asserted rather than created here: seeding a record into a tree the build
    # never made is a fixture-order mistake, not a state to paper over.
    marker = identity_dir.parent / CONTROL_TREE_MARKER_NAME
    assert marker.is_file(), (
        f"osprey build did not provision the control-context tree at "
        f"{identity_dir.parent} (no {CONTROL_TREE_MARKER_NAME}); a narrowing seeded "
        "into an unprovisioned tree reads as unavailable for every owner"
    )
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    ensure_control_target_dir(config, repo, identity, relative_to=repo)
    record = ControlContext(
        target=TARGET_VA,
        generation=1,
        owner=None,
        posture=dict.fromkeys(CONTROL_TARGETS, POSTURE_SANDBOX),
    )
    return write_record(record, path=record_path)


def _assert_roster_personas(repo: Path) -> None:
    """Hold this module's two roster names to the personas the preset binds them to.

    Everything the owner rows do downstream is decided by the FIRING user's
    persona rather than by their name: which container holds the dispatcher
    bearer, and which one carries a ``web.panels.events`` entry for the terminal
    to resolve the ``/panel/events/*`` hop against. Both names appear in pairs
    everywhere else in this module, so an inverted label is invisible until a
    row reaches for that hop — and then the deployment answers it correctly,
    with a 404 from the tier that is supposed to lack it, an hour into a deploy.

    Checked on the render rather than on the preset text, so a preset that
    re-binds a name is caught here too, and checked before ``up`` so the
    contradiction costs two minutes instead of an hour.
    """
    rendered = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    roster = {
        str(entry.get("name")): str(entry.get("persona"))
        for entry in (rendered["modules"]["web_terminals"]["users"] or [])
    }
    expected = {READONLY_USER: "readonly", READWRITE_USER: "readwrite"}
    actual = {user: roster.get(user) for user in expected}
    assert actual == expected, (
        f"this module's roster labels contradict the render: expected {expected}, "
        f"the deployment binds {actual} (whole roster: {roster}). The tier follows "
        f"the persona, not the name — swap READONLY_USER/READWRITE_USER to match."
    )

    # The read-write persona's own render must carry the events panel, because
    # that entry is the whole of what the terminal resolves the dispatcher hop
    # against. Absent here means every owner row below gets a 404 that is the
    # terminal answering correctly about a panel it was never told.
    persona_config = yaml.safe_load(
        (_persona_dir(repo, READWRITE_PROJECT) / "config.yml").read_text(encoding="utf-8")
    )
    events = ((persona_config.get("web") or {}).get("panels") or {}).get("events") or {}
    assert events.get("enabled") and events.get("url"), (
        f"the read-write persona render declares no reachable EVENTS panel "
        f"(web.panels.events = {events!r}), so /panel/events/mcp cannot resolve in "
        f"{READWRITE_USER}'s terminal. The entry is projected from the hosting "
        f"deployment for a profile that selects the panel — see "
        f"osprey.deployment.reach.project_attached_overrides."
    )


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Init + build + ``osprey up`` a control-assistant stack; tear down after."""
    if not os.environ.get("ALS_APG_API_KEY"):
        pytest.skip("ALS_APG_API_KEY not set")

    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("dispatch_deploy_build")
    repo = base / PROJECT_NAME

    # The preset's multi-user web tier deploys as shipped; only its
    # host-global identifiers are remapped to e2e-unique values (see
    # COEXISTENCE in the module docstring): the container-name prefix, and the
    # one port base every published port of the stack derives from. Dotted LEAF
    # keys on purpose -- each edit states only its own leaf and leaves its
    # subtree's siblings intact (same convention as tests/e2e/_orm_stack.py).
    # The persona catalog's own `project` / `project_path` are deliberately
    # NOT stated: `osprey init` writes them from the repo's name, and the
    # build renders each persona exactly there.
    edits = {
        "config": {
            "facility.prefix": WEB_PREFIX,
            PORT_BASE_CONFIG_KEY: PORT_BASE,
        }
    }

    # Two steps, because the surface has two: `init` writes the repo's source
    # zone from the preset, `build` renders build/ from it — including one
    # persona project per `personas/` delta.
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
            # The Channel Access port PORT_BASE cannot move (see VA_CA_PORT).
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

    # Append the owner probes to the deployment's own trigger set BEFORE the
    # build, so the render the dispatcher loads carries them.
    _append_owner_probe_triggers(repo)

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

    # Before anything is deployed: the render must agree with this module about
    # which roster user holds which tier, and the read-write persona must carry
    # the events panel the owner rows hop through.
    _assert_roster_personas(repo)

    # The repo root's .env is the deployment's whole secret store — the file the
    # worker's env_file (compose template) delivers to the container so
    # inject_provider_env can resolve the provider key, and the file `osprey up`
    # refuses to start without. The compose templates have no token default
    # (they fail closed), and `up` would otherwise auto-generate a random token;
    # we write fixed tokens here so the bearer below is predictable, and pass the
    # provider secret through. The endpoint override rides the same .env: without
    # it the container-side config expansion falls back to the provider's
    # built-in default gateway.
    base_url_line = (
        f"ALS_APG_BASE_URL={os.environ['ALS_APG_BASE_URL']}\n"
        if os.environ.get("ALS_APG_BASE_URL")
        else ""
    )
    # APPENDED, never rewritten: `osprey init` seeded this file and `osprey
    # build` appended the keys the virtual-accelerator containers boot from
    # (VA_CHANNELS_FILE, VA_LATTICE). Rewriting it would drop those, and both
    # VA instances would refuse to start against an unnamed manifest.
    with (repo / ".env").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n# ── e2e fixture ──\n"
            "EVENT_DISPATCHER_TOKEN=dev-token\n"
            "DISPATCH_WORKER_TOKEN=dev-token\n"
            f"ALS_APG_API_KEY={os.environ['ALS_APG_API_KEY']}\n" + base_url_line
        )

    # Narrow the firing user on the host tree the build just provisioned, before
    # anything is deployed: the dispatch worker binds this tree read-only at
    # start, so the record has to exist as a file the container's group can read
    # by the time a job owned by that user attempts its first write.
    _narrow_on_the_host(repo, FIRING_USER)

    # Force a fresh image build so the deployed services run CURRENT source. The
    # dispatcher runs <project>-dispatch:local; the worker runs the unified
    # project image <project>:local (built by `up --dev` from the repo root).
    # Both tags are project-prefixed, derived via resolve_project_name exactly as
    # the compose templates / project build do.
    # `osprey up` does not pass --build to compose, so it would otherwise reuse
    # existing images and silently test stale code. The freshly-built dev wheel
    # invalidates the relevant build-cache layers on rebuild.
    project = resolve_project_name({"project_name": PROJECT_NAME})
    for image in (f"{project}-dispatch:local", f"{project}:local"):
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)

    # The web-container names are host-global fixed identifiers, so a crashed
    # prior run could have left the exact-named containers behind. Remove them
    # by exact name before deploy so this run never adopts a stale container
    # (guardrail: exact-named only, never a prune or wildcard).
    for user in (READONLY_USER, READWRITE_USER):
        subprocess.run(["docker", "rm", "-f", _web_container(user)], capture_output=True, text=True)
    subprocess.run(["docker", "rm", "-f", NGINX_CONTAINER], capture_output=True, text=True)

    try:
        up = _run(
            [str(osprey_bin), "up", "-d", "--dev"],
            cwd=repo,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey up failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{DISPATCHER_URL}/health", HEALTH_TIMEOUT_SEC)
        # The dispatcher being healthy does not mean the worker is: the worker runs
        # the heavier project image and boots later. Gate on its proxied feed so
        # tests never race a still-warming worker (and a worker that never comes up
        # fails here with container logs, not as a bare 502 mid-test).
        _wait_for_worker_feed(HEALTH_TIMEOUT_SEC)
        yield repo
    finally:
        # `osprey down` tears down the services stack AND the web stack
        # (deploy_down_web_terminals); the exact-named sweeps after it are
        # belt-and-suspenders for a down that failed midway. Volumes are
        # deliberately kept by `down`, so this project's own — the per-user
        # terminal volumes and the dispatch workspace alike — are removed
        # exact-named via the shared label-scoped sweep (tests/e2e/_volumes.py);
        # the persona image tags are namespaced by this repo's name (see
        # COEXISTENCE), so removing them cannot untag a real deployment's
        # images.
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        for user in (READONLY_USER, READWRITE_USER):
            subprocess.run(
                ["docker", "rm", "-f", _web_container(user)], capture_output=True, text=True
            )
        remove_project_volumes(resolve_project_name({"project_name": PROJECT_NAME}))
        subprocess.run(["docker", "rm", "-f", NGINX_CONTAINER], capture_output=True, text=True)
        for image in (READONLY_IMAGE, READWRITE_IMAGE):
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)


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
    """Return container state + recent worker/dispatcher logs for failure output.

    A full-Docker e2e that fails mid-run is nearly undiagnosable from the pytest
    traceback alone (the interesting state is inside the containers). Surfacing
    ``docker ps`` plus the tail of both service logs turns an opaque ``502`` into
    an actionable report (e.g. the worker crash-looping vs. merely slow to boot).
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
        logs = subprocess.run(
            ["docker", "logs", "--tail", "40", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        tail = ((logs.stdout or "") + (logs.stderr or "")).strip() or "(no logs)"
        lines.append(f"--- {name} (last 40 log lines) ---\n{tail}")
    return "\n".join(lines)


def _wait_for_worker_feed(timeout: float) -> None:
    """Wait until the dispatcher can proxy the worker run feed (HTTP 200).

    The fixture's ``/health`` gate proves only the DISPATCHER is up, but every run
    assertion reads ``/dashboard/runs``, which the dispatcher proxies to the
    worker. The worker runs the full project image and can take appreciably longer
    to become ready than the dispatcher — until it does, the proxy returns ``502``.
    Gate on a ``200`` here so the test never races a still-warming worker, and a
    genuine worker startup failure surfaces here (with container logs) instead of
    as a bare ``502`` mid-test.
    """
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


def _fire(trigger: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{DISPATCHER_URL}/webhook/{trigger}",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
        assert resp.status == 202, f"{trigger}: expected 202 from webhook, got {resp.status}"
        fired = json.loads(resp.read().decode("utf-8"))
    assert fired.get("dispatched") is True, f"{trigger}: {fired}"


def _worker_artifact_files() -> list[str]:
    """List artifact files the worker persisted to its workspace volume.

    Read directly from the worker container because the dispatcher run feed
    reports only status/tool counts, not whether a real artifact landed. This is
    what distinguishes a genuine ``save-report`` (which must persist via the
    ``mcp__osprey_workspace__`` artifact tool) from a hollow "completed" run that
    only claimed success — see the assertion in ``test_full_stack_dispatch``.
    """
    proc = subprocess.run(
        ["docker", "exec", WORKER_CONTAINER, "ls", f"{WORKER_AGENT_DATA}/artifacts"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.split() if line.endswith(".md")]


def _worker_mcp_surface() -> str:
    """Report whether the worker container actually has the workspace MCP server.

    An empty artifact list has two causes that deserve opposite responses: the
    ``osprey_workspace`` server never reached the container (a real provisioning
    defect), or it was provisioned and the agent simply never called the save
    tool (model behaviour, which is why this module is marked flaky). An empty
    list alone cannot tell them apart, so probe the container and let the failure
    report the cause it actually hit rather than asserting the likelier guess.

    A third cause — the server was provisioned but not yet connected when the
    agent's first turn went out, so the tool was never in its toolset — no
    longer reaches this assertion: the worker refuses such a run as an
    ``infrastructure`` error (see ``sdk_runner._stream_with_ready_mcp``), which
    the status assertion above reports with the server named. The persisted run
    record's ``mcp_servers`` snapshot shows what the barrier saw either way.
    """
    proc = subprocess.run(
        [
            "docker",
            "exec",
            WORKER_CONTAINER,
            "sh",
            "-c",
            f"find /app/{PROJECT_NAME} -maxdepth 2 -name .mcp.json -print -exec cat {{}} +",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip() or "(no output)"
        return f"CAUSE UNDETERMINED: could not probe {WORKER_CONTAINER}: {err}"
    found = proc.stdout.strip()
    if not found:
        return (
            f"CAUSE = provisioning: no .mcp.json anywhere under /app/{PROJECT_NAME}, so "
            "mcp__osprey_workspace__artifact_register never existed and the agent's Write "
            "fallback was denied by the allowlist."
        )
    if "osprey_workspace" not in found:
        return (
            "CAUSE = provisioning: .mcp.json exists but declares no osprey_workspace "
            f"server:\n{found}"
        )
    return (
        "CAUSE = agent behaviour: the workspace MCP server IS provisioned (.mcp.json "
        "declares osprey_workspace), so the save tool existed and the run completed "
        "without calling it. That is a model flake, not a harness defect -- which is "
        "what this module's flaky(reruns=1) covers."
    )


#: Upper bound on each captured blob in a failure message, so a verbose run
#: record or a chatty worker cannot bury the assertion it is attached to.
_EVIDENCE_CAP = 6000


def _docker_capture(argv: list[str]) -> str:
    """Run one ``docker`` command and return its trimmed output, never raising.

    Every helper here runs in a failing assertion's message, where an exception
    would replace the failure it was meant to explain.
    """
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(could not run {' '.join(argv[:3])}: {exc})"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip() or "(no output)"
    if len(out) > _EVIDENCE_CAP:
        out = out[-_EVIDENCE_CAP:]
        out = f"...(truncated to the last {_EVIDENCE_CAP} chars)\n{out}"
    return out


def _worker_run_evidence(run: dict | None) -> str:
    """Return what the worker itself recorded about the ``save-report`` run.

    The ``.mcp.json`` probe above says whether the save tool *could* have
    existed; it cannot say whether the agent saw it or what the run did instead.
    Both are on the worker: the run feed carries the tool count and the agent's
    final text, the persisted record at ``<agent_data>/dispatch/<run_id>.json``
    carries every tool call, and the container log says which MCP servers were
    ready before the first turn. The stack is torn down before any CI diagnostics
    step runs, so a failure that does not carry these here loses them for good.
    """
    lines = ["=== save-report run evidence ==="]
    if run is None:
        lines.append("run feed: no save-report run in the feed")
        return "\n".join(lines)
    feed = {
        key: run.get(key)
        for key in ("run_id", "status", "tool_count", "num_turns", "duration_sec", "error")
    }
    lines.append(f"run feed: {feed}")
    text = (run.get("text_output") or "").strip()
    if text:
        lines.append(f"agent text_output:\n{text[:_EVIDENCE_CAP]}")
    run_id = run.get("run_id")
    if run_id:
        record = _docker_capture(
            [
                "docker",
                "exec",
                WORKER_CONTAINER,
                "cat",
                f"{WORKER_AGENT_DATA}/dispatch/{run_id}.json",
            ]
        )
        lines.append(f"--- persisted run record ({run_id}.json) ---\n{record}")
    listing = _docker_capture(
        ["docker", "exec", WORKER_CONTAINER, "ls", "-la", f"{WORKER_AGENT_DATA}/artifacts"]
    )
    lines.append(f"--- {WORKER_AGENT_DATA}/artifacts ---\n{listing}")
    log_tail = _docker_capture(["docker", "logs", "--tail", "150", WORKER_CONTAINER])
    lines.append(f"--- {WORKER_CONTAINER} (last 150 log lines) ---\n{log_tail}")
    return "\n".join(lines)


def _runs_by_trigger() -> dict[str, dict]:
    """Snapshot the dispatcher run feed keyed by trigger_name (latest wins).

    The dispatcher's /dashboard/runs proxies the worker feed and enriches each
    run with the trigger_name that produced it. It is a bearer-gated read endpoint,
    so the snapshot must send the same EVENT_DISPATCHER_TOKEN written to .env above.
    """
    req = urllib.request.Request(  # noqa: S310
        f"{DISPATCHER_URL}/dashboard/runs",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            runs = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # A transient gateway error means the dispatcher momentarily could not
        # reach the worker upstream (e.g. mid-restart). Treat it as "no feed yet"
        # so the caller's poll loop retries instead of aborting the whole run.
        if exc.code in (502, 503, 504):
            return {}
        raise
    except (urllib.error.URLError, ConnectionError):
        return {}
    by_trigger: dict[str, dict] = {}
    for run in runs:
        name = run.get("trigger_name")
        if not name:
            continue
        # Latest wins by created_at, explicitly. The feed arrives newest-first,
        # so a plain overwrite would keep the OLDEST run per trigger -- and on a
        # flaky() rerun (module fixture not torn down) the poll loop would
        # instantly re-read attempt 1's failed run instead of tracking the run
        # this attempt just fired, making the retry vacuous.
        held = by_trigger.get(name)
        if held is None or (run.get("created_at") or 0) > (held.get("created_at") or 0):
            by_trigger[name] = run
    return by_trigger


def test_full_stack_dispatch(deployed_stack: Path) -> None:
    """All four shipped triggers behave correctly through the real Docker stack."""
    # Fire the three completing triggers + the denied one.
    _fire("hello-dispatch", {})
    _fire("triage-event", _DEMO_PAYLOAD)
    _fire("save-report", _DEMO_PAYLOAD)
    _fire(_DENIED_TRIGGER, {})

    # Poll until each completing trigger has a terminal run.
    deadline = time.monotonic() + RUN_TIMEOUT_SEC
    by_trigger: dict[str, dict] = {}
    while time.monotonic() < deadline:
        by_trigger = _runs_by_trigger()
        if all(
            by_trigger.get(t, {}).get("status") in ("completed", "error")
            for t in _COMPLETING_TRIGGERS
        ):
            break
        time.sleep(3.0)

    # If the feed never converged, attach container state so a CI failure reads as
    # "worker crashed / never produced the run" rather than an opaque status miss.
    converged = all(
        by_trigger.get(t, {}).get("status") in ("completed", "error") for t in _COMPLETING_TRIGGERS
    )
    diag = "" if converged else "\n" + _dump_stack_diagnostics("run feed did not converge")

    for trigger in _COMPLETING_TRIGGERS:
        run = by_trigger.get(trigger)
        assert run is not None, (
            f"{trigger}: no run appeared in the dispatcher feed: {by_trigger}{diag}"
        )
        assert run.get("status") == "completed", (
            f"{trigger}: expected completed, got status={run.get('status')!r} "
            f"error={run.get('error')!r}{diag}"
        )

    # save-report must persist via the workspace artifact tool, not merely report
    # "completed". Without the worker's startup artifact provisioning, .mcp.json is
    # absent, mcp__osprey_workspace__artifact_register does not exist, the agent's
    # Write fallback is denied by the allowlist, and the run hollow-completes with
    # no artifact on disk. Asserting a real .md artifact landed guards that path.
    artifacts = _worker_artifact_files()
    assert artifacts, (
        "save-report completed but no .md artifact was persisted to the worker "
        f"workspace.\n{_worker_mcp_surface()}\n"
        f"{_worker_run_evidence(by_trigger.get('save-report'))}"
    )

    # The denylisted trigger is rejected at the worker /dispatch endpoint BEFORE
    # any run record is created, so it must never surface as a completed run.
    denied = by_trigger.get(_DENIED_TRIGGER)
    assert denied is None or denied.get("status") != "completed", (
        f"denied-tool-demo should be rejected by the denylist, not completed: {denied!r}"
    )


# ---------------------------------------------------------------------------
# Multi-user web tier (T1-T4) — the same deploy stands up nginx + one terminal
# container per roster user from the preset's persona catalog. T1 exercises the
# running topology; T2-T4 are render-layer assertions on the persona projects
# `osprey build` rendered into the repo's build/ zone.
# ---------------------------------------------------------------------------


def _inspect_container(name: str, fmt: str) -> str | None:
    result = subprocess.run(
        ["docker", "inspect", "--type", "container", "-f", fmt, name],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _wait_for_container_health(container: str, timeout: float) -> None:
    """Poll ``.State.Health.Status`` until ``healthy`` or timeout.

    A container can be created and serving before Docker flips its healthcheck
    STATUS off ``starting`` (the healthcheck runs only on its interval, after
    ``start_period``), so an instant equality assert is racy.
    """
    deadline = time.monotonic() + timeout
    last = "(no status yet)"
    while time.monotonic() < deadline:
        if _inspect_container(container, "{{.Id}}") is None:
            last = "(container not present)"
        else:
            last = _inspect_container(container, "{{.State.Health.Status}}") or "(no health field)"
            if last == "healthy":
                return
        time.sleep(2.0)
    raise AssertionError(
        f"{container} did not reach 'healthy' within {timeout:.0f}s (last status: {last!r})"
    )


def _fetch_in_container(container: str, url: str, timeout: float) -> str:
    """Poll ``url`` from INSIDE ``container`` (docker exec + curl) until HTTP
    200 (or timeout); return the response body.

    curl is guaranteed present — the container's own compose healthcheck uses
    it. Probing in-container keeps the check independent of Docker Desktop's
    host-networking setting (see the LANDING_URL comment).
    """
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "exec", container, "curl", "-fsS", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return result.stdout
        last_err = (result.stderr or result.stdout).strip() or f"rc={result.returncode}"
        time.sleep(1.0)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {url} in {container} (last: {last_err})"
    )


def _persona_dir(repo: Path, persona_project: str) -> Path:
    """The rendered persona project directory.

    The catalog pins each ``project_path`` to ``build/<repo>-<persona>``, which
    is where the build renders it: a persona project is build output like every
    other render, so it lives in the repo's OUTPUT zone.
    """
    persona_dir = repo / "build" / persona_project
    assert persona_dir.is_dir(), (
        f"persona project {persona_project!r} was not rendered at {persona_dir}"
    )
    return persona_dir


def _writes_enabled(render: Path) -> bool:
    config = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))
    return bool((config.get("control_system") or {}).get("writes_enabled", False))


def _permissions(render: Path) -> dict:
    settings_path = render / ".claude" / "settings.json"
    assert settings_path.is_file(), f"no settings.json rendered at {settings_path}"
    return json.loads(settings_path.read_text(encoding="utf-8")).get("permissions", {})


def test_web_tier_topology(deployed_stack: Path) -> None:
    """T1: nginx + both per-user containers come up healthy off the overridden
    prefix, both persona images were built locally, and the landing page lists
    both roster users."""
    config = yaml.safe_load((deployed_stack / "build" / "config.yml").read_text(encoding="utf-8"))
    prefix = ((config.get("facility") or {}).get("prefix") or "").strip()
    assert prefix == WEB_PREFIX, (
        f"facility.prefix override did not land in the rendered config: {prefix!r}"
    )

    expected = [NGINX_CONTAINER, _web_container(READONLY_USER), _web_container(READWRITE_USER)]
    missing = [name for name in expected if _inspect_container(name, "{{.Id}}") is None]
    assert not missing, f"container(s) not created by 'osprey up': {missing}"
    for name in expected:
        _wait_for_container_health(name, CONTAINER_HEALTH_TIMEOUT_SEC)

    # Both per-user images were built locally as `<persona-project>-<persona>:local`
    # (build_persona_images) — the readonly and readwrite tiers are genuinely two
    # distinct images, the core promise of the local-mode persona build.
    for image in (READONLY_IMAGE, READWRITE_IMAGE):
        inspect = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, text=True, timeout=15
        )
        assert inspect.returncode == 0, f"persona image {image} was not built by 'osprey up'"

    landing = _fetch_in_container(NGINX_CONTAINER, LANDING_URL, HEALTH_TIMEOUT_SEC)
    for user in (READONLY_USER, READWRITE_USER):
        assert user in landing, f"landing page does not list roster user {user!r}"


def test_readonly_persona_denies_writes(deployed_stack: Path) -> None:
    """T2: the rendered read-only persona project pins writes off and its rendered
    settings.json carries the channel-write tool in permissions.deny — the
    render-time layer that actually enforces the read-only posture. The
    deployment's provider must also have reached the render (the up-time
    credential preflight already depends on it)."""
    readonly_dir = _persona_dir(deployed_stack, READONLY_PROJECT)
    config = yaml.safe_load((readonly_dir / "config.yml").read_text(encoding="utf-8"))
    assert (config.get("claude_code") or {}).get("provider") == "als-apg", (
        "the deployment's `--set provider=als-apg` did not reach the rendered persona project"
    )
    assert _writes_enabled(readonly_dir) is False, (
        "readonly tier must render control_system.writes_enabled: false"
    )
    perms = _permissions(readonly_dir)
    assert CHANNEL_WRITE_TOOL in perms.get("deny", []), (
        f"readonly tier must deny {CHANNEL_WRITE_TOOL!r} in rendered settings.json, "
        f"but deny list is: {perms.get('deny', [])}"
    )


def test_readwrite_persona_arms_writes(deployed_stack: Path) -> None:
    """T3: positive control for T2 — the same render pipeline does NOT deny the
    write tool for the readwrite tier (so T2's deny is a real posture
    difference, not a render that denied the tool everywhere), and the write
    path stays supervised: the tool remains on the ask (human-approval) path."""
    readwrite_dir = _persona_dir(deployed_stack, READWRITE_PROJECT)
    assert _writes_enabled(readwrite_dir) is True, (
        "readwrite tier must render control_system.writes_enabled: true"
    )
    perms = _permissions(readwrite_dir)
    assert CHANNEL_WRITE_TOOL not in perms.get("deny", []), (
        f"readwrite tier must not deny {CHANNEL_WRITE_TOOL!r}, but deny list is: "
        f"{perms.get('deny', [])}"
    )
    assert CHANNEL_WRITE_TOOL in perms.get("ask", []), (
        f"readwrite tier must keep {CHANNEL_WRITE_TOOL!r} on the ask path, but ask "
        f"list is: {perms.get('ask', [])}"
    )


def test_tiers_share_identical_mcp_surface(deployed_stack: Path) -> None:
    """T4: both persona projects declare the IDENTICAL control-system server
    set — that tier boundary is enforcement (``writes_enabled``), never a
    quietly different tool surface.

    The dispatcher wire is the one entry that differs, and it must: it follows
    the persona's EVENTS panel (see the owner-attribution note above), so the
    persona that may fire triggers declares it and the read-only persona — which
    holds no panel to reach it through and no credential to present — does not.
    Both halves are asserted, so neither an extra difference nor a dispatcher
    entry leaking into the read-only render can pass.
    """

    def mcp_server_keys(render: Path) -> set[str]:
        mcp_path = render / ".mcp.json"
        assert mcp_path.is_file(), f"no .mcp.json rendered at {mcp_path}"
        return set(json.loads(mcp_path.read_text(encoding="utf-8")).get("mcpServers", {}))

    readonly_keys = mcp_server_keys(_persona_dir(deployed_stack, READONLY_PROJECT))
    readwrite_keys = mcp_server_keys(_persona_dir(deployed_stack, READWRITE_PROJECT))
    assert DISPATCHER_MCP_SERVER in readwrite_keys, (
        "the read-write persona selects the EVENTS panel, so its render must declare "
        f"the dispatcher wire: readwrite={sorted(readwrite_keys)}"
    )
    assert DISPATCHER_MCP_SERVER not in readonly_keys, (
        "the read-only persona carries no EVENTS panel, so a dispatcher entry in its "
        f"render is a server it cannot reach: readonly={sorted(readonly_keys)}"
    )
    assert readonly_keys == readwrite_keys - {DISPATCHER_MCP_SERVER}, (
        "apart from the dispatcher wire the two tiers must declare the identical MCP "
        "server set (the boundary is writes_enabled, not tool absence): "
        f"readonly={sorted(readonly_keys)} readwrite={sorted(readwrite_keys)}"
    )


# ---------------------------------------------------------------------------
# Owner attribution on the dispatcher wire (O1-O4).
#
# The wire under test runs: an agent in a web-terminal session calls
# ``manual_fire`` through that terminal's own panel proxy at
# ``/panel/events/mcp``; the proxy strips whatever the caller claimed and mints
# ``X-Osprey-Owner`` from the account the container acts as; the dispatcher
# reads that header in ``manual_fire``, hands it to ``fire_callback``, and it
# travels on the worker request body to ``sdk_runner``, which exports it as
# ``OSPREY_CONTROL_OWNER`` into the run's environment. From there the
# connector's owner ladder resolves that person and reads THEIR narrowing out of
# the read-only control-context tree.
#
# Nothing on that path records the owner in a run feed or a history entry, so
# these rows read it where it lands: O1 in the run's own environment, O2 in what
# the run's first control-system write was answered with. O3 is the control —
# the same two probes through the webhook, the cron-shaped door that mints no
# owner — without which O1 and O2 would pass just as well on a wire that stamped
# every run with one name. O4 pins that the hop is credentialed at all.
# ---------------------------------------------------------------------------

#: An MCP client for one ``manual_fire`` call, run INSIDE a web-terminal
#: container.
#:
#: In-container for the same reason the landing-page probe is (see LANDING_URL):
#: the proxy hop is loopback-only by design, and the credential and the port it
#: needs are that container's own environment — so no secret crosses into the
#: test process, and the row goes through the same door a session's agent uses.
#: Streamable HTTP takes three requests (initialize, the initialized
#: notification, then the call) and answers in SSE frames, so the raw bodies come
#: back for the caller to unwrap.
_MANUAL_FIRE_CLIENT = r"""
import json, os, sys, urllib.error, urllib.request

mode, trigger = sys.argv[1], sys.argv[2]
# The event payload, as JSON. The dispatcher folds it into the prompt the
# worker runs, so a probe whose instruction needs a value the trigger file
# could not know is told it here.
payload = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
secret = (os.environ.get("OSPREY_TERMINAL_SECRET") or "").strip()

# Two variables name a port in a per-user terminal container and they are not
# the same question: OSPREY_TERMINAL_WEB_PORT is the DECLARED port the server
# binds (web_cmd.resolve_web_port treats a declaration as authoritative), while
# OSPREY_WEB_PORT is what the rendered .mcp.json points the session's agent at.
# They agree in a healthy render; try the declared one first and fall back, and
# report which one answered so a disagreement is visible rather than guessed at.
candidates = []
for name in ("OSPREY_TERMINAL_WEB_PORT", "OSPREY_WEB_PORT"):
    value = (os.environ.get(name) or "").strip()
    if value and value not in [p for _n, p in candidates]:
        candidates.append((name, value))

port_name, port = (candidates[0] if candidates else ("", ""))
url = "http://127.0.0.1:%s/panel/events/mcp" % port


def post(body, session=None):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if mode == "authenticated":
        headers["X-Osprey-Terminal-Secret"] = secret
    if session:
        headers["Mcp-Session-Id"] = session
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, raw
    except Exception as exc:
        return 0, {}, "%s: %s" % (type(exc).__name__, exc)


INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "osprey-e2e", "version": "0"},
    },
}

out = {"candidates": candidates, "have_secret": bool(secret)}
status, headers, body = 0, {}, "(no port variable set in this container)"
for port_name, port in candidates:
    url = "http://127.0.0.1:%s/panel/events/mcp" % port
    status, headers, body = post(INITIALIZE)
    # Status 0 is a transport failure (nothing listening there yet); any HTTP
    # status means this is the port the terminal serves, refusal included.
    if status:
        break
out["port"] = port
out["port_from"] = port_name
out["initialize"] = {"status": status, "body": body[:2000]}
if not status:
    # Nothing answered on any candidate. Say what IS listening, so the next look
    # starts from the container's own sockets rather than from a guess.
    out["listening"] = os.popen(
        "ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null || true"
    ).read()[:2000]
session = headers.get("mcp-session-id")
if status == 200 and session:
    post({"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
    status, _headers, body = post(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "manual_fire", "arguments": {"name": trigger, "payload": payload}},
        },
        session,
    )
    out["call"] = {"status": status, "body": body[:4000]}
print(json.dumps(out))
"""


def _run_manual_fire_client(mode: str, trigger: str, payload: dict | None = None) -> dict:
    """One invocation of the in-container MCP client."""
    result = subprocess.run(
        ["docker", "exec", _web_container(FIRING_USER), "python3", "-c", _MANUAL_FIRE_CLIENT]
        + [mode, trigger, json.dumps(payload or {})],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"the in-container MCP client failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _manual_fire_through_the_panel_proxy(
    trigger: str, *, authenticated: bool = True, payload: dict | None = None
) -> dict:
    """Call ``manual_fire`` for *trigger* from inside the firing user's terminal.

    Waits for that container to be healthy and for its terminal to answer first.
    The owner rows do not depend on T1 having run: a ``-k`` selection deselects
    it, and the per-user terminals boot well after the dispatcher and the worker
    the module's fixture already gates on — so without this wait the first row to
    run raced a container that had not yet bound its port, and read the race as a
    broken proxy hop.
    """
    _wait_for_container_health(_web_container(FIRING_USER), CONTAINER_HEALTH_TIMEOUT_SEC)
    mode = "authenticated" if authenticated else "anonymous"
    deadline = time.monotonic() + HEALTH_TIMEOUT_SEC
    fired = _run_manual_fire_client(mode, trigger, payload)
    while not fired.get("initialize", {}).get("status") and time.monotonic() < deadline:
        # Status 0 is a transport failure, which at this point means the terminal
        # is still coming up inside a container Docker already calls healthy.
        time.sleep(3.0)
        fired = _run_manual_fire_client(mode, trigger, payload)
    if authenticated and fired.get("initialize", {}).get("status") not in (200, None):
        # The terminal answered and refused. Carry its own log out with the
        # answer: a refusal at this hop is about what that container knows —
        # which panels it resolved, which credential admitted the request — and
        # the stack is torn down before anyone can go and look.
        fired["terminal_log"] = _docker_capture(
            ["docker", "logs", "--tail", "200", _web_container(FIRING_USER)]
        )
    return fired


def _wait_for_terminal_run(trigger: str, timeout: float = RUN_TIMEOUT_SEC) -> dict:
    """Poll the dispatcher feed until *trigger* has a terminal run; return it."""
    deadline = time.monotonic() + timeout
    run: dict = {}
    while time.monotonic() < deadline:
        run = _runs_by_trigger().get(trigger, {})
        if run.get("status") in ("completed", "error"):
            return run
        time.sleep(3.0)
    raise AssertionError(
        f"{trigger}: no terminal run within {timeout:.0f}s "
        f"(last: {run or 'no run in the feed'})\n"
        f"{_dump_stack_diagnostics(f'{trigger} did not converge')}"
    )


def _persisted_run_record(run_id: str) -> dict:
    """The worker's own record of a run — the one artifact carrying tool results.

    The dispatcher feed reports status and a tool COUNT; what a tool ANSWERED
    lives only in ``<agent_data>/dispatch/<run_id>.json``, which the worker
    writes with a ``{name, input, result}`` entry per call. These rows assert on
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


def _owner_run_evidence(trigger: str, run: dict, record: dict | None = None) -> str:
    """Everything a failing owner row needs, gathered before the stack goes away."""
    feed = {key: run.get(key) for key in ("run_id", "status", "tool_count", "error")}
    lines = [f"=== {trigger} run evidence ===", f"feed: {feed}"]
    text = (run.get("text_output") or "").strip()
    if text:
        lines.append(f"agent text_output:\n{text[:_EVIDENCE_CAP]}")
    if record is not None:
        calls = json.dumps(record.get("tool_calls"), indent=2)
        lines.append(f"persisted tool_calls:\n{calls[:_EVIDENCE_CAP]}")
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


def test_manual_fire_owner_reaches_the_dispatch_run(deployed_stack: Path) -> None:
    """O1: a job fired through a terminal's panel proxy runs as that terminal's user.

    The whole wire in one assertion: the proxy minted the owner from the account
    its container acts as, the dispatcher read it off the header, the worker
    accepted it on the request body (an undeclared field would have been dropped
    silently by pydantic), and ``sdk_runner`` exported it — so the run's own
    environment names the firing user.
    """
    fired = _manual_fire_through_the_panel_proxy(ENV_PROBE_PANEL_TRIGGER)
    assert fired.get("initialize", {}).get("status") == 200, (
        f"the MCP handshake through /panel/events/mcp failed: {fired}"
    )
    call = fired.get("call") or {}
    assert call.get("status") == 200, f"manual_fire through the panel proxy failed: {fired}"
    payloads = sse_payloads(call.get("body") or "")
    assert payloads, f"manual_fire returned no SSE payload: {call.get('body')!r}"
    answer = tool_result(payloads)
    assert answer.get("dispatched") is True, (
        f"manual_fire did not dispatch {ENV_PROBE_PANEL_TRIGGER}: {answer}"
    )
    assert answer.get("trigger") == ENV_PROBE_PANEL_TRIGGER, (
        f"manual_fire dispatched {answer.get('trigger')!r} rather than "
        f"{ENV_PROBE_PANEL_TRIGGER!r}: {answer}"
    )

    run = _wait_for_terminal_run(ENV_PROBE_PANEL_TRIGGER)
    record = _persisted_run_record(run["run_id"])
    answers = _tool_answers(record, "__execute")
    evidence = _owner_run_evidence(ENV_PROBE_PANEL_TRIGGER, run, record)
    assert answers, (
        "the panel-fired environment probe never called the python executor, so the "
        f"run's environment was never read.\n{evidence}"
    )
    assert any(f"OWNER={FIRING_USER}" in answer for answer in answers), (
        f"the panel-fired job's environment does not name {FIRING_USER!r}: the owner "
        f"minted on the proxy hop did not reach OSPREY_CONTROL_OWNER in the run.\n{evidence}"
    )


def test_dispatch_owner_absent_for_a_cron_shaped_fire(deployed_stack: Path) -> None:
    """O3a: the webhook door mints nothing, so the job it fires belongs to nobody.

    The control for O1. Without it, a wire that stamped every run with one name
    would satisfy O1 exactly as the real one does.
    """
    _fire(ENV_PROBE_CRON_TRIGGER, {})
    run = _wait_for_terminal_run(ENV_PROBE_CRON_TRIGGER)
    record = _persisted_run_record(run["run_id"])
    answers = _tool_answers(record, "__execute")
    evidence = _owner_run_evidence(ENV_PROBE_CRON_TRIGGER, run, record)
    assert answers, (
        "the webhook-fired environment probe never called the python executor, so the "
        f"run's environment was never read.\n{evidence}"
    )
    assert any(f"OWNER={NO_OWNER_MARKER}" in answer for answer in answers), (
        "a webhook fire carries no owner, so OSPREY_CONTROL_OWNER must be unset in the "
        f"run it dispatches.\n{evidence}"
    )


def test_manual_fire_owner_narrowing_refuses_the_jobs_first_write(deployed_stack: Path) -> None:
    """O2: the firing user's chip gates every write the job they fired attempts.

    The point of carrying the owner at all. The firing user is narrowed to
    read-only on the host tree (the fixture wrote the record the header chip
    writes), the job runs as that user, and its first control-system write is
    refused by the reference monitor before the control system is asked.

    Which refusal is correct is a fact about the HOST rather than about this
    deployment: the tree reaches the container through a group, and a remapped
    bind ownership (Docker Desktop) makes it unreadable, which fails closed with
    ``control_context_unavailable`` instead. The entrypoint's own access probe
    says which, and this row asserts that one — by the clause each arm CLOSES on,
    never by the word "narrowing", which the unavailable arm's prose also carries.
    """
    fired = _manual_fire_through_the_panel_proxy(WRITE_PROBE_PANEL_TRIGGER)
    assert (fired.get("call") or {}).get("status") == 200, (
        f"manual_fire through the panel proxy failed: {fired}"
    )

    run = _wait_for_terminal_run(WRITE_PROBE_PANEL_TRIGGER)
    record = _persisted_run_record(run["run_id"])
    answers = _tool_answers(record, "__channel_write")
    evidence = _owner_run_evidence(WRITE_PROBE_PANEL_TRIGGER, run, record)
    assert answers, (
        "the panel-fired write probe never called the channel-write tool, so no write "
        f"was ever judged.\n{evidence}"
    )
    readable = _control_tree_is_readable_in_the_worker()
    expected = NARROWING_CLAUSE if readable else UNAVAILABLE_CLAUSE
    why = (
        f"{FIRING_USER} is narrowed to read-only, so the write must be refused with the "
        "store's narrowing verdict"
        if readable
        else "the worker's entrypoint reported it cannot read the control-context tree, "
        "so every owned write must fail closed on control_context_unavailable"
    )
    joined = "\n".join(answers)
    assert LIMITS_REFUSAL not in joined, (
        f"the write to {PROBE_CHANNEL} was refused by the limits hook before the tool "
        "ran, so this row judged nothing about the narrowing. Pick a channel this "
        "deployment's data/channel_limits.json marks writable and a value inside its "
        f"range.\n{joined[:_EVIDENCE_CAP]}\n{evidence}"
    )
    assert any(expected in answer for answer in answers), (
        f"{why}. The channel-write tool answered:\n{joined[:_EVIDENCE_CAP]}\n{evidence}"
    )


def test_dispatch_owner_less_write_is_held_to_the_deployment_ceiling(
    deployed_stack: Path,
) -> None:
    """O3b: the control for O2 — an owner-less job reads nobody's narrowing.

    The firing user is narrowed; a cron-shaped fire of the same probe must not
    inherit that narrowing, because the run belongs to nobody and the deployment
    ceiling is what governs it. Asserted as the absence of BOTH store clauses: an
    owner-less run reads no record at all, so neither verdict can be its reason
    for refusing.
    """
    _fire(WRITE_PROBE_CRON_TRIGGER, {})
    run = _wait_for_terminal_run(WRITE_PROBE_CRON_TRIGGER)
    record = _persisted_run_record(run["run_id"])
    answers = _tool_answers(record, "__channel_write")
    evidence = _owner_run_evidence(WRITE_PROBE_CRON_TRIGGER, run, record)
    assert answers, (
        "the webhook-fired write probe never called the channel-write tool, so no write "
        f"was ever judged.\n{evidence}"
    )
    joined = "\n".join(answers)
    for clause in (NARROWING_CLAUSE, UNAVAILABLE_CLAUSE):
        assert clause not in joined, (
            "an owner-less job's write was judged against a recorded write state — it "
            "must be held to the deployment ceiling alone. The channel-write tool "
            f"answered:\n{joined[:_EVIDENCE_CAP]}\n{evidence}"
        )
    # The absence of both clauses is not enough on its own: a probe refused for
    # some unrelated reason — a read-only address, a limits violation, a
    # transport error — carries neither clause and would let this row pass having
    # judged nothing. The control is a control only if the write the narrowed
    # firing user was refused actually SUCCEEDS when nobody owns the run, so the
    # tool's own success envelope is asserted too (channel_write answers
    # ``{"status": "success", "description": "Wrote N channel(s)", ...}``).
    assert LIMITS_REFUSAL not in joined, (
        f"the write to {PROBE_CHANNEL} was refused by the limits hook before the tool "
        "ran, so this row judged nothing. Pick a channel this deployment's "
        "data/channel_limits.json marks writable and a value inside its range.\n"
        f"{joined[:_EVIDENCE_CAP]}\n{evidence}"
    )
    assert any_answer_succeeded(answers), (
        f"the owner-less write to {PROBE_CHANNEL} did not succeed. A cron-shaped fire "
        "belongs to nobody, so no recorded write state governs it and the deployment "
        "ceiling alone decides — which here permits the write. A refusal of ANY kind "
        "means either the ceiling is not what this row assumes or the probe never "
        f"reached the machine.\n{joined[:_EVIDENCE_CAP]}\n{evidence}"
    )


def test_manual_fire_through_the_panel_proxy_needs_a_credential(deployed_stack: Path) -> None:
    """O4: the hop to the dispatcher's MCP transport is gated at the terminal.

    ``/panel/events/mcp`` is the one route in the panel tier that fires work
    rather than arranging a page, and the dispatcher's own bearer never leaves
    the proxy — so an uncredentialed caller inside the container is refused at
    the terminal's gate and never reaches the dispatcher at all.
    """
    fired = _manual_fire_through_the_panel_proxy(ENV_PROBE_PANEL_TRIGGER, authenticated=False)
    assert fired.get("initialize", {}).get("status") == 401, (
        f"an uncredentialed call to /panel/events/mcp must be refused with 401: {fired}"
    )
    assert "call" not in fired, f"an uncredentialed call reached manual_fire: {fired}"


# ---------------------------------------------------------------------------
# The chip's own write (O5).
#
# O2 seeds the firing user's record with ``write_record`` on the host, which
# proves what a lane does with a narrowing but not that the CHIP produces one.
# ---------------------------------------------------------------------------
#: A chip toggle, run INSIDE a web-terminal container.
#:
#: In-container for the reason the manual-fire client is: the terminal's own
#: port and secret are that container's environment, so no per-user secret
#: crosses into the test process and the toggle goes through the same door the
#: header chip uses. ``target=all`` rather than one name: the assertion is about
#: whose record moves, and pinning a target would make the row a fact about
#: which machine this deployment happens to point at.
_CHIP_TOGGLE_CLIENT = r"""
import json, os, urllib.error, urllib.request

secret = (os.environ.get("OSPREY_TERMINAL_SECRET") or "").strip()
# Same two-variable question as the manual-fire client above, answered the same
# way: the DECLARED port first, the agent-facing one as the fallback.
candidates = []
for name in ("OSPREY_TERMINAL_WEB_PORT", "OSPREY_WEB_PORT"):
    value = (os.environ.get(name) or "").strip()
    if value and value not in [p for _n, p in candidates]:
        candidates.append((name, value))

out = {"status": 0, "body": "(no port variable set in this container)"}
for port_name, port in candidates:
    req = urllib.request.Request(
        "http://127.0.0.1:%s/api/terminal/posture" % port,
        data=json.dumps({"target": "all", "posture": "sandbox"}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-Osprey-Terminal-Secret": secret},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = {"status": resp.status, "body": resp.read().decode("utf-8", "replace")}
    except urllib.error.HTTPError as exc:
        out = {"status": exc.code, "body": exc.read().decode("utf-8", "replace")}
    except Exception as exc:
        out = {"status": 0, "body": "%s: %s" % (type(exc).__name__, exc)}
    # Any HTTP status means this is the port the terminal serves, refusal
    # included; only a transport failure is worth trying the next candidate on.
    if out["status"]:
        out["port"], out["port_from"] = port, port_name
        break
print(json.dumps(out))
"""


def _toggle_the_chip_in(user: str) -> dict:
    """POST the chip's own endpoint from inside *user*'s terminal container."""
    _wait_for_container_health(_web_container(user), CONTAINER_HEALTH_TIMEOUT_SEC)
    result = subprocess.run(
        ["docker", "exec", _web_container(user), "python3", "-c", _CHIP_TOGGLE_CLIENT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"the chip client could not run in {_web_container(user)}: "
        f"rc={result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    answer = json.loads(result.stdout.strip().splitlines()[-1])
    if answer.get("status") != 200:
        answer["terminal_log"] = _docker_capture(
            ["docker", "logs", "--tail", "200", _web_container(user)]
        )
    return answer


def _host_record_path(repo: Path, identity: str) -> Path:
    """Where *identity*'s control-context record lives on the HOST tree."""
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    return control_target_identity_dir(config, repo, identity) / RECORD_FILENAME


def test_a_chip_toggle_lands_the_owners_own_record_on_the_host_tree(
    deployed_stack: Path,
) -> None:
    """O5: the chip writes the record a lane reads, per user, group-readable.

    The toggle is made inside the READ-ONLY user's own terminal container,
    through the endpoint the header chip posts to, and the record appears on the
    HOST tree under that user's name. That tier is the honest place for it: the
    fixture already seeded the read-write user's record with ``write_record``
    before the deploy, so a toggle there could not tell a working chip from a
    no-op, and narrowing is the one posture gesture no tier gates — only
    ``writes`` is answered ``writes_disabled``.

    The mode is the assertion, not decoration. The record is written 0640
    because the dispatch worker drops to uid 1000 and reads this file across
    uids through the shared group: under umask 077 a record left at 0600 turns
    every owned lane write into ``control_context_unavailable`` while a CI
    runner at 022 stays green, so a suite that does not pin the mode cannot see
    the failure its own umask hides.

    Per user, not per deployment: the other roster user's record must be exactly
    as it was. Without that half a tree that wrote one shared record would
    satisfy the first half precisely as the real one does — and that other
    record is the narrowing O2 is judged against, so this row also says the chip
    did not quietly widen it.

    Ownership is deliberately not asserted. On a runtime that remaps bind
    ownership (Docker Desktop) the uid on the host is a fact about the id map;
    the mode bits survive it, and they are what the group read depends on.
    """
    other = _host_record_path(deployed_stack, READWRITE_USER)
    before = other.read_bytes() if other.is_file() else None

    # Non-vacuity. A record this toggle did not move would satisfy every
    # assertion below, so the state it starts from is asserted rather than
    # assumed: nothing has narrowed this user yet on a fresh deploy.
    record_path = _host_record_path(deployed_stack, READONLY_USER)
    if record_path.is_file():
        seed = json.loads(record_path.read_text(encoding="utf-8"))
        assert not all(
            (seed.get("posture") or {}).get(target) == POSTURE_SANDBOX for target in CONTROL_TARGETS
        ), (
            f"{READONLY_USER} was already narrowed on every target before the toggle, so "
            f"this row cannot tell a working chip from a no-op: {seed}"
        )

    answer = _toggle_the_chip_in(READONLY_USER)
    assert answer["status"] == 200, (
        f"the chip toggle in {_web_container(READONLY_USER)} was refused: {answer}"
    )

    assert record_path.is_file(), (
        f"the chip toggle answered 200 but wrote no record at {record_path} — the "
        f"terminal's control-context directory is not bound to that host path.\n"
        f"{_docker_capture(['docker', 'logs', '--tail', '200', _web_container(READONLY_USER)])}"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    posture = record.get("posture") or {}
    # ``all`` means "narrow whatever can be narrowed" and reports the rest in
    # ``skipped`` rather than refusing, so a target left armed is a failure only
    # when the terminal did not say why it stayed that way.
    reported = json.loads(answer["body"])
    skipped = {str(row.get("target")) for row in (reported.get("skipped") or [])}
    armed = [target for target in CONTROL_TARGETS if posture.get(target) != POSTURE_SANDBOX]
    assert not [target for target in armed if target not in skipped], (
        f"a [ Sandbox everything ] toggle left {sorted(armed)} armed with no reason "
        f"given for it; the terminal reported skipped={sorted(skipped)}: {record}"
    )
    assert len(armed) < len(CONTROL_TARGETS), (
        f"the toggle narrowed nothing at all — every target is still armed: {record}"
    )
    mode = stat.S_IMODE(record_path.stat().st_mode)
    assert mode == 0o640, (
        f"{record_path} is {oct(mode)}; the record must be 0640 so the worker's "
        "uid 1000 can read it through the shared group — at 0600 every owned lane "
        "write refuses control_context_unavailable on a host whose umask is 077"
    )

    after = other.read_bytes() if other.is_file() else None
    assert after == before, (
        f"{READONLY_USER}'s chip moved {READWRITE_USER}'s record: posture is per "
        f"identity, so one operator narrowing themselves must leave every other "
        f"roster user exactly as they were.\nbefore={before!r}\nafter={after!r}"
    )


# ---------------------------------------------------------------------------
# The owner on a QUEUED PLAN (Q1-Q2).
#
# O1-O3 follow the owner into a run's environment and into the verdict on a
# direct control-system write. A plan is the other thing a job can send toward
# the machine, and it travels by a different carrier: the agent's Bluesky MCP
# server stamps ``X-Osprey-Owner`` on the enqueue, the bridge lifts the name
# onto the item, and the item outlives the run that composed it — so the queue
# a human reads is what still says who put each plan there.
#
# The preset deploys that bridge beside the dispatcher, so these rows need no
# second stack: they compose a plan through the bridge's own draft surface,
# fire the probe through both doors, and read the queue back.
#
# The queue is STOPPED first, and that is a precondition rather than a subject:
# the shipped preset arms the queue at boot, where an add IS the launch and the
# item leaves the queue for the run feed. A stopped queue holds the item where
# the owner can be read off it. Nothing else in this module touches the queue.
# ---------------------------------------------------------------------------


def _queue_snapshot() -> dict[str, Any]:
    status, body = _queue_drive.request(BRIDGE_URL, "/queue", "GET")
    assert status == 200, f"GET /queue failed: {status} {body}"
    return body


def _stop_the_queue() -> None:
    """Disarm the deployment's queue, so an add composes rather than launches.

    The Stop is ungated by design — halting is never something to hold an
    operator at — and on an idle queue it is purely the disarm.
    """
    status, body = _queue_drive.request(BRIDGE_URL, "/queue/stop", "POST", timeout=60.0)
    assert status == 200, f"POST /queue/stop failed: {status} {body}"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not _queue_drive.queue_is_armed(BRIDGE_URL):
            return
        time.sleep(1.0)
    raise AssertionError(
        "the queue was still armed 30s after a Stop, so an add would launch the "
        f"plan instead of holding it: {_queue_snapshot().get('status')}"
    )


def _a_short_plan(repo: Path) -> dict[str, Any]:
    """``grid_scan`` arguments naming this deployment's OWN devices.

    Read back from the device file the build staged and from the deployment's
    own ``channel_limits.json`` rather than authored here: the enqueue is
    validated against the names the worker registered, so a plan composed from
    a hardcoded facility channel would be refused ``unknown_device`` and these
    rows would fail on an address rather than on the owner. The sweep is the
    middle half of the axis's own band, and two points, because nothing here
    ever runs the plan.

    The device NAMES come from the roster the build derived
    (``_orm_stack.staged_devices``); the band VALUES come from the limits
    projection, which gates a subset of those channels and enumerates none of
    them (see ``_orm_stack.channel_limits``). The two are not the same set, so
    the axis is the first staged corrector the limits file actually BOUNDS —
    indexing the projection by the first staged name would raise deep inside a
    deploy, naming nothing about the owner.
    """
    correctors, bpms = _orm_stack.staged_devices(repo)
    assert correctors and bpms, (
        f"the build staged no settable/readable device pair under {repo}, so "
        "there is no plan to compose"
    )
    limits = _orm_stack.channel_limits(repo)
    axis = next(
        (
            (name, entry)
            for name, (setpoint_address, _readback) in correctors.items()
            if isinstance(entry := limits.get(setpoint_address), dict)
            and "min_value" in entry
            and "max_value" in entry
        ),
        None,
    )
    assert axis is not None, (
        "no staged corrector carries a channel_limits band, so this plan has no "
        f"axis to sweep (staged correctors: {sorted(correctors)})"
    )
    axis_name, entry = axis
    low, high = float(entry["min_value"]), float(entry["max_value"])
    return {
        "readbacks": [next(iter(bpms))],
        "axes": [
            {
                "setpoint": axis_name,
                "start": low + 0.375 * (high - low),
                "stop": low + 0.625 * (high - low),
                "num_points": 2,
            }
        ],
    }


def _drain_and_stage(repo: Path) -> int:
    """Empty the pending queue and stage one plan; return its draft revision."""
    _queue_drive.drain_pending_queue(BRIDGE_URL)
    remaining = _queue_snapshot().get("items") or []
    assert not remaining, f"the queue still holds {len(remaining)} item(s) after draining"
    return _queue_drive.stage_draft(
        BRIDGE_URL, "grid_scan", _a_short_plan(repo), client_id=DRAFT_CLIENT_ID
    )


def _one_queued_item_after(trigger: str, run: dict, record: dict) -> dict:
    """The single item *trigger*'s run left on the queue.

    The queue was drained to empty before the fire, so exactly one item is the
    whole claim that the agent's enqueue is what produced it. Any other count
    reports the queue AND what the queue-add tool answered, because a run whose
    add was refused and a run that queued twice are indistinguishable from the
    count alone.
    """
    answers = _tool_answers(record, "__queue_add")
    evidence = _owner_run_evidence(trigger, run, record)
    assert answers, (
        f"the {trigger} probe never called the queue-add tool, so no plan was "
        f"ever enqueued.\n{evidence}"
    )
    joined = "\n".join(answers)
    items = _queue_snapshot().get("items") or []
    assert len(items) == 1, (
        f"the {trigger} probe left {len(items)} item(s) on a queue drained to "
        "empty before the fire; exactly one is what the enqueue produced. The "
        f"queue-add tool answered:\n{joined[:_EVIDENCE_CAP]}\nqueue: {items}\n{evidence}"
    )
    return items[0]


def test_a_panel_fired_run_that_enqueues_a_plan_stamps_the_firing_user(
    deployed_stack: Path,
) -> None:
    """Q1: the plan a job queues carries the name of the human who fired the job.

    The owner's second carrier, end to end on real containers: the proxy minted
    the name from the firing user's own terminal, the dispatcher read it off the
    header, the worker exported it, the agent's Bluesky server stamped it onto
    the enqueue, and the bridge lifted it onto the item. The queue an operator
    reads is where it lands, which is the point of carrying it — an item
    outlives the run that composed it, so the queue is the only surface that can
    still say whose plan this is.

    Attribution is not authorization, and this row is where the difference
    shows: the firing user is narrowed to read-only — the same record O2 judges
    a write against — and the enqueue is permitted anyway, because composing a
    queue moves nothing. What the narrowing withholds is the launch token, and
    a stopped queue needs none.
    """
    _queue_drive.wait_for_worker_environment(BRIDGE_URL)
    _stop_the_queue()
    revision = _drain_and_stage(deployed_stack)

    fired = _manual_fire_through_the_panel_proxy(
        QUEUE_PROBE_PANEL_TRIGGER, payload={"draft_revision": revision}
    )
    assert (fired.get("call") or {}).get("status") == 200, (
        f"manual_fire through the panel proxy failed: {fired}"
    )

    run = _wait_for_terminal_run(QUEUE_PROBE_PANEL_TRIGGER)
    record = _persisted_run_record(run["run_id"])
    item = _one_queued_item_after(QUEUE_PROBE_PANEL_TRIGGER, run, record)
    assert item.get("owner") == FIRING_USER, (
        f"the queued plan names {item.get('owner')!r} as its owner, not "
        f"{FIRING_USER!r}. The job was fired from that user's terminal, so the "
        f"item their agent enqueued is theirs.\nitem: {item}\n"
        f"{_owner_run_evidence(QUEUE_PROBE_PANEL_TRIGGER, run, record)}"
    )


def test_a_cron_shaped_run_queues_a_plan_that_belongs_to_nobody(
    deployed_stack: Path,
) -> None:
    """Q2: the control for Q1 — an owner-less job queues an owner-less plan.

    Without it, a wire that stamped every enqueue with one name would satisfy
    Q1 exactly as the real one does. The webhook door mints nothing, so the run
    it dispatches carries no owner and the item its agent queues names nobody:
    the queue port is the facility's, a plan can reach it from outside OSPREY,
    and such an item is attributed to no one rather than to whoever fired last.

    Asserted as the ABSENCE of the key, which is how the bridge relays an item
    that named nobody — an empty string or a null there would be an owner
    nobody set.
    """
    _queue_drive.wait_for_worker_environment(BRIDGE_URL)
    _stop_the_queue()
    revision = _drain_and_stage(deployed_stack)

    _fire(QUEUE_PROBE_CRON_TRIGGER, {"draft_revision": revision})

    run = _wait_for_terminal_run(QUEUE_PROBE_CRON_TRIGGER)
    record = _persisted_run_record(run["run_id"])
    item = _one_queued_item_after(QUEUE_PROBE_CRON_TRIGGER, run, record)
    assert "owner" not in item, (
        f"a webhook fire carries no owner, so the plan its run queued must name "
        f"nobody — this item names {item.get('owner')!r}.\nitem: {item}\n"
        f"{_owner_run_evidence(QUEUE_PROBE_CRON_TRIGGER, run, record)}"
    )
