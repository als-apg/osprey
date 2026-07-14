"""Full-stack Docker e2e: overlay skills + project ``data/`` are visible in-worker.

This is the authoritative two-party gate for the dispatch-worker image
unification. After unification the worker runs the *project* image
(``proj:local``, built from the project via ``COPY .``) rather than the old lean
``osprey-dispatch:local`` — so the whole project tree, including facility
``.claude`` overlays and the ``data/`` bundle, is baked into the container the
agent actually runs in. A config-only render (the pre-unification behaviour)
would silently drop both, and no mock can falsify that: only a live container
dispatch proves the overlay artifacts and data files reached the agent.

The test mutates a freshly built control-assistant project *before* deploy so it
carries three things that a hollow/degraded worker could not satisfy:

  * an OVERLAY SKILL (``.claude/skills/facility-marker``) that is NOT an OSPREY
    built-in, whose instructions produce an observable marker token;
  * a project ``data/channel_limits.json`` overwritten with a small fixture
    carrying a distinctive sentinel value;
  * a custom ``overlay-visibility`` trigger that routes to the worker, invokes
    the overlay skill, and carries a ``surface_prompt`` fragment.

Firing the trigger and reading the worker's persisted per-run record (its
``tool_calls``, with the tool-call *results* the agent actually loaded
in-container) then proves, deterministically:

  (a) the OVERLAY SKILL was invoked AND present -> a ``Skill`` tool-call
      references facility-marker, and OVERLAY_MARKER (a token that exists only
      inside the baked-in skill file) surfaces in the run;
  (b) the project ``data/`` was read AND present -> a ``Read`` tool-call
      references channel_limits.json, and DATA_SENTINEL (a token that exists only
      inside the baked-in data file) surfaces in the run;
  (c) the ``surface_prompt`` reached the agent -> the surface token surfaces (the
      surface fragment instructs the agent to emit it, so the token appearing is
      the observable proof the fragment landed in the system prompt).

The tool-call proofs (a)/(b) are robust to report formatting — a ``Read`` result
is the file's own bytes; only (c) depends on agent behaviour.

Gating: needs Docker and ``ALS_APG_API_KEY``. We do NOT gate on a host
``claude`` binary — the CLI lives inside the built image, not on the runner.
"""

from __future__ import annotations

import json
import os
import shutil
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

DISPATCHER_URL = "http://localhost:8020"
TOKEN = "dev-token"  # matches the .env tokens written below

# Container image build (Node + Claude CLI install) is slow on a cold cache.
DEPLOY_UP_TIMEOUT_SEC = 900
HEALTH_TIMEOUT_SEC = 180.0
RUN_TIMEOUT_SEC = 300.0

# The unified worker runs the PROJECT image; its in-container project dir and
# artifact dir are named after the project ("proj"), not the old "/app/project".
# The compose template renders the worker container_name as
# ``<project>-dispatch-worker-1`` (services/dispatch_worker/docker-compose.yml.j2),
# so derive both it and the in-container paths from the one project name rather
# than hardcode a host-global name that breaks once the template is namespaced.
PROJECT_NAME = "proj"
WORKER_CONTAINER = f"{PROJECT_NAME}-dispatch-worker-1"
WORKER_ARTIFACT_DIR = f"/app/{PROJECT_NAME}/_agent_data/artifacts"
# The worker persists each completed run (full result incl. tool_calls) here.
WORKER_DISPATCH_DIR = f"/app/{PROJECT_NAME}/_agent_data/dispatch"

# The named volume the worker mounts at ``/app/proj/_agent_data``. Its name is
# ``<compose-project>_<volume>_<replica>`` = ``services`` (the compose files live
# under ``build/services/``) + ``dispatch_workspace`` + ``_1`` (one worker).
#
# ``osprey deploy down`` does NOT remove named volumes, so a volume left by a
# prior deployment survives. The worker runs as a NON-ROOT user and relies on a
# FRESH volume inheriting the image's ``osprey``-owned ``_agent_data`` (see
# ``docker-compose.yml.j2``: the image ``chown``s the tree before ``USER
# osprey``, and Docker copies that ownership into a volume only on first
# population). A volume left root-owned by an older root-writing deployment makes
# the worker's ``_persist_run`` fail with ``PermissionError`` — the run still
# completes, but its per-run JSON never lands, so this test would read an empty
# record and mis-report the agent as having taken no actions. Removing the volume
# before deploy restores the documented fresh-volume precondition.
WORKER_WORKSPACE_VOLUME = "services_dispatch_workspace_1"

# Custom trigger + the observable tokens the overlay skill is instructed to emit.
OVERLAY_TRIGGER = "overlay-visibility"
OVERLAY_MARKER = "OVERLAY_MARKER_OK"  # produced only by the overlay skill
DATA_SENTINEL = "CHANLIM_SENTINEL_7F3A"  # lives only in the project data/ file
SURFACE_TOKEN = "SURFACE=e2e-generic-webhook"  # emitted only if surface_prompt lands

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
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


# SKILL.md for the overlay skill. Its frontmatter mirrors the shape of the
# bundled skills (name / description / allowed-tools). The body instructs the
# agent to perform observable, specific actions that only succeed if the overlay
# and the project data/ file are both present in the container.
_OVERLAY_SKILL_MD = f"""\
---
name: facility-marker
description: >
  Facility-specific end-to-end visibility check. When invoked, read the project
  channel-limits data file and save a report proving the overlay skill and the
  project data bundle are both reachable in the running worker container.
allowed-tools: Read, mcp__osprey_workspace__artifact_save
---

# Facility Marker

When this skill is invoked, do exactly the following, in order:

1. Read the file `data/channel_limits.json` from the working directory.
2. Note the value of its top-level `_sentinel` field.
3. Save a short markdown report using the workspace artifact tool
   (`mcp__osprey_workspace__artifact_save`) with `content` (inline markdown).
   The report body MUST contain, each on its own line and verbatim:
   - the literal token `{OVERLAY_MARKER}`
   - the `_sentinel` value you read from the data file
   - the exact `{SURFACE_TOKEN}` line if your operating instructions told you
     to include a SURFACE token (see your system context)
4. Confirm the artifact you created.
"""

# Small, known-content replacement for the project data/ file. The sentinel is a
# distinctive value that appears NOWHERE else, so its presence in the persisted
# artifact is proof the in-container data/ file was actually read.
_DATA_FIXTURE = {
    "_sentinel": DATA_SENTINEL,
    "defaults": {"writable": True},
    "E2E:MAG:01:CURRENT:SP": {"min_value": 1.0, "max_value": 2.0},
    "E2E:MAG:02:CURRENT:SP": {"min_value": 3.0, "max_value": 4.0},
}

# The surface fragment is a system-prompt addition. It is written so that its
# effect is self-proving: if it reaches the agent, the agent is told to emit the
# SURFACE token, which we then observe in the artifact body.
_SURFACE_PROMPT = (
    "Operational context: you are handling a generic webhook surface. "
    f"Whenever you write a report, include the exact token {SURFACE_TOKEN} "
    "verbatim on its own line in the report body."
)

# The custom trigger: routes to the worker (via the dispatcher.dispatch_target
# already set in the shipped triggers file), invokes the overlay skill, allows
# the tools the skill needs, and carries the surface fragment.
_OVERLAY_TRIGGER_ENTRY = {
    "name": OVERLAY_TRIGGER,
    "source": "webhook",
    "action": {
        "prompt": (
            "Invoke the facility-marker skill now using the Skill tool, and follow "
            "its steps exactly: read the file data/channel_limits.json from the "
            "working directory, then save a markdown report via the workspace "
            "artifact_save tool. Do not skip any step."
        ),
        "surface_prompt": _SURFACE_PROMPT,
        "allowed_tools": [
            "Skill",
            "Read",
            "mcp__osprey_workspace__artifact_save",
        ],
    },
}


def _remove_worker_workspace_volume() -> None:
    """Remove the worker's persisted ``_agent_data`` named volume, if present.

    Guarantees the fresh-volume precondition the non-root worker relies on to own
    (and write) its ``_agent_data`` (see ``WORKER_WORKSPACE_VOLUME``). ``rm -f``
    tolerates an absent volume; it only fails if a live container still holds the
    volume, in which case the post-deploy checks surface the stale-ownership
    symptom with a clear diagnostic rather than the misleading "no tool_calls".
    """
    subprocess.run(
        ["docker", "volume", "rm", "-f", WORKER_WORKSPACE_VOLUME],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _mutate_project(project_dir: Path) -> None:
    """Inject the overlay skill, known data fixture, and custom trigger.

    Applied to the built project *before* ``osprey deploy up`` so the worker
    image (``COPY .``) bakes them in. None of the three are OSPREY built-ins, so
    a config-only / hollow worker could not satisfy the resulting assertions.
    """
    skill_path = project_dir / ".claude" / "skills" / "facility-marker" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(_OVERLAY_SKILL_MD, encoding="utf-8")

    data_path = project_dir / "data" / "channel_limits.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(_DATA_FIXTURE, indent=2), encoding="utf-8")

    # Append the custom trigger to the shipped triggers file, preserving its
    # dispatcher block (which carries the dispatch-worker-1:9190 routing).
    triggers_path = project_dir / "triggers.yml"
    doc = yaml.safe_load(triggers_path.read_text(encoding="utf-8")) or {}
    doc.setdefault("triggers", []).append(_OVERLAY_TRIGGER_ENTRY)
    triggers_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build + mutate + ``osprey deploy up`` a control-assistant stack.

    The mutation (overlay skill + data fixture + custom trigger) is applied
    between ``osprey build`` and ``osprey deploy up`` so the freshly built worker
    image bakes it in.
    """
    if not os.environ.get("ALS_APG_API_KEY"):
        pytest.skip("ALS_APG_API_KEY not set")

    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("dispatch_overlay_build")
    project_dir = base / PROJECT_NAME

    build = _run(
        [
            str(osprey_bin),
            "build",
            PROJECT_NAME,
            "--preset",
            "control-assistant",
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(base),
            "--force",
        ],
        cwd=base,
        timeout=300,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    # Inject the overlay skill, known data fixture, and custom trigger into the
    # built project so the worker image bakes them in.
    _mutate_project(project_dir)

    # The worker's env_file (compose template) delivers this .env's vars so
    # inject_provider_env can resolve the provider key. The compose templates
    # have no token default (they fail closed), and `deploy up` would otherwise
    # auto-generate a random token; we write fixed tokens here so the bearer
    # below is predictable, and pass the provider secret through.
    (project_dir / ".env").write_text(
        "EVENT_DISPATCHER_TOKEN=dev-token\n"
        "DISPATCH_WORKER_TOKEN=dev-token\n"
        f"ALS_APG_API_KEY={os.environ['ALS_APG_API_KEY']}\n",
        encoding="utf-8",
    )

    # Force fresh image builds so the deployed stack runs CURRENT source AND our
    # mutated project. `osprey deploy up --dev` rebuilds these, but removing any
    # stale images first guarantees the worker bakes the mutation above. Ignore
    # failure if an image is absent.
    subprocess.run(["docker", "rmi", "-f", "proj:local"], capture_output=True, text=True)
    subprocess.run(["docker", "rmi", "-f", "osprey-dispatch:local"], capture_output=True, text=True)

    # Drop any stale worker workspace volume so the fresh deploy repopulates it
    # from the image (osprey-owned _agent_data). Without this, a volume left
    # root-owned by an older deployment makes the non-root worker unable to
    # persist run records (see WORKER_WORKSPACE_VOLUME).
    _remove_worker_workspace_volume()

    try:
        up = _run(
            [str(osprey_bin), "deploy", "up", "-d", "--dev"],
            cwd=project_dir,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey deploy up failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{DISPATCHER_URL}/health", HEALTH_TIMEOUT_SEC)
        yield project_dir
    finally:
        down = _run([str(osprey_bin), "deploy", "down"], cwd=project_dir, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey deploy down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        # Leave no stale workspace volume behind so the next local run starts
        # from the documented fresh-volume state (see WORKER_WORKSPACE_VOLUME).
        _remove_worker_workspace_volume()


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


def _fire(trigger: str, payload: dict) -> str:
    """Fire a webhook trigger and return the dispatch_id the dispatcher assigned.

    Tracking THIS fire's dispatch_id (then resolving it to the worker run_id) is
    the only reliable way to assert on the run we caused: the feed accumulates a
    run per fire (including across flaky reruns), so keying by trigger name alone
    is ambiguous.
    """
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
    dispatch_id = fired.get("dispatch_id") or ""
    assert dispatch_id, f"{trigger}: webhook returned no dispatch_id: {fired}"
    return dispatch_id


def _worker_artifact_files() -> list[str]:
    """List ``.md`` artifact files the worker persisted to its workspace volume.

    Read directly from the worker container (the unified project image), whose
    artifact dir is namespaced under the project name (``/app/proj/...``).
    """
    proc = subprocess.run(
        ["docker", "exec", WORKER_CONTAINER, "ls", WORKER_ARTIFACT_DIR],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.split() if line.endswith(".md")]


def _worker_artifact_text() -> str:
    """Concatenate the bodies of all ``.md`` artifacts persisted by the worker.

    The overlay skill is instructed to emit its marker, the data sentinel, and
    the surface token into the report body, so the combined artifact text is the
    single deterministic observable for all three assertions.
    """
    parts: list[str] = []
    for name in _worker_artifact_files():
        proc = subprocess.run(
            ["docker", "exec", WORKER_CONTAINER, "cat", f"{WORKER_ARTIFACT_DIR}/{name}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            parts.append(proc.stdout)
    return "\n".join(parts)


def _dispatcher_get_json(path: str) -> Any:
    """GET a bearer-gated dispatcher endpoint and decode its JSON body.

    The dispatcher read endpoints require the same EVENT_DISPATCHER_TOKEN written
    to .env above.
    """
    req = urllib.request.Request(  # noqa: S310
        f"{DISPATCHER_URL}{path}",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _await_worker_run_id(dispatch_id: str, timeout: float = 45.0) -> str:
    """Resolve a dispatch_id to the worker run_id it produced.

    ``dispatch_to_worker`` returns the worker's 202 body (``run_id``) as soon as
    the worker accepts the prompt — well before the agent finishes — so the pool
    result carries the run_id within a second or two of the fire.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            res = _dispatcher_get_json(f"/dispatch/{dispatch_id}")
        except (urllib.error.URLError, OSError):
            res = {}
        run_id = ((res or {}).get("result") or {}).get("run_id")
        if run_id:
            return run_id
        time.sleep(1.0)
    return ""


def _run_by_id(run_id: str) -> dict:
    """Return the dispatcher-feed entry for a specific worker run_id (or ``{}``)."""
    for run in _dispatcher_get_json("/dashboard/runs"):
        if run.get("run_id") == run_id:
            return run
    return {}


def _clear_worker_artifacts() -> None:
    """Delete stale ``.md`` artifacts so ONLY this run's outputs are asserted on.

    The worker's ``_agent_data`` lives on a NAMED volume that ``osprey deploy
    down`` does not remove, so a prior e2e can leave reports behind and the
    concatenating reader would pick them up. Tolerates a non-zero rc when the
    dir is empty or absent (``rm -f`` + a shell that never fails the exec).
    """
    # Run as root: prior-session artifacts may be owned by a different uid, and a
    # non-root exec would fail to unlink them (silently, under ``|| true``).
    subprocess.run(
        [
            "docker",
            "exec",
            "-u",
            "root",
            WORKER_CONTAINER,
            "sh",
            "-c",
            f"rm -rf {WORKER_ARTIFACT_DIR}/*.md {WORKER_ARTIFACT_DIR}/artifacts.json || true",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _worker_run_record(run_id: str) -> dict:
    """Read the worker's persisted run JSON (full result incl. ``tool_calls``).

    The dispatcher feed only exposes a ``tool_count`` summary, but the worker
    persists the whole result dict — ``tool_calls`` as ``{name, input, result}``
    — to ``<project>/_agent_data/dispatch/<run_id>.json``. That file is the
    deterministic observable this test asserts on: tool-call *results* carry the
    file/skill content the agent actually loaded in-container, independent of
    how the agent formats any report.
    """
    for _ in range(5):  # tiny retry for the persist-after-status window
        proc = subprocess.run(
            ["docker", "exec", WORKER_CONTAINER, "cat", f"{WORKER_DISPATCH_DIR}/{run_id}.json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                pass
        time.sleep(1.0)
    return {}


def _tool_calls_referencing(tool_calls: list[dict], name: str, needle: str) -> list[dict]:
    """Tool calls whose ``name`` matches and whose input JSON contains ``needle``."""
    out = []
    for tc in tool_calls:
        if tc.get("name") != name:
            continue
        if needle in json.dumps(tc.get("input") or {}):
            out.append(tc)
    return out


def _run_haystack(record: dict, artifact_body: str) -> str:
    """Concatenate every text surface a marker token could deterministically land in.

    Includes tool-call inputs AND results (deterministic — a ``Read`` result is
    the file's own bytes, a ``Skill`` result is the loaded skill body), plus the
    agent's ``text_output`` and the persisted artifact body. A sentinel that
    exists only inside a baked-in file can appear here only if that file was
    physically present and read in the running container.
    """
    parts: list[str] = [artifact_body, record.get("text_output") or ""]
    for tc in record.get("tool_calls") or []:
        parts.append(json.dumps(tc.get("input") or {}))
        result = tc.get("result")
        if isinstance(result, str):
            parts.append(result)
    return "\n".join(parts)


def _worker_logs_for(run_id: str) -> str:
    """Tail worker container logs mentioning ``run_id`` (SDK run outcome/errors)."""
    proc = subprocess.run(
        ["docker", "logs", "--tail", "400", WORKER_CONTAINER],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    short = run_id.split("-")[0] if run_id else ""
    hits = [ln for ln in combined.splitlines() if short and short in ln]
    return "\n".join(hits[-15:]) if hits else "(no log lines referenced this run_id)"


def _diagnostics(dispatch_id: str, run: dict, record: dict, artifact_body: str) -> str:
    """Render a decisive evidence dump (surfaced by pytest capture on failure)."""
    run_id = run.get("run_id") or record.get("run_id") or ""
    lines = [
        "=== dispatch overlay diagnostics ===",
        f"dispatch_id={dispatch_id!r} run_id={run_id!r}",
        f"feed run: status={run.get('status')!r} tool_count={run.get('tool_count')!r} "
        f"num_turns={run.get('num_turns')!r} duration_sec={run.get('duration_sec')!r} "
        f"error={run.get('error')!r}",
        f"record: status={record.get('status')!r} num_turns={record.get('num_turns')!r} "
        f"cost_usd={record.get('cost_usd')!r} duration_sec={record.get('duration_sec')!r} "
        f"error={record.get('error')!r}",
        f"record stderr[:600]={(record.get('stderr') or '')[:600]!r}",
    ]
    # Full feed landscape — every run the worker knows about, to expose stale
    # persisted runs vs. this session's fresh fire.
    try:
        feed = _dispatcher_get_json("/dashboard/runs")
        lines.append(f"feed ({len(feed)} runs):")
        for r in feed[:12]:
            lines.append(
                f"  run_id={r.get('run_id')!r} status={r.get('status')!r} "
                f"tool_count={r.get('tool_count')!r} trigger={r.get('trigger_name')!r} "
                f"age_sec={r.get('age_sec')!r}"
            )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never mask the real failure
        lines.append(f"feed fetch failed: {exc!r}")

    tcs = record.get("tool_calls") or []
    lines.append(f"persisted tool_calls ({len(tcs)}):")
    for i, tc in enumerate(tcs):
        result = tc.get("result")
        rsnip = result[:400] + "…" if isinstance(result, str) and len(result) > 400 else result
        lines.append(
            f"  [{i}] name={tc.get('name')!r} input={json.dumps(tc.get('input') or {})[:300]} "
            f"result={rsnip!r}"
        )
    lines.append(f"text_output[:500]={(record.get('text_output') or '')[:500]!r}")
    lines.append(f"artifact_body[:800]={artifact_body[:800]!r}")
    skills = subprocess.run(
        ["docker", "exec", WORKER_CONTAINER, "ls", "-la", f"/app/{PROJECT_NAME}/.claude/skills"],
        capture_output=True,
        text=True,
    )
    lines.append(f"skills dir:\n{skills.stdout or skills.stderr}")
    md = subprocess.run(
        ["docker", "exec", WORKER_CONTAINER, "sh", "-c", f"find {WORKER_ARTIFACT_DIR} -type f"],
        capture_output=True,
        text=True,
    )
    lines.append(f"artifact files:\n{md.stdout or md.stderr}")
    lines.append(f"worker logs for run:\n{_worker_logs_for(run_id)}")
    return "\n".join(lines)


@pytest.mark.flaky(reruns=2)
def test_overlay_skill_and_data_visible_in_worker(deployed_stack: Path) -> None:
    """A dispatched agent can invoke an overlay skill and read a project data file.

    Proves the unified worker image carries the facility ``.claude`` overlays and
    the project ``data/`` bundle. The primary observables are the worker's
    persisted per-run ``tool_calls`` (deterministic: a tool-call *result* is the
    content the agent actually loaded in-container, not agent-authored prose):

      (a) overlay skill invoked  -> a ``Skill`` tool-call references facility-marker,
          and OVERLAY_MARKER (a token that exists ONLY in the baked-in skill file)
          surfaces in the run — proving the overlay was present AND loaded;
      (b) project data/ was read -> a ``Read`` tool-call references
          channel_limits.json, and its result carries DATA_SENTINEL (a token that
          exists ONLY in the baked-in data file) — proving data/ was present and read;
      (c) surface_prompt landed  -> the SURFACE token (which the surface fragment
          instructs the agent to emit) surfaces in the run.

    Assertions (a)/(b) never depend on report formatting; (c) is the one
    behaviour-dependent signal (the only observable effect of a system-prompt
    fragment is the agent emitting the token), retained as a real end-to-end proof.
    """
    # Isolate this run: the worker _agent_data lives on a named volume that
    # `osprey deploy down` does not clear, so wipe stale artifacts first.
    _clear_worker_artifacts()

    # Fire and track THIS dispatch by id -> worker run_id, so a stale/other run in
    # the feed can never be mistaken for the run we caused.
    dispatch_id = _fire(OVERLAY_TRIGGER, {})
    run_id = _await_worker_run_id(dispatch_id)
    assert run_id, f"{OVERLAY_TRIGGER}: dispatch {dispatch_id!r} never yielded a worker run_id"

    # Poll THIS run_id until it reaches a terminal state.
    deadline = time.monotonic() + RUN_TIMEOUT_SEC
    run: dict = {}
    while time.monotonic() < deadline:
        run = _run_by_id(run_id)
        if run.get("status") in ("completed", "error"):
            break
        time.sleep(3.0)

    record = _worker_run_record(run_id)
    artifact_body = _worker_artifact_text()
    diag = _diagnostics(dispatch_id, run, record, artifact_body)
    print(diag)  # noqa: T201 - surfaced by pytest capture on failure

    assert run, f"{OVERLAY_TRIGGER}: run {run_id!r} never appeared in the dispatcher feed\n{diag}"

    assert run.get("status") == "completed", (
        f"{OVERLAY_TRIGGER}: expected completed, got status={run.get('status')!r} "
        f"error={run.get('error')!r}\n{diag}"
    )

    tool_calls = record.get("tool_calls") or []
    # Distinguish a persist failure from a genuinely empty run: if the dispatcher
    # feed saw tool activity but the persisted record is empty, the agent DID act
    # and only the on-disk record is missing — almost always a PermissionError
    # writing WORKER_DISPATCH_DIR on a stale root-owned volume. Call that out
    # explicitly so it is never again mistaken for "the agent took no actions".
    if not tool_calls and run.get("tool_count"):
        pytest.fail(
            f"run {run_id!r} completed with tool_count={run.get('tool_count')!r} in "
            f"the dispatcher feed, but its persisted record at {WORKER_DISPATCH_DIR} "
            "has no tool_calls: the worker failed to PERSIST the run (typically a "
            f"PermissionError writing {WORKER_DISPATCH_DIR} because the "
            f"{WORKER_WORKSPACE_VOLUME} volume was left root-owned by a prior "
            "deployment). The agent acted; the record is the problem — check the "
            f"worker logs for 'Failed to persist run'.\n{diag}"
        )
    assert tool_calls, (
        f"run completed but no tool_calls were persisted for run {run_id!r} at "
        f"{WORKER_DISPATCH_DIR} — the agent took no actions in-container.\n{diag}"
    )

    haystack = _run_haystack(record, artifact_body)

    # (a) Overlay skill invoked AND present. The Skill tool-call proves the agent
    #     invoked facility-marker; OVERLAY_MARKER (a token defined ONLY inside the
    #     baked-in skill file) surfacing anywhere in the run proves the skill file
    #     was physically present in the container and its body was loaded — a
    #     config-only / hollow worker would carry no such skill.
    skill_calls = _tool_calls_referencing(tool_calls, "Skill", "facility-marker")
    assert skill_calls, (
        "no Skill tool-call referencing 'facility-marker' was recorded; the overlay "
        f"skill was not invoked.\n{diag}"
    )
    assert OVERLAY_MARKER in haystack, (
        f"overlay marker {OVERLAY_MARKER!r} (defined only in the baked-in skill file) "
        f"did not surface; the overlay skill was not present/loaded in-container.\n{diag}"
    )

    # (b) Project data/ read AND present. The Read tool-call proves the agent read
    #     channel_limits.json; DATA_SENTINEL (a token that lives ONLY in the baked-in
    #     data file) surfacing proves the file was physically present and its content
    #     was returned to the agent.
    read_calls = _tool_calls_referencing(tool_calls, "Read", "channel_limits.json")
    assert read_calls, (
        "no Read tool-call referencing 'channel_limits.json' was recorded; the "
        f"project data/ file was not read.\n{diag}"
    )
    assert DATA_SENTINEL in haystack, (
        f"data sentinel {DATA_SENTINEL!r} (present only in the baked-in data file) did "
        f"not surface; data/channel_limits.json was not readable in-container.\n{diag}"
    )

    # (c) surface_prompt landed. The fragment instructs the agent to emit the SURFACE
    #     token; its presence is the only observable effect that the fragment reached
    #     the system prompt. Behaviour-dependent, but a real end-to-end signal.
    assert SURFACE_TOKEN in haystack, (
        f"surface token {SURFACE_TOKEN!r} did not surface; the trigger surface_prompt "
        f"did not reach the agent's system prompt.\n{diag}"
    )
