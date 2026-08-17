"""Real-container sandbox-escape e2e (task 2.10) -- the authoritative proof of
the authoring-sandbox feature's central invariant: an agent-authored
session-tier plan can never reach real hardware unless it is validated AND
that exact validated content is what actually launches.

Deploys the VA-backed turn-key plan-stack (``tests/e2e/_orm_stack.py`` -- the
same real Virtual Accelerator + bluesky-bridge container pair
``test_orm_roundtrip.py`` uses) and drives the session-authoring HTTP surface
(``POST /plans/session``, ``POST /plans/validate``, then the queue path
``PATCH /draft`` -> ``POST /queue/items`` -> ``POST /queue/start``) end to end
against it, then reads a real corrector setpoint back
over Channel Access -- from this test process, independent of the bridge --
to prove a rejected write never lands, not merely that the bridge's own HTTP
responses claim it didn't.

Mocked-client tests (``tests/services/bluesky_bridge/test_plan_validation.py``,
``test_session_load_gate.py``, ``test_launch_validation_gate.py``) only
exercise OSPREY's own half of this contract in-process. This is the other
half: a real deployed bridge container, a real deployed IOC, and an
independent CA read that never goes through the bridge at all.

CRITICAL INTEGRATION CONTRACT (see ``plan_validation.py``'s module docstring
and the P5 Phase 2 research digest): the bytes ``validate_plan``
hashes for its validation record must be byte-identical to what the session
directory's load gate and the ENQUEUE gate (validation.py, called
from the queue's ``POST /queue/items``) re-hash from
disk. ``POST /plans/session`` writes the body once; ``POST /plans/validate``
re-reads and hashes that SAME file -- never a body passed separately -- so
"validated bytes == file bytes" is structural here, not a test convention
this e2e has to arrange for itself.

LOAD-BEARING SECURITY ASSUMPTION -- read before touching this file's
obfuscation-residual test: the stage-1/stage-2 static validator (AST import
walk + CA/connector substring-and-regex pattern scan) is NOT a containment
boundary. A sufficiently obfuscated body (reflected ``__import__``, a
getattr'd/concatenated attribute name for the call itself) can evade both
stages by construction -- neither stage's source text or AST ever contains a
literal "epics"/"caput" token for such a body. This is the DOCUMENTED,
ACCEPTED residual (task 2.1's ``TestKnownObfuscationResidual``, xfail,
strict) -- the real backstop for this exact case is human approval
RENDERING THE PLAN SOURCE at launch (task 2.6), not this validator, not the
session-layer load gate, not this test. ``test_obfuscated_residual_is_a_
documented_known_uncaught_case`` below records that case; it does NOT assert
the sandbox catches it -- doing so would misrepresent what the feature
actually guarantees.

Container safety: every docker invocation here names an exact
container/image -- never a wildcard, never ``system prune``/``--volumes``.
Teardown goes through ``osprey down``, matching every other e2e in
this directory. ``BRIDGE_PORT`` below is distinct from every sibling e2e
module's pinned port; the VA's Channel Access port is NOT freely overridable
(see ``_orm_stack.VA_CA_PORT``'s docstring) so this test shares that fixed
port with ``test_orm_roundtrip.py``/``test_va_substrate_equivalence.py`` --
safe sequentially (each tears its own container down by exact name before
the next starts), not intended to run concurrently with them on one host.

Gating: needs Docker; the VA image is amd64-only (PyAT/softioc have no
aarch64 wheels), so it builds/boots under QEMU emulation on Apple Silicon --
as heavy as ``test_va_substrate_equivalence.py``. Advisory CI lane (see
ci.yml's ``bluesky-sandbox-escape-e2e`` job); run locally with
``E2E_REUSE_IMAGES=1`` set for fast iteration once the image cache is warm.

GAP FOUND WHILE WRITING THIS E2E, NOW FIXED: ``plan_validation.py``'s stage-1
``_ALLOWED_TOP_LEVEL_MODULES`` originally never added ``typing``/
``__future__``/``logging`` -- the shipped ``plans_core/orm.py``
exemplar this test's positive plan body mirrors uses all three
(`from __future__ import annotations`, `from typing import Any`,
`import logging`), so an author copying that exact, idiomatic house style
verbatim (as the writing-bluesky-plans skill's own format spec shows) would
have had an otherwise entirely benign plan rejected at stage 1 for reasons
unrelated to control-system safety. That allowlist gap has since landed
(task 2.1's `_ALLOWED_TOP_LEVEL_MODULES` now includes all three) --
``_POSITIVE_PLAN_BODY`` below uses the full idiomatic style unmodified.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.e2e import _orm_stack, _queue_drive
from tests.e2e._deploy_diagnostics import queue_stack_logs

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

# Distinct from every sibling e2e module's pinned bridge port (_orm_stack.py's
# 18102, test_bluesky_deploy.py's 18090, test_va_substrate_equivalence.py's
# 18099, test_tiled_roundtrip.py's 18101, test_bluesky_catalog_e2e.py's 18103).
BRIDGE_PORT = 18105
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

#: Compose project this suite deploys under. Container names follow
#: ``<project>-<service>``, so anything naming a deployed container derives it
#: from here rather than repeating the literal.
PROJECT_NAME = "sandbox-escape"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 1200  # amd64-emulated VA image build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0
SCAN_TIMEOUT_SEC = 120.0

# Three correctors total: one reserved exclusively as the escape/residual
# probe TARGET (never launched in this test, by either the negative
# or the obfuscation-residual case), two driven for real by the positive
# author -> validate -> enqueue -> read round trip. Disjoint by
# construction, so a run-order change can never let the positive run's
# legitimate write be mistaken for evidence the negative case's write landed.
CORRECTOR_COUNT = 3
BPM_COUNT = 2

# Within the corrector channel_limits band (+-12A, see
# tests/va/e2e/test_limits_enforcement.py) but far from the identity-state
# baseline (0.0 A) -- unambiguous if it ever actually lands.
POISON_CURRENT = 5.0

SPAN_A = 2.0
NUM_POINTS = 3

_ESCAPE_PLAN_NAME = "sandbox_escape_probe"
_RESIDUAL_PLAN_NAME = "sandbox_escape_obfuscated_residual"
_POSITIVE_PLAN_NAME = "session_orbit_probe"


# ---------------------------------------------------------------------------
# Malicious / residual plan bodies
# ---------------------------------------------------------------------------
def _escape_plan_body(target_sp: str, poison_current: float) -> str:
    """MUST-CATCH plan body: reaches raw Channel Access AT MODULE SCOPE.

    ``epics.caput(...)`` sits directly at module level, outside ``build_plan``
    entirely -- if the session-tier LOAD gate (task 2.4) ever ``exec_module``'d
    this file despite it lacking a passing validation record, the poison write
    would fire on the very first ``GET /plans`` call that re-scans the session
    directory (``get_facility_plans()`` re-scans on every call), with no
    launch needed at all. This also makes ``import epics`` itself
    task 2.1's REJECT case (its unit table asserts against exactly this
    import) -- caught at stage 1, long before any exec is even attempted.
    """
    return f'''"""MUST-CATCH plan body for the sandbox-escape e2e
(tests/e2e/test_bluesky_sandbox_escape_e2e.py). Never meant to run -- see this
test module's docstring."""

from __future__ import annotations

from typing import Any

import epics
from pydantic import BaseModel

# Fires the instant this module is ever exec'd -- not gated behind
# build_plan at all, so a load-gate bypass would be provable without this
# test ever needing to launch a run.
epics.caput({target_sp!r}, {poison_current!r})


class PARAMS(BaseModel):
    """No parameters needed -- this body never legitimately runs."""


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    yield from ()
'''


def _obfuscated_residual_plan_body(target_sp: str, poison_current: float) -> str:
    """KNOWN-UNCAUGHT residual plan body -- see this module's docstring.

    Extends task 2.1's exact documented obfuscation
    (``test_plan_validation.py::TestKnownObfuscationResidual``, xfail,
    strict) -- a getattr/string-concatenation reflected ``__import__("epics")``
    call, which neither stage 1's AST import walk nor stage 2's
    substring/regex pattern scan ever see (no literal ``import`` statement,
    no "epics." substring) -- one step further, ALSO reaching the write
    itself via a getattr'd, concatenated attribute name (``"ca" + "put"``) so
    the literal substring "caput(" stage 2 scans for never appears in this
    source either. Deliberately NOT asserted refused anywhere in this test
    module -- see ``test_obfuscated_residual_is_a_documented_known_uncaught_case``.
    """
    return f'''"""KNOWN-UNCAUGHT residual plan body for the sandbox-escape e2e
(tests/e2e/test_bluesky_sandbox_escape_e2e.py). Documented, accepted residual --
see this test module's docstring. Never launched by this test."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PARAMS(BaseModel):
    """No parameters needed -- this body never legitimately runs."""


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    _builtins = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    _import_name = "".join(["__", "imp", "ort", "__"])
    _module_name = "".join(["ep", "ics"])
    _reflected = _builtins[_import_name](_module_name)
    _write = getattr(_reflected, "".join(["ca", "put"]))
    _write({target_sp!r}, {poison_current!r})
    yield from ()
'''


# orm-shaped (mirrors plans_core/orm.py's PARAMS/
# build_plan in spirit, and now also its typing/logging imports verbatim --
# see the module docstring's "GAP FOUND ... NOW FIXED" note): device-agnostic,
# resolves correctors/readbacks by string name against whatever `devices`
# dict the bridge passes in. Authored via write_plan (which prepends
# the generated PLAN_METADATA block), so only the author's own body -- no
# PLAN_METADATA -- lives here.
#
# SEPARATE, STRUCTURAL CONSTRAINT (found running this e2e for real, distinct
# from the now-fixed allowlist gap): unlike `plans_core/orm.py`
# (a shipped file, never passed through this prepending), this body's own
# text is NOT position 0 in the file `write_session_plan` actually writes --
# `POST /plans/session` assembles `f"PLAN_METADATA = {metadata!r}\\n\\n{body}"`,
# so the generated `PLAN_METADATA` assignment always sits ahead of it. Python
# requires `from __future__ import ...` to be the file's first statement
# (only a docstring/comments may precede it) -- with PLAN_METADATA occupying
# that slot, ANY session-authored body containing `from __future__ import
# annotations` fails stage 3's dry-run with a SyntaxError, regardless of the
# allowlist. This is an inherent consequence of the metadata-prepending
# design, not a bug worth fixing (`list[str]`/`dict[str, Any]` hints work
# natively on Python 3.9+ without it -- see PEP 585), so this body simply
# omits the future import; `typing`/`logging` have no such positional rule
# and are used exactly as the shipped exemplar does.
#
# What this body mirrors is the exemplar's IMPORTS and module shape -- the
# only thing the stage-1 allowlist this test exists to probe can see. It does
# not carry the shipped plan's read-relative-restore sweep idiom, which adds
# no import and so is invisible to that allowlist.
_POSITIVE_PLAN_BODY = '''"""Session-authored positive plan body for the sandbox-escape e2e
(tests/e2e/test_bluesky_sandbox_escape_e2e.py) -- mirrors plans_core/
orm.py's PARAMS/build_plan, proving the author -> validate ->
enqueue -> read path works end to end for a legitimately-authored
session plan, in the same deployed stack the negative case runs against.
"""

import logging
from typing import Any

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class PARAMS(BaseModel):
    correctors: list[str] = Field(..., min_length=1)
    readbacks: list[str] = Field(..., min_length=1)
    span_a: float = Field(..., gt=0, le=10.0)
    num: int = Field(..., ge=3)

    @model_validator(mode="after")
    def _disjoint(self) -> "PARAMS":
        overlap = set(self.correctors) & set(self.readbacks)
        if overlap:
            raise ValueError(f"correctors and readbacks must be disjoint (overlap: {sorted(overlap)})")
        return self


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    correctors = [(name, devices[name]) for name in params.correctors]
    corrector_devices = [corrector for _, corrector in correctors]
    readback_devices = [devices[name] for name in params.readbacks]
    step = (2 * params.span_a) / (params.num - 1)
    currents = [-params.span_a + i * step for i in range(params.num)]
    all_devices = corrector_devices + readback_devices

    @bpp.stage_decorator(all_devices)
    @bpp.run_decorator()
    def _sweep():
        for name, corrector in correctors:
            try:
                for current in currents:
                    yield from bps.mv(corrector, current)
                    yield from bps.trigger_and_read(all_devices)
            finally:
                try:
                    yield from bps.mv(corrector, 0.0)
                except Exception:
                    logger.warning("failed to restore corrector %s to 0", name, exc_info=True)

    return _sweep()
'''


# ---------------------------------------------------------------------------
# HTTP helpers (mirrors test_orm_roundtrip.py / test_bluesky_catalog_e2e.py)
# ---------------------------------------------------------------------------
def _get(path: str) -> tuple[int, Any]:
    req = urllib.request.Request(f"{BRIDGE_URL}{path}", method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(path: str, body: dict, headers: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        f"{BRIDGE_URL}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _patch(path: str, body: dict) -> tuple[int, Any]:
    """A PATCH against the bridge (the shared draft is the only PATCH surface)."""
    req = urllib.request.Request(  # noqa: S310
        f"{BRIDGE_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="PATCH",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _refusal_code(body: Any) -> Any:
    """The machine-readable code off a refusal body (``{"detail": {"code": ...}}``).

    ``None`` for any other shape, so an assertion reports the drifted body
    rather than a ``KeyError`` from inside this helper.
    """
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail.get("code") if isinstance(detail, dict) else None


def _enqueue_session_plan(plan_name: str, plan_args: dict) -> tuple[int, Any]:
    """Stage ``plan_name`` in the shared draft and try to enqueue it.

    The queue path takes plan name and args from the server-side draft snapshot
    at a pinned revision, never from the enqueue body — so reaching the enqueue
    gate at all means going through the draft first.
    """
    status, patched = _patch(
        "/draft",
        {
            "plan_name": plan_name,
            "plan_args_patch": plan_args,
            "client_id": "sandbox-escape-e2e",
        },
    )
    assert status == 200, f"PATCH /draft failed for {plan_name!r}: {status} {patched}"
    return _post("/queue/items", {"draft_revision": patched["revision"]})


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


def _minted_token(repo: Path) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = repo / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_LAUNCH_TOKEN")
    assert token, "BLUESKY_LAUNCH_TOKEN missing/empty in the deployment repo's .env"
    return token


def _channel_limits(repo: Path) -> dict[str, Any]:
    """The limits database the deployed containers actually read.

    ``control_system.limits_checking.database_path`` resolves against
    ``CONFIG_FILE``'s directory, and the render points that at
    ``<repo>/build/config.yml`` -- so the render's copy, not the operator-owned
    source under ``<repo>/data/``, is the file whose channels the containers see.
    """
    return json.loads((repo / "build" / "data" / "channel_limits.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Real EPICS CA probe -- from this test PROCESS, never through the bridge.
# Subprocess-based (mirrors tests/va/e2e/conftest.py's `_readiness_pv_served`):
# an in-process pyepics CA context in the main pytest process can deadlock
# unrelated executor-thread CA calls elsewhere in the same run, so every read
# below runs out-of-process against the VA's published CA port.
# ---------------------------------------------------------------------------
def _caget(address: str, *, timeout: float = 5.0) -> float | None:
    code = (
        "import sys, epics\n"
        f"v = epics.caget({address!r}, timeout={timeout!r}, connection_timeout={timeout!r})\n"
        "sys.stdout.write(repr(v))\n"
    )
    env = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{_orm_stack.VA_CA_PORT}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    env.pop("EPICS_CA_ADDR_LIST", None)
    env.pop("EPICS_CA_SERVER_PORT", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout + 10.0,
        env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"caget({address!r}) subprocess failed (rc={proc.returncode}): "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return ast.literal_eval(proc.stdout.strip())


# ---------------------------------------------------------------------------
# Fixture: deploy the VA-backed stack once, shared by every test below.
# ---------------------------------------------------------------------------
@dataclass
class DeployedSandboxStack:
    repo: Path
    escape_target_sp: str
    escape_target_rb: str
    positive_correctors: dict[str, tuple[str, str]]
    positive_bpms: dict[str, str]


@pytest.fixture(scope="module")
def deployed_sandbox_stack(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DeployedSandboxStack]:
    base = tmp_path_factory.mktemp("sandbox_escape_build")
    # The deployment REPO: `osprey up` runs here, `.env` lives here, and the
    # render `osprey build` produced is `<repo>/build`.
    repo = _orm_stack.build_project_subprocess(
        PROJECT_NAME, output_dir=base, bridge_port=BRIDGE_PORT, timeout=BUILD_TIMEOUT_SEC
    )

    limits = _channel_limits(repo)
    correctors = _orm_stack.select_correctors(limits, count=CORRECTOR_COUNT)
    bpms = _orm_stack.select_bpms(limits, count=BPM_COUNT)
    escape_name, (escape_sp, escape_rb) = sorted(correctors.items())[0]
    positive_correctors = {name: pair for name, pair in correctors.items() if name != escape_name}
    # Writes the repo root's `.env` — the deployment's whole secret store, and
    # the file `osprey up` refuses to start without.
    _orm_stack.write_substrate_env(repo, correctors=correctors, bpms=bpms)

    osprey_bin = _orm_stack.find_osprey_console_script()

    # Force fresh --dev builds so the deployed containers run CURRENT source
    # (osprey up does not pass --build to compose, so it would otherwise
    # reuse a stale cached image). Exact-named images only.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(
            ["docker", "rmi", "-f", _orm_stack.va_image("sandbox-escape")],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["docker", "rmi", "-f", _orm_stack.bridge_image("sandbox-escape")],
            capture_output=True,
            text=True,
        )

    try:
        up = subprocess.run(
            [str(osprey_bin), "up", "-d", "--dev"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        # HTTP readiness is not enqueue readiness -- the worker namespace an
        # enqueue validates against exists only once the RE worker environment
        # is open, and the bridge opens that off the readiness path. See
        # `_queue_drive.wait_for_worker_environment`.
        try:
            _queue_drive.wait_for_worker_environment(BRIDGE_URL)
        except AssertionError as exc:
            pytest.fail(f"{exc}\n{queue_stack_logs(_orm_stack.project_prefix(PROJECT_NAME))}")
        yield DeployedSandboxStack(
            repo=repo,
            escape_target_sp=escape_sp,
            escape_target_rb=escape_rb,
            positive_correctors=positive_correctors,
            positive_bpms=bpms,
        )
    finally:
        down = subprocess.run(
            [str(osprey_bin), "down"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )


# ---------------------------------------------------------------------------
# Negative (MUST-CATCH, strict): the escape plan is refused by all three
# gates, and the real corrector it targets never moves.
# ---------------------------------------------------------------------------
def test_sandbox_escape_is_caught_and_no_write_reaches_the_ioc(
    deployed_sandbox_stack: DeployedSandboxStack,
) -> None:
    """Every gate refuses ``sandbox_escape_probe``, and its target corrector
    never moves -- read directly over Channel Access, independent of the
    bridge's own claims.
    """
    target_sp = deployed_sandbox_stack.escape_target_sp
    sp_baseline = _caget(target_sp)
    assert sp_baseline is not None, f"could not read baseline {target_sp!r} over CA"

    # --- author: writes a file, never execs it ---
    status, body = _post(
        "/plans/session",
        {
            "name": _ESCAPE_PLAN_NAME,
            "description": "MUST-CATCH sandbox-escape probe (never meant to run)",
            "category": "test",
            "required_devices": [],
            "writes": True,
            "body": _escape_plan_body(target_sp, POISON_CURRENT),
        },
    )
    assert status == 200, f"POST /plans/session failed: {status} {body}"

    # --- gate (a): validate_plan rejects it (stage 1: import epics) ---
    status, body = _post("/plans/validate", {"name": _ESCAPE_PLAN_NAME})
    assert status == 200, f"POST /plans/validate failed: {status} {body}"
    assert body["passed"] is False, f"escape plan validation unexpectedly passed: {body}"
    assert any("epics" in reason for reason in body["reasons"]), (
        f"expected an 'epics' import rejection reason, got: {body['reasons']}"
    )

    # --- gate (b): the session-tier LOAD gate never exec_module's it, so it
    # never appears in GET /plans (checked twice -- get_facility_plans()
    # re-scans the session directory on every call, so a gate that only
    # worked "once" would still leak the plan in on a later poll). ---
    for _ in range(2):
        status, plans = _get("/plans")
        assert status == 200, f"GET /plans failed: {status} {plans}"
        names = {p["name"] for p in plans}
        assert _ESCAPE_PLAN_NAME not in names, (
            f"{_ESCAPE_PLAN_NAME!r} is discoverable via GET /plans despite failing "
            f"validation -- the session-tier load gate did not refuse it: {sorted(names)}"
        )

    # --- gate (c): the execution gate refuses it ---
    # The gate that mattered here was never "this URL 409s" — it was "an
    # unvalidated plan cannot reach hardware". Execution moved from the bridge
    # to the queue server, so this asserts BOTH halves of that move: the retired
    # direct-execute route is gone and says so machine-readably, and the path
    # that replaced it refuses the same plan for the same reason.
    token = _minted_token(deployed_sandbox_stack.repo)

    # (c1) the retired route is an unconditional 410 -- never a silent 404, and
    # never something a caller could mistake for "this run does not exist".
    status, body = _post("/runs/not-a-real-run-id/launch", {}, headers={"X-Launch-Token": token})
    assert status == 410, f"expected 410 Gone from the retired launch route, got {status}: {body}"
    assert _refusal_code(body) == "use_the_queue", (
        f"the retired route must say where the capability went: {body}"
    )

    # (c2) the queue path cannot even be ENTERED with this plan. `POST
    # /queue/items` takes the plan from the shared draft at a pinned revision,
    # never from its own body, and the draft resolves a plan name against the
    # SAME trust-resolved catalog gate (b) just proved this plan is absent
    # from. So the escape plan is refused at composition — it is not
    # composable, let alone queueable.
    #
    # This is a stronger statement than a 409 at enqueue, and it is the
    # honest one for THIS plan: `session_plan_unvalidated` at enqueue is
    # reachable only for a plan that WAS validated and then had its bytes
    # change underneath the pin, which is a different scenario and is covered
    # against a real queue server by
    # tests/e2e/test_bluesky_queue_e2e.py::test_5_session_plan_with_stale_validation_is_refused_at_enqueue.
    status, body = _patch(
        "/draft",
        {
            "plan_name": _ESCAPE_PLAN_NAME,
            "plan_args_patch": {},
            "client_id": "sandbox-escape-e2e",
        },
    )
    assert status == 422, (
        f"an unvalidated session plan must not even be stageable in the draft, got {status}: {body}"
    )

    # And nothing named it ever reached the queue -- the refusals are not
    # merely status codes on a surface that quietly accepted the work anyway.
    status, queue = _get("/queue")
    assert status == 200, f"GET /queue failed: {status} {queue}"
    assert not any(item.get("name") == _ESCAPE_PLAN_NAME for item in queue["items"]), (
        f"the escape plan is sitting in the queue despite every gate: {queue['items']}"
    )

    # --- the IOC probe: the IOC itself confirms nothing ever landed, whether
    # or not any of the HTTP gates above had actually worked. ---
    sp_after = _caget(target_sp)
    assert sp_after == pytest.approx(sp_baseline), (
        f"{target_sp} changed from {sp_baseline} to {sp_after} despite every gate "
        "refusing the escape plan -- a write reached the IOC"
    )


# ---------------------------------------------------------------------------
# Obfuscation residual: DOCUMENTED, ACCEPTED, KNOWN-UNCAUGHT. NOT asserted
# refused -- see this module's docstring and plan_validation.py's own.
# ---------------------------------------------------------------------------
def test_obfuscated_residual_is_a_documented_known_uncaught_case(
    deployed_sandbox_stack: DeployedSandboxStack,
) -> None:
    """Records the getattr/string-concat obfuscation residual. Deliberately
    does NOT assert the validator refuses it -- stages 1-2 are AST/regex
    checks, not a containment boundary (see module docstring); asserting
    refusal here would misrepresent what this feature actually guarantees.
    This plan is never launched by this test either way, so nothing
    here depends on -- or claims anything about -- whether the dry run's own
    independent EPICS_CA_* neutralization (plan_validation.py) happens to
    let stage 3 complete or fail for this particular body.
    """
    target_sp = deployed_sandbox_stack.escape_target_sp

    status, body = _post(
        "/plans/session",
        {
            "name": _RESIDUAL_PLAN_NAME,
            "description": "Known-uncaught obfuscation residual (never launched)",
            "category": "test",
            "required_devices": [],
            "writes": True,
            "body": _obfuscated_residual_plan_body(target_sp, POISON_CURRENT),
        },
    )
    assert status == 200, f"POST /plans/session failed: {status} {body}"

    # Bounded dry-run timeout: keeps this test's worst case (the reflected
    # caput's connection attempt against an intentionally inert CA env)
    # bounded, without asserting anything about how it resolves.
    status, body = _post("/plans/validate", {"name": _RESIDUAL_PLAN_NAME, "dry_run_timeout": 10.0})
    assert status == 200, f"POST /plans/validate failed: {status} {body}"
    assert isinstance(body.get("content_hash"), str) and body["content_hash"], (
        f"validate response missing a content_hash: {body}"
    )
    print(  # noqa: T201 - informational only, not asserted on
        f"obfuscated residual {_RESIDUAL_PLAN_NAME!r}: passed={body['passed']!r} "
        f"reasons={body['reasons']!r} (documented known-uncaught case; not asserted)"
    )

    # Non-asserting probe: this plan is never launched by this test,
    # so nothing should have moved regardless of the validate outcome above --
    # but this is recorded as an observation, not an assertion (see docstring).
    sp_value = _caget(target_sp)
    print(f"{target_sp} reads {sp_value!r} after the obfuscated-residual validate call")  # noqa: T201


# ---------------------------------------------------------------------------
# Positive: author -> validate -> enqueue -> drain -> read, over the same
# deployed stack. May flake on the drain->read leg (bounded run timing);
# the negative case above stays strict.
# ---------------------------------------------------------------------------
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_session_plan_author_validate_launch_read_round_trip(
    deployed_sandbox_stack: DeployedSandboxStack,
) -> None:
    correctors = deployed_sandbox_stack.positive_correctors
    bpms = deployed_sandbox_stack.positive_bpms

    status, body = _post(
        "/plans/session",
        {
            "name": _POSITIVE_PLAN_NAME,
            "description": "Legit session-authored orbit-response probe",
            "category": "accelerator",
            "required_devices": ["correctors", "readbacks"],
            "writes": True,
            "body": _POSITIVE_PLAN_BODY,
        },
    )
    assert status == 200, f"POST /plans/session failed: {status} {body}"

    status, body = _post(
        "/plans/validate",
        {
            "name": _POSITIVE_PLAN_NAME,
            "sample_args": {
                "correctors": list(correctors)[:1],
                "readbacks": list(bpms)[:1],
                "span_a": 1.0,
                "num": 3,
            },
        },
    )
    assert status == 200, f"POST /plans/validate failed: {status} {body}"
    assert body["passed"] is True, f"legit session plan failed validation: {body['reasons']}"

    status, plans = _get("/plans")
    assert status == 200, f"GET /plans failed: {status} {plans}"
    by_name = {p["name"]: p for p in plans}
    assert _POSITIVE_PLAN_NAME in by_name, (
        f"validated session plan {_POSITIVE_PLAN_NAME!r} not discoverable: {sorted(by_name)}"
    )
    assert by_name[_POSITIVE_PLAN_NAME]["provenance"] == "session", (
        f"expected provenance 'session', got {by_name[_POSITIVE_PLAN_NAME]['provenance']!r}"
    )

    plan_args = {
        "correctors": list(correctors),
        "readbacks": list(bpms),
        "span_a": SPAN_A,
        "num": NUM_POINTS,
    }
    # Execution is the queue's: stage the plan in the shared draft, enqueue at
    # the pinned revision (which mints the run id), then arm the queue. Same
    # guarantees the retired two-step mint-then-launch flow had — the pinned
    # revision and the launch token — with the plan running in the queue
    # server's worker rather than in the bridge process.
    status, body = _enqueue_session_plan(_POSITIVE_PLAN_NAME, plan_args)
    assert status == 200, f"POST /queue/items failed: {status} {body}"
    run_id = body["run_id"]

    token = _minted_token(deployed_sandbox_stack.repo)
    status, body = _post("/queue/start", {}, headers={"X-Launch-Token": token})
    assert status == 200, f"POST /queue/start failed: {status} {body}"

    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    status_body: dict = {}
    while time.monotonic() < deadline:
        _, status_body = _get(f"/runs/{run_id}")
        if status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(0.5)
    assert status_body.get("status") == "completed", (
        f"session_orbit_probe run did not complete within {SCAN_TIMEOUT_SEC:.0f}s "
        f"(status={status_body})"
    )

    status, data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    expected_rows = len(correctors) * NUM_POINTS
    assert data["row_count"] == expected_rows, (
        f"expected {expected_rows} rows, got {data['row_count']}: {data}"
    )
    assert len(data["rows"]) == expected_rows
