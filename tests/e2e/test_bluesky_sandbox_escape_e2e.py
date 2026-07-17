"""Real-container sandbox-escape e2e (task 2.10) -- the authoritative proof of
the authoring-sandbox feature's central invariant: an agent-authored
session-tier plan can never reach real hardware unless it is validated AND
that exact validated content is what actually launches.

Deploys the VA-backed turn-key scan-stack (``tests/e2e/_orm_stack.py`` -- the
same real Virtual Accelerator + bluesky-bridge container pair
``test_orm_roundtrip.py`` uses) and drives the session-authoring HTTP surface
(``POST /plans/session``, ``POST /plans/validate``, ``POST /runs`` ->
``launch``) end to end against it, then reads a real corrector setpoint back
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
directory's load gate (task 2.4) and the launch gate (task 2.5) re-hash from
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
Teardown goes through ``osprey deploy down``, matching every other e2e in
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

from tests.e2e import _orm_stack

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

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 1200  # amd64-emulated VA image build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0
SCAN_TIMEOUT_SEC = 120.0

# Three correctors total: one reserved exclusively as the escape/residual
# probe TARGET (never launched in this test, by either the negative
# or the obfuscation-residual case), two driven for real by the positive
# author -> validate -> launch -> read round trip. Disjoint by
# construction, so a run-order change can never let the positive scan's
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
# resolves correctors/detectors by string name against whatever `devices`
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
_POSITIVE_PLAN_BODY = '''"""Session-authored positive plan body for the sandbox-escape e2e
(tests/e2e/test_bluesky_sandbox_escape_e2e.py) -- mirrors plans_core/
orm.py's PARAMS/build_plan, proving the author -> validate ->
launch -> read path works end to end for a legitimately-authored
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
    detectors: list[str] = Field(..., min_length=1)
    span_a: float = Field(..., gt=0, le=10.0)
    num: int = Field(..., ge=3)

    @model_validator(mode="after")
    def _disjoint(self) -> "PARAMS":
        overlap = set(self.correctors) & set(self.detectors)
        if overlap:
            raise ValueError(f"correctors and detectors must be disjoint (overlap: {sorted(overlap)})")
        return self


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    correctors = [(name, devices[name]) for name in params.correctors]
    corrector_devices = [corrector for _, corrector in correctors]
    detector_devices = [devices[name] for name in params.detectors]
    step = (2 * params.span_a) / (params.num - 1)
    currents = [-params.span_a + i * step for i in range(params.num)]
    all_devices = corrector_devices + detector_devices

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


def _minted_token(project_dir: Path) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_LAUNCH_TOKEN")
    assert token, "BLUESKY_LAUNCH_TOKEN missing/empty in the project .env"
    return token


def _channel_limits(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8"))


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
    project_dir: Path
    escape_target_sp: str
    escape_target_rb: str
    positive_correctors: dict[str, tuple[str, str]]
    positive_bpms: dict[str, str]


@pytest.fixture(scope="module")
def deployed_sandbox_stack(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DeployedSandboxStack]:
    base = tmp_path_factory.mktemp("sandbox_escape_build")
    project_dir = _orm_stack.build_project_subprocess(
        "sandbox-escape", output_dir=base, bridge_port=BRIDGE_PORT, timeout=BUILD_TIMEOUT_SEC
    )

    limits = _channel_limits(project_dir)
    correctors = _orm_stack.select_correctors(limits, count=CORRECTOR_COUNT)
    bpms = _orm_stack.select_bpms(limits, count=BPM_COUNT)
    escape_name, (escape_sp, escape_rb) = sorted(correctors.items())[0]
    positive_correctors = {name: pair for name, pair in correctors.items() if name != escape_name}
    _orm_stack.write_scan_env(project_dir, correctors=correctors, bpms=bpms)

    osprey_bin = _orm_stack.find_osprey_console_script()

    # Force fresh --dev builds so the deployed containers run CURRENT source
    # (osprey deploy up does not pass --build to compose, so it would
    # otherwise reuse a stale cached image). Exact-named images only.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(["docker", "rmi", "-f", _orm_stack.VA_IMAGE], capture_output=True, text=True)
        subprocess.run(
            ["docker", "rmi", "-f", _orm_stack.BRIDGE_IMAGE], capture_output=True, text=True
        )

    try:
        up = subprocess.run(
            [str(osprey_bin), "deploy", "up", "-d", "--dev"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey deploy up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        yield DeployedSandboxStack(
            project_dir=project_dir,
            escape_target_sp=escape_sp,
            escape_target_rb=escape_rb,
            positive_correctors=positive_correctors,
            positive_bpms=bpms,
        )
    finally:
        down = subprocess.run(
            [str(osprey_bin), "deploy", "down"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey deploy down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
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

    # --- gate (c): launch 409s (the launch-validation gate, task 2.5) ---
    status, body = _post("/runs", {"plan_name": _ESCAPE_PLAN_NAME, "plan_args": {}})
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    token = _minted_token(deployed_sandbox_stack.project_dir)
    status, body = _post(f"/runs/{run_id}/launch", {}, headers={"X-Launch-Token": token})
    assert status == 409, (
        f"expected 409 launching an unvalidated session plan, got {status}: {body}"
    )
    assert "validation record" in body.get("detail", ""), (
        f"409 detail doesn't name the validation-record gate: {body}"
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
# Positive: author -> validate -> launch -> read, over the same
# deployed stack. May flake on the launch->read leg (bounded scan timing);
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
            "required_devices": ["correctors", "detectors"],
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
                "detectors": list(bpms)[:1],
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
        "detectors": list(bpms),
        "span_a": SPAN_A,
        "num": NUM_POINTS,
    }
    status, body = _post("/runs", {"plan_name": _POSITIVE_PLAN_NAME, "plan_args": plan_args})
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    token = _minted_token(deployed_sandbox_stack.project_dir)
    status, body = _post(f"/runs/{run_id}/launch", {}, headers={"X-Launch-Token": token})
    assert status == 200, f"launch failed: {status} {body}"

    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    status_body: dict = {}
    while time.monotonic() < deadline:
        _, status_body = _get(f"/runs/{run_id}")
        if status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(0.5)
    assert status_body.get("status") == "completed", (
        f"session_orbit_probe scan did not complete within {SCAN_TIMEOUT_SEC:.0f}s "
        f"(status={status_body})"
    )

    status, data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    expected_rows = len(correctors) * NUM_POINTS
    assert data["row_count"] == expected_rows, (
        f"expected {expected_rows} rows, got {data['row_count']}: {data}"
    )
    assert len(data["rows"]) == expected_rows
