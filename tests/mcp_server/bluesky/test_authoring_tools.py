"""Tests for the session-plan authoring/validation surface:
`write_plan` / `validate_plan` MCP tools, their bridge routes
(`POST /plans/session`, `POST /plans/validate`), and their permission tier.

Two layers are covered:

- MCP-tool level (`osprey.mcp_server.bluesky.tools.authoring`): payload shape and
  error-envelope mapping, with `_http_post_json` mocked out (no bridge process
  needed) — mirrors `test_queue_tools.py`'s conventions.
- Bridge-route level (`osprey.services.bluesky_bridge.app`): exercised via
  FastAPI's `TestClient` against the real routes, proving the write path never
  imports/execs the authored body, the HASH CONTRACT (the bytes validated are
  the exact bytes persisted and later re-hashed), and that neither route is
  gated on `control_system.writes_enabled` or `BLUESKY_LAUNCH_TOKEN`.

The full write-then-validate round trip needs a real bluesky dry run, so
those tests are guarded by `pytest.importorskip`, matching every other
bluesky-capable test in this suite (see `test_plan_validation.py`).
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import authoring
from osprey.registry.mcp import FRAMEWORK_SERVERS
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.plan_metadata import parse_plan_metadata_dict
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.validation_record import validation_records
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MOD = "osprey.mcp_server.bluesky.tools.authoring"

_BENIGN_BODY = textwrap.dedent(
    """\
    from bluesky import plan_stubs as bps
    from bluesky import preprocessors as bpp
    from pydantic import BaseModel, Field

    from osprey.services.bluesky_bridge.plan_fields import (
        MovableChannels,
        ReadableChannels,
        scan_metadata,
    )


    class PARAMS(BaseModel):
        correctors: MovableChannels = Field(..., min_length=1)
        monitors: ReadableChannels = Field(..., min_length=1)
        num: int = Field(..., ge=1)


    def build_plan(devices, params):
        corrector = devices[params.correctors[0]]
        monitor = devices[params.monitors[0]]

        @bpp.stage_decorator([corrector, monitor])
        @bpp.run_decorator(
            md=scan_metadata(
                movable=params.correctors,
                readable=params.monitors,
                points=params.num,
            )
        )
        def _sweep():
            for i in range(params.num):
                yield from bps.mv(corrector, float(i))
                yield from bps.trigger_and_read([corrector, monitor])

        return _sweep()
    """
)
_BENIGN_SAMPLE_ARGS = {"correctors": ["c1"], "monitors": ["d1"], "num": 3}


# =========================================================================
# Registry permission tier: both tools approval-only, no
# _WRITES_CHECK — must work identically whether writes_enabled is on or off.
# =========================================================================


def test_both_tools_are_approval_ask_tier_never_writes_checked():
    bluesky = FRAMEWORK_SERVERS["bluesky"]
    assert "write_plan" in bluesky.permissions_ask
    assert "validate_plan" in bluesky.permissions_ask

    by_matcher = {rule.matcher: rule for rule in bluesky.hooks_pre}
    for tool in ("write_plan", "validate_plan"):
        matcher = f"mcp__bluesky__{tool}"
        assert matcher in by_matcher, f"no hooks_pre rule for {matcher}"
        commands = [h.command for h in by_matcher[matcher].hooks]
        assert any("osprey_approval.py" in c for c in commands), (
            f"{matcher} must carry the approval hook"
        )
        assert not any("osprey_writes_check.py" in c for c in commands), (
            f"{matcher} must NEVER be kill-switch-gated — it reaches no hardware"
        )


def test_both_tools_get_distinct_independently_allowlistable_short_names():
    bluesky = FRAMEWORK_SERVERS["bluesky"]
    # Distinct from the queue/stop tools' own tier, and from each other, so an
    # operator can permit authoring without also permitting execution.
    names = {"queue_add", "queue_start", "queue_stop", "stop_run", "write_plan", "validate_plan"}
    assert len(names) == 6
    assert set(bluesky.permissions_ask) >= names


# =========================================================================
# MCP-tool level: payload shape + error mapping (HTTP boundary mocked)
# =========================================================================


def _write_fn():
    return get_tool_fn(authoring.write_plan)


def _validate_fn():
    return get_tool_fn(authoring.validate_plan)


@pytest.fixture(autouse=True)
def _reset_scan_context():
    yield
    reset_server_context()


async def test_write_plan_posts_the_structured_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(200, {"name": "tiny", "content_hash": "deadbeef"}),
    ) as m:
        result = await _write_fn()(
            name="tiny",
            writes=True,
            body="def build_plan(devices, params):\n    yield\n",
            description="A tiny plan.",
        )

    assert m.call_args.args[0] == "/plans/session"
    payload = m.call_args.args[1]
    # The whole authoring surface: no channel declaration travels through this
    # call at all — it lives in the body's own role-typed PARAMS fields.
    assert payload == {
        "name": "tiny",
        "description": "A tiny plan.",
        "writes": True,
        "body": "def build_plan(devices, params):\n    yield\n",
    }
    data = extract_response_dict(result)
    assert data == {"name": "tiny", "content_hash": "deadbeef"}


async def test_write_plan_rejected_maps_to_error_envelope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(
            400,
            {"detail": "invalid plan name '1bad': must be a valid Python identifier"},
        ),
    ):
        with assert_raises_error(error_type="plan_write_rejected") as ctx:
            await _write_fn()(name="1bad", writes=False, body="")
    assert "invalid plan name" in ctx["envelope"]["error_message"]


async def test_validate_plan_posts_name_and_sample_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initialize_server_context()

    body = {"passed": True, "reasons": [], "content_hash": "deadbeef"}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        result = await _validate_fn()(name="tiny", sample_args={"correctors": ["c1"]})

    assert m.call_args.args[0] == "/plans/validate"
    assert m.call_args.args[1] == {
        "name": "tiny",
        "sample_args": {"correctors": ["c1"]},
        "dry_run_timeout": 30.0,
    }
    data = extract_response_dict(result)
    assert data["passed"] is True


async def test_validate_plan_unknown_name_maps_to_error_envelope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(404, {"detail": "unknown session plan 'nope'"}),
    ):
        with assert_raises_error(error_type="unknown_session_plan") as ctx:
            await _validate_fn()(name="nope")
    assert "unknown session plan" in ctx["envelope"]["error_message"]


# =========================================================================
# Bridge-route level: real routes via TestClient, no HTTP mocking.
# =========================================================================


@pytest.fixture(autouse=True)
def _isolated_session_dir_and_records(tmp_path, monkeypatch):
    """Every test gets its own session-plan directory and a clean record store.

    The bridge's `validation_records` is a module-level singleton (like
    `runs.py`'s registry) — clearing it around each test keeps a passing
    record from one test invisible to (and unpolluted by) another.
    """
    monkeypatch.setenv("BLUESKY_SESSION_PLAN_DIR", str(tmp_path / "session_plans"))
    with validation_records.lock:
        validation_records._passing_hashes.clear()
    yield
    with validation_records.lock:
        validation_records._passing_hashes.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_write_session_plan_persists_generated_metadata_plus_body(
    client: TestClient, tmp_path: Path
):
    resp = client.post(
        "/plans/session",
        json={
            "name": "tiny_sweep",
            "description": "A tiny sweep.",
            "writes": True,
            "body": "def build_plan(devices, params):\n    yield\n",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "tiny_sweep"

    from osprey.services.bluesky_bridge.session_dir import resolve_session_plan_dir

    persisted = (resolve_session_plan_dir() / "tiny_sweep.py").read_text(encoding="utf-8")
    assert "PLAN_METADATA = {" in persisted
    assert "'name': 'tiny_sweep'" in persisted
    assert "'writes': True" in persisted
    assert persisted.endswith("def build_plan(devices, params):\n    yield\n")
    # HASH CONTRACT: the returned hash is the hash of the EXACT persisted bytes.
    assert data["content_hash"] == hash_plan_body(persisted)


def test_generated_metadata_block_is_exactly_the_three_declared_fields(
    client: TestClient,
) -> None:
    """The generated block is the plan's whole authoring declaration: name,
    description, writes — nothing about which channels it touches, which the
    body's role-typed `PARAMS` fields carry instead. Parsed with the real
    loader-side parser, so a key reintroduced here fails as the load gate
    would (`PlanMetadata` forbids extras), not just as a changed literal."""
    resp = client.post(
        "/plans/session",
        json={
            "name": "three_field_plan",
            "description": "A read-only plan.",
            "writes": False,
            "body": "def build_plan(devices, params):\n    yield\n",
        },
    )
    assert resp.status_code == 200, resp.text

    from osprey.services.bluesky_bridge.session_dir import resolve_session_plan_dir

    persisted = (resolve_session_plan_dir() / "three_field_plan.py").read_text(encoding="utf-8")
    block = persisted.split("\n\n", 1)[0]
    metadata = ast.literal_eval(block.split("=", 1)[1].strip())
    assert metadata == {
        "name": "three_field_plan",
        "description": "A read-only plan.",
        "writes": False,
    }
    parsed = parse_plan_metadata_dict(metadata, source="three_field_plan.py")
    assert parsed.model_dump() == metadata


def test_write_session_plan_overwrite_changes_the_hash(client: TestClient):
    first = client.post(
        "/plans/session",
        json={
            "name": "reauthored",
            "writes": False,
            "body": "x = 1\n",
        },
    ).json()
    second = client.post(
        "/plans/session",
        json={
            "name": "reauthored",
            "writes": False,
            "body": "x = 2\n",
        },
    ).json()
    assert first["content_hash"] != second["content_hash"]


def test_write_session_plan_rejects_an_unsafe_name(client: TestClient):
    resp = client.post(
        "/plans/session",
        json={
            "name": "../escape",
            "writes": False,
            "body": "x = 1\n",
        },
    )
    assert resp.status_code == 400


def test_write_session_plan_rejects_a_trailing_newline_name(client: TestClient):
    """Regression: `$` matches at end-of-string OR just before a single
    trailing "\\n", so a naive `^...$` anchor would let "foo\\n" through —
    producing a file literally named "foo\\n.py" and a `PLAN_METADATA["name"]`
    that is not a valid identifier. The route must anchor with `\\Z` (or
    `re.fullmatch`) instead, so this must be rejected exactly like any other
    non-identifier name."""
    resp = client.post(
        "/plans/session",
        json={
            "name": "foo\n",
            "writes": False,
            "body": "x = 1\n",
        },
    )
    assert resp.status_code == 400


def test_write_session_plan_rejects_an_overlong_name(client: TestClient):
    """An absurdly long name must fail closed with a clean 400, not surface
    as an unhandled 500 (e.g. an OSError from `Path.write_text` on a
    filesystem that rejects overlong filenames)."""
    resp = client.post(
        "/plans/session",
        json={
            "name": "a" * 5000,
            "writes": False,
            "body": "x = 1\n",
        },
    )
    assert resp.status_code == 400


def test_write_session_plan_never_imports_or_execs_the_body(client: TestClient, tmp_path: Path):
    """A sentinel top-level side effect in the authored body must NEVER fire —
    proving the write route only writes bytes, never imports/execs them."""
    sentinel_path = tmp_path / "sentinel.txt"
    body = (
        f'from pathlib import Path\nPath(r"{sentinel_path}").write_text("fired")\n\n'
        "def build_plan(devices, params):\n    yield\n"
    )
    resp = client.post(
        "/plans/session",
        json={
            "name": "sentinel_plan",
            "writes": False,
            "body": body,
        },
    )
    assert resp.status_code == 200
    assert not sentinel_path.exists(), "write_session_plan must never exec the authored body"


def test_write_session_plan_rejects_a_retired_category_key(client: TestClient):
    """A stale client still POSTing the retired `category`/`required_devices`
    keys must fail loudly (422 naming the key), not get a silent 200 with the
    surplus dropped -- `PlanSessionWriteRequest` is `extra="forbid"`, uniform
    with `PlanMetadata`."""
    resp = client.post(
        "/plans/session",
        json={
            "name": "stale_client_plan",
            "writes": False,
            "body": "def build_plan(devices, params):\n    yield\n",
            "category": "diagnostics",
        },
    )
    assert resp.status_code == 422
    assert "category" in resp.text


def test_validate_unknown_session_plan_is_404(client: TestClient):
    resp = client.post("/plans/validate", json={"name": "never_written"})
    assert resp.status_code == 404


def test_validate_session_plan_rejects_a_retired_required_devices_key(client: TestClient):
    """Same fail-loudly posture as the write route: `PlanValidateRequest` is
    also `extra="forbid"`, so a stale client still sending `required_devices`
    gets a 422 naming the key instead of a silently-accepted request."""
    resp = client.post(
        "/plans/validate",
        json={"name": "never_written", "required_devices": ["c1"]},
    )
    assert resp.status_code == 422
    assert "required_devices" in resp.text


def test_session_authoring_routes_ignore_writes_enabled_and_launch_token(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Neither route is gated on control_system.writes_enabled or
    BLUESKY_LAUNCH_TOKEN — both must keep working with writes fully off and
    no launch token configured at all (their protection is the loopback
    bind + MCP approval hook, not a token or the writes kill switch)."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "config.yml").write_text("control_system:\n  writes_enabled: false\n")
    monkeypatch.chdir(project_dir)
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)

    write_resp = client.post(
        "/plans/session",
        json={
            "name": "writes_off_ok",
            "writes": False,
            "body": "def build_plan(devices, params):\n    yield\n",
        },
    )
    assert write_resp.status_code == 200

    validate_resp = client.post(
        "/plans/validate", json={"name": "writes_off_ok", "sample_args": {}}
    )
    # Reaches the validator (no 401/403/503 from any token/writes gate);
    # the plan itself fails stage 3 (no PARAMS/real generator) but that is
    # an authoring-quality rejection, not an authorization one.
    assert validate_resp.status_code == 200


# =========================================================================
# Full pipeline (needs a real bluesky dry run): write -> validate -> HASH
# CONTRACT between the persisted file and the recorded validation hash.
#
# Only THIS test needs bluesky/ophyd_async importable (stage 3's dry run
# actually drives a RunEngine) — every other test in this file is
# bluesky-independent and must keep running even where those extras aren't
# installed, so the importorskip guard lives inside this one test only
# rather than at module scope (unlike `test_plan_validation.py`, where every
# test is dry-run-adjacent and a module-level guard is appropriate).
# =========================================================================


def test_hash_contract_write_then_validate_same_hash(client: TestClient):
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    write_resp = client.post(
        "/plans/session",
        json={
            "name": "tiny_session_sweep",
            "description": "Session-authored sweep for the hash-contract test.",
            "writes": True,
            "body": _BENIGN_BODY,
        },
    )
    assert write_resp.status_code == 200
    written = write_resp.json()

    from osprey.services.bluesky_bridge.session_dir import resolve_session_plan_dir

    persisted = (resolve_session_plan_dir() / "tiny_session_sweep.py").read_text(encoding="utf-8")
    assert written["content_hash"] == hash_plan_body(persisted)

    validate_resp = client.post(
        "/plans/validate",
        json={"name": "tiny_session_sweep", "sample_args": _BENIGN_SAMPLE_ARGS},
    )
    assert validate_resp.status_code == 200
    validated = validate_resp.json()

    assert validated["passed"] is True, validated["reasons"]
    # THE HASH CONTRACT: the hash validate recorded is the hash of the exact
    # persisted file content, re-hashed independently of the write response.
    assert validated["content_hash"] == hash_plan_body(persisted)
    assert validated["content_hash"] == written["content_hash"]
    assert validation_records.has_passing_record(validated["content_hash"]) is True


def test_a_failing_validation_records_nothing(client: TestClient):
    client.post(
        "/plans/session",
        json={
            "name": "unsafe_plan",
            "writes": True,
            "body": "import epics\n\n\ndef build_plan(devices, params):\n    epics.caput('X', 1)\n    yield\n",
        },
    )
    resp = client.post("/plans/validate", json={"name": "unsafe_plan"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is False
    assert data["reasons"]
    assert validation_records.has_passing_record(data["content_hash"]) is False
