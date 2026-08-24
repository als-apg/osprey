"""The directory-layer scan behind `GET /plans/{name}/source`.

`app._find_layer_source_path` reads each candidate file's declared plan name
through the shared source-only reader
(`plan_metadata.extract_plan_name_from_source`) rather than through a private
`ast` walk of its own, so there is one authority for how that key is read.

The scan asks one question — *which file declares this name?* — and so is
deliberately as permissive as an import, not as strict as the full
`PlanMetadata` contract. The two are not the same set: a file whose
``PLAN_METADATA`` holds a non-literal value under some key other than ``name``
imports cleanly and registers as a launchable plan, yet cannot be read by the
full-contract reader. Serving its source is what keeps the human approval
prompt's source excerpt intact, so this module pins that case first.

The route is also not registration-gated — it scans disk directly, so a file
the loader quarantines still has its source served. That is the honest answer
for a file plainly sitting in a trusted plan directory; the alternative is a 404
saying no source file exists. (The registry is read for one thing only: which of
several same-name candidates to serve.)

Where the scan must agree with the loader exactly is plan-name shadowing, and
it gets that agreement from the registry itself rather than by re-deriving the
loader's rule: for a registered name the candidate whose tier matches the
registered spec's provenance is served, so the file served is the file that
would run even when the higher-trust file is one the loader quarantined. Only
for an unregistered name does the scan fall back to highest trust — which is
what keeps a quarantined file's source reachable at all.

Plan files here are pure pydantic/stdlib (no bluesky import), mirroring
`test_plan_source_endpoint.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import app

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    plan_loader.reset_facility_plans()
    yield
    plan_loader.reset_facility_plans()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _plan_source(metadata_block: str, name: str) -> str:
    """A complete, loadable plan file whose only variable is its metadata block."""
    return (
        f"{metadata_block}\n\n\ndef build_plan(devices, params):\n    return {{'plan': {name!r}}}\n"
    )


_VALID_METADATA = (
    "PLAN_METADATA = {\n"
    '    "name": "layer_plan",\n'
    '    "description": "A directory-layer test plan.",\n'
    '    "writes": False,\n'
    "}"
)

# Reads fine as a Python import — `_DESC` is bound before `PLAN_METADATA` — but
# the dict is not a literal, so the full-contract reader cannot evaluate it.
# The loader still registers this plan, so the source route must still find it.
_NON_LITERAL_VALUE_METADATA = (
    '_DESC = "A directory-layer test plan."\n'
    "\n"
    "PLAN_METADATA = {\n"
    '    "name": "layer_plan",\n'
    '    "description": _DESC,\n'
    '    "writes": False,\n'
    "}"
)

# Right name, missing the required `writes` key — located by this name-only
# scan, rejected by the `PlanMetadata` contract the loader applies.
_MISSING_FIELD_METADATA = (
    "PLAN_METADATA = {\n"
    '    "name": "layer_plan",\n'
    '    "description": "A directory-layer test plan.",\n'
    "}"
)

# Right name, plus a key outside the contract (`extra="forbid"`).
_UNKNOWN_KEY_METADATA = (
    "PLAN_METADATA = {\n"
    '    "name": "layer_plan",\n'
    '    "description": "A directory-layer test plan.",\n'
    '    "writes": False,\n'
    '    "provenance": "facility",\n'
    "}"
)


def _shadowed_metadata(description: str) -> str:
    """A valid block under the shared name `layer_plan`, told apart by its text."""
    return (
        "PLAN_METADATA = {\n"
        '    "name": "layer_plan",\n'
        f'    "description": "{description}",\n'
        '    "writes": False,\n'
        "}"
    )


def _shipped_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shipped_dir = tmp_path / "shipped"
    shipped_dir.mkdir()
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    return shipped_dir


def test_a_registered_plan_with_a_non_literal_value_is_still_served(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression case: a plan the loader accepts must stay reachable here.

    Its ``description`` is a module-level constant rather than a literal, which
    an import resolves and a whole-block literal read cannot. Reading the whole
    block to answer an identity question would 404 this plan — stripping the
    source excerpt out of the human approval prompt for a plan that is
    registered, listed, and launchable.
    """
    shipped_dir = _shipped_layer(tmp_path, monkeypatch)
    (shipped_dir / "layer_plan.py").write_text(
        _plan_source(_NON_LITERAL_VALUE_METADATA, "layer_plan")
    )

    assert "layer_plan" in plan_loader.get_facility_plans().plans

    resp = client.get("/plans/layer_plan/source")
    assert resp.status_code == 200, resp.text
    assert resp.json()["provenance"] == "shipped"
    assert "_DESC" in resp.json()["source"]


@pytest.mark.parametrize(
    ("label", "metadata_block"),
    [
        ("missing_field", _MISSING_FIELD_METADATA),
        ("unknown_key", _UNKNOWN_KEY_METADATA),
    ],
)
def test_a_quarantined_file_still_has_its_source_served(
    label: str,
    metadata_block: str,
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file whose ``PLAN_METADATA`` names a plan but breaks the contract is
    quarantined by the loader — and its source is served anyway.

    This route is not registration-gated: it scans the plan directories
    directly and never consults the registry. For a file sitting in a trusted
    plan directory under the asked-for name, returning its text is the honest
    answer; a 404 would claim no source file exists for a file that plainly
    does. The file is complete in every other respect (it defines
    `build_plan`), so its metadata is the only reason it fails to register.
    """
    shipped_dir = _shipped_layer(tmp_path, monkeypatch)
    (shipped_dir / "layer_plan.py").write_text(_plan_source(metadata_block, "layer_plan"))

    assert "layer_plan" not in plan_loader.get_facility_plans().plans, label

    resp = client.get("/plans/layer_plan/source")
    assert resp.status_code == 200, f"{label}: {resp.text}"
    assert resp.json()["provenance"] == "shipped", label


def test_valid_layer_plan_is_still_located(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control case: a contract-passing file registers AND is served."""
    shipped_dir = _shipped_layer(tmp_path, monkeypatch)
    (shipped_dir / "some_file_stem.py").write_text(_plan_source(_VALID_METADATA, "layer_plan"))

    assert "layer_plan" in plan_loader.get_facility_plans().plans

    resp = client.get("/plans/layer_plan/source")
    assert resp.status_code == 200, resp.text
    assert resp.json()["provenance"] == "shipped"


@pytest.mark.parametrize(
    ("label", "bad_source"),
    [
        ("unparseable", "def build_plan(  # unbalanced\n"),
        ("no_metadata", "def build_plan(devices, params):\n    return {}\n"),
        ("computed_metadata", "PLAN_METADATA = dict(name='layer_plan')\n"),
        ("non_literal_name", "_NAME = 'layer_plan'\nPLAN_METADATA = {'name': _NAME}\n"),
        ("annotated_assignment", "PLAN_METADATA: dict = {'name': 'layer_plan'}\n"),
        ("unhashable_key", "PLAN_METADATA = {[1]: 2}\n"),
        ("other_name", "PLAN_METADATA = {'name': 'a_different_plan'}\n"),
    ],
)
def test_an_unidentifiable_file_does_not_abort_the_scan(
    label: str,
    bad_source: str,
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every way a file can fail to yield a plan name is a skip, not a failure
    of the whole scan: a later file in the same directory is still found.

    ``unhashable_key`` is the case that would raise `TypeError` out of
    `ast.literal_eval` if this scan evaluated the whole block; the name-only
    reader never evaluates it. ``annotated_assignment`` is a standing,
    deliberate limit of both readers — such a plan registers and launches
    normally and only its source lookup misses.

    Named to sort ahead of the good file, so the good file is only reachable
    if the scan survived the bad one.
    """
    shipped_dir = _shipped_layer(tmp_path, monkeypatch)
    (shipped_dir / "a_broken.py").write_text(bad_source)
    (shipped_dir / "z_good.py").write_text(_plan_source(_VALID_METADATA, "layer_plan"))

    resp = client.get("/plans/layer_plan/source")
    assert resp.status_code == 200, f"{label}: {resp.text}"
    assert resp.json()["provenance"] == "shipped", label


def test_a_broken_layer_does_not_hide_a_later_layer(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skip carries across layer boundaries too: a shipped file whose name
    cannot be read does not stop the higher-trust facility layer from being
    reached."""
    shipped_dir = _shipped_layer(tmp_path, monkeypatch)
    (shipped_dir / "layer_plan.py").write_text("PLAN_METADATA = dict(name='layer_plan')\n")

    facility_dir = tmp_path / "facility"
    facility_dir.mkdir()
    (facility_dir / "layer_plan.py").write_text(_plan_source(_VALID_METADATA, "layer_plan"))
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(facility_dir))

    resp = client.get("/plans/layer_plan/source")
    assert resp.status_code == 200, resp.text
    assert resp.json()["provenance"] == "facility"


def test_a_shadowed_plan_serves_the_file_that_actually_runs(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When two tiers declare the same plan name, this route must serve the file
    the loader registered — the one that would actually run.

    `plan_loader._register_plan` resolves a same-name collision by highest
    trust, so the facility file wins the registry. The scan resolves it the same
    way. Serving the shipped file here would show an operator the source of a
    file that is not the one about to execute: wrong evidence at an approval
    gate, worse than no evidence.
    """
    shipped_dir = _shipped_layer(tmp_path, monkeypatch)
    (shipped_dir / "layer_plan.py").write_text(
        "# SHIPPED FILE\n" + _plan_source(_shadowed_metadata("The shipped original."), "layer_plan")
    )

    facility_dir = tmp_path / "facility"
    facility_dir.mkdir()
    (facility_dir / "layer_plan.py").write_text(
        "# FACILITY FILE\n"
        + _plan_source(_shadowed_metadata("The facility override."), "layer_plan")
    )
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(facility_dir))

    # What the loader registered, i.e. what a launch would run...
    spec = plan_loader.get_facility_plans().plans["layer_plan"]
    assert spec.provenance == "facility"
    assert spec.description == "The facility override."

    # ...is the file this route serves.
    resp = client.get("/plans/layer_plan/source")
    assert resp.status_code == 200, resp.text
    assert resp.json()["provenance"] == "facility"
    assert "# FACILITY FILE" in resp.json()["source"]
    assert "# SHIPPED FILE" not in resp.json()["source"]


def test_a_name_this_scan_cannot_locate_is_a_404(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No file declares the asked-for name, so the route degrades to a 404."""
    shipped_dir = _shipped_layer(tmp_path, monkeypatch)
    (shipped_dir / "layer_plan.py").write_text(_plan_source(_VALID_METADATA, "layer_plan"))

    assert client.get("/plans/some_other_plan/source").status_code == 404


def test_the_session_layer_is_not_scanned_by_the_layer_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_find_layer_source_path` deliberately skips the `session` layer — the
    route resolves a session plan by filename first, and a session file must
    never be served as if it carried a directory tier's implicit trust."""
    from osprey.services.bluesky_bridge.app import _find_layer_source_path

    _shipped_layer(tmp_path, monkeypatch)
    session_dir = tmp_path / "plans_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "layer_plan.py").write_text(_plan_source(_VALID_METADATA, "layer_plan"))

    assert _find_layer_source_path("layer_plan") is None


def _facility_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int) -> list[Path]:
    """``count`` facility dirs, in `BLUESKY_PLAN_DIRS` order (all equal trust)."""
    dirs = [tmp_path / f"facility_{index}" for index in range(count)]
    for directory in dirs:
        directory.mkdir()
    monkeypatch.setenv(_PLAN_DIRS_ENV, os.pathsep.join(str(d) for d in dirs))
    return dirs


def test_a_quarantined_shadow_does_not_win_over_the_registered_file(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Higher trust alone does not decide which file is served — the registry does.

    The facility file shadows the shipped one by name, but breaks the
    `PlanMetadata` contract, so the loader quarantines it and registers the
    SHIPPED plan. A scan that re-derived the loader's rule by trust tier would
    serve the facility file here — the file that is not the one about to run,
    reported under a provenance `GET /plans` contradicts. The scan reads the
    loader's actual decision instead, so the two agree.
    """
    shipped_dir = _shipped_layer(tmp_path, monkeypatch)
    (shipped_dir / "layer_plan.py").write_text(
        "# SHIPPED FILE\n" + _plan_source(_shadowed_metadata("The shipped original."), "layer_plan")
    )

    facility_dir = _facility_layers(tmp_path, monkeypatch, 1)[0]
    (facility_dir / "layer_plan.py").write_text(
        "# FACILITY FILE\n" + _plan_source(_UNKNOWN_KEY_METADATA, "layer_plan")
    )

    spec = plan_loader.get_facility_plans().plans["layer_plan"]
    assert spec.provenance == "shipped"
    assert spec.description == "The shipped original."

    resp = client.get("/plans/layer_plan/source")
    assert resp.status_code == 200, resp.text
    assert "# SHIPPED FILE" in resp.json()["source"]
    assert "# FACILITY FILE" not in resp.json()["source"]
    # The agreement itself, not just today's value: the source route and
    # `GET /plans` must never report two different tiers for one name.
    assert resp.json()["provenance"] == spec.provenance


def test_an_equal_trust_collision_serves_the_registered_later_file(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two facility dirs declare the same name, and the later-scanned file wins.

    `plan_loader._register_plan` lets an equal-trust redefinition override, so
    the second directory in `BLUESKY_PLAN_DIRS` owns the registration — and the
    served file has to be that one.
    """
    _shipped_layer(tmp_path, monkeypatch)
    first, second = _facility_layers(tmp_path, monkeypatch, 2)
    (first / "layer_plan.py").write_text(
        "# FIRST DIR\n" + _plan_source(_shadowed_metadata("The first facility dir."), "layer_plan")
    )
    (second / "layer_plan.py").write_text(
        "# SECOND DIR\n"
        + _plan_source(_shadowed_metadata("The second facility dir."), "layer_plan")
    )

    spec = plan_loader.get_facility_plans().plans["layer_plan"]
    assert spec.provenance == "facility"
    assert spec.description == "The second facility dir."

    resp = client.get("/plans/layer_plan/source")
    assert resp.status_code == 200, resp.text
    assert "# SECOND DIR" in resp.json()["source"]
    assert "# FIRST DIR" not in resp.json()["source"]
    assert resp.json()["provenance"] == spec.provenance


def test_an_unregistered_equal_trust_collision_serves_the_later_file(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same tie-break in the fallback path, where no registration exists.

    Both files are quarantined, so the registry cannot decide anything and the
    scan falls back to highest trust, last match winning. Pinning it here rather
    than only in the registered case is what makes the `>=` (not `>`) tie-break
    observable at all: in the registered case the registry short-circuits it.
    """
    _shipped_layer(tmp_path, monkeypatch)
    first, second = _facility_layers(tmp_path, monkeypatch, 2)
    (first / "layer_plan.py").write_text(
        "# FIRST DIR\n" + _plan_source(_UNKNOWN_KEY_METADATA, "layer_plan")
    )
    (second / "layer_plan.py").write_text(
        "# SECOND DIR\n" + _plan_source(_UNKNOWN_KEY_METADATA, "layer_plan")
    )

    assert "layer_plan" not in plan_loader.get_facility_plans().plans

    resp = client.get("/plans/layer_plan/source")
    assert resp.status_code == 200, resp.text
    assert "# SECOND DIR" in resp.json()["source"]
    assert "# FIRST DIR" not in resp.json()["source"]
    assert resp.json()["provenance"] == "facility"
