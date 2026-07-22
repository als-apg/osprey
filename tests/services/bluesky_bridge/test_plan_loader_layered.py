"""Coverage for the layered directory plan loader (task 1.2): ordered
directory-layer scanning, per-file quarantine, fail-closed trust-collision
resolution, and the preserved legacy single-module contract folded in as a
one-entry `facility`-tier layer.

Directory layers are exercised by monkeypatching `plan_loader._SHIPPED_PLANS_DIR`
(the in-image core dir doesn't exist in this checkout yet — task 1.5) and by
setting `BLUESKY_PLAN_DIRS` (env, `facility` tier) / `bluesky.plan_dirs` in a
temp config.yml (`preset` tier). All plan files here are pure pydantic/stdlib
— no bluesky import — so this suite runs in the bluesky-less lane alongside
`test_plan_injection.py`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

import pytest
import yaml

from osprey.services.bluesky_bridge import plan_loader

_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"


@pytest.fixture(autouse=True)
def _isolated_plan_loader(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean loader cache and no leftover layer env vars."""
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    plan_loader.reset_facility_plans()
    yield
    plan_loader.reset_facility_plans()


def _valid_plan_source(name: str, *, with_params: bool = False) -> str:
    """A minimal, well-formed directory-layer plan file defining ``name``."""
    params_block = (
        "class PARAMS(BaseModel):\n    amplitude: float = 2.0\n\n\n" if with_params else ""
    )
    return (
        "from pydantic import BaseModel\n\n\n"
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A layered test plan.",\n'
        '    "category": "accelerator",\n'
        '    "required_devices": [],\n'
        '    "writes": False,\n'
        "}\n\n\n"
        f"{params_block}"
        "def build_plan(devices, params):\n"
        f'    return {{"plan": {name!r}}}\n'
    )


def _write(directory: Path, filename: str, source: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(source)
    return path


def _synthetic_module_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]
    return f"_osprey_bridge_plan_{digest}"


def _write_config(tmp_path: Path, plan_dirs: list[str]) -> Path:
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"bluesky": {"plan_dirs": plan_dirs}}))
    return config_file


def test_layered_scan_registers_plans_from_each_directory_with_correct_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped_dir = tmp_path / "shipped"
    preset_dir = tmp_path / "preset"
    facility_dir = tmp_path / "facility"
    _write(shipped_dir, "core_plan.py", _valid_plan_source("core_plan"))
    _write(preset_dir, "preset_plan.py", _valid_plan_source("preset_plan"))
    _write(facility_dir, "facility_plan.py", _valid_plan_source("facility_plan"))

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    monkeypatch.setenv("OSPREY_CONFIG", str(_write_config(tmp_path, [str(preset_dir)])))
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(facility_dir))

    facility = plan_loader.get_facility_plans()

    assert facility.plans["core_plan"].provenance == "shipped"
    assert facility.plans["preset_plan"].provenance == "preset"
    assert facility.plans["facility_plan"].provenance == "facility"


def test_same_stem_files_across_layers_both_load_without_sys_modules_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped_dir = tmp_path / "shipped"
    facility_dir = tmp_path / "facility"
    shipped_path = _write(shipped_dir, "myplan.py", _valid_plan_source("plan_a"))
    facility_path = _write(facility_dir, "myplan.py", _valid_plan_source("plan_b"))

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(facility_dir))

    facility = plan_loader.get_facility_plans()

    assert facility.plans["plan_a"].provenance == "shipped"
    assert facility.plans["plan_b"].provenance == "facility"

    name_a = _synthetic_module_name(shipped_path)
    name_b = _synthetic_module_name(facility_path)
    assert name_a in sys.modules
    assert name_b in sys.modules
    assert sys.modules[name_a] is not sys.modules[name_b]
    del sys.modules[name_a]
    del sys.modules[name_b]


def test_malformed_file_is_quarantined_and_other_plans_still_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    layer_dir = tmp_path / "layer"
    _write(layer_dir, "good_plan.py", _valid_plan_source("good_plan"))
    _write(layer_dir, "syntax_error.py", "def broken(:\n    pass\n")
    _write(layer_dir, "missing_metadata.py", "def build_plan(devices, params):\n    return None\n")
    _write(
        layer_dir,
        "missing_build_plan.py",
        "PLAN_METADATA = {'name': 'no_build', 'description': 'd', 'category': 'accelerator', "
        "'required_devices': [], 'writes': False}\n",
    )

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", layer_dir)

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        facility = plan_loader.get_facility_plans()

    assert "good_plan" in facility.plans
    assert "no_build" not in facility.plans
    quarantine_warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and "quarantining" in r.message
    ]
    assert len(quarantine_warnings) == 3


def test_higher_trust_directory_layer_overrides_lower_trust_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    shipped_dir = tmp_path / "shipped"
    facility_dir = tmp_path / "facility"
    _write(shipped_dir, "a.py", _valid_plan_source("shared_name"))
    _write(facility_dir, "b.py", _valid_plan_source("shared_name"))

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(facility_dir))

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        facility = plan_loader.get_facility_plans()

    assert facility.plans["shared_name"].provenance == "facility"
    assert any(r.levelno == logging.WARNING and "overrides" in r.message for r in caplog.records)


def test_lower_trust_directory_layer_is_rejected_when_legacy_module_already_owns_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The legacy single-module contract is a `facility`-tier layer scanned
    first — a `shipped`-tier directory file defining the same name must be
    rejected outright, not silently clobber it."""
    legacy_dir = tmp_path / "legacy"
    legacy_path = _write(
        legacy_dir,
        "legacy_plans.py",
        "from osprey.services.bluesky_bridge.plan_types import PlanSpec\n"
        "from pydantic import BaseModel\n\n"
        "class P(BaseModel):\n"
        "    pass\n\n"
        "PLANS = {'shared_name': PlanSpec(name='shared_name', plan=lambda d, p: None, "
        "schema=P, description='from legacy module')}\n\n"
        "def get_devices():\n"
        "    return {}\n",
    )
    shipped_dir = tmp_path / "shipped"
    _write(shipped_dir, "a.py", _valid_plan_source("shared_name"))

    monkeypatch.setenv(_PLAN_MODULE_ENV, str(legacy_path))
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        facility = plan_loader.get_facility_plans()

    assert facility.plans["shared_name"].description == "from legacy module"
    assert facility.plans["shared_name"].provenance == "facility"
    assert any(r.levelno == logging.ERROR and "rejecting" in r.message for r in caplog.records)


def test_legacy_single_module_contract_still_loads_and_is_tagged_facility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the pre-existing `PLANS`/`get_devices()` contract still
    loads via `load_facility_plans`, and `get_facility_plans()` folds it in."""
    module_path = _write(
        tmp_path / "facility_repo",
        "my_facility_plans.py",
        "from osprey.services.bluesky_bridge.plan_types import PlanSpec\n"
        "from pydantic import BaseModel\n\n"
        "class WiggleParams(BaseModel):\n"
        "    amplitude: float = 1.0\n\n"
        "def _wiggle_plan(devices, params):\n"
        "    return {'device': devices['wiggler'], 'amplitude': params.amplitude}\n\n"
        "PLANS = {'wiggle': PlanSpec(name='wiggle', plan=_wiggle_plan, schema=WiggleParams, "
        "description='A facility-specific plan.')}\n\n"
        "def get_devices():\n"
        "    return {'wiggler': 'fake-wiggler-device'}\n",
    )

    direct = plan_loader.load_facility_plans(str(module_path))
    assert set(direct.plans) == {"wiggle"}
    assert direct.plans["wiggle"].provenance == "facility"
    assert direct.devices == {"wiggler": "fake-wiggler-device"}

    monkeypatch.setenv(_PLAN_MODULE_ENV, str(module_path))
    facility = plan_loader.get_facility_plans()
    assert "wiggle" in facility.plans
    assert facility.devices == {"wiggler": "fake-wiggler-device"}


def test_params_not_a_type_is_quarantined_and_other_plans_still_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    layer_dir = tmp_path / "layer"
    _write(layer_dir, "good_plan.py", _valid_plan_source("good_plan"))
    _write(
        layer_dir,
        "params_not_a_type.py",
        "PLAN_METADATA = {\n"
        "    'name': 'bad_params',\n"
        "    'description': 'd',\n"
        "    'category': 'accelerator',\n"
        "    'required_devices': [],\n"
        "    'writes': False,\n"
        "}\n\n"
        "PARAMS = 123\n\n"
        "def build_plan(devices, params):\n"
        "    return None\n",
    )

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", layer_dir)

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        facility = plan_loader.get_facility_plans()

    assert "good_plan" in facility.plans
    assert "bad_params" not in facility.plans
    assert any(r.levelno == logging.WARNING and "quarantining" in r.message for r in caplog.records)


def test_params_instance_not_subclass_is_quarantined_and_other_plans_still_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`PARAMS` must be a `BaseModel` *subclass*, not an instance of one."""
    layer_dir = tmp_path / "layer"
    _write(layer_dir, "good_plan.py", _valid_plan_source("good_plan"))
    _write(
        layer_dir,
        "params_is_instance.py",
        "from pydantic import BaseModel\n\n\n"
        "class _P(BaseModel):\n"
        "    pass\n\n\n"
        "PLAN_METADATA = {\n"
        "    'name': 'bad_params_instance',\n"
        "    'description': 'd',\n"
        "    'category': 'accelerator',\n"
        "    'required_devices': [],\n"
        "    'writes': False,\n"
        "}\n\n"
        "PARAMS = _P()\n\n"
        "def build_plan(devices, params):\n"
        "    return None\n",
    )

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", layer_dir)

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        facility = plan_loader.get_facility_plans()

    assert "good_plan" in facility.plans
    assert "bad_params_instance" not in facility.plans
    assert any(r.levelno == logging.WARNING and "quarantining" in r.message for r in caplog.records)


def test_system_exit_at_import_time_is_quarantined_and_siblings_still_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression for the `SystemExit`-at-import-time hardening: a plan file
    calling `sys.exit()` must be quarantined like any other bad file, not
    abort the rest of the directory scan."""
    layer_dir = tmp_path / "layer"
    _write(layer_dir, "good_plan.py", _valid_plan_source("good_plan"))
    _write(layer_dir, "exits_on_import.py", "import sys\n\nsys.exit(1)\n")

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", layer_dir)

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        facility = plan_loader.get_facility_plans()

    assert "good_plan" in facility.plans
    assert any(r.levelno == logging.WARNING and "quarantining" in r.message for r in caplog.records)


def test_equal_trust_collision_lets_the_later_scanned_directory_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Two `facility`-tier directories (both from `BLUESKY_PLAN_DIRS`) defining
    the same plan name is an equal-trust collision: the later-scanned
    directory wins, with a warning (same-tier dirs are all operator-
    controlled, so scan order is the only tie-breaker)."""
    first_dir = tmp_path / "facility_first"
    second_dir = tmp_path / "facility_second"
    _write(first_dir, "a.py", _valid_plan_source("shared_equal"))
    _write(second_dir, "b.py", _valid_plan_source("shared_equal"))

    monkeypatch.setenv(_PLAN_DIRS_ENV, os.pathsep.join([str(first_dir), str(second_dir)]))

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        facility = plan_loader.get_facility_plans()

    assert facility.plans["shared_equal"].provenance == "facility"
    assert any(
        r.levelno == logging.WARNING and "redefined at equal trust" in r.message
        for r in caplog.records
    )


def _write_excluded_config(tmp_path: Path, excluded_plans) -> Path:
    """Write a temp config.yml carrying ``bluesky.excluded_plans`` (list or scalar)."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"bluesky": {"excluded_plans": excluded_plans}}))
    return config_file


def test_resolve_excluded_plans_env_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "orm")
    assert plan_loader._resolve_excluded_plans() == {"orm"}


def test_resolve_excluded_plans_config_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSPREY_CONFIG", str(_write_excluded_config(tmp_path, ["orm"])))
    assert plan_loader._resolve_excluded_plans() == {"orm"}


def test_resolve_excluded_plans_unions_env_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "orm")
    monkeypatch.setenv("OSPREY_CONFIG", str(_write_excluded_config(tmp_path, ["grid_scan"])))
    assert plan_loader._resolve_excluded_plans() == {"orm", "grid_scan"}


def test_resolve_excluded_plans_config_bare_string_is_coerced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSPREY_CONFIG", str(_write_excluded_config(tmp_path, "orm")))
    assert plan_loader._resolve_excluded_plans() == {"orm"}


def test_resolve_excluded_plans_unset_env_and_no_config_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(plan_loader._EXCLUDED_PLANS_ENV, raising=False)
    assert plan_loader._resolve_excluded_plans() == set()


def test_resolve_excluded_plans_empty_env_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "")
    assert plan_loader._resolve_excluded_plans() == set()


def test_resolve_excluded_plans_trailing_pathsep_contributes_no_empty_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "orm" + os.pathsep)
    assert plan_loader._resolve_excluded_plans() == {"orm"}


_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_EXCLUSION_NO_MATCH_MARKER = "matched no registered plan"


def _register_session_plan(session_dir: Path, filename: str, name: str) -> Path:
    """Write a session-tier plan file and record a passing validation for it,
    so `get_facility_plans()`'s load-time gate lets it register."""
    source = _valid_plan_source(name)
    path = _write(session_dir, filename, source)
    plan_loader.validation_records.record(plan_loader.hash_plan_body(source))
    return path


def _exclusion_no_match_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and _EXCLUSION_NO_MATCH_MARKER in r.getMessage()
    ]


def test_excluded_plan_absent_from_catalog_while_siblings_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped_dir = tmp_path / "shipped"
    _write(shipped_dir, "keep_me.py", _valid_plan_source("keep_me"))
    _write(shipped_dir, "drop_me.py", _valid_plan_source("drop_me"))

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "drop_me")

    facility = plan_loader.get_facility_plans()

    assert "drop_me" not in facility.plans
    assert "keep_me" in facility.plans


def test_exclusion_removes_a_shipped_tier_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped_dir = tmp_path / "shipped"
    _write(shipped_dir, "core_plan.py", _valid_plan_source("core_plan"))

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "core_plan")

    assert "core_plan" not in plan_loader.get_facility_plans().plans


def test_exclusion_removes_a_facility_tier_plan_via_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facility_dir = tmp_path / "facility"
    _write(facility_dir, "facility_plan.py", _valid_plan_source("facility_plan"))

    monkeypatch.setenv(_PLAN_DIRS_ENV, str(facility_dir))
    monkeypatch.setenv("OSPREY_CONFIG", str(_write_excluded_config(tmp_path, ["facility_plan"])))

    assert "facility_plan" not in plan_loader.get_facility_plans().plans


def test_exclusion_removes_a_session_tier_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "session"
    _register_session_plan(session_dir, "authored.py", "authored_plan")

    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(session_dir))

    # Sanity: it registers when not excluded.
    assert "authored_plan" in plan_loader.get_facility_plans().plans

    plan_loader.reset_facility_plans()
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "authored_plan")

    assert "authored_plan" not in plan_loader.get_facility_plans().plans


def test_unknown_excluded_name_warns_exactly_once_across_repeated_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    shipped_dir = tmp_path / "shipped"
    session_dir = tmp_path / "session"
    _write(shipped_dir, "keep_me.py", _valid_plan_source("keep_me"))

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(session_dir))
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "does_not_exist")

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        for _ in range(3):
            plan_loader.get_facility_plans()

    assert len(_exclusion_no_match_warnings(caplog)) == 1


def test_exclusion_warning_refires_after_the_registry_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The warn-once guard is keyed on the live `registry.keys()`; adding a new
    (session-tier) plan changes that key and must re-warn."""
    shipped_dir = tmp_path / "shipped"
    session_dir = tmp_path / "session"
    _write(shipped_dir, "keep_me.py", _valid_plan_source("keep_me"))
    session_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(session_dir))
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "does_not_exist")

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        plan_loader.get_facility_plans()
        plan_loader.get_facility_plans()
        assert len(_exclusion_no_match_warnings(caplog)) == 1

        # The session layer is re-scanned every call: adding a validated plan
        # changes registry.keys() without a cache reset, so the guard re-fires.
        _register_session_plan(session_dir, "extra.py", "session_extra")
        plan_loader.get_facility_plans()

    assert len(_exclusion_no_match_warnings(caplog)) == 2


def test_empty_exclusion_set_is_a_silent_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    shipped_dir = tmp_path / "shipped"
    _write(shipped_dir, "keep_me.py", _valid_plan_source("keep_me"))

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    monkeypatch.delenv(plan_loader._EXCLUDED_PLANS_ENV, raising=False)

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        facility = plan_loader.get_facility_plans()

    assert "keep_me" in facility.plans
    assert _exclusion_no_match_warnings(caplog) == []


def test_excluding_the_surviving_name_of_a_trust_collision_filters_it_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A higher-trust `facility` file wins the `shared_name` collision over a
    `shipped` file; excluding `shared_name` must drop the survivor and the
    rejected lower-trust loser must not resurface."""
    shipped_dir = tmp_path / "shipped"
    facility_dir = tmp_path / "facility"
    _write(shipped_dir, "a.py", _valid_plan_source("shared_name"))
    _write(facility_dir, "b.py", _valid_plan_source("shared_name"))

    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(facility_dir))
    monkeypatch.setenv(plan_loader._EXCLUDED_PLANS_ENV, "shared_name")

    facility = plan_loader.get_facility_plans()

    assert "shared_name" not in facility.plans


def test_shipped_plans_register_through_the_real_shipped_dir() -> None:
    """Sanity check: `plan_loader.py` is the sole plan registry — the shipped
    `orm`/`grid_scan` plans (in `plans_core/`) register through the ordinary
    `shipped`-tier directory scan, same as any other layer."""
    pytest.importorskip("bluesky")

    facility = plan_loader.get_facility_plans()
    assert set(facility.plans) == {"orm", "grid_scan"}
    assert all(spec.provenance == "shipped" for spec in facility.plans.values())
