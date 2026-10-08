"""Deleted surfaces stay deleted.

Each row names a module that survives and a name it no longer defines. The
module is imported and asked for the name, so a symbol brought back under its
old spelling fails here even where no text sweep reaches it. Some surfaces
are checked by parsing, by path or by module lookup instead, because importing
them would load a server library or they no longer exist. Once the old model
code is gone, no name of the retired engine and serving stack remains in the
shipped source.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

from tests.facility._batches import BATCHES, CURRENT_BATCH

REPO = Path(__file__).resolve().parents[2]

#: The serving runner; it imports the Channel Access server extension at
#: module level, so it is parsed, never imported.
SERVING_RUNNER = REPO / "src/osprey/services/virtual_accelerator/serving/runner.py"

#: Repo-relative paths that no longer exist.
DELETED_PATHS: tuple[str, ...] = (
    "src/osprey/services/virtual_accelerator/ioc",
    "src/osprey/services/virtual_accelerator/serving/model_stub.py",
    "src/osprey/services/virtual_accelerator/serving/pvdb.py",
    "src/osprey/services/virtual_accelerator/serving/write_path.py",
)

#: Dotted module paths that no longer resolve.
DELETED_MODULES: tuple[str, ...] = (
    "osprey.connectors.channel_taxonomy",
    "osprey.simulation.archiver_seed",
    "osprey.simulation.channel_schema",
    "osprey.simulation.engine",
    "osprey.simulation.expressions",
    "osprey.simulation.facility_spec",
    "osprey.simulation.machine",
    "osprey.simulation.procedural",
    "osprey.simulation.series",
    "osprey_connectors.channel_taxonomy",
    "osprey_connectors.simulation.archiver_seed",
    "osprey_connectors.simulation.engine",
    "osprey_connectors.simulation.expressions",
    "osprey_connectors.simulation.machine",
    "osprey_connectors.simulation.procedural",
)

#: The names of the retired engine and serving stack; none remains in the
#: shipped source once the stage that deletes the old model code closes.
RETIRED_SOURCE_NAMES = (
    "SimulationEngine|CohostRunner|CohostDriver|derive_record_type|classify_channel|"
    "VA_LATTICE|VA_BPM_ERRORS|VA_CORR_GAIN|VA_STANDIN_BPM_ERRORS|VA_NOISE_LEVEL|"
    "VA_ENTRYPOINT_MODULE|VA_CHANNELS_FILE|VA_STUCK_SETPOINTS|ordinalInFacility"
)

#: The stage whose close removes the last shipped spelling of those names.
RETIRED_SOURCE_STAGE = "7e"

#: The shipped source the sweep reads; the key manifest records deleted keys
#: by name, so it is left out.
RETIRED_SOURCE_PATHSPEC: tuple[str, ...] = (
    "src",
    "packages",
    ":(exclude)src/osprey/profiles/config_key_manifest.yml",
)

#: Surviving module -> names it no longer defines.
DELETED_NAMES: dict[str, tuple[str, ...]] = {
    "osprey.cli.build_profile_va_faults": (
        "STANDIN_BPM_ERRORS_ENV",
        "effective_standin_bpm_errors",
    ),
    "osprey.deployment.compose_generator": ("_standin_perturbation",),
    "osprey.deployment.container_lifecycle": (
        "_SOLVED_BASELINES_KIND",
        "_STANDIN_OFFSET_AXES",
        "_STANDIN_TRANSFORM_KIND",
        "_baselines_fingerprint",
        "_preflight_build_derived_env",
        "_preflight_served_lattice",
        "_served_monitor_readings",
        "_solved_monitor_baselines",
        "_standin_bpm_error_spec",
        "_standin_seed_transform",
    ),
    "osprey.deployment.reset": ("DERIVED_ENV_BANNER",),
    "osprey_connectors.dotenv": (
        "BUILD_DERIVED_BANNER",
        "BUILD_DERIVED_KEYS",
        "VA_LATTICE_DEFAULT",
        "VA_LATTICE_KEY",
        "resolved_va_lattice",
    ),
}


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module, names in sorted(DELETED_NAMES.items()) for name in names],
)
def test_the_deleted_name_is_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize("path", DELETED_PATHS)
def test_the_deleted_path_is_gone(path: str) -> None:
    # A directory left holding only bytecode caches is gone as source.
    target = REPO / path
    assert not target.is_file()
    assert not any(target.rglob("*.py"))


@pytest.mark.parametrize("module", DELETED_MODULES)
def test_the_deleted_module_is_gone(module: str) -> None:
    assert importlib.util.find_spec(module) is None


def retired_source_hits(root: Path) -> list[str]:
    """Every ``path:line`` of the shipped source spelling a retired name."""
    result = subprocess.run(
        ["git", "grep", "-n", "-E", RETIRED_SOURCE_NAMES, "--", *RETIRED_SOURCE_PATHSPEC],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    return [":".join(line.split(":", 2)[:2]) for line in result.stdout.splitlines()]


@pytest.mark.skipif(
    CURRENT_BATCH < BATCHES.index(RETIRED_SOURCE_STAGE),
    reason="the old model code still ships until its deleting stage closes",
)
def test_no_retired_name_remains_in_the_shipped_source() -> None:
    assert retired_source_hits(REPO) == []


def test_the_retired_source_sweep_finds_a_planted_name(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\nengine = SimulationEngine()\n")
    (tmp_path / "src/osprey/profiles").mkdir(parents=True)
    (tmp_path / "src/osprey/profiles/config_key_manifest.yml").write_text("VA_NOISE_LEVEL\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)

    assert retired_source_hits(tmp_path) == ["src/a.py:2"]


def test_the_rows_are_sorted() -> None:
    assert list(DELETED_NAMES) == sorted(DELETED_NAMES)
    assert list(DELETED_PATHS) == sorted(DELETED_PATHS)
    assert list(DELETED_MODULES) == sorted(DELETED_MODULES)
    for module, names in DELETED_NAMES.items():
        assert list(names) == sorted(names), module


def test_the_serving_runner_exports_only_the_model_runner() -> None:
    tree = ast.parse(SERVING_RUNNER.read_text(encoding="utf-8"))
    public_classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    }
    exported = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    ]

    assert public_classes == {"ModelRunner"}
    assert exported == [["ModelRunner"]]
