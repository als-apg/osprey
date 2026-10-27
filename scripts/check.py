#!/usr/bin/env python3
"""Answer "is this change clean" with one command, from one declaration.

A change is clean when every check below that applies to it ran here and passed.
Each check is declared once, in ``CHECKS``, and every caller runs it from there::

    check.py quick [--changed BASE [--worktree] | --base BASE] [--fix] [-- PATH...]
    check.py full  [--changed BASE [--worktree] | --base BASE]         [-- PATH...]
    check.py run ID... [--base BASE] [--shard cli|rest-a|rest-b] [-- EXEC-FLAGS...]
    check.py list [--tier quick|full] [--changed BASE] [--json]

Four kinds of check:

* ``invariant`` -- holds for the repository as a whole and always runs: the
  lockfile, ruff, the config-key guard, the changelog and plugin-version gates,
  and the rule that no tracked file is gitignored.
* ``static`` -- one per language or rule: mypy, the import contracts, the
  private-name imports, tsc, eslint, prettier.
* ``tests`` -- one per runner: the parallel pytest sweep, the serial
  real-terminal and wall-clock suites, the live Channel Access gate, Vitest,
  and the dock-engine's own Vitest suite.
* ``build`` -- the documentation, the dock-engine bundle, and the packages.

Two tiers. ``quick`` is what a commit needs, ``full`` is what a push needs and
adds the serial suites and the builds. ``run`` runs named checks as declared and
is what a CI step calls; flags after ``--`` are execution flags only (workers,
verbosity, coverage, timeouts), because a selection flag would make the caller a
second declaration.

``--changed BASE`` scopes a tier to the range ``BASE...HEAD``, or with
``--worktree`` to the merge base against the working tree, and then also hands
every check that judges a range the working tree (staged, unstaged and
untracked), so one verdict judges one range. A check whose scope
the range does not touch is reported as not touched; a path the range deleted or
moved away touches the scope it left. The test runners get only the tests the
changed paths map to. A path the mapping cannot place widens that runner to its
whole selection; scoping never narrows. A script a check runs through selects
``tests/scripts/``; a file under ``tests/<dir>/`` selects that directory when it
holds a test module and widens otherwise. A base also arms the
checks that judge a range. Paths after ``--`` are test files added to what the
range selects.

``list --changed BASE`` adds which checks and CI lanes the range ``BASE...HEAD``
touches, read as a tier reads it; it runs nothing.

``--base BASE`` arms every check that judges a range against ``BASE`` and leaves
the runners on their whole selection, so ``full --base BASE`` judges the whole
tree and the range in one verdict. ``--changed`` and ``--base`` may not name
different bases.

Exit status: 0 clean, 1 a check failed, 2 refused. A refusal means the command
could not judge: a missing toolchain, a base that does not resolve, an unknown
check, a flag that is not an execution flag. Every check of the run is attempted
even after an earlier one failed, and the last line is the verdict. A failed
test check names its failing tests, read from the report its runner writes for
the command. A refusal names every check it cannot judge and what to install.

What cannot run here is named, never counted as passed: a check bound to a venue
this machine is not, a check whose install is absent, a check that needs a base
when none was given, and the CI lanes this command never runs. A check that
proves something only against one install refuses any other install, and under
``run`` refuses its absence too; when it runs, the verdict names the install it
ran against.

Standard library only: this runs before the project's dependencies are installed
and must still report when installing them is what broke.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import ModuleType
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]

TIERS = ("quick", "full")
KINDS = ("invariant", "static", "tests", "build")

#: Exit statuses.
CLEAN, FAILED_EXIT, REFUSED_EXIT = 0, 1, 2


@dataclass(frozen=True)
class Check:
    """One check: the one name every caller uses and the one spelling it runs."""

    id: str
    kind: str
    tiers: frozenset[str]
    #: The command; ``{base}`` is filled from ``--changed`` or ``--base``. Empty
    #: for a check implemented in this module.
    argv: tuple[str, ...]
    #: Path globs that make the check relevant to a range; empty means always.
    scope: tuple[str, ...] = ()
    #: What must be present to judge; unmet is a refusal.
    needs: tuple[str, ...] = ()
    #: A check that judges a range declares how it judges the working tree: the
    #: command lines that replace ``argv`` under ``--worktree``. ``{base}`` is
    #: filled as in ``argv``; a line holding ``{untracked}`` is repeated once per
    #: untracked path and dropped when there is none.
    worktree: tuple[tuple[str, ...], ...] = ()
    #: A platform the check can only run on; unmet is named, not failed.
    venue: str | None = None
    #: Wall-clock bound through ``run_bounded.py``; 0 means unbounded.
    bound_s: int = 0
    #: The machine-readable report the runner writes and the command reads back,
    #: a key of ``REPORT_FLAGS``; ``None`` for a check that writes none.
    report: str | None = None
    #: What ``--fix`` runs to repair what this check reports; it writes files.
    fix: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class Lane:
    """A CI job this command never runs and always names."""

    job: str
    why: str
    #: Path globs; a range that touches one is told to run the job on CI.
    scope: tuple[str, ...]


@dataclass(frozen=True)
class UnitSelection:
    """The one statement of the parallel pytest sweep."""

    #: The paths the whole sweep collects.
    roots: tuple[str, ...]
    #: Each path the sweep leaves out, with the check or lane that runs it.
    ignores: tuple[tuple[str, str], ...]
    marker: str
    quick_marker: str
    #: Shard name to (paths, extra ignores); the shards cover the roots exactly once.
    shards: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]
    dist: tuple[str, ...]


BOTH = frozenset(TIERS)
FULL = frozenset({"full"})

LIVE_CA_VENUE = "linux-x86_64"

#: The venue's requirement in words, for the line that names an unmet one.
VENUE_NEEDS = {LIVE_CA_VENUE: "needs linux x86_64"}

#: The dock-engine package: its TypeScript sources, the bundle built from them,
#: its Vitest suite and the cases both runtimes read.
DOCK_ENGINE = "packages/dock-engine/**"

#: The scope of ``tsc``: the include globs of both ``tsconfig.json`` files and their inputs.
TSC_SCOPE = (
    "src/osprey/interfaces/vendor-globals.d.ts",
    "src/osprey/interfaces/**/static/js/**/*.js",
    "src/osprey/interfaces/bluesky_web/panels/**/*.js",
    "tests/**/*.js",
    "tests/**/*.mjs",
    "tsconfig.json",
    DOCK_ENGINE,
    "package*.json",
)

#: The ruff checks print GitHub annotations when they run on GitHub Actions.
GITHUB_FORMAT_CHECKS = ("ruff", "private-imports")

CHECKS: tuple[Check, ...] = (
    Check("lockfile", "invariant", BOTH, ("uv", "lock", "--check"), needs=("uv",)),
    Check(
        "ruff",
        "invariant",
        BOTH,
        ("uv", "run", "ruff", "check", "."),
        needs=("uv",),
        fix=(("uv", "run", "ruff", "check", ".", "--fix"),),
    ),
    Check(
        "ruff-format",
        "invariant",
        BOTH,
        ("uv", "run", "ruff", "format", "--check", "."),
        needs=("uv",),
        fix=(("uv", "run", "ruff", "format", "."),),
    ),
    Check(
        "config-keys",
        "invariant",
        BOTH,
        ("uv", "run", "python", "scripts/check_config_keys.py", "--back-test"),
        needs=("uv", "history"),
    ),
    Check(
        "changelog",
        "invariant",
        BOTH,
        ("python3", "scripts/changelog_fragments.py", "check", "--base", "{base}"),
        needs=("base",),
        worktree=(
            (
                "python3",
                "scripts/changelog_fragments.py",
                "check",
                "--base",
                "{base}",
                "--worktree",
            ),
        ),
    ),
    Check(
        "plugin-version",
        "invariant",
        BOTH,
        ("python3", "scripts/plugin_version.py", "check", "--base", "{base}"),
        needs=("base",),
        worktree=(
            ("python3", "scripts/plugin_version.py", "check", "--base", "{base}", "--worktree"),
        ),
    ),
    # Implemented here (BUILTINS): git lists what is both tracked and ignored.
    Check("tracked-ignored", "invariant", BOTH, ()),
    Check(
        "mypy",
        "static",
        BOTH,
        ("uv", "run", "python", "scripts/mypy_gate.py"),
        scope=("*.py", "pyproject.toml", "uv.lock"),
        needs=("uv",),
    ),
    Check(
        "import-contracts",
        "static",
        BOTH,
        ("uv", "run", "lint-imports"),
        scope=("*.py", "pyproject.toml", "uv.lock"),
        needs=("uv",),
    ),
    Check(
        "private-imports",
        "static",
        BOTH,
        ("uv", "run", "ruff", "check", "--preview", "--select=PLC2701", "."),
        scope=("*.py", "pyproject.toml"),
        needs=("uv",),
    ),
    Check("tsc", "static", BOTH, ("npm", "run", "typecheck"), scope=TSC_SCOPE, needs=("node",)),
    Check(
        "eslint",
        "static",
        BOTH,
        ("npm", "run", "lint"),
        scope=("*.js", "*.mjs", "*.ts", "eslint.config.js", "tools/eslint/**", "package*.json"),
        needs=("node",),
    ),
    Check(
        "prettier",
        "static",
        BOTH,
        ("npm", "run", "format:check"),
        scope=("*.js", "*.mjs", ".prettierrc.json", ".prettierignore", "package*.json"),
        needs=("node",),
        fix=(("npm", "run", "format"),),
    ),
    Check(
        "pytest",
        "tests",
        BOTH,
        ("uv", "run", "pytest"),
        needs=("uv",),
        bound_s=5400,
        report="junit-pytest",
    ),
    Check(
        "pytest-pty",
        "tests",
        FULL,
        ("uv", "run", "pytest", "tests/pty/", "-m", "pty"),
        scope=("tests/pty/**", "src/osprey/cli/**"),
        needs=("uv",),
        bound_s=600,
        report="junit-pytest",
    ),
    # A wall-clock budget timed beside loaded workers measures the workers, so
    # the sweep deselects the marker and this suite runs alone.
    Check(
        "pytest-wallclock",
        "tests",
        FULL,
        ("uv", "run", "pytest", "tests/integration", "-m", "wallclock"),
        scope=("tests/integration/**", "*.py", "pyproject.toml", "uv.lock"),
        needs=("uv",),
        bound_s=900,
        report="junit-pytest",
    ),
    Check(
        "live-ca",
        "tests",
        FULL,
        ("uv", "run", "python", "scripts/va/live_ca/gate.py", "--pva"),
        scope=(
            "tests/va/test_apply_fault.py",
            "tests/va/test_lume_pva_seam.py",
            "scripts/va/live_ca/**",
            "src/osprey/services/virtual_accelerator/**",
        ),
        needs=("uv", "va-extra"),
        venue=LIVE_CA_VENUE,
    ),
    Check(
        "vitest", "tests", BOTH, ("npx", "vitest", "run"), needs=("node",), report="junit-vitest"
    ),
    # The root Vitest config does not reach the package; it carries its own.
    Check(
        "dock-engine-tests",
        "tests",
        BOTH,
        ("npx", "vitest", "run", "-c", "packages/dock-engine/vitest.config.ts"),
        scope=(DOCK_ENGINE, "package*.json"),
        needs=("node",),
        report="junit-vitest",
    ),
    Check(
        "dock-engine-bundle",
        "build",
        BOTH,
        ("node", "packages/dock-engine/scripts/build.mjs", "--check"),
        scope=(DOCK_ENGINE, "package*.json"),
        needs=("node",),
    ),
    Check(
        "docs",
        "build",
        FULL,
        ("uv", "run", "make", "html", "SPHINXOPTS=-W --keep-going"),
        scope=("docs/**", "*.rst", "*.md"),
        needs=("uv", "docs-extra", "graphviz"),
    ),
    Check(
        "package",
        "build",
        FULL,
        (),
        scope=("pyproject.toml", "packages/**", "src/**", "MANIFEST*"),
        needs=("uv",),
    ),
)

#: Each check whose scope is computed from the tree being judged, to the
#: function that computes it from the tree's file list. Its declared scope is
#: empty and is never read.
DERIVED_SCOPES: dict[str, Callable[[Sequence[str]], tuple[str, ...]]] = {}

#: The directory a check's command runs in, relative to the repository root.
WORKDIRS = {"docs": "docs"}

#: The checks a ``run`` caller may pass execution flags to: the pytest runs.
EXEC_FLAG_CHECKS = ("pytest", "pytest-pty", "pytest-wallclock")

#: The quick tier's bound for the parallel sweep; the full tier's is the check's own.
#: It stops a sweep that outlives its last test (a teardown hang), never a slow
#: one, so it sits at about twice the sweep's runtime on a loaded host.
QUICK_PYTEST_BOUND_S = 1800

#: Where the parallel sweep finds the tests of a changed source file, first match
#: wins. ``package`` maps the nearest package to its directory under ``tests/``;
#: ``file`` maps a top-level ``scripts/<name>.py`` to ``tests/scripts/test_<name>.py``.
#: A source file under a root whose rule finds nothing widens the sweep.
TEST_MIRRORS: tuple[tuple[str, str], ...] = (
    ("src/osprey/", "package"),
    ("scripts/", "file"),
)

#: The test directories of the ``rest-a`` shard.
_REST_A = ("tests/facility", "tests/mcp_server", "tests/deployment")

UNIT_SELECTION = UnitSelection(
    # An explicit path replaces testpaths, so the connectors package's own
    # suite runs only where it is named.
    roots=("tests/", "packages/osprey-connectors/tests"),
    ignores=(
        ("tests/e2e", "e2e-tests"),
        ("tests/va/test_apply_fault.py", "live-ca"),
        ("tests/va/test_lume_pva_seam.py", "live-ca"),
        ("tests/services/channel_finder/graph_index/test_scale.py", "channel-finder-benchmarks"),
        ("tests/integration/test_build_boot.py", "build-boot-smoke"),
        ("tests/integration/test_preset_static.py", "static-checks"),
        ("tests/integration/test_preset_build.py", "static-checks"),
    ),
    marker="not pty and not wallclock",
    quick_marker="not pty and not wallclock and not slow",
    shards={
        "cli": (("tests/cli",), ()),
        "rest-a": (_REST_A, ()),
        "rest-b": (("tests/", "packages/osprey-connectors/tests"), ("tests/cli", *_REST_A)),
    },
    dist=("--dist", "loadgroup"),
)

#: The execution flags a tier passes to the pytest checks; a ``run`` caller passes its own.
TIER_EXEC_FLAGS = {
    "quick": {"pytest": ("-n", "auto", "--maxfail=1", "--tb=line", "-q")},
    "full": {
        "pytest": ("-n", "auto", "--tb=short", "--timeout=600", "--timeout-method=signal"),
        "pytest-pty": ("--tb=short",),
        "pytest-wallclock": ("--tb=short",),
    },
}

#: The CI jobs that call this command; every other required job is a lane.
CALLER_JOBS = ("test", "lint", "docs", "package", "frontend-js")

#: The paths of the scan stack: a range that touches one needs the container lanes.
SCAN_STACK = (
    "src/osprey/services/bluesky_bridge/**",
    "src/osprey/interfaces/bluesky_web/**",
    "src/osprey/templates/services/bluesky**",
    "src/osprey/mcp_server/bluesky/**",
    "tests/e2e/test_bluesky**",
    "tests/e2e/test_tiled_roundtrip**",
    "tests/e2e/test_grid_scan_roundtrip**",
    "tests/e2e/_orm_stack**",
    "tests/e2e/_queue_drive**",
)

_PLUGIN_TREES = ("plugins/**", ".claude-plugin/**", ".agents/plugins/**")


def _e2e(job: str, *tests: str, why: str = "containers") -> Lane:
    return Lane(job, why, tuple(f"tests/e2e/{name}" for name in tests))


def _scan(job: str, *tests: str, why: str = "containers") -> Lane:
    return Lane(job, why, (*SCAN_STACK, *(f"tests/e2e/{name}" for name in tests)))


def _va(job: str, *tests: str) -> Lane:
    return Lane(job, "containers", tuple(f"tests/va/e2e/{name}" for name in tests))


LANES: tuple[Lane, ...] = (
    Lane(
        "dependency-floor",
        "dependency floor",
        (
            "pyproject.toml",
            "uv.lock",
            "tests/connectors/test_archiver_timezone.py",
            "tests/connectors/test_mock_connector.py",
            "tests/connectors/test_mya_registration.py",
            "packages/osprey-connectors/tests/osprey_connectors_tests/test_archiver_timerange.py",
            "packages/osprey-connectors/tests/osprey_connectors_tests/test_doocs_archiver_connector.py",
            "packages/osprey-connectors/tests/osprey_connectors_tests/test_epics_archiver_connector.py",
            "packages/osprey-connectors/tests/osprey_connectors_tests/test_mya_archiver_connector.py",
            "tests/connectors/test_simulation_integration.py",
            "tests/utils/test_timeseries.py",
        ),
    ),
    Lane(
        "connectors-standalone",
        "installs the connectors alone",
        ("packages/osprey-connectors/**", "pyproject.toml", "uv.lock"),
    ),
    Lane(
        "static-checks",
        "builds a project",
        ("tests/integration/test_preset_static.py", "tests/integration/test_preset_build.py"),
    ),
    Lane(
        "visual-theming",
        "browser",
        (
            "src/osprey/interfaces/**",
            "tests/interfaces/**/*.py",
            "tests/interfaces/design_system/baselines/**",
            "tests/docs/screenshots/**",
        ),
    ),
    Lane("build-boot-smoke", "matrix", ("tests/integration/test_build_boot.py",)),
    Lane("e2e-tests", "models", ("tests/e2e/**",)),
    Lane("e2e-no-model", "containers", ("tests/e2e/**",)),
    Lane("deploy-e2e", "containers", ("src/osprey/**",)),
    _e2e("dispatch-deploy-e2e", "test_dispatch_deploy.py"),
    _e2e("single-user-narrowing-e2e", "test_single_user_narrowing_e2e.py"),
    _e2e("dispatch-overlay-e2e", "test_dispatch_overlay_visibility.py"),
    _scan("bluesky-deploy-e2e", "test_bluesky_deploy.py"),
    _scan("bluesky-web-deploy-e2e", "test_bluesky_web_deploy.py"),
    _e2e("va-substrate-equivalence-e2e", "test_va_substrate_equivalence.py"),
    _scan(
        "orm-roundtrip-e2e",
        "test_orm_roundtrip.py",
        "test_grid_scan_roundtrip.py",
        "test_bump_roundtrip.py",
        "test_preflight_probe.py",
    ),
    _scan("bluesky-queue-e2e", "test_bluesky_queue_e2e.py"),
    _e2e("scan-agentic-e2e", "test_plan_stack_agentic.py", why="models"),
    _e2e("archiver-world-e2e", "test_archiver_world_e2e.py"),
    _e2e("graph-reseed-e2e", "test_graph_reseed_stamp.py"),
    _scan("tiled-roundtrip-e2e", "test_tiled_roundtrip.py"),
    _scan("bluesky-catalog-e2e", "test_bluesky_catalog_e2e.py"),
    _scan("bluesky-sandbox-escape-e2e", "test_bluesky_sandbox_escape_e2e.py"),
    _e2e("nextcloud-talk-bridge-e2e", "test_nextcloud_talk_bridge_e2e.py"),
    _e2e("gchat-bridge-e2e", "test_gchat_bridge_e2e.py", "fixtures/test_gchat_fixture_smoke.py"),
    _e2e(
        "teams-bridge-e2e",
        "test_teams_bridge_e2e.py",
        "fixtures/test_teams_fixture_smoke.py",
        why="loopback services",
    ),
    _e2e("dockerfile-e2e", "test_dockerfile_e2e.py"),
    _e2e("privilege-split-e2e", "test_privilege_split.py", "test_audit_two_container.py"),
    _e2e("execution-environment-parity-e2e", "test_execution_environment_parity.py"),
    _e2e("agentic-per-preset", "test_preset_agentic.py", why="models"),
    _e2e("deploy-writeback-e2e", "test_deploy_writeback_e2e.py"),
    _e2e("multi-user-deploy-lifecycle-e2e", "test_deploy_lifecycle.py"),
    _e2e("multi-user-deploy-lifecycle-e2e-podman", "test_deploy_lifecycle.py"),
    _e2e("auth-perimeter-e2e", "web_terminals/test_auth_perimeter.py"),
    _e2e("terminal-auth-multiuser-e2e", "web_terminals/test_terminal_auth_multiuser_e2e.py"),
    _e2e("full-chain-auth-e2e", "test_full_chain_auth.py"),
    _e2e("two-shape-boot-e2e", "test_two_shape_boot.py"),
    Lane(
        "qmd-sidecar-e2e",
        "containers",
        ("tests/integration/test_qmd_models_mount.py", "tests/e2e/test_deploy_lifecycle.py"),
    ),
    _e2e("jupyter-panel-e2e", "test_jupyter_panel_lifecycle.py"),
    _va(
        "target-switch-e2e",
        "test_target_switch.py",
        "test_bench_ioc_live.py",
        "test_live_standin.py",
        "test_per_target_posture.py",
        "test_model_status_va.py",
    ),
    _e2e("target-switch-agentic-e2e", "test_target_switch_agentic.py", why="models"),
    _va(
        "va-live-e2e",
        "test_approval_smoke.py",
        "test_limits_enforcement.py",
        "test_orbit_response.py",
        "test_full_sweep.py",
        "test_finder_live_reads.py",
        "test_scenario_reload.py",
        "test_sim_apply_va.py",
        "test_va_put_surface.py",
        "test_planted_journal.py",
        "test_substrate_parity.py",
        "test_chain_boot.py",
        "test_pyaml_measure_guarded.py",
        "test_mml_trees_boot.py",
        "test_check_b_served.py",
    ),
    _va("va-bluesky-deploy-e2e", "test_bluesky_lanes.py"),
    Lane("plugin-smoke", "plugin install", _PLUGIN_TREES),
    Lane(
        "channel-finder-benchmarks",
        "on-demand benchmark",
        (
            "tests/services/channel_finder/graph_index/test_scale.py",
            "tests/e2e/claude_code/test_channel_finder_mcp_benchmarks.py",
        ),
    ),
)

#: The connectors wheel's import walk, shared with release-connectors.yml so the
#: wheel a connectors tag publishes passes the check every pull request runs.
#: Run with ``-I`` by the bare venv's interpreter: neither the checkout nor
#: ``PYTHONPATH`` can lend it a module the wheel does not ship.
LEAN_IMPORT_WALK = "scripts/ci/connectors_lean_import_walk.py"

#: Run with the framework wheel as its argument: the wheel bundles dock_engine
#: and its package data.
DOCK_ENGINE_WHEEL_PROBE = """\
import sys, zipfile
(wheel,) = sys.argv[1:]
names = zipfile.ZipFile(wheel).namelist()
assert 'dock_engine/__init__.py' in names, f'{wheel} carries no dock_engine package'
cases = [n for n in names if n.startswith('dock_engine/cases/validate/') and n.endswith('.json')]
assert cases, f'{wheel} carries no dock_engine/cases/validate/ case'
assert 'dock_engine/manifest.json' in names, f'{wheel} carries no dock_engine/manifest.json'
print(f'DOCK ENGINE IN FRAMEWORK WHEEL OK ({len(cases)} validate cases)')
"""

#: Run by a venv holding the framework, connectors and pyaml-cs-osprey wheels.
PYAML_WHEEL_PROBE = """\
import importlib, pkgutil
import pyaml_cs_osprey
def _raise(name):
    raise ImportError(f'failed to walk {name}')
modules = [m.name for m in pkgutil.walk_packages(pyaml_cs_osprey.__path__, 'pyaml_cs_osprey.', onerror=_raise)]
for name in modules:
    importlib.import_module(name)
print(f'PYAML WHEEL OK ({len(modules)} modules)')
import dock_engine
assert dock_engine.cases_dir().is_dir(), dock_engine.cases_dir()
assert len(dock_engine.rules_digest()) == 64
print(f'DOCK ENGINE IMPORT OK ({dock_engine.__version__})')
"""

#: The package build, in order; each step needs the one before it. ``{scratch}``
#: is a fresh temporary directory, and a ``dist/`` glob is expanded when the
#: step runs.
PACKAGE_STEPS: tuple[tuple[str, ...], ...] = (
    ("uv", "build"),
    ("uvx", "twine", "check", "dist/*"),
    ("uv", "build", "--package", "osprey-connectors"),
    ("uv", "venv", "{scratch}/lean", "--python", "3.11"),
    (
        "uv",
        "pip",
        "install",
        "--python",
        "{scratch}/lean/bin/python",
        "dist/osprey_connectors-*.whl",
    ),
    ("{scratch}/lean/bin/python", "-I", LEAN_IMPORT_WALK),
    ("python3", "-c", DOCK_ENGINE_WHEEL_PROBE, "dist/osprey_framework-*.whl"),
    ("uv", "build", "--package", "pyaml-cs-osprey"),
    ("uv", "venv", "{scratch}/pyaml", "--python", "3.11"),
    (
        *("uv", "pip", "install", "--python", "{scratch}/pyaml/bin/python"),
        *("dist/osprey_framework-*.whl", "dist/osprey_connectors-*.whl"),
        "dist/pyaml_cs_osprey-*.whl",
    ),
    ("{scratch}/pyaml/bin/python", "-c", PYAML_WHEEL_PROBE),
)

#: Prints the keys of the documentation's ``redirects`` table, space separated.
REDIRECT_KEYS_PROBE = (
    "uv",
    "run",
    "python",
    "-c",
    "import os, runpy; os.chdir('docs/source'); "
    "print(' '.join(runpy.run_path('conf.py')['redirects']))",
)

DOCS_HTML = "docs/build/html"

#: Importable only in a venv synced with the docs extra.
DOCS_EXTRA_PROBE = ("uv", "run", "python", "-c", "import myst_parser")

#: Importable only in a venv synced with the virtual-accelerator extra, on its
#: platform. The modules are the ones the live gate requires.
LIVE_CA_PROBE = ("uv", "run", "python", "-c", "import pcaspy, p4p, lume_pva_apg")

#: Every command ``--fix`` can run, in declaration order; the checks never write.
FIX_COMMANDS: tuple[tuple[str, ...], ...] = tuple(argv for c in CHECKS for argv in c.fix)

#: What a failed check that declares a fix tells its reader to run.
FIX_HINT = "uv run python scripts/check.py quick --fix"

#: The module that owns the JavaScript module-to-test mapping.
GUARD_MODULE = "tests/interfaces/test_js_module_tests.py"

#: Suffixes whose change reaches no test runner.
INERT_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".woff", ".woff2", ".ttf", ".pdf"}
)

JS_SUFFIXES = (".js", ".mjs", ".cjs")

#: Files whose change reaches every Vitest file.
VITEST_WIDE_NAMES = frozenset({"vitest.config.js", "package.json", "package-lock.json"})
VITEST_SETUP = "tests/vitest.setup.mjs"

#: The interface every other interface imports through the design-system alias.
SHARED_INTERFACE = "design_system"

WHOLE, PATHS, NOT_TOUCHED = "whole", "paths", "not touched"


class Refused(Exception):
    """The command cannot judge; the message says what to do about it."""


@dataclass(frozen=True)
class Selection:
    """What a tier does with one check: run it whole, on these paths, or not at all."""

    state: str
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """A tier resolved against a range: one selection per check, and the lanes to name."""

    tier: str
    selections: dict[str, Selection]
    lanes: tuple[str, ...]
    scoped: bool


@dataclass(frozen=True)
class Context:
    """The facts of one run that the checks share."""

    tier: str = "full"
    base: str | None = None
    shard: str | None = None
    #: Per check id; ``"*"`` holds a ``run`` caller's flags for every pytest check.
    exec_flags: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: The directory the runners write their reports into, one subdirectory per check.
    reports: str = "{reports}"
    #: The range ends at the working tree, not at HEAD.
    worktree: bool = False
    #: The checks were named by a ``run`` caller, so an install a tier does
    #: without is a refusal.
    named: bool = False


#: Each report to the flags that make its runner write it; appended last on the line.
REPORT_FLAGS: dict[str, tuple[str, ...]] = {
    # xunit1 is the family whose file attribute rebuilds the node id exactly.
    "junit-pytest": ("--junitxml={reports}/{id}/junit.xml", "-o", "junit_family=xunit1"),
    "junit-vitest": (
        "--reporter=default",
        "--reporter=junit",
        "--outputFile.junit={reports}/{id}/junit.xml",
    ),
}


# ---------------------------------------------------------------------------
# Processes


def _say(line: str) -> None:
    print(f"check: {line}", flush=True)


def _run(argv: Sequence[str], cwd: Path | None = None) -> int:
    """Run *argv* with this process's stdout and stderr, and return its exit status.

    The child's output is never captured, redirected or discarded: what a check
    prints is part of what it reports.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        return subprocess.run(list(argv), cwd=cwd or ROOT).returncode
    except FileNotFoundError:
        print(f"check: {argv[0]}: command not found", file=sys.stderr, flush=True)
        return 127


def _capture(argv: Sequence[str]) -> tuple[int, str]:
    """Run a probe and return its exit status and stdout."""
    try:
        done = subprocess.run(list(argv), cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return 127, ""
    return done.returncode, done.stdout


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        raise Refused(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def _git_paths(subcommand: str, *args: str) -> list[str]:
    """The paths a git listing names, read NUL-separated so no path is split or quoted."""
    return [path for path in _git(subcommand, "-z", *args).split("\0") if path]


# ---------------------------------------------------------------------------
# Needs and venues


def _need_uv() -> str | None:
    if shutil.which("uv"):
        return None
    return "uv is not on PATH: install uv, then run uv sync --extra dev"


def _need_node() -> str | None:
    if shutil.which("node") and shutil.which("npm") and (ROOT / "node_modules").is_dir():
        return None
    return "the Node toolchain is not installed: install Node, then run npm ci"


def _need_history() -> str | None:
    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        return "this clone is shallow: run git fetch --unshallow"
    return None


def _need_docs_extra() -> str | None:
    if _capture(DOCS_EXTRA_PROBE)[0] == 0:
        return None
    return "the docs extra is not installed: run uv sync --extra docs"


def _need_graphviz() -> str | None:
    if shutil.which("dot"):
        return None
    return "graphviz is not on PATH: the docs build draws its diagrams with dot; install graphviz"


def _need_va_extra() -> str | None:
    if _capture(LIVE_CA_PROBE)[0] == 0:
        return None
    return "the virtual-accelerator extra is not installed: run uv sync --extra virtual-accelerator"


#: Each need to its probe: ``None`` when met, else the refusal. ``base`` is not a
#: probe, it is an argument.
NEED_PROBES: dict[str, Callable[[], str | None]] = {
    "uv": _need_uv,
    "node": _need_node,
    "history": _need_history,
    "docs-extra": _need_docs_extra,
    "graphviz": _need_graphviz,
    "va-extra": _need_va_extra,
}


def _on_live_ca_venue() -> bool:
    """The platform the virtual-accelerator extra's marker installs on."""
    return sys.platform == "linux" and platform.machine() == "x86_64"


#: Each venue to the probe that says whether this machine is one.
VENUE_PROBES: dict[str, Callable[[], bool]] = {LIVE_CA_VENUE: _on_live_ca_venue}


#: A probe's exit status when what it looks for is not installed at all.
ABSENT_EXIT = 2


@dataclass(frozen=True)
class Install:
    """An install a check proves something only against.

    A tier does without it: when it is absent the check is named as not run
    here. A ``run`` caller asked for the proof, so there its absence is a
    refusal, and so is any install other than the proven one, everywhere.
    """

    #: Exit 0 when the proven install is there, ``ABSENT_EXIT`` when nothing
    #: is, anything else when another one is; the last line it prints says which.
    probe: tuple[str, ...]
    #: What to install, for a refusal.
    instruction: Callable[[], str]
    #: The verdict's line for a run against the proven install, given whether
    #: the check passed.
    attestation: Callable[[bool], str]


#: Each check that needs an install, to that install.
INSTALLS: dict[str, Install] = {}


def probe_install(install: Install) -> tuple[int, str]:
    """The exit status of *install*'s probe and the line it printed last."""
    status, out = _capture(install.probe)
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    said = lines[-1] if lines else f"{shlex.join(install.probe)} exited {status}"
    return status, said


def require_base(base: str) -> None:
    """Refuse a base that names no commit."""
    done = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise Refused(f"{base} does not resolve to a commit: fetch it or name another base")


# ---------------------------------------------------------------------------
# Selection


def matches(path: str, pattern: str) -> bool:
    """Whether *path* matches the scope glob *pattern*.

    ``**`` crosses directories and ``*`` does not. A pattern without a slash is
    matched against the file name at any depth.
    """
    target = path if "/" in pattern else path.rsplit("/", 1)[-1]
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.fullmatch("".join(parts), target) is not None


def touches(paths: Iterable[str], scope: Sequence[str]) -> bool:
    """Whether any of *paths* falls in *scope*; an empty scope is every change."""
    if not scope:
        return True
    return any(matches(path, pattern) for path in paths for pattern in scope)


def _range(base: str, worktree: bool) -> str:
    """The git revision argument for the range: committed state, or the working tree."""
    if worktree:
        return _git("merge-base", base, "HEAD").strip()
    return f"{base}...HEAD"


def touched(base: str, worktree: bool = False, *, scope: bool = False) -> list[str]:
    """The paths the range changed and still holds, repo-relative.

    Committed state is ``BASE...HEAD``. With *worktree* it is the merge base
    against the working tree, so staged, unstaged and untracked files count.

    With *scope* it is every path the range changed, the deleted ones too, and
    a move names both the path it left and the one it reached: a change that
    takes a file out of a scope touches that scope. A runner can be handed
    only a file that is there, so its selection reads the default form.
    """
    if scope:
        listing = ("--name-only", "--no-renames")
    else:
        listing = ("--name-only", "--diff-filter=d")
    paths = set(_git_paths("diff", *listing, _range(base, worktree)))
    if worktree:
        paths.update(_git_paths("ls-files", "--others", "--exclude-standard"))
    return sorted(paths)


def repository_files(worktree: bool = False) -> list[str]:
    """Every tracked path, plus the untracked ones when the working tree is judged."""
    files = _git_paths("ls-files")
    if worktree:
        files += _git_paths("ls-files", "--others", "--exclude-standard")
    return files


def _guard() -> ModuleType | None:
    """The module that owns the JavaScript module-to-test mapping, loaded by
    path, or ``None`` while the tree holds no such module."""
    if not (ROOT / GUARD_MODULE).is_file():
        return None
    spec = importlib.util.spec_from_file_location("js_module_tests", ROOT / GUARD_MODULE)
    if spec is None or spec.loader is None:
        raise Refused(f"cannot load {GUARD_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hosted_outside_sweep(path: str) -> bool:
    """Whether the parallel sweep leaves *path* to another check or to a lane."""
    for ignored, _host in UNIT_SELECTION.ignores:
        if path == ignored or path.startswith(ignored.rstrip("/") + "/"):
            return True
    return path.startswith("tests/pty/")


def _is_pytest_file(name: str) -> bool:
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py")


def _holds_test_module(directory: str, files: Sequence[str]) -> bool:
    """Whether *directory* holds a test file the sweep would collect."""
    prefix = directory + "/"
    return any(
        name.startswith(prefix) and _is_pytest_file(name.rsplit("/", 1)[-1]) for name in files
    )


def _mirror_root(path: str) -> tuple[str, str] | None:
    """The ``TEST_MIRRORS`` entry whose source root holds *path*, if any."""
    return next((entry for entry in TEST_MIRRORS if path.startswith(entry[0])), None)


def _mirror(path: str, files: Sequence[str]) -> str | None:
    """The tests that mirror *path* under its ``TEST_MIRRORS`` rule, if they exist."""
    entry = _mirror_root(path)
    if entry is None:
        return None
    root, rule = entry
    if rule == "file":
        relative = PurePosixPath(path).relative_to(root)
        if len(relative.parts) != 1 or relative.suffix != ".py":
            return None
        mirror = f"tests/{root.rstrip('/')}/test_{relative.stem}.py"
        return mirror if mirror in set(files) else None
    known = set(files)
    top = root.rstrip("/")
    package = PurePosixPath(path).parent
    while package.as_posix() != top and f"{package}/__init__.py" not in known:
        package = package.parent
    if package.as_posix() == top:
        return None
    mirror = "tests/" + package.relative_to(top).as_posix()
    return mirror if any(name.startswith(mirror + "/") for name in files) else None


def _select_pytest(paths: Sequence[str], files: Sequence[str]) -> Selection:
    chosen: list[str] = []
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        suffix = PurePosixPath(path).suffix
        if _hosted_outside_sweep(path):
            continue
        if suffix == ".py":
            if path.startswith("tests/") and _is_pytest_file(name):
                chosen.append(path)
                continue
            if path in SHAPING_SCRIPTS:
                chosen.append("tests/scripts")
                continue
            if name == "conftest.py" or _mirror_root(path) is None:
                return Selection(WHOLE)
        elif name in ("pyproject.toml", "uv.lock"):
            return Selection(WHOLE)
        elif suffix in JS_SUFFIXES or suffix in INERT_SUFFIXES:
            continue
        elif path.startswith("tests/") and len(PurePosixPath(path).parts) > 2:
            directory = "tests/" + PurePosixPath(path).parts[1]
            if not _holds_test_module(directory, files):
                return Selection(WHOLE)
            chosen.append(directory)
            continue
        elif path.startswith(("packages/", "tests/")):
            return Selection(WHOLE)
        elif not path.startswith("src/osprey/"):
            continue
        mirror = _mirror(path, files)
        if mirror is None:
            return Selection(WHOLE)
        chosen.append(mirror)
    return _paths(chosen)


def _select_vitest(paths: Sequence[str], files: Sequence[str]) -> Selection:
    guard = _guard()
    if guard is None:
        # Without the mapping no JavaScript change can be placed, so any one
        # widens Vitest to its whole selection.
        reaches = any(
            path.endswith(JS_SUFFIXES)
            or path.rsplit("/", 1)[-1] in VITEST_WIDE_NAMES
            or path == VITEST_SETUP
            for path in paths
        )
        return Selection(WHOLE) if reaches else Selection(NOT_TOUCHED)
    chosen: list[str] = []
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        if path.endswith(guard.TEST_SUFFIXES):
            if path.startswith(("tests/", guard.INTERFACES_ROOT)):
                chosen.append(path)
        elif guard.modules([path]):
            interface = guard.interface_of(path)
            if interface == SHARED_INTERFACE:
                return Selection(WHOLE)
            own = guard.covering_tests(path, files)
            if own:
                chosen += own
                continue
            fallback = f"{guard.TESTS_ROOT}{interface}/"
            if not any(f.startswith(fallback) and f.endswith(guard.TEST_SUFFIXES) for f in files):
                return Selection(WHOLE)
            chosen.append(fallback)
        elif name in VITEST_WIDE_NAMES or path == VITEST_SETUP:
            return Selection(WHOLE)
        elif path.endswith(JS_SUFFIXES) and path.startswith(("tests/", "src/")):
            return Selection(WHOLE)
    return _paths(chosen)


def _paths(chosen: Iterable[str]) -> Selection:
    unique = tuple(dict.fromkeys(chosen))
    return Selection(PATHS, unique) if unique else Selection(NOT_TOUCHED)


def _route_named(named: Sequence[str]) -> dict[str, list[str]]:
    """Send each named test file to its runner by suffix."""
    routed: dict[str, list[str]] = {"pytest": [], "vitest": []}
    for path in named:
        if path.endswith((".test.js", ".test.mjs")):
            routed["vitest"].append(path)
        elif path.endswith(".py"):
            routed["pytest"].append(path)
        else:
            raise Refused(f"{path}: not a test file; name a .py, .test.js or .test.mjs path")
    return routed


def tier_checks(tier: str) -> list[Check]:
    """The checks of *tier*, in declaration order."""
    return [check for check in CHECKS if tier in check.tiers]


def select(
    tier: str,
    paths: Sequence[str] | None,
    named: Sequence[str] = (),
    files: Sequence[str] | None = None,
    reach: Sequence[str] | None = None,
) -> Plan:
    """Resolve *tier* against the changed *paths*, or run it whole when there are none.

    *paths* are the changed files the range still holds, which the runners'
    selections read. *reach* is every path the range changed, deleted and
    moved-away ones included, which every scope reads; it is *paths* when not
    given. *named* are test files added to what the range selects. *files* is
    the repository's file list, read from git when not given.
    """
    routed = _route_named(named)
    if paths is None:
        whole = {check.id: Selection(WHOLE) for check in tier_checks(tier)}
        lanes = tuple(lane.job for lane in LANES) if tier == "full" else ()
        return Plan(tier, whole, lanes, scoped=False)
    if files is None:
        files = repository_files()
    if reach is None:
        reach = paths
    selections: dict[str, Selection] = {}
    for check in tier_checks(tier):
        if check.id == "pytest":
            chosen = _select_pytest(paths, files)
        elif check.id == "vitest":
            chosen = _select_vitest(paths, files)
        elif check.id in DERIVED_SCOPES:
            derived = DERIVED_SCOPES[check.id](files)
            chosen = Selection(WHOLE) if touches(reach, derived) else Selection(NOT_TOUCHED)
        elif touches(reach, check.scope):
            chosen = Selection(WHOLE)
        else:
            chosen = Selection(NOT_TOUCHED)
        extra = routed.get(check.id, [])
        if extra and chosen.state != WHOLE:
            chosen = _paths([*chosen.paths, *extra])
        selections[check.id] = chosen
    lanes = tuple(lane.job for lane in LANES if touches(reach, lane.scope))
    return Plan(tier, selections, lanes, scoped=True)


# ---------------------------------------------------------------------------
# Commands


def _is_exec_flag(flag: str) -> bool:
    return (
        flag in ("-v", "-q")
        or re.fullmatch(r"-n(\d+|auto|logical)", flag) is not None
        or flag.startswith(("--tb=", "--durations=", "--timeout=", "--timeout-method="))
        or flag.startswith(("--maxfail=", "--cov"))
    )


def validate_exec_flags(flags: Sequence[str]) -> None:
    """Refuse anything that is not an execution flag."""
    index = 0
    while index < len(flags):
        flag = flags[index]
        following = flags[index + 1] if index + 1 < len(flags) else ""
        if flag == "-n" and re.fullmatch(r"\d+|auto|logical", following):
            index += 2
        elif _is_exec_flag(flag):
            index += 1
        else:
            raise Refused(
                f"{flag}: not an execution flag; what a check selects is declared in "
                "scripts/check.py"
            )


def unit_argv(
    tier: str,
    targets: Sequence[str] = (),
    shard: str | None = None,
    unit: UnitSelection = UNIT_SELECTION,
) -> list[str]:
    """The parallel sweep's selection: paths, ignores and marker expression."""
    roots, shard_ignores = unit.shards[shard] if shard else (unit.roots, ())
    ignores = [*shard_ignores, *(path for path, _host in unit.ignores)]
    marker = unit.quick_marker if tier == "quick" else unit.marker
    return [
        *(targets or roots),
        *(f"--ignore={path}" for path in ignores),
        "-m",
        marker,
    ]


def _bounded(seconds: int, argv: Sequence[str]) -> list[str]:
    return ["uv", "run", "python", "scripts/run_bounded.py", str(seconds), "--", *argv]


def _shaping_scripts(checks: Iterable[Check]) -> frozenset[str]:
    """The scripts a check runs through: a change to one can change any check's verdict."""
    argvs = [*(c.argv for c in checks), _bounded(0, [])]
    named = {
        arg for argv in argvs for arg in argv if arg.startswith("scripts/") and arg.endswith(".py")
    }
    return frozenset({*named, Path(__file__).resolve().relative_to(ROOT).as_posix()})


#: Derived from ``CHECKS``, so a check that names a new script makes it shaping.
SHAPING_SCRIPTS = _shaping_scripts(CHECKS)


def report_flags(check: Check, context: Context) -> list[str]:
    """The flags that make *check*'s runner write its report into the run's directory."""
    if not check.report:
        return []
    return [
        flag.replace("{reports}", context.reports).replace("{id}", check.id)
        for flag in REPORT_FLAGS[check.report]
    ]


def _worktree_lines(lines: Iterable[Sequence[str]], base: str) -> list[list[str]]:
    """*lines* with ``{base}`` filled and each ``{untracked}`` line once per untracked path."""
    untracked: list[str] | None = None
    filled: list[list[str]] = []
    for line in lines:
        line = [part.replace("{base}", base) for part in line]
        if "{untracked}" not in line:
            filled.append(line)
            continue
        if untracked is None:
            untracked = _git_paths("ls-files", "--others", "--exclude-standard")
        filled += [[path if part == "{untracked}" else part for part in line] for path in untracked]
    return filled


def commands(check: Check, selection: Selection, context: Context) -> list[list[str]]:
    """The effective command lines of *check* under *selection*, without running them."""
    flags = list(context.exec_flags.get(check.id, context.exec_flags.get("*", ())))
    base = context.base or "{base}"
    written = report_flags(check, context)
    if check.id == "pytest":
        bound = QUICK_PYTEST_BOUND_S if context.tier == "quick" else check.bound_s
        selected = unit_argv(context.tier, selection.paths, context.shard)
        argv = [*check.argv, *selected, *flags, *UNIT_SELECTION.dist, *written]
        return [_bounded(bound, argv)]
    if check.id in ("pytest-pty", "pytest-wallclock"):
        return [_bounded(check.bound_s, [*check.argv, *flags, *written])]
    if check.id == "vitest":
        return [[*check.argv, *selection.paths, *written]]
    if check.id == "package":
        return [list(step) for step in PACKAGE_STEPS]
    if check.id in GITHUB_FORMAT_CHECKS and os.environ.get("GITHUB_ACTIONS"):
        return [[*check.argv, "--output-format=github"]]
    if context.worktree and check.worktree:
        return [[*line, *written] for line in _worktree_lines(check.worktree, base)]
    if not check.argv:
        return []
    return [[*(part.replace("{base}", base) for part in check.argv), *written]]


# ---------------------------------------------------------------------------
# Checks implemented here


def missing_redirect_stubs(keys: Iterable[str], html_root: Path) -> list[str]:
    """The redirect keys whose stub page the documentation build did not write."""
    return [key for key in keys if not (html_root / f"{key}.html").is_file()]


def _check_redirect_stubs(_context: Context) -> int:
    status, out = _capture(REDIRECT_KEYS_PROBE)
    if status != 0:
        _say("docs: cannot read the redirects table of docs/source/conf.py")
        return FAILED_EXIT
    missing = missing_redirect_stubs(out.split(), ROOT / DOCS_HTML)
    if missing:
        _say(f"docs: redirect stubs missing from the build: {' '.join(missing)}")
        return FAILED_EXIT
    return CLEAN


def _expand(step: Sequence[str], scratch: str) -> list[str]:
    """Fill ``{scratch}`` and expand a ``dist/`` glob against the repository root."""
    argv: list[str] = []
    for part in step:
        part = part.replace("{scratch}", scratch)
        if part.startswith("dist/") and "*" in part:
            found = sorted(glob.glob(part, root_dir=ROOT))
            argv += found or [part]
        else:
            argv.append(part)
    return argv


def _check_package(_context: Context) -> int:
    # dist/ outlives a branch switch and `uv build` only adds to it, so anything
    # left there would be checked as this tree's.
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="package-check-") as scratch:
        for step in PACKAGE_STEPS:
            status = _run(_expand(step, scratch))
            if status != 0:
                return status
    return CLEAN


#: Lists every tracked path a ``.gitignore`` rule matches.
TRACKED_IGNORED = ("git", "ls-files", "-ci", "--exclude-standard")


def _check_tracked_ignored(_context: Context) -> int:
    status, out = _capture(TRACKED_IGNORED)
    if status != 0:
        _say(f"tracked-ignored: {shlex.join(TRACKED_IGNORED)} exited {status}")
        return FAILED_EXIT
    ignored = out.split()
    if ignored:
        _say("tracked-ignored: tracked files matched by .gitignore (narrow the rule or untrack")
        _say(f"tracked-ignored: the file): {' '.join(ignored)}")
        return FAILED_EXIT
    return CLEAN


#: Run in the documentation directory; needs the docs extra, as the build does.
DOCS_CLEAN = ("uv", "run", "make", "clean")


def _prune_bytecode(_context: Context) -> int:
    # A package whose sources were deleted still imports, as a namespace
    # package, while its directory holds a __pycache__; a test that the
    # package is gone then fails here and passes on a fresh checkout.
    package = ROOT / "src" / "osprey"
    for cache in sorted(package.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    emptied = sorted((path for path in package.rglob("*") if path.is_dir()), reverse=True)
    for directory in emptied:
        try:
            directory.rmdir()
        except OSError:
            continue
    return CLEAN


def _clean_docs_build(_context: Context) -> int:
    # The build tree outlives a branch switch and the build only adds to it, so
    # a page left there would be judged as this tree's.
    _say(f"docs: {shlex.join(DOCS_CLEAN)}")
    return _run(DOCS_CLEAN, ROOT / WORKDIRS["docs"])


#: What a check does before its own command; a failure here fails the check.
PREPARATIONS: dict[str, Callable[[Context], int]] = {
    "pytest": _prune_bytecode,
    "pytest-pty": _prune_bytecode,
    "docs": _clean_docs_build,
}

#: Checks, or the part of one, implemented in this module; each runs after the
#: check's own command, if it has one, passed.
BUILTINS: dict[str, Callable[[Context], int]] = {
    "tracked-ignored": _check_tracked_ignored,
    "docs": _check_redirect_stubs,
    "package": _check_package,
}


# ---------------------------------------------------------------------------
# Running

#: What can happen to a check; nothing else is ever reported.
RAN, FAILED, NOT_RUN, REFUSED = "ran", "failed", "not run", "refused"
STATES = (RAN, FAILED, NOT_RUN, REFUSED)

#: Why a check did not run, each with the words the report uses.
OFF_VENUE, NOT_INSTALLED, NO_BASE, DESELECTED = (
    "off venue",
    "not installed",
    "no base",
    "deselected",
)
REASONS = {
    NOT_TOUCHED: "not touched",
    OFF_VENUE: "not run here",
    NOT_INSTALLED: "not run here",
    NO_BASE: "not run, no base given",
    DESELECTED: "not run, the marker expression deselects every test in these paths",
}
#: The reasons a line names the check after its words, as ``not run here: ID``.
_NAMED_AFTER = (OFF_VENUE, NOT_INSTALLED)

#: The most failing tests a line names; the rest are counted.
FAILURES_SHOWN = 5
#: run_bounded.py's status for a command stopped at its bound.
TIMED_OUT = 124


@dataclass(frozen=True)
class Outcome:
    """What happened to one check: one state, and only the facts that state carries."""

    id: str
    state: str
    #: ``NOT_RUN`` only: a key of ``REASONS``.
    reason: str | None = None
    #: ``DESELECTED`` only: the checks and CI lanes that run the deselected tests.
    hosts: tuple[str, ...] = ()
    #: ``FAILED`` only: the exit status, never 0.
    status: int = 0
    #: ``FAILED`` only: the failing test ids, from the runner's report.
    failures: tuple[str, ...] = ()
    #: ``REFUSED`` only: what to install, as a ``NEED_PROBES`` answer.
    need: str = ""
    #: What the check's install probe said: on ``RAN`` and ``FAILED``, the line
    #: naming the install the check ran against; on ``NOT_INSTALLED``, the line
    #: saying it is absent.
    attestation: str = ""

    def __post_init__(self) -> None:
        problems = []
        if self.state not in STATES:
            problems.append(f"unknown state {self.state!r}")
        if (self.state == NOT_RUN) != (self.reason in REASONS):
            problems.append(f"reason {self.reason!r} on state {self.state!r}")
        if self.hosts and self.reason != DESELECTED:
            problems.append("hosts on a check that was not deselected")
        if (self.reason == NOT_INSTALLED) and not self.attestation:
            problems.append("no probe line on a check whose install is absent")
        if self.attestation and not (self.state in (RAN, FAILED) or self.reason == NOT_INSTALLED):
            problems.append(f"attestation on state {self.state!r}")
        if (self.state == FAILED) != (self.status != 0):
            problems.append(f"status {self.status} on state {self.state!r}")
        if self.failures and self.state != FAILED:
            problems.append(f"failures on state {self.state!r}")
        if (self.state == REFUSED) != bool(self.need):
            problems.append(f"need {self.need!r} on state {self.state!r}")
        if problems:
            raise ValueError(f"{self.id}: {'; '.join(problems)}")


def unmet_needs(checks: Iterable[Check]) -> list[tuple[str, str]]:
    """Each check of *checks* with an unmet need and its refusal, probing each need once."""
    answers: dict[str, str | None] = {}
    unmet: list[tuple[str, str]] = []
    for check in checks:
        for need in check.needs:
            if need == "base":
                continue
            if need not in answers:
                answers[need] = NEED_PROBES[need]()
            refusal = answers[need]
            if refusal:
                unmet.append((check.id, refusal))
                break
    return unmet


def _hosts_over(paths: Sequence[str]) -> list[str]:
    """The checks with a derived scope and the CI lanes whose scope holds one of
    *paths*, a file or a directory, or lies beneath one of the directories: what
    runs the tests the sweep deselects."""
    inside = [candidate for path in paths for candidate in (path, path.rstrip("/") + "/")]
    above = tuple(path.rstrip("/") + "/" for path in paths)

    def hosts(scope: Sequence[str]) -> bool:
        return touches(inside, scope) or any(pattern.startswith(above) for pattern in scope)

    files = repository_files() if DERIVED_SCOPES else []
    checks = [check_id for check_id, derive in DERIVED_SCOPES.items() if hosts(derive(files))]
    lanes = [lane.job for lane in LANES if hosts(lane.scope)]
    return list(dict.fromkeys([*checks, *lanes]))


def _pytest_id(case: ElementTree.Element) -> str:
    """The node id of an xunit1 ``testcase``: its file, its class path and its name."""
    path, classname, name = (case.get(key, "") for key in ("file", "classname", "name"))
    if not path:
        return "::".join(part for part in (classname, name) if part)
    if not classname:
        return path
    module = path.removesuffix(".py").replace("/", ".")
    if classname == module:
        inner = ""
    elif classname.startswith(module + "."):
        inner = classname[len(module) + 1 :]
    else:
        inner = classname
    return "::".join(part for part in (path, inner.replace(".", "::"), name) if part)


def _vitest_id(case: ElementTree.Element) -> str:
    """A Vitest ``testcase``'s id: its file, then its name unless that is the file."""
    classname, name = case.get("classname", ""), case.get("name", "")
    return classname if classname == name else f"{classname} > {name}"


def failing_tests(check: Check, reports_dir: str) -> tuple[str, ...]:
    """The failing test ids in the reports *check*'s runner wrote under *reports_dir*.

    A missing, empty or unreadable report names nothing; the exit status stands.
    """
    named = _pytest_id if check.report == "junit-pytest" else _vitest_id
    found: list[str] = []
    for path in sorted(Path(reports_dir, check.id).glob("*.xml")):
        try:
            cases = list(ElementTree.parse(path).iter("testcase"))
        except (ElementTree.ParseError, OSError):
            continue
        found += [
            named(case)
            for case in cases
            if case.find("failure") is not None or case.find("error") is not None
        ]
    return tuple(dict.fromkeys(found))


def _run_check(check: Check, selection: Selection, context: Context) -> Outcome:
    status = PREPARATIONS[check.id](context) if check.id in PREPARATIONS else CLEAN
    if not check.argv:
        _say(f"{check.id}: running")
    elif status == 0:
        cwd = ROOT / WORKDIRS[check.id] if check.id in WORKDIRS else None
        for argv in commands(check, selection, context):
            _say(f"{check.id}: {shlex.join(argv)}")
            status = _run(argv, cwd)
            if status != 0:
                break
    if status == 0 and check.id in BUILTINS:
        status = BUILTINS[check.id](context)
    if status == 0:
        return Outcome(check.id, RAN)
    if status == 5 and selection.state == PATHS and check.id == "pytest":
        # pytest collected the paths and the marker expression left none of
        # their tests: nothing ran, so the check is named and not counted.
        return Outcome(check.id, NOT_RUN, DESELECTED, hosts=tuple(_hosts_over(selection.paths)))
    named = check.report and status != TIMED_OUT
    failures = failing_tests(check, context.reports) if named else ()
    return Outcome(check.id, FAILED, status=status, failures=failures)


def _classify(check: Check, selection: Selection, context: Context) -> Outcome | None:
    """Why *check* will not run here, or ``None`` when it will."""
    if selection.state == NOT_TOUCHED:
        return Outcome(check.id, NOT_RUN, NOT_TOUCHED)
    if "base" in check.needs and not context.base:
        return Outcome(check.id, NOT_RUN, NO_BASE)
    if check.venue and not VENUE_PROBES[check.venue]():
        return Outcome(check.id, NOT_RUN, OFF_VENUE)
    return None


def execute(
    planned: Sequence[tuple[Check, Selection]], context: Context, fix: bool = False
) -> list[Outcome]:
    """Run every planned check, each one whatever the earlier ones did.

    A check that will not run here is classified first, and only the needs of
    the rest are probed. When one is unmet, nothing runs and every check with an
    unmet need is refused.
    """
    with tempfile.TemporaryDirectory(prefix="check-reports-") as reports:
        return _execute(planned, dataclasses.replace(context, reports=reports), fix)


def _execute(
    planned: Sequence[tuple[Check, Selection]], context: Context, fix: bool
) -> list[Outcome]:
    outcomes: dict[str, Outcome] = {}
    running: list[tuple[Check, Selection]] = []
    for check, selection in planned:
        skipped = _classify(check, selection, context)
        if skipped:
            outcomes[check.id] = skipped
        else:
            running.append((check, selection))
    refused = dict(unmet_needs(check for check, _selection in running))
    installed: dict[str, Install] = {}
    for check, _selection in running:
        install = INSTALLS.get(check.id)
        if install is None or check.id in refused:
            continue
        status, said = probe_install(install)
        if status == 0:
            installed[check.id] = install
        elif status == ABSENT_EXIT and not context.named:
            outcomes[check.id] = Outcome(check.id, NOT_RUN, NOT_INSTALLED, attestation=said)
        else:
            refused[check.id] = f"{said}: {install.instruction()}"
    if refused:
        return [
            Outcome(check.id, REFUSED, need=refused[check.id])
            for check, _selection in running
            if check.id in refused
        ]
    if fix:
        for check, _selection in running:
            for argv in check.fix:
                _say(f"fix: {shlex.join(argv)}")
                _run(argv)
    for check, selection in running:
        if check.id in outcomes:
            continue
        outcome = _run_check(check, selection, context)
        if check.id in installed and outcome.state in (RAN, FAILED):
            line = installed[check.id].attestation(outcome.state == RAN)
            outcome = dataclasses.replace(outcome, attestation=line)
        outcomes[check.id] = outcome
    return [outcomes[check.id] for check, _selection in planned]


_BY_ID = {check.id: check for check in CHECKS}


def _names(failures: Sequence[str]) -> str:
    """The first ``FAILURES_SHOWN`` failing tests, then a count of the rest."""
    hidden = len(failures) - FAILURES_SHOWN
    return ", ".join([*failures[:FAILURES_SHOWN], *([f"+{hidden}"] if hidden > 0 else [])])


def _fact(outcome: Outcome) -> str:
    """The parenthesised fact of *outcome*'s line, or ``""`` when it has none."""
    declared = _BY_ID.get(outcome.id)
    if outcome.state == FAILED:
        if outcome.failures:
            return _names(outcome.failures)
        if outcome.status == TIMED_OUT and declared and declared.bound_s:
            return "still running at its bound"
        return f"exit {outcome.status}"
    if outcome.state == REFUSED:
        return outcome.need
    if outcome.reason == OFF_VENUE and declared and declared.venue:
        return VENUE_NEEDS[declared.venue]
    if outcome.reason == NOT_INSTALLED:
        return outcome.attestation
    if outcome.reason == DESELECTED and outcome.hosts:
        checks = [host for host in outcome.hosts if host in _BY_ID]
        lanes = [host for host in outcome.hosts if host not in _BY_ID]
        said = [f"run by {', '.join(checks)}"] if checks else []
        said += [f"run on CI: {', '.join(lanes)}"] if lanes else []
        return ", ".join(said)
    return ""


def _repair(outcome: Outcome) -> str:
    """The command that repairs *outcome*, or ``""`` when its check declares no fix."""
    declared = _BY_ID.get(outcome.id)
    if outcome.state == FAILED and declared and declared.fix:
        return f"fix: {FIX_HINT}"
    return ""


def _line(outcome: Outcome) -> str:
    """The per-check line of *outcome*."""
    detail = "; ".join(part for part in (_fact(outcome), _repair(outcome)) if part)
    suffix = f" ({detail})" if detail else ""
    if outcome.reason in _NAMED_AFTER:
        return f"{REASONS[outcome.reason]}: {outcome.id}{suffix}"
    words = REASONS[outcome.reason] if outcome.reason else outcome.state
    return f"{outcome.id}: {words}{suffix}"


def _refused_parts(outcomes: Sequence[Outcome]) -> list[str]:
    """One ``refused`` part per need, naming every check it refused."""
    by_need: dict[str, list[str]] = {}
    for outcome in outcomes:
        if outcome.state == REFUSED:
            by_need.setdefault(outcome.need, []).append(outcome.id)
    return [f"refused: {', '.join(ids)} ({need})" for need, ids in by_need.items()]


def verdict(outcomes: Sequence[Outcome], lanes: Sequence[str], scoped: bool) -> str:
    """The last line: clean or not, and everything that was not run here."""
    refused = _refused_parts(outcomes)
    if refused:
        return "; ".join(refused)
    failed = []
    for o in outcomes:
        if o.state == FAILED:
            detail = "; ".join(part for part in (_names(o.failures), _repair(o)) if part)
            failed.append(f"{o.id} ({detail})" if detail else o.id)
    parts = [f"not clean: failed: {', '.join(failed)}" if failed else "clean"]
    parts += [o.attestation for o in outcomes if o.state in (RAN, FAILED) and o.attestation]
    named: dict[str, list[str]] = {}
    for reason in (OFF_VENUE, NOT_INSTALLED, NO_BASE, DESELECTED):
        for outcome in outcomes:
            if outcome.reason == reason:
                fact = _fact(outcome)
                said = f"{outcome.id} ({fact})" if fact else outcome.id
                named.setdefault(REASONS[reason], []).append(said)
    parts += [f"{words}: {', '.join(ids)}" for words, ids in named.items()]
    if lanes:
        reason = "this range touches" if scoped else "never run here"
        parts.append(f"run on CI, {reason}: {', '.join(lanes)}")
    return "; ".join(parts)


def report(outcomes: Sequence[Outcome], lanes: Sequence[str], scoped: bool) -> int:
    """Print one line per check and the verdict, and return the exit status.

    A refused run prints to stderr, as every refusal does; any other to stdout.
    """
    refused = any(o.state == REFUSED for o in outcomes)
    stream = sys.stderr if refused else sys.stdout
    for outcome in outcomes:
        print(f"check: {_line(outcome)}", file=stream, flush=True)
    print(f"check: {verdict(outcomes, lanes, scoped)}", file=stream, flush=True)
    if refused:
        return REFUSED_EXIT
    return FAILED_EXIT if any(o.state == FAILED for o in outcomes) else CLEAN


# ---------------------------------------------------------------------------
# Command line


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check.py",
        description="Run the repository's checks from their one declaration.",
    )
    verbs = parser.add_subparsers(dest="verb", required=True)
    for tier in TIERS:
        verb = verbs.add_parser(tier, help=f"run the {tier} tier; test files may follow --")
        verb.add_argument(
            "--changed", metavar="BASE", help="scope the tier to BASE...HEAD; implies --base BASE"
        )
        verb.add_argument(
            "--base",
            metavar="BASE",
            help="arm the checks that judge a range against BASE; the runners stay whole",
        )
        verb.add_argument(
            "--worktree", action="store_true", help="judge the working tree, not HEAD"
        )
        if tier == "quick":
            verb.add_argument(
                "--fix",
                action="store_true",
                help="let the checks that declare a fix rewrite files first",
            )
    run = verbs.add_parser("run", help="run the named checks; execution flags may follow --")
    run.add_argument("ids", nargs="+", metavar="ID")
    run.add_argument("--base", metavar="BASE", help="the base of the checks that judge a range")
    run.add_argument("--shard", choices=sorted(UNIT_SELECTION.shards))
    listing = verbs.add_parser("list", help="print the declaration")
    listing.add_argument("--tier", choices=TIERS)
    listing.add_argument(
        "--changed", metavar="BASE", help="also name the checks and lanes BASE...HEAD touches"
    )
    listing.add_argument("--json", action="store_true")
    return parser


def _split(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """The arguments before the first ``--`` and those after it."""
    argv = list(argv)
    if "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1 :]
    return argv, []


@dataclass(frozen=True)
class _Request:
    planned: list[tuple[Check, Selection]]
    context: Context
    lanes: tuple[str, ...] = ()
    scoped: bool = False
    fix: bool = False


def _request(argv: Sequence[str]) -> _Request:
    """Parse a ``quick``, ``full`` or ``run`` command line into what it will run."""
    head, tail = _split(argv)
    args = _parser().parse_args(head)
    by_id = {check.id: check for check in CHECKS}
    if args.verb == "run":
        unknown = [name for name in args.ids if name not in by_id]
        if unknown:
            raise Refused(f"unknown check: {', '.join(unknown)}; see check.py list")
        validate_exec_flags(tail)
        chosen = [by_id[name] for name in args.ids]
        if tail and not any(check.id in EXEC_FLAG_CHECKS for check in chosen):
            raise Refused(f"execution flags apply to {', '.join(EXEC_FLAG_CHECKS)} only")
        unbased = [c.id for c in chosen if "base" in c.needs and not args.base]
        if unbased:
            raise Refused(f"{', '.join(unbased)}: needs --base BASE")
        context = Context("full", args.base, args.shard, {"*": tuple(tail)}, named=True)
        return _Request([(check, Selection(WHOLE)) for check in chosen], context)
    if args.changed and args.base and args.changed != args.base:
        raise Refused("--changed and --base name different bases")
    base = args.changed or args.base
    if args.worktree and not args.changed:
        raise Refused("--worktree needs --changed BASE")
    paths = files = reach = None
    if args.changed:
        paths = touched(args.changed, args.worktree)
        reach = touched(args.changed, args.worktree, scope=True)
        files = repository_files(args.worktree)
    plan = select(args.verb, paths, tail, files, reach)
    context = Context(
        args.verb, base, None, dict(TIER_EXEC_FLAGS[args.verb]), worktree=args.worktree
    )
    planned = [(by_id[name], selection) for name, selection in plan.selections.items()]
    return _Request(planned, context, plan.lanes, plan.scoped, getattr(args, "fix", False))


def resolve(argv: Sequence[str]) -> list[list[str]]:
    """The effective commands of a ``quick``, ``full`` or ``run`` line, without running them.

    A check the range does not touch contributes no line, and ``list`` runs nothing.
    """
    if _parser().parse_args(_split(argv)[0]).verb == "list":
        return []
    request = _request(argv)
    return [
        argv
        for check, selection in request.planned
        if selection.state != NOT_TOUCHED
        for argv in commands(check, selection, request.context)
    ]


def declaration(tier: str | None = None) -> dict[str, object]:
    """The declaration as data; a derived scope is the one computed for this tree."""
    files = repository_files() if DERIVED_SCOPES else []

    def plain(item: Check | Lane) -> dict[str, object]:
        data = dataclasses.asdict(item)
        if "tiers" in data:
            data["tiers"] = sorted(data["tiers"])
        if isinstance(item, Check) and item.id in DERIVED_SCOPES:
            data["scope"] = list(DERIVED_SCOPES[item.id](files))
        return data

    return {
        "tiers": list(TIERS),
        "checks": [plain(check) for check in CHECKS if tier is None or tier in check.tiers],
        "unit_selection": dataclasses.asdict(UNIT_SELECTION),
        "lanes": [plain(lane) for lane in LANES],
    }


def touched_by(base: str, tier: str = "full") -> list[str]:
    """The ids of the checks of *tier* and of the CI lanes the range ``BASE...HEAD`` touches."""
    files = repository_files()
    plan = select(tier, touched(base), (), files, touched(base, scope=True))
    checks = [name for name, chosen in plan.selections.items() if chosen.state != NOT_TOUCHED]
    return [*checks, *plan.lanes]


def _list(tier: str | None, as_json: bool, changed: str | None = None) -> int:
    hit = touched_by(changed, tier or "full") if changed else None
    if as_json:
        data = declaration(tier)
        if hit is not None:
            data["touched"] = hit
        print(json.dumps(data, indent=2))
        return CLEAN
    for check in CHECKS:
        if tier is None or tier in check.tiers:
            spelling = shlex.join(check.argv) if check.argv else "(in scripts/check.py)"
            tiers = ",".join(t for t in TIERS if t in check.tiers)
            print(f"{check.id:<20}{check.kind:<11}{tiers:<12}{spelling}")
    if hit is not None:
        print(f"touched: {', '.join(hit)}")
    return CLEAN


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    argv = sys.argv[1:] if argv is None else list(argv)
    head, _tail = _split(argv)
    args = _parser().parse_args(head)
    try:
        if args.verb == "list":
            if args.changed:
                require_base(args.changed)
            return _list(args.tier, args.json, args.changed)
        base = args.base if args.verb == "run" else (args.changed or args.base)
        if base:
            require_base(base)
        request = _request(argv)
        outcomes = execute(request.planned, request.context, request.fix)
    except Refused as refusal:
        print(f"check: refused: {refusal}", file=sys.stderr, flush=True)
        return REFUSED_EXIT
    return report(outcomes, request.lanes, request.scoped)


if __name__ == "__main__":
    sys.exit(main())
