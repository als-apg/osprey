"""The framework's checks: what holds for the repository as a whole and what
judges the framework's code, each declared with everything the check command
does with it.

``scripts/check.py`` collects this module first and every other
``repo_checks.py`` after it; this module's docstring in ``check.py`` names the
four kinds and the two tiers. A check another package owns is declared in that
package's own ``repo_checks.py``.

Standard library only, for the reason ``scripts/check.py`` gives.
"""

from __future__ import annotations

import dataclasses
import glob
import importlib.util
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType

import check_kit as kit
from check_kit import (
    BOTH,
    CLEAN,
    FAILED_EXIT,
    FULL,
    NOT_TOUCHED,
    PATHS,
    UV_RUN,
    WHOLE,
    Check,
    Context,
    EnvironmentFault,
    Lane,
    Refused,
    Selection,
    Venue,
)


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
    #: Trim name to the test files it leaves out of the roots. Selected with
    #: ``--shard``, but outside the partition: one job over everything else.
    trims: dict[str, tuple[str, ...]]


#: The tests that pin this declaration and the repository-wide rules, and
#: every test that reads the whole tests tree: a change anywhere in the tree
#: can break them, so no change selection may leave them out.
INVARIANT_TESTS = (
    "tests/docs/test_shipped_identity_literals.py",
    "tests/scripts/test_ruff_file_selection.py",
    "tests/scripts/test_check.py",
    "tests/scripts/test_changelog_fragments.py",
    "tests/interfaces/test_js_module_tests.py",
    "tests/deployment/test_ci_workflow_wiring.py",
    "tests/deployment/test_retired_spellings_at_parse_point.py",
    "tests/deployment/web_terminals/test_roster_entries.py",
    "tests/infrastructure/test_import_time_audit.py",
    "tests/infrastructure/test_stand_in_canaries.py",
    "tests/test_offline_build_env.py",
    "tests/test_api_marker_placement.py",
    "tests/test_graphdb_container.py",
    "tests/test_mongo_container.py",
    "tests/test_no_legacy_symbols.py",
)

LIVE_CA_VENUE = "linux-x86_64"
DARWIN_VENUE = "darwin"

#: The readonly gate's warm-up module: which stdlib lookups load native code
#: lazily depends on the OS and the interpreter.
READONLY_WARMUPS_TEST_FILE = "tests/runtime/test_raw_put_block_readonly_warmups.py"

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
GITHUB_FORMAT = ("--output-format=github",)

#: The documentation's directory, where its build runs.
DOCS_DIR = "docs"

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

#: The files the macOS cells leave to ubuntu: each was read, and none reaches
#: anything whose outcome depends on the OS — no platform branch, no real
#: child process it signals, no path or permission semantics macOS spells
#: differently, no native binary. A file is listed whole or not at all, and a
#: new test file runs on macOS until someone reads it.
_MACOS_REPEATS_UBUNTU = (
    # In-process `osprey build --skip-deps` renders, which install nothing;
    # the build's directory swap stays on macOS in test_build_swap.py.
    "tests/cli/test_answer_provenance_render.py",
    "tests/cli/test_build_cmd.py",
    "tests/cli/test_build_config_edit_session.py",
    "tests/cli/test_build_graph_index.py",
    "tests/cli/test_build_graphdb_removal.py",
    "tests/cli/test_build_hierarchy_embedding.py",
    "tests/cli/test_build_preset.py",
    "tests/cli/test_build_profile_ariel_sync.py",
    "tests/cli/test_build_zero_arg.py",
    "tests/cli/test_channel_finder_graph_tools.py",
    "tests/cli/test_claude_code_integration.py",
    "tests/cli/test_claude_regen.py",
    "tests/cli/test_config_records_environment.py",
    "tests/cli/test_convention_artifacts_by_name.py",
    "tests/cli/test_deployed_services_injection.py",
    "tests/cli/test_deployment_project_name.py",
    "tests/cli/test_e2e_preset_login_tiers.py",
    "tests/cli/test_explicit_config_equivalence.py",
    "tests/cli/test_facility_rule_source_zone.py",
    "tests/cli/test_facility_show.py",
    "tests/cli/test_facility_validate.py",
    "tests/cli/test_graph_agent_surface.py",
    "tests/cli/test_hooks_channel.py",
    "tests/cli/test_init_revalidate.py",
    "tests/cli/test_init_revalidate_fill.py",
    "tests/cli/test_init_revalidate_write_keys.py",
    "tests/cli/test_interpreter_role_split.py",
    "tests/cli/test_json_contracts.py",
    "tests/cli/test_killswitch_on_tier.py",
    "tests/cli/test_layer_emission.py",
    "tests/cli/test_live_standin_overrides.py",
    "tests/cli/test_live_standin_render.py",
    "tests/cli/test_login_tiers_render.py",
    "tests/cli/test_mml_build_recipes.py",
    "tests/cli/test_model_warnings_report.py",
    "tests/cli/test_port_base_both_halves.py",
    "tests/cli/test_posture_floor.py",
    "tests/cli/test_posture_render_matrix.py",
    "tests/cli/test_preset_drift.py",
    "tests/cli/test_preset_render.py",
    "tests/cli/test_profile_annotations.py",
    "tests/cli/test_profile_data_root.py",
    "tests/cli/test_profile_to_claude_contract.py",
    "tests/cli/test_profile_va_archiver_block.py",
    "tests/cli/test_rendered_identity_literals.py",
    "tests/cli/test_renders_projection.py",
    "tests/cli/test_set_config_dotted.py",
    "tests/cli/test_set_verb.py",
    "tests/cli/test_source_zone_materialization.py",
    "tests/cli/test_store_file_deny.py",
    "tests/cli/test_terminology_tables.py",
    "tests/cli/test_tier_app_presets.py",
    "tests/cli/test_va_build_step.py",
    "tests/cli/test_va_default_config.py",
    "tests/cli/test_variant_selection.py",
    "tests/deployment/test_container_project_layout.py",
    "tests/deployment/test_control_assistant_end_state.py",
    "tests/deployment/test_live_standin_render.py",
    "tests/deployment/test_login_tiers_staged_render.py",
    "tests/deployment/test_render_e2e.py",
    "tests/deployment/test_render_reach_drift.py",
    "tests/deployment/test_render_reach_endpoints.py",
    "tests/deployment/test_render_reach_paths.py",
    "tests/facility/test_build_stops.py",
    "tests/facility/test_chain.py",
    "tests/facility/test_demo_census.py",
    "tests/facility/test_knowledge_links.py",
    "tests/facility/test_simulator_view.py",
    "tests/integration/test_data_checksum_stability.py",
    "tests/integration/test_profile_roundtrip.py",
    "tests/test_e2e_ariel_db_pin.py",
    # Facility logic in process; pyAT is only compared against itself here,
    # while the fixed-reference replays stay on macOS.
    "tests/connectors/test_archiver_scenario_signatures.py",
    "tests/facility/test_cf_index_middle_layer.py",
    "tests/facility/test_demo_seed_rules.py",
    "tests/facility/test_injector_sources.py",
    "tests/facility/test_layer_mml.py",
    "tests/facility/test_layer_mml_decks.py",
    "tests/facility/test_layer_mml_identity.py",
    "tests/facility/test_layer_mml_import.py",
    "tests/facility/test_layer_mml_tune.py",
    "tests/facility/test_layer_mml_wiring.py",
    "tests/facility/test_line_sources.py",
    "tests/facility/test_minimal_trees.py",
    "tests/facility/test_mml_layer_seed_once.py",
    "tests/facility/test_served_models.py",
    "tests/services/channel_finder/graph_index/test_build.py",
    "tests/simulation/test_single_pass_plugin.py",
    "tests/simulation/test_waveform_equivalence.py",
    "tests/templates/test_pyat_specialist.py",
    # Scans of the source tree: AST, text, git ls-files, import graphs.
    "tests/benchmark/test_matrix_lanes.py",
    "tests/cli/test_lifecycle_invariants.py",
    "tests/cli/test_printed_copy_style.py",
    "tests/config/test_bluesky_keys_retired.py",
    "tests/config/test_facility_keys_retired.py",
    "tests/config/test_injected_keys_are_manifest_rows.py",
    "tests/config/test_simulation_keys_retired.py",
    "tests/connectors/test_framework_connectors_compat.py",
    "tests/connectors/test_public_api.py",
    "tests/connectors/test_safety_file_single_home.py",
    "tests/deployment/test_ci_workflow_wiring.py",
    "tests/deployment/test_retired_spellings_at_parse_point.py",
    "tests/deployment/test_runtime_env_digest.py",
    "tests/docs/test_shipped_identity_literals.py",
    "tests/infrastructure/test_import_time_audit.py",
    "tests/interfaces/test_js_module_tests.py",
    "tests/mcp_server/test_channel_finder_tool_vocabulary.py",
    "tests/registry/test_tool_classification_drift.py",
    "tests/scripts/test_check.py",
    "tests/scripts/test_config_key_guard.py",
    "tests/services/ariel_search/test_import_graph.py",
    "tests/services/facility_knowledge/test_import_isolation.py",
    "tests/test_no_legacy_symbols.py",
    "tests/test_retired_store_words.py",
    # Logic over a faked runtime, subprocess or clock, or a POSIX-wide protocol.
    "tests/cli/test_lifecycle_promotions.py",
    "tests/connectors/test_mock_connector.py",
    "tests/deployment/test_container_lifecycle.py",
    "tests/deployment/test_down_restart_gates.py",
    "tests/deployment/test_status_sections.py",
    "tests/deployment/test_up_as_built.py",
    "tests/hooks/test_approval_outcome.py",
    "tests/hooks/test_approval_queue_enrichment.py",
    "tests/hooks/test_hook_agreement.py",
    "tests/hooks/test_store_path_pattern.py",
    "tests/integration/test_write_verdict_surfaces.py",
    "tests/interfaces/ariel/test_routes.py",
    "tests/interfaces/web_terminal/test_config_routes.py",
    "tests/interfaces/web_terminal/test_facility_timezone_stamp.py",
    "tests/interfaces/web_terminal/test_proxy.py",
    "tests/interfaces/web_terminal/test_tour.py",
    "tests/mcp_server/python_executor/test_sandbox_single_posture.py",
    "tests/mcp_server/test_bluesky_client_tools.py",
    "tests/mcp_server/test_error_envelope_ipc.py",
    "tests/mcp_server/test_executor_token_regression.py",
    "tests/mcp_server/test_fastmcp_contract.py",
    "tests/pty/test_stub_coverage.py",
    "tests/services/ariel_search/test_image_lane.py",
    "tests/services/ariel_search/test_ingestion.py",
    "tests/services/bluesky_bridge/test_plan_validation.py",
    "tests/services/bluesky_bridge/test_planspec_render.py",
    "tests/templates/test_control_system_safety_rule.py",
    "tests/test_e2e_provider_gate.py",
    "tests/utils/test_import_env_hygiene.py",
    "tests/va/test_suite_still_scenario.py",
    # Chromium only: the engine is the same on both hosts.
    "tests/interfaces/ariel/test_embedded_nav.py",
    "tests/interfaces/artifacts/test_opaque_origin_browser.py",
    "tests/interfaces/design_system/test_osprey_drawer.py",
    "tests/interfaces/web_terminal/test_panels_browser.py",
    # Shell a Linux image runs, as PID 1 or a RUN step: ubuntu is the faithful host.
    "tests/cli/test_dockerfile_template.py",
    "tests/deployment/test_entrypoint_script.py",
    "tests/deployment/test_image_member_install.py",
)

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
    trims={"macos": _MACOS_REPEATS_UBUNTU},
    dist=("--dist", "loadgroup"),
)

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


# ---------------------------------------------------------------------------
# Selection


def _guard() -> ModuleType | None:
    """The module that owns the JavaScript module-to-test mapping, loaded by
    path, or ``None`` while the tree holds no such module."""
    if not (kit.ROOT / GUARD_MODULE).is_file():
        return None
    spec = importlib.util.spec_from_file_location("js_module_tests", kit.ROOT / GUARD_MODULE)
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


def _paths(chosen: Iterable[str]) -> Selection:
    unique = tuple(dict.fromkeys(chosen))
    return Selection(PATHS, unique) if unique else Selection(NOT_TOUCHED)


def _select_pytest(
    paths: Sequence[str], files: Sequence[str], shaping: frozenset[str] = frozenset()
) -> Selection:
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
            if path in shaping:
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


def _select_vitest(
    paths: Sequence[str], files: Sequence[str], _shaping: frozenset[str] = frozenset()
) -> Selection:
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
        elif (
            name in VITEST_WIDE_NAMES
            or path == VITEST_SETUP
            or path.endswith(JS_SUFFIXES)
            and path.startswith(("tests/", "src/"))
        ):
            return Selection(WHOLE)
    return _paths(chosen)


# ---------------------------------------------------------------------------
# Commands


def unit_argv(
    tier: str,
    targets: Sequence[str] = (),
    shard: str | None = None,
    unit: UnitSelection = UNIT_SELECTION,
) -> list[str]:
    """The parallel sweep's selection: paths, ignores and marker expression."""
    if shard in unit.trims:
        roots, shard_ignores = unit.roots, unit.trims[shard]
    else:
        roots, shard_ignores = unit.shards[shard] if shard else (unit.roots, ())
    ignores = [*shard_ignores, *(path for path, _host in unit.ignores)]
    marker = unit.quick_marker if tier == "quick" else unit.marker
    return [
        *(targets or roots),
        *(f"--ignore={path}" for path in ignores),
        "-m",
        marker,
    ]


def _pytest_command(
    check: Check,
    selection: Selection,
    context: Context,
    flags: Sequence[str],
    written: Sequence[str],
) -> list[list[str]]:
    bound = QUICK_PYTEST_BOUND_S if context.tier == "quick" else check.bound_s
    selected = unit_argv(context.tier, selection.paths, context.shard)
    argv = [*check.argv, *selected, *flags, *UNIT_SELECTION.dist, *written]
    return [kit.bounded(bound, argv)]


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


def _package_command(*_run: object) -> list[list[str]]:
    return [list(step) for step in PACKAGE_STEPS]


# ---------------------------------------------------------------------------
# Checks implemented here

#: Prints the keys of the documentation's ``redirects`` table, space separated.
REDIRECT_KEYS_PROBE = (
    *UV_RUN,
    "python",
    "-c",
    "import os, runpy; os.chdir('docs/source'); "
    "print(' '.join(runpy.run_path('conf.py')['redirects']))",
)

DOCS_HTML = "docs/build/html"


def missing_redirect_stubs(keys: Iterable[str], html_root: Path) -> list[str]:
    """The redirect keys whose stub page the documentation build did not write."""
    return [key for key in keys if not (html_root / f"{key}.html").is_file()]


def _check_redirect_stubs(_context: Context) -> int:
    status, out = kit.capture(REDIRECT_KEYS_PROBE)
    if status != 0:
        kit.say("docs: cannot read the redirects table of docs/source/conf.py")
        return FAILED_EXIT
    missing = missing_redirect_stubs(out.split(), kit.ROOT / DOCS_HTML)
    if missing:
        kit.say(f"docs: redirect stubs missing from the build: {' '.join(missing)}")
        return FAILED_EXIT
    return CLEAN


def _expand(step: Sequence[str], scratch: str) -> list[str]:
    """Fill ``{scratch}`` and expand a ``dist/`` glob against the repository root."""
    argv: list[str] = []
    for part in step:
        part = part.replace("{scratch}", scratch)
        if part.startswith("dist/") and "*" in part:
            found = sorted(glob.glob(part, root_dir=kit.ROOT))
            argv += found or [part]
        else:
            argv.append(part)
    return argv


def _check_package(_context: Context) -> int:
    # dist/ outlives a branch switch and `uv build` only adds to it, so anything
    # left there would be checked as this tree's.
    shutil.rmtree(kit.ROOT / "dist", ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="package-check-") as scratch:
        for step in PACKAGE_STEPS:
            status = kit.run(_expand(step, scratch))
            if status != 0:
                return status
    return CLEAN


#: Lists every tracked path a ``.gitignore`` rule matches.
TRACKED_IGNORED = ("git", "ls-files", "-ci", "--exclude-standard")


def _check_tracked_ignored(_context: Context) -> int:
    status, out = kit.capture(TRACKED_IGNORED)
    if status != 0:
        kit.say(f"tracked-ignored: {shlex.join(TRACKED_IGNORED)} exited {status}")
        return FAILED_EXIT
    ignored = out.split()
    if ignored:
        kit.say("tracked-ignored: tracked files matched by .gitignore (narrow the rule or untrack")
        kit.say(f"tracked-ignored: the file): {' '.join(ignored)}")
        return FAILED_EXIT
    return CLEAN


#: Run in the documentation directory; needs the docs extra, as the build does.
DOCS_CLEAN = (*UV_RUN, "make", "clean")


def _prune_bytecode(_context: Context) -> int:
    # A package whose sources were deleted still imports, as a namespace
    # package, while its directory holds a __pycache__; a test that the
    # package is gone then fails here and passes on a fresh checkout.
    package = kit.ROOT / "src" / "osprey"
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
    kit.say(f"docs: {shlex.join(DOCS_CLEAN)}")
    return kit.run(DOCS_CLEAN, kit.ROOT / DOCS_DIR)


# ---------------------------------------------------------------------------
# Needs and venues

#: Importable only in a venv synced with the docs extra.
DOCS_EXTRA_PROBE = (*UV_RUN, "python", "-c", "import myst_parser")

#: Importable only in a venv synced with the virtual-accelerator extra, on its
#: platform. The modules are the ones the live gate requires.
LIVE_CA_PROBE = (*UV_RUN, "python", "-c", "import pcaspy, p4p, lume_pva_apg")

#: Prints the directory whose ``claude`` reports the pin, or the product's
#: refusal on stderr.
CLAUDE_PIN_PROBE = (*UV_RUN, "python", "scripts/pinned_claude.py")

#: The compare of ``node_modules`` with ``package-lock.json``, stdlib only.
JS_DEPS_CURRENT = Path(__file__).resolve().with_name("js_deps_current.py")

#: Below this a run can die halfway through its temp tree; one figure, no knob.
FREE_SPACE_FLOOR_GIB = 5

#: How long each docker command of the storage probe may take.
DOCKER_PROBE_TIMEOUT_S = 60

#: The docker group as the tests spell it, read as text: collecting the
#: selection to ask would cost more than the probe it arms.
DOCKER_GROUP_PATTERN = r'mark\.xdist_group\(\s*"docker"'

#: ``--no-sync``: a plain ``uv run`` is what would hide a stale install, or install it.
INSTALLED_VERSIONS_PROBE = (*UV_RUN, "--no-sync", "python", "scripts/installed_versions.py")

#: Runs only in a venv synced with the dev extra. actionlint disables its
#: shellcheck rule without a word when shellcheck is absent and still exits 0, so
#: this probe is what keeps the check from passing over unread ``run:`` blocks.
WORKFLOW_LINT_PROBE = (*UV_RUN, "shellcheck", "--version")


def _need_uv() -> str | None:
    if shutil.which("uv"):
        return None
    return "uv is not on PATH: install uv, then run uv sync --extra dev"


def _need_gitleaks() -> str | None:
    if shutil.which("gitleaks"):
        return None
    return (
        "gitleaks is not on PATH: run brew install gitleaks, "
        "or install a release from https://github.com/gitleaks/gitleaks/releases"
    )


def _need_workflow_linters() -> str | None:
    if kit.capture(WORKFLOW_LINT_PROBE)[0] == 0:
        return None
    return "the workflow linters are not installed: run uv sync --extra dev"


def _probe(argv: Sequence[str], timeout: float | None = None) -> tuple[int, str, str]:
    """Run a probe and return its exit status, stdout and stderr."""
    try:
        done = subprocess.run(
            list(argv), cwd=kit.ROOT, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        return 127, "", ""
    return done.returncode, done.stdout, done.stderr


def _need_claude_pin() -> str | None:
    """Met when a ``claude`` reporting the pin is found; its directory then goes
    first on ``PATH``, so every check this command starts runs it."""
    status, out, err = _probe(CLAUDE_PIN_PROBE)
    if status != 0:
        return " ".join(err.split()) or f"{shlex.join(CLAUDE_PIN_PROBE)} exited {status}"
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines or not Path(lines[-1]).is_dir():
        return f"{shlex.join(CLAUDE_PIN_PROBE)} printed no directory"
    directory = lines[-1]
    os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
    kit.say(f"claude-pin: {directory} first on PATH")
    return None


def _opt_in_js_check(context: Context) -> int:
    """Open the test that fetches and runs the toolchain ``osprey app check`` ships."""
    os.environ["OSPREY_JS_CHECK"] = "1"
    return _prune_bytecode(context)


def _js_deps_current() -> ModuleType:
    """The lockfile compare, loaded by path: scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("js_deps_current", JS_DEPS_CURRENT)
    if spec is None or spec.loader is None:
        raise Refused(f"cannot load {JS_DEPS_CURRENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _need_node() -> str | None:
    """Met when ``node_modules`` holds what ``package-lock.json`` pins; runs
    ``npm ci`` when it does not."""
    if not (shutil.which("node") and shutil.which("npm")):
        return "the Node toolchain is not installed: install Node, then run npm ci"
    first_difference = _js_deps_current().first_difference
    difference = first_difference(kit.ROOT)
    if difference is None:
        return None
    kit.say(f"node: {difference}; npm ci")
    status = kit.run(("npm", "ci"))
    if status != 0:
        return f"npm ci failed (exit {status}): {difference}"
    again = first_difference(kit.ROOT)
    if again is not None:
        return f"node_modules still differs from package-lock.json after npm ci: {again}"
    return None


def _device(path: str | Path) -> int:
    return os.stat(path).st_dev


def _need_free_space() -> str | None:
    """Met when the temp directory and the checkout each hold the floor; each
    device is read once."""
    try:
        places: dict[int, str | Path] = {}
        for place in (tempfile.gettempdir(), kit.ROOT):
            places.setdefault(_device(place), place)
        for path in places.values():
            free = shutil.disk_usage(path).free
            if free < FREE_SPACE_FLOOR_GIB * 2**30:
                raise EnvironmentFault(
                    "free space",
                    f"{free // 2**20} MiB free on {path}, floor {FREE_SPACE_FLOOR_GIB} GiB",
                )
    except OSError as error:
        raise EnvironmentFault("free space", str(error)) from error
    return None


def _need_docker_storage() -> str | None:
    """Met when a volume can be created and removed, and on a machine with no
    daemon to ask, whose Docker tests skip. It starts no container."""
    if not shutil.which("docker"):
        return None
    try:
        if _probe(("docker", "info"), DOCKER_PROBE_TIMEOUT_S)[0] != 0:
            return None
    except subprocess.TimeoutExpired:
        return None
    volume = f"osprey-check-probe-{os.getpid()}"
    for verb in ("create", "rm"):
        try:
            status, _out, err = _probe(("docker", "volume", verb, volume), DOCKER_PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            fact = f"did not answer in {DOCKER_PROBE_TIMEOUT_S} s"
        else:
            if status == 0:
                continue
            fact = err.strip().rpartition(": ")[2] or f"exited {status}"
        raise EnvironmentFault("docker storage", f"volume {verb} {fact}")
    return None


def _docker_group_needs(
    roots: Callable[[Context], Iterable[str]],
) -> Callable[[Selection, Context], tuple[str, ...]]:
    """The hook of a check whose whole selection is *roots*: it needs Docker
    storage when a selected file, or a conftest above one, carries the group."""

    def needs(selection: Selection, context: Context) -> tuple[str, ...]:
        paths = selection.paths if selection.state == PATHS else roots(context)
        targets = [path.partition("::")[0].rstrip("/") for path in paths]
        above = {
            conftest
            for target in targets
            for parent in PurePosixPath(target).parents
            if (kit.ROOT / (conftest := str(parent / "conftest.py"))).is_file()
        }
        status, out, _err = _probe(
            ("git", "grep", "--untracked", "-lE", DOCKER_GROUP_PATTERN, "--", *targets, *above)
        )
        hit = status == 0 and any(line.endswith(".py") for line in out.splitlines())
        return ("docker-storage",) if hit else ()

    return needs


def _stale_installs() -> list[list[str]]:
    """The reader's rows. It prints none on a checkout with no environment yet:
    the check's own ``uv run`` creates that one current."""
    rows = (line.split("\t") for line in _probe(INSTALLED_VERSIONS_PROBE)[1].splitlines())
    return [row for row in rows if len(row) == 3]


def _need_install_current() -> str | None:
    """Met when the install is what the tree describes; re-syncs it when it is not."""
    stale = _stale_installs()
    if not stale:
        return None
    for dist, installed, described in stale:
        kit.say(f"install-current: {dist} installed {installed}, tree describes {described}")
    reinstall = [word for row in stale for word in ("--reinstall-package", row[0])]
    # --inexact: a plain sync removes the extras this environment was given.
    status = kit.run(("uv", "sync", "--frozen", "--inexact", *reinstall))
    still = _stale_installs() or (stale if status != 0 else [])
    if not still:
        return None
    dist, installed, described = still[0]
    raise EnvironmentFault(
        "install currency",
        f"{dist} installed {installed}, tree describes {described} after uv sync",
    )


def _need_history() -> str | None:
    if kit.git("rev-parse", "--is-shallow-repository").strip() == "true":
        return "this clone is shallow: run git fetch --unshallow"
    return None


def _need_docs_extra() -> str | None:
    if kit.capture(DOCS_EXTRA_PROBE)[0] == 0:
        return None
    return "the docs extra is not installed: run uv sync --extra docs"


def _need_graphviz() -> str | None:
    if shutil.which("dot"):
        return None
    return "graphviz is not on PATH: the docs build draws its diagrams with dot; install graphviz"


def _need_va_extra() -> str | None:
    if kit.capture(LIVE_CA_PROBE)[0] == 0:
        return None
    return "the virtual-accelerator extra is not installed: run uv sync --extra virtual-accelerator"


#: Each need to its probe: ``None`` when met, else the refusal.
NEEDS = {
    "uv": _need_uv,
    "gitleaks": _need_gitleaks,
    "node": _need_node,
    "claude-pin": _need_claude_pin,
    "history": _need_history,
    "docs-extra": _need_docs_extra,
    "graphviz": _need_graphviz,
    "workflow-linters": _need_workflow_linters,
    "va-extra": _need_va_extra,
    "free-space": _need_free_space,
    "docker-storage": _need_docker_storage,
    "install-current": _need_install_current,
}


def _on_live_ca_venue() -> bool:
    """The platforms the virtual-accelerator extra's marker installs on."""
    return sys.platform == "linux" and platform.machine() in ("x86_64", "aarch64")


def _on_darwin_venue() -> bool:
    return sys.platform == "darwin"


VENUES = {
    LIVE_CA_VENUE: Venue(_on_live_ca_venue, "needs linux x86_64 or aarch64"),
    DARWIN_VENUE: Venue(_on_darwin_venue, "needs macOS"),
}


# ---------------------------------------------------------------------------
# The checks

CHECKS: tuple[Check, ...] = (
    Check("lockfile", "invariant", BOTH, ("uv", "lock", "--check"), needs=("uv",)),
    Check(
        "ruff",
        "invariant",
        BOTH,
        (*UV_RUN, "ruff", "check", "."),
        needs=("uv",),
        fix=((*UV_RUN, "ruff", "check", ".", "--fix"),),
        annotate=GITHUB_FORMAT,
    ),
    Check(
        "ruff-format",
        "invariant",
        BOTH,
        (*UV_RUN, "ruff", "format", "--check", "."),
        needs=("uv",),
        fix=((*UV_RUN, "ruff", "format", "."),),
    ),
    Check(
        "config-keys",
        "invariant",
        BOTH,
        (*UV_RUN, "python", "scripts/check_config_keys.py", "--back-test"),
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
    # Implemented in Python: git lists what is both tracked and ignored.
    Check("tracked-ignored", "invariant", BOTH, (), builtin=_check_tracked_ignored),
    Check(
        "gitleaks",
        "invariant",
        BOTH,
        ("gitleaks", "git", "--log-opts={base}..HEAD", "--no-banner", "--redact"),
        needs=("base", "gitleaks"),
        # ``git --staged`` scans the index, ``git --pre-commit`` the unstaged
        # diff, and ``dir`` one untracked path: ``dir`` given several paths or
        # none scans the whole directory.
        worktree=(
            ("gitleaks", "git", "--log-opts={base}..HEAD", "--no-banner", "--redact"),
            ("gitleaks", "git", "--staged", "--no-banner", "--redact"),
            ("gitleaks", "git", "--pre-commit", "--no-banner", "--redact"),
            ("gitleaks", "dir", "{untracked}", "--no-banner", "--redact"),
        ),
    ),
    Check(
        "invariant-tests",
        "invariant",
        BOTH,
        (*UV_RUN, "pytest", *INVARIANT_TESTS),
        needs=("uv",),
        report="junit-pytest",
    ),
    Check(
        "mypy",
        "static",
        BOTH,
        (*UV_RUN, "python", "scripts/mypy_gate.py"),
        scope=("*.py", "pyproject.toml", "uv.lock"),
        needs=("uv",),
    ),
    Check(
        "import-contracts",
        "static",
        BOTH,
        (*UV_RUN, "lint-imports"),
        scope=("*.py", "pyproject.toml", "uv.lock"),
        needs=("uv",),
    ),
    Check(
        "private-imports",
        "static",
        BOTH,
        (*UV_RUN, "ruff", "check", "--preview", "--select=PLC2701", "."),
        scope=("*.py", "pyproject.toml"),
        needs=("uv",),
        annotate=GITHUB_FORMAT,
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
        "actionlint",
        "static",
        BOTH,
        (*UV_RUN, "actionlint"),
        scope=(".github/workflows/**",),
        needs=("uv", "workflow-linters"),
    ),
    # The parallel sweep: its selection is UNIT_SELECTION, a range picks its
    # tests through TEST_MIRRORS, and a run whose paths the marker expression
    # empties is named, not failed.
    Check(
        "pytest",
        "tests",
        BOTH,
        (*UV_RUN, "pytest"),
        needs=("uv", "claude-pin"),
        bound_s=5400,
        report="junit-pytest",
        tier_flags={
            "quick": ("-n", "auto", "--maxfail=1", "--tb=line", "-q"),
            "full": ("-n", "auto", "--tb=short", "--timeout=600", "--timeout-method=signal"),
        },
        empty_exit=5,
        named_suffixes=(".py",),
        named_node_ids=True,
        named_dirs=_is_pytest_file,
        selector=_select_pytest,
        command=_pytest_command,
        prepare=_prune_bytecode,
        shards=(*UNIT_SELECTION.shards, *UNIT_SELECTION.trims),
        exec_flags=True,
    ),
    Check(
        "pytest-pty",
        "tests",
        FULL,
        (*UV_RUN, "pytest", "tests/pty/", "-m", "pty"),
        scope=("tests/pty/**", "src/osprey/cli/**"),
        needs=("uv",),
        bound_s=600,
        report="junit-pytest",
        tier_flags={"full": ("--tb=short",)},
        prepare=_prune_bytecode,
        exec_flags=True,
    ),
    # A wall-clock budget timed beside loaded workers measures the workers, so
    # the sweep deselects the marker and this suite runs alone.
    Check(
        "pytest-wallclock",
        "tests",
        FULL,
        (*UV_RUN, "pytest", "tests/integration", "-m", "wallclock"),
        scope=("tests/integration/**", "*.py", "pyproject.toml", "uv.lock"),
        needs=("uv",),
        bound_s=900,
        report="junit-pytest",
        tier_flags={"full": ("--tb=short",)},
        exec_flags=True,
    ),
    # The readonly gate refuses every shared-library load once user code starts,
    # so a stdlib lookup that loads native code lazily must be warmed before it
    # closes. The sweep runs the warm-ups on the tree's interpreter; this runs
    # them under 3.14 on macOS in a throwaway environment.
    Check(
        "readonly-warmups-py314",
        "tests",
        FULL,
        (
            *UV_RUN,
            "--isolated",
            "--python",
            "3.14",
            "--extra",
            "dev",
            "pytest",
            READONLY_WARMUPS_TEST_FILE,
            "-v",
        ),
        scope=(READONLY_WARMUPS_TEST_FILE, "src/osprey/runtime/**"),
        needs=("uv",),
        venue=DARWIN_VENUE,
        bound_s=900,
        report="junit-pytest",
    ),
    Check(
        "live-ca",
        "tests",
        FULL,
        (*UV_RUN, "python", "scripts/va/live_ca/gate.py", "--pva"),
        scope=(
            "tests/va/test_apply_fault.py",
            "tests/va/test_lume_pva_seam.py",
            "scripts/va/live_ca/**",
            "src/osprey/services/virtual_accelerator/**",
        ),
        needs=("uv", "va-extra"),
        venue=LIVE_CA_VENUE,
    ),
    # A range picks its files through the JavaScript module-to-test mapping.
    Check(
        "vitest",
        "tests",
        BOTH,
        ("npx", "vitest", "run"),
        needs=("node",),
        report="junit-vitest",
        named_suffixes=(".test.js", ".test.mjs"),
        selector=_select_vitest,
    ),
    # The unit lanes leave this test closed, having no reason to fetch a JS toolchain.
    Check(
        "app-check-js",
        "tests",
        FULL,
        (
            *UV_RUN,
            "--extra",
            "dev",
            "pytest",
            "tests/cli/test_app_cmd.py",
            "-k",
            "pinned_toolchain",
        ),
        scope=(
            "src/osprey/apps/js_check/**",
            "src/osprey/cli/app_cmd.py",
            "tests/cli/test_app_cmd.py",
            "tests/apps/fixtures/js_check_app/**",
        ),
        needs=("uv", "node"),
        bound_s=600,
        report="junit-pytest",
        prepare=_opt_in_js_check,
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
    # The build runs from a clean tree, and every redirect key must leave its
    # stub page in it.
    Check(
        "docs",
        "build",
        FULL,
        (*UV_RUN, "make", "html", "SPHINXOPTS=-W --keep-going"),
        scope=("docs/**", "*.rst", "*.md"),
        needs=("uv", "docs-extra", "graphviz"),
        workdir=DOCS_DIR,
        prepare=_clean_docs_build,
        builtin=_check_redirect_stubs,
    ),
    # A link no reader can follow fails the run; an address that answers a
    # reader and not a checker is in `linkcheck_ignore` in docs/source/conf.py.
    Check(
        "linkcheck",
        "build",
        FULL,
        (*UV_RUN, "make", "linkcheck"),
        scope=("docs/**",),
        needs=("uv", "docs-extra"),
        workdir=DOCS_DIR,
    ),
    # Implemented in Python: PACKAGE_STEPS, run in order.
    Check(
        "package",
        "build",
        FULL,
        (),
        scope=("pyproject.toml", "packages/**", "src/**", "MANIFEST*"),
        needs=("uv",),
        command=_package_command,
        builtin=_check_package,
    ),
)

#: The CI jobs that call the check command; every other required job is a lane.
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
        "framework-at-connectors-floor",
        "installs the published connectors floor",
        (
            "src/osprey/**",
            "packages/pyaml-cs-osprey/**",
            "packages/osprey-connectors/tests/osprey_connectors_tests/compat/**",
            "tests/connectors/**",
            "tests/registry/**",
            "scripts/ci/connectors_compat_lane.py",
            "scripts/ci/connectors_compat_matrix.py",
            "pyproject.toml",
            "uv.lock",
        ),
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
    _e2e("agent-record-e2e", "test_agent_record_e2e.py"),
    _e2e("host-record-e2e", "test_host_record_e2e.py", why="models"),
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
    Lane(
        "qmd-cache-warmup",
        "main-branch cache",
        ("src/osprey/templates/services/qmd/**", ".github/actions/build-qmd-image/**"),
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

#: What ``list --json`` prints beside the checks and the lanes.
LISTED = {
    "unit_selection": dataclasses.asdict(UNIT_SELECTION),
    "invariant_tests": list(INVARIANT_TESTS),
}
