"""Release-gate test: no legacy renamed symbols survive.

Walks ``src/``, ``tests/``, ``docs/source/`` and ``packages/`` (sources only;
vendored trees, ``changelog.d/`` fragments and a package's ``CHANGELOG.md``,
which record removals by name, are skipped) and asserts that none of the legacy names appear anywhere.

Nine rename generations are guarded. The first eight are plain substrings
with no per-file exclusions (only the docs redirect sources, which name retired
pages by design, are masked) — if a legitimate match shows up later, the rename
surfaced it: the ``prompts``-era rename; the
``scan`` -> ``bluesky`` rename that generalized the Bluesky plan/run subsystem
(a plan is an arbitrary generator, not only a scan); the
``motor``/``detector`` -> ``setpoint``/``readback`` rename that put the bridge
in the control room's vocabulary; that same generalization carried through
the *prose*, which the second generation renamed the symbols for and then left
behind; the substrate-env -> devices-file move, which deleted the three
passthrough variables and the helpers that produced and read them; the
control-context rename, which made the control target and the write posture one
record per *deployment* and retired every name that called that deployment-wide
thing a session's; the build-artifact catalog's move into the harness
adapter; and the agent record, which replaced the telemetry store's modules.
The store's own vocabulary (its service names, tables, reader keys and the
old audit ledger directory) is guarded in ``tests/test_retired_store_words.py``,
which allows the refusal code and its tests to name it. The genuine
tokens that legitimately survive — the ``scan``/``grid_scan``
*plan names*, their ``Scan*Params`` schemas, physics scan docs, the many
unrelated senses of "scan" (a directory scan, a pattern scan, a CRT scanline),
and every upstream ``bluesky``/``ophyd`` name such as ``SimMotor`` or
``EpicsMotor`` — are NOT listed here.

The ninth, the write-posture generation, retired every reader of the old
``writes_enabled`` posture when the tier key and ``write_verdict`` became the
one answer. Its names are ordinary words (``effective_writes``,
``target_posture``), so it matches whole identifiers only, and it has a closed
exemption map (path -> names). A stale exemption — a path that is gone, or a
name it no longer spells — fails, so the map can only shrink.

Deliberately absent: the ``scan-agentic-e2e`` CI job id. GitHub branch
protection matches required checks on that exact name, so the job keeps it
while the test file it runs is renamed.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

from tests._text_sweep import text_files

LEGACY_SUBSTRINGS = (
    # prompts-era rename
    "PromptCatalog",
    "PromptArtifact",
    "services.prompts",
    "services/prompts",
    "from osprey.services.prompts",
    "osprey prompts",
    "/api/prompts",
    "prompts-gallery",
    "PromptGalleryService",
    "prompts.css",
    "prompts_cmd",
    # scan -> bluesky rename: mislabeled subsystem identifiers (never genuine)
    "scan_panels",
    "SCAN_PANELS",
    "scan-panels",
    "ScanPanelsConfig",
    "_inject_scan_panels",
    # bluesky_panels -> bluesky_web rename: the sidecar is named for its role
    # (the browser-facing half of the bluesky stack), not for the panel it
    # serves — "panel" is the web terminal's word for a tab.
    "bluesky_panels",
    "BLUESKY_PANELS",
    "bluesky-panels",
    "BlueskyPanelsConfig",
    "_inject_bluesky_panels",
    "mcp__scan__",
    "osprey.mcp_server.scan",
    "mcp_server/scan",
    "create_scan_intent",
    "list_scan_plans",
    "read_scan_data",
    "launch_scan",
    "stop_scan",
    "scan_status",
    "BlueskyScanner",
    "FakeScanner",
    "ScanContext",
    "scanner_factory",
    "scanner_bluesky",
    "BLUESKY_DEMO_SCANNER",
    "write_bluesky_plan",
    "validate_bluesky_plan",
    # motor/detector -> setpoint/readback rename: the bluesky bridge speaks the
    # control room's vocabulary, and beamline device words are retired for good.
    # Upstream bluesky/ophyd names are untouched by this list -- nothing here is
    # a name OSPREY does not own.
    "BLUESKY_EPICS_MOTORS",
    "BLUESKY_EPICS_DETECTORS",
    "parse_motor_specs",
    "parse_detector_specs",
    "format_motors_env",
    "format_detectors_env",
    "MockMotor",
    "MockDetector",
    # scan -> plan in the prose: a plan is an arbitrary generator, and the
    # earlier generation above renamed only the identifiers. Gated here are the
    # names that generation left behind. The compound noun "scan plan" is NOT
    # gated: `tune-scan plan` and `grid-scan plan` are genuine, so a substring
    # gate on it would need the per-file exclusions this module refuses to have.
    "operating-bluesky-scans",
    "run-first-scan",
    "test_scan_stack_agentic",
    "scan-bridge",
    # substrate env -> devices file: the worker's device set is authored in one
    # file the deployment mounts, not inferred from PV spellings handed to the
    # container in three env vars. The variables, and the helpers that derived,
    # formatted, wrote and parsed them, are deleted rather than aliased -- they
    # never left ``[Unreleased]``, so no deployment can be setting them. The
    # bare ``BLUESKY_EPICS`` prefix is NOT gated: the render tests assert the
    # names are absent, and asserting an absence means spelling the prefix.
    "BLUESKY_EPICS_SUBSTRATE",
    "BLUESKY_EPICS_SETPOINTS",
    "BLUESKY_EPICS_READBACKS",
    "specs_from_env",
    "is_substrate_enabled",
    "format_setpoints_env",
    "format_readbacks_env",
    "derive_substrate_env",
    "write_substrate_env",
    # control-context rename: the control target and the write posture became
    # ONE record per deployment, so every name calling that deployment-wide
    # thing a session's is retired. Nothing here was ever released, so the old
    # names are deleted rather than aliased.
    #
    # The bare ``session_store`` substring is deliberately NOT gated:
    # ``web_auth.py``'s ``SessionStore`` is the browser's session store, a
    # genuinely different thing, and gating it would need the per-file
    # exclusions this module refuses to have.
    "session_control_target",
    "osprey_connectors.session_store",
    "osprey_connectors/session_store",
    "_session_store_permits",
    "parse_store(",
    "test_session_store_parity",
    "REASON_SESSION_TARGET_MISMATCH",
    # The bare name covers every shape it took: the attribute, the keyword
    # argument, the payload key, and the helpers and reasons built on it.
    "session_target",
    # build-artifact catalog moved inside the harness adapter
    "services.build_artifacts",
    "services/build_artifacts",
    # the agent record replaced the telemetry store: the modules that
    # configured, credentialed, linted and exported it, and the session
    # reconstruction query and page that read it, are deleted.
    "telemetry_config",
    "telemetry_credentials",
    "telemetry_lint",
    "telemetry_export",
    "reconstruct_session",
    "reconstruct-session",
)

# write-posture generation: every reader of the old ``writes_enabled`` posture,
# deleted rather than aliased when ``control_system.writes`` (the tier) and
# ``write_verdict`` became the one answer. Matched on WHOLE identifiers: the
# names are ordinary words, and ``effective_writes`` is a prefix of nothing
# live only by luck. Bare ``writes_enabled`` and ``WRITES_KEY`` are NOT listed:
# the first is a live keyword and report field, the second the live tier key.
WRITE_POSTURE_IDENTIFIERS = (
    # osprey_connectors.types readers
    "WRITES_ENABLED_KEY",
    "TYPE_WRITES_ENABLED_LEAF",
    "type_writes_enabled",
    "target_writes_enabled",
    "target_writes_enabled_key",
    "session_posture",
    "any_armed_target_requires_limits_database",
    "any_target_writes_enabled",
    "any_armed_target_checks_limits",
    "writes_enabled_key",
    "writes_enabled_answering_key",
    "writes_enabled_remedy",
    # osprey_connectors.posture_store readers
    "effective_writes",
    "store_permits",
    "store_verdict",
    "store_verdict_detail",
    "record_verdict_detail",
    "StoreVerdict",
    "StoreVerdictDetail",
    "recorded_posture",
    "target_posture",
    "launch_permits",
    "launch_narrowed_target",
    "parse_launch_posture",
    # osprey_connectors.ipc
    "config_writes_enabled",
    # target_eligibility
    "effective_writes_for_target",
    "narrowing_refusal",
    # python executor
    "ExecutionControlConfig",
    "get_execution_control_config",
    "_writes_disabled_result",
    # the stdlib twin's pre-switch posture functions
    "WRITES_ENABLED_LEAF",
    "effective_writes_for",
    "most_restrictive_posture",
    "target_sandboxed",
    "type_posture",
    "writes_posture",
    "_global_posture",
    "_states_posture",
    "_posture_for",
)

# Closed exemption map, repo-relative path -> the names that path may spell.
# Every entry is live: test_write_posture_exemptions_are_live fails on a path
# that is gone or a name it no longer contains, so the map only shrinks.
WRITE_POSTURE_EXEMPTIONS: dict[str, frozenset[str]] = {
    # Transition tests: they assert the retired name is absent, and asserting
    # an absence means spelling the name.
    "tests/hooks/test_target_state_reader.py": frozenset(
        {
            "effective_writes_for",
            "most_restrictive_posture",
            "recorded_posture",
            "target_posture",
            "target_sandboxed",
            "type_posture",
            "writes_posture",
        }
    ),
    "tests/interfaces/web_terminal/test_posture_get_contract.py": frozenset(
        {"narrowing_refusal", "session_posture"}
    ),
}

ROOTS = ("src", "tests", "docs/source", "packages")
# Never walked: caches, vendored and built trees, and changelog fragments (a
# removal entry names what it removed; the root ``changelog.d`` is outside the
# roots for the same reason). A package's ``CHANGELOG.md`` is where its
# fragments are folded at release, so it is skipped by name for that reason too.
SKIP_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        ".venv",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "changelog.d",
    }
)
SKIP_FILES = frozenset({"CHANGELOG.md"})
SCAN_SUFFIXES = (
    ".py",
    ".md",
    ".rst",
    ".js",
    ".css",
    ".html",
    ".j2",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
)
# Transition tests must reference legacy names by design — they assert the
# legacy module no longer imports, the legacy route no longer registers, etc.
# Skip them here so the gate runs against all *other* sources.
_REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_ALLOWLIST = {
    Path(__file__).resolve(),
    (_REPO_ROOT / "tests/agent_runner/test_build_artifacts_imports.py").resolve(),
}


# A redirect's source is a retired URL by definition: the docs ``redirects``
# dict keeps old page paths alive so links into them still land. Its keys are
# masked before matching; its targets and every other line are still judged.
_DOCS_CONF = (_REPO_ROOT / "docs/source/conf.py").resolve()


def _redirect_sources(content: str) -> list[str]:
    """The key literals of the module-level ``redirects`` dict in ``content``."""
    for node in ast.parse(content).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "redirects":
            assert isinstance(value, ast.Dict), "redirects must be a dict literal"
            return [
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
    return []


def _judged_text(path: Path, content: str) -> str:
    """``content`` with the docs redirect sources masked out."""
    if path.resolve() != _DOCS_CONF:
        return content
    for source in _redirect_sources(content):
        content = content.replace(f'"{source}":', '"":')
    return content


def _scan_roots(repo_root: Path = _REPO_ROOT, roots: tuple[str, ...] = ROOTS) -> list[Path]:
    def skip(path: Path) -> bool:
        return (
            path.resolve() in SELF_ALLOWLIST
            or ".bak" in path.name
            or path.name in SKIP_FILES
            or bool(SKIP_DIRS.intersection(path.relative_to(repo_root).parts))
        )

    if roots is not ROOTS:
        return text_files(repo_root, roots, suffixes=SCAN_SUFFIXES, skip=skip)
    # The repository's own scan names ``ROOTS`` at the walk, so check.py's
    # tree-reader detector sees that this module reads the whole tests tree.
    return text_files(repo_root, ROOTS, suffixes=SCAN_SUFFIXES, skip=skip)


def _identifier(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")


_WRITE_POSTURE_PATTERNS = {name: _identifier(name) for name in WRITE_POSTURE_IDENTIFIERS}
# One pass per file: the alternation finds every whole identifier that is a
# retired name; longest names first so no alternative shadows a longer one.
_WRITE_POSTURE_ANY = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(re.escape(n) for n in sorted(WRITE_POSTURE_IDENTIFIERS, key=len, reverse=True))
    + r")(?![A-Za-z0-9_])"
)


def _legacy_offenders_in(path: Path, content: str) -> list[str]:
    offenders: list[str] = []
    for needle in LEGACY_SUBSTRINGS:
        if needle in content:
            # Capture a few line numbers for diagnosis
            lines = [
                f"{i}: {ln}" for i, ln in enumerate(content.splitlines(), start=1) if needle in ln
            ][:3]
            offenders.append(f"{path}: {needle!r}\n  " + "\n  ".join(lines))
    return offenders


def _write_posture_offenders_in(path: Path, content: str, allowed: frozenset[str]) -> list[str]:
    # The alternation is the slow part of the sweep, and a whole identifier it
    # finds is a substring first: a file that holds none of the names is passed
    # over without it.
    if not any(name in content for name in WRITE_POSTURE_IDENTIFIERS):
        return []
    offenders: list[str] = []
    for name in sorted(set(_WRITE_POSTURE_ANY.findall(content)) - allowed):
        pattern = _WRITE_POSTURE_PATTERNS[name]
        lines = [
            f"{i}: {ln}" for i, ln in enumerate(content.splitlines(), start=1) if pattern.search(ln)
        ][:3]
        offenders.append(f"{path}: {name!r}\n  " + "\n  ".join(lines))
    return offenders


def _sweep(
    repo_root: Path = _REPO_ROOT,
    roots: tuple[str, ...] = ROOTS,
    exemptions: dict[str, frozenset[str]] = WRITE_POSTURE_EXEMPTIONS,
) -> tuple[list[str], list[str]]:
    """The legacy-substring and the write-posture offenders, from one read of each file."""
    legacy: list[str] = []
    posture: list[str] = []
    for path in _scan_roots(repo_root, roots):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        legacy += _legacy_offenders_in(path, _judged_text(path, content))
        allowed = exemptions.get(path.relative_to(repo_root).as_posix(), frozenset())
        posture += _write_posture_offenders_in(path, content, allowed)
    return legacy, posture


@functools.cache
def _repository_sweep() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """:func:`_sweep` of the repository itself, read once for the two tests that assert it.

    Reading the scanned tree is most of what either test costs, and the tree
    does not change while the suite runs.
    """
    legacy, posture = _sweep()
    return tuple(legacy), tuple(posture)


def _write_posture_offenders(
    repo_root: Path = _REPO_ROOT,
    roots: tuple[str, ...] = ROOTS,
    exemptions: dict[str, frozenset[str]] = WRITE_POSTURE_EXEMPTIONS,
) -> list[str]:
    return _sweep(repo_root, roots, exemptions)[1]


def test_no_legacy_symbols() -> None:
    offenders = _repository_sweep()[0]
    assert not offenders, "Legacy renamed symbols remain:\n" + "\n".join(offenders)


def test_no_write_posture_identifiers() -> None:
    offenders = _repository_sweep()[1]
    assert not offenders, "Retired write-posture names remain:\n" + "\n".join(offenders)


@pytest.mark.parametrize("rel_path", sorted(WRITE_POSTURE_EXEMPTIONS))
def test_write_posture_exemptions_are_live(rel_path: str) -> None:
    """An exemption whose path is gone, or that no longer spells a name, fails."""
    names = WRITE_POSTURE_EXEMPTIONS[rel_path]
    assert names, rel_path
    assert names <= set(WRITE_POSTURE_IDENTIFIERS), sorted(names - set(WRITE_POSTURE_IDENTIFIERS))
    path = _REPO_ROOT / rel_path
    assert path.is_file(), f"stale exemption: {rel_path} no longer exists"
    parts = Path(rel_path).parts
    scanned = (
        any(rel_path.startswith(f"{root}/") for root in ROOTS)
        and not SKIP_DIRS.intersection(parts)
        and path.suffix in SCAN_SUFFIXES
    )
    assert scanned, f"exemption {rel_path} is outside the scanned tree"
    content = path.read_text(encoding="utf-8", errors="ignore")
    stale = sorted(n for n in names if not _WRITE_POSTURE_PATTERNS[n].search(content))
    assert not stale, f"stale exemption: {rel_path} no longer spells {stale}"


def test_planted_name_in_packages_fails(tmp_path: Path) -> None:
    """Negative control: the gate sees a retired name planted under packages/."""
    pkg = tmp_path / "packages" / "osprey-demo" / "src" / "osprey_demo"
    pkg.mkdir(parents=True)
    (pkg / "planted.py").write_text("def launch_permits(section):\n    return True\n")
    # Whole identifiers only: a longer name containing a retired one is not it.
    (pkg / "near_miss.py").write_text("my_launch_permits = launch_permits_v2 = 1\n")
    # An exemption covers only the names it lists.
    (pkg / "exempt.py").write_text("store_permits = 1\nStoreVerdict = 2\n")
    # Changelog fragments record removals by name and are never walked.
    log = tmp_path / "packages" / "osprey-demo" / "changelog.d"
    log.mkdir()
    (log / "x.removed.md").write_text("`launch_permits` is gone.\n")
    # Nor is the package changelog those fragments are folded into.
    (tmp_path / "packages" / "osprey-demo" / "CHANGELOG.md").write_text(
        "`launch_permits` is gone.\n"
    )
    exemptions = {
        "packages/osprey-demo/src/osprey_demo/exempt.py": frozenset({"store_permits"}),
    }

    offenders = _write_posture_offenders(tmp_path, ("packages",), exemptions)

    flagged = sorted((Path(o.split(":", 1)[0]).name, o.split("'")[1]) for o in offenders)
    assert flagged == [("exempt.py", "StoreVerdict"), ("planted.py", "launch_permits")]
