"""Release-gate test: no legacy renamed symbols survive.

Walks ``src/``, ``tests/``, and ``docs/`` (sources only) and asserts that none
of the following substrings appear anywhere. Intentionally has no per-file
exclusions — if a legitimate match shows up later, the rename surfaced it.

Six rename generations are guarded: the ``prompts``-era rename; the
``scan`` -> ``bluesky`` rename that generalized the Bluesky plan/run subsystem
(a plan is an arbitrary generator, not only a scan); the
``motor``/``detector`` -> ``setpoint``/``readback`` rename that put the bridge
in the control room's vocabulary; that same generalization carried through
the *prose*, which the second generation renamed the symbols for and then left
behind; the substrate-env -> devices-file move, which deleted the three
passthrough variables and the helpers that produced and read them; and the
control-context rename, which made the control target and the write posture one
record per *deployment* and retired every name that called that deployment-wide
thing a session's. The genuine
tokens that legitimately survive — the ``scan``/``grid_scan``
*plan names*, their ``Scan*Params`` schemas, physics scan docs, the many
unrelated senses of "scan" (a directory scan, a pattern scan, a CRT scanline),
and every upstream ``bluesky``/``ophyd`` name such as ``SimMotor`` or
``EpicsMotor`` — are NOT listed here.

Deliberately absent: the ``scan-agentic-e2e`` CI job id. GitHub branch
protection matches required checks on that exact name, so the job keeps it
while the test file it runs is renamed.
"""

from __future__ import annotations

from pathlib import Path

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
    # The bare ``session_store`` and ``session_target`` substrings are
    # deliberately NOT gated. ``web_auth.py``'s ``SessionStore`` is the
    # browser's session store, a genuinely different thing, and
    # ``test_posture_get_contract.py`` must spell ``session_target`` and
    # ``session_target_label`` to assert the posture payload no longer answers
    # them. Both would need the per-file exclusions this module refuses to have.
    "session_control_target",
    "osprey_connectors.session_store",
    "osprey_connectors/session_store",
    "_session_store_permits",
    "parse_store(",
    "test_session_store_parity",
    "session_target_mismatch",
    "REASON_SESSION_TARGET_MISMATCH",
    "resolve_session_target",
    "test_resolve_session_target",
    # Bare ``session_target`` cannot be gated (see above), so the two shapes it
    # actually took are gated instead: the dataclass attribute read and the
    # keyword argument. Between them they cover every site the rename touched
    # except the payload key, which the contract test guards by name.
    "situation.session_target",
    "session_target=",
)

ROOTS = ("src", "tests", "docs/source")
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
    (_REPO_ROOT / "tests/services/test_build_artifacts_imports.py").resolve(),
    (_REPO_ROOT / "tests/interfaces/web_terminal/test_scaffold_routes_registration.py").resolve(),
}


def _scan_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    files: list[Path] = []
    for root in ROOTS:
        base = repo_root / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in SCAN_SUFFIXES:
                continue
            if path.resolve() in SELF_ALLOWLIST:
                continue
            if "__pycache__" in path.parts or ".bak" in path.name:
                continue
            files.append(path)
    return files


def test_no_legacy_symbols() -> None:
    offenders: list[str] = []
    for path in _scan_roots():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for needle in LEGACY_SUBSTRINGS:
            if needle in content:
                # Capture a few line numbers for diagnosis
                lines = [
                    f"{i}: {ln}"
                    for i, ln in enumerate(content.splitlines(), start=1)
                    if needle in ln
                ][:3]
                offenders.append(f"{path}: {needle!r}\n  " + "\n  ".join(lines))
    assert not offenders, "Legacy renamed symbols remain:\n" + "\n".join(offenders)
