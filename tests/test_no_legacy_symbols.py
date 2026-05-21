"""Release-gate test: no legacy ``prompts``-era symbols survive the rename.

Walks ``src/``, ``tests/``, and ``docs/`` (sources only) and asserts that none
of the following substrings appear anywhere. Intentionally has no per-file
exclusions — if a legitimate match shows up later, the rename surfaced it.
"""

from __future__ import annotations

from pathlib import Path

LEGACY_SUBSTRINGS = (
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
    assert not offenders, "Legacy prompts-era symbols remain:\n" + "\n".join(offenders)
