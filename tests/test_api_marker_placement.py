"""Placement guard: credentialed-resource tests must live under ``tests/e2e/``.

The fast lane is invoked as ``pytest tests/ --ignore=tests/e2e`` (see
``CLAUDE.md`` and ``.github/workflows/ci.yml``). That command must not make
credentialed provider calls *regardless of which keys a developer has
exported*.

The ``requires_<resource>`` skip-gates in ``tests/conftest.py`` are
one-directional: each skips when its resource is *absent*, but silently
exercises the real resource when it is *present*. For a credentialed marker
(``requires_api`` / ``requires_als_apg`` / ``requires_anthropic``) that means a
live LLM call — which passes on a clean machine yet fails on a blocked or
expired key for anyone who has a key in their shell. That non-determinism is
what this guard prevents.

We **default-deny** the entire ``requires_*`` marker family outside
``tests/e2e/`` rather than enumerate the credentialed ones, so a newly-added
credentialed marker is guarded automatically. Only keyless/local resources are
allowlisted below — they have no credential and no gateway-auth failure mode,
so they are fine in the fast lane (e.g. the ARIEL integration tests that gate
on a local Ollama server). If you add a new keyless resource marker, add it to
``_KEYLESS_ALLOWLIST``; if it needs a credential, put the test under
``tests/e2e/`` instead.
"""

from __future__ import annotations

import re
from pathlib import Path

# Any `pytest.mark.requires_<resource>` usage. Captures the marker name so we
# can allowlist keyless resources. The `mark.` prefix excludes the unrelated
# provider attribute `requires_api_key` (which appears as `.requires_api_key`,
# never `mark.requires_...`).
_MARKER_RE = re.compile(r"mark\.(requires_\w+)")

# Keyless / local resources — no credential, no gateway-auth failure mode, so
# leaving them in the fast lane is safe. Everything else in the requires_*
# family is treated as credentialed and must live under tests/e2e/.
_KEYLESS_ALLOWLIST = frozenset({"requires_ollama"})

_TESTS_ROOT = Path(__file__).resolve().parent
_E2E_ROOT = _TESTS_ROOT / "e2e"


def _fast_lane_test_files() -> list[Path]:
    """Every ``.py`` source the fast lane could collect (all of tests/ except
    the path-excluded ``tests/e2e/`` subtree, and this guard file itself)."""
    self_path = Path(__file__).resolve()
    files: list[Path] = []
    for path in _TESTS_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        resolved = path.resolve()
        if _E2E_ROOT in resolved.parents:
            continue
        if resolved == self_path:  # this guard names the markers in prose above
            continue
        files.append(path)
    return files


def test_no_credentialed_markers_outside_e2e() -> None:
    offenders: list[str] = []
    for path in _fast_lane_test_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits: list[str] = []
        for i, ln in enumerate(content.splitlines(), start=1):
            for marker in _MARKER_RE.findall(ln):
                if marker not in _KEYLESS_ALLOWLIST:
                    hits.append(f"{i}: {ln.strip()}")
                    break
        if hits:
            rel = path.relative_to(_TESTS_ROOT)
            offenders.append(f"tests/{rel}:\n  " + "\n  ".join(hits))
    assert not offenders, (
        "Credentialed `requires_*` tests must live under tests/e2e/ so the fast "
        "lane (`pytest tests/ --ignore=tests/e2e`) makes no credentialed call "
        "when a key is present. Move these to tests/e2e/ (or, if the resource is "
        "keyless/local, add its marker to _KEYLESS_ALLOWLIST):\n" + "\n".join(offenders)
    )
