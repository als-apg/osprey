"""Stub a ``build/`` inside a deployment repo, for CLI tests that need one.

Every repo-scoped verb reads the *render* — ``<repo>/build/config.yml`` —
rather than a directory it was handed, so a CLI test that used to write one
``config.yml`` into ``tmp_path`` now needs a repo with a render in it. The
``lifecycle_repo`` fixture supplies the repo (source zone, state zone, an
anchored ``.gitignore``); this supplies the render, which the fixture
deliberately does not: ``build/`` is 100% derived, and the exemplar models the
repo *before* any build has run.

Nothing here runs a real ``osprey build`` — that is Task 1.5's subject and far
too slow for a unit test. The verbs cannot tell the difference: they read a
rendered ``config.yml`` and a provenance manifest, which is exactly what this
writes, with the stamped fingerprint under the caller's control so drift is a
one-line parameter.

Lives in a ``_``-prefixed module rather than ``conftest.py`` because it is a
plain function, not a fixture — callers want it inside a test body, often more
than once per test. Same convention as :mod:`tests.cli._scoped_subprocess`.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The smallest rendered config that still resolves a provider. Enough for the
#: pre-flight probes and the provider lookups the launch verbs run before they
#: hand off; tests needing more pass their own ``config``.
STUB_CONFIG = "claude_code:\n  provider: anthropic\n"


def installed_version() -> str:
    """The framework version a build stamps when it is rendered right now."""
    from osprey.cli.templates.manifest import get_framework_release_version

    return get_framework_release_version()


def profile_hash(repo: Path) -> str:
    """The fingerprint ``osprey build`` would stamp for this repo's profile."""
    from osprey.deployment.staleness import profile_fingerprint

    fingerprint = profile_fingerprint(repo / "profile.yml")
    assert fingerprint is not None, f"profile at {repo} must resolve"
    return fingerprint.profile_hash


def stub_build(
    repo: Path,
    *,
    stamped_hash: str | None = "current",
    version: str | None = None,
    config: str = STUB_CONFIG,
    with_config: bool = True,
) -> Path:
    """Write a ``build/`` the drift check and the launch verbs can both read.

    Args:
        repo: The deployment repo root (must hold ``profile.yml``).
        stamped_hash: ``"current"`` stamps the profile's real fingerprint — a
            CLEAN build. Any other string stands for a build the profile has
            moved on from. ``None`` stamps no fingerprint at all, which is the
            unverifiable case.
        version: OSPREY version to stamp. Defaults to the installed one, so
            version skew is opt-in rather than an accident of the test machine.
        config: Contents of ``build/config.yml``.
        with_config: When False, leave the manifest without a rendered config
            beside it — a build in name only.

    Returns:
        The build directory.
    """
    build = repo / "build"
    build.mkdir(parents=True, exist_ok=True)
    if with_config:
        (build / "config.yml").write_text(config, encoding="utf-8")

    creation: dict[str, object] = {"osprey_version": version or installed_version()}
    if stamped_hash is not None:
        creation["preset_hash"] = profile_hash(repo) if stamped_hash == "current" else stamped_hash
    (build / ".osprey-manifest.json").write_text(
        json.dumps({"creation": creation}), encoding="utf-8"
    )
    return build
