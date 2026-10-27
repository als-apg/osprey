"""Stub or run a ``build/`` inside a deployment repo, for CLI tests that need one.

Every repo-scoped verb reads the *render* — ``<repo>/build/config.yml`` —
rather than a directory it was handed, so a CLI test that used to write one
``config.yml`` into ``tmp_path`` now needs a repo with a render in it. The
``lifecycle_repo`` fixture supplies the repo (source zone, state zone, an
anchored ``.gitignore``); this supplies the render, which the fixture
deliberately does not: ``build/`` is 100% derived, and the exemplar models the
repo *before* any build has run.

:func:`stub_build` runs no real ``osprey build``, which is far too slow for a
unit test that only needs a render to read. The verbs cannot tell the
difference: they read a rendered ``config.yml`` and a provenance manifest,
which is exactly what it writes, with the stamped fingerprint under the
caller's control so drift is a one-line parameter.

For the tests that do run one, :func:`invoke_build` is the in-process
``osprey build`` with the install and the lifecycle hooks skipped,
:func:`invoke_build_in` the same from a working directory instead of ``--repo``, and
:func:`rendered_config` reads back the ``build/config.yml`` it rendered.

Lives in a ``_``-prefixed module rather than ``conftest.py`` because these are
plain functions, not fixtures — callers want them inside a test body, often more
than once per test. Same convention as :mod:`tests.cli._scoped_subprocess`.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import yaml
from click.testing import CliRunner, Result

#: The smallest rendered config that still resolves a provider. Enough for the
#: pre-flight probes and the provider lookups the launch verbs run before they
#: hand off; tests needing more pass their own ``config``.
STUB_CONFIG = "claude_code:\n  provider: anthropic\n"

#: What an in-process build skips: the dependency install and the lifecycle hooks.
CI_FLAGS = ("--skip-deps", "--skip-lifecycle")


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


def invoke_build(repo: Path | None = None, *args: str) -> Result:
    """Run ``osprey build`` in-process with :data:`CI_FLAGS`.

    Args:
        repo: The repo, passed as ``--repo``. ``None`` passes none, so the build
            finds its repo from the working directory the caller set.
        *args: Further arguments, after the flags.

    Returns:
        The CliRunner result.
    """
    from osprey.cli.build_cmd import build

    repo_args = [] if repo is None else ["--repo", str(repo)]
    # The flags spelled out rather than as `*CI_FLAGS`: tests/test_offline_build_env.py
    # reads build argv from source and credits only a literal `--skip-deps`.
    return CliRunner().invoke(build, [*repo_args, "--skip-deps", "--skip-lifecycle", *args])


def invoke_build_in(cwd: Path, *args: str) -> Result:
    """Run :func:`invoke_build` with no ``--repo``, from *cwd*.

    The build finds its repo from the working directory, as an operator's
    bare ``osprey build`` does; the previous one is back once it returns.
    """
    with contextlib.chdir(cwd):
        return invoke_build(None, *args)


def rendered_config(repo: Path) -> dict[str, Any]:
    """The ``build/config.yml`` a build rendered for *repo*, parsed."""
    config: dict[str, Any] = yaml.safe_load((repo / "build" / "config.yml").read_text("utf-8"))
    return config
