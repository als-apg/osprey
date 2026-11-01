"""Wheel artifacts staged into image build contexts: the dev wheel and deployment packages.

A ``--dev`` deploy installs a locally-built osprey wheel into container build
contexts instead of the released PyPI pin. This module owns that wheel's whole
lifecycle: building it at most once per process
(:func:`_build_dev_wheel_cached`), caching the build across the many contexts a
single deploy stages (:data:`_wheel_build_cache`), copying it into a context
alongside its base-requirements manifest
(:func:`_copy_local_framework_for_override`,
:func:`_write_local_requirements_manifest`), and enumerating what staging added
so callers can clean up exactly that (:func:`_staged_dev_artifact_paths`).

The per-context *decision* to stage — gated on ``--dev`` and the presence of a
``Dockerfile`` — stays in :mod:`osprey.deployment.compose_generator`
(``_stage_dev_wheel_for_context``), where it is wired into the compose render
flow; that module re-exports the copy and cache-reset helpers below so existing
import and monkeypatch call sites keep resolving against it.

It also builds a deployment's own packages (:func:`build_deployment_packages`)
into the directory every image context carries, with their manifest.
"""

import atexit
import contextlib
import os
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from osprey.cli.phase_reporter import report_step
from osprey.deployment.errors import DevModeUnavailableError
from osprey.deployment.members import WORKSPACE_MEMBERS
from osprey_connectors.logger import get_logger

logger = get_logger("deployment.compose")


def preflight_dev_mode():
    """Verify ``--dev`` can be honored, or raise before any deploy work begins.

    Every precondition here is knowable in microseconds — an import-path check,
    a ``stat``, and a module-spec lookup — so the check runs at the top of a
    deploy rather than surfacing seven services in, after a subprocess launch.

    This exists because the failure it guards is invisible by construction: a
    ``--dev`` deploy that cannot stage a wheel still brings containers up
    *successfully*, running the pinned PyPI release. Nothing downstream looks
    wrong; the deployment simply is not testing the code the user meant to test.

    :raises DevModeUnavailableError: If osprey is not installed editable, its
        source root has no ``pyproject.toml``, or the ``build`` package is
        missing.
    """
    import importlib.util

    try:
        import osprey
    except ImportError as e:  # pragma: no cover — osprey is importing this module
        raise DevModeUnavailableError(
            "the osprey package could not be imported",
            "Reinstall osprey: uv pip install -e <path-to-osprey-checkout>",
        ) from e

    module_path = Path(osprey.__file__).parent
    path_str = str(module_path)
    if "site-packages" in path_str or "dist-packages" in path_str:
        raise DevModeUnavailableError(
            f"osprey is installed from PyPI, not as an editable checkout ({path_str})",
            "--dev builds a wheel from a local osprey source tree, which a "
            "released install does not have.\n"
            "Reinstall editable: uv pip install -e <path-to-osprey-checkout>",
        )

    source_root = module_path.parent.parent
    if not (source_root / "pyproject.toml").exists():
        raise DevModeUnavailableError(
            f"no pyproject.toml found at the osprey source root ({source_root})",
            "--dev needs a complete source checkout to build a wheel from.\n"
            "Reinstall editable from a full checkout: uv pip install -e <path-to-osprey-checkout>",
        )

    if importlib.util.find_spec("build") is None:
        raise DevModeUnavailableError(
            "the 'build' package is not installed in the environment running osprey",
            "--dev builds a wheel from your local osprey checkout so the "
            "containers run your code instead of the pinned PyPI release.\n"
            "Install it with:  uv sync --extra dev\n"
            "        or with:  uv pip install build",
        )


# Staged next to every dev wheel (see ``_copy_local_framework_for_override``
# below): the wheel's own base dependency list, which the Dockerfiles' deps
# layer installs with the build toolchain still present. Without it the deps
# layer primes only the RELEASED PyPI pin's dependencies, and any base dep the
# local wheel adds that the release lacks (e.g. a native sdist like
# accelerator-toolbox) would have to compile in the later toolchain-less
# wheel-install layer — and fail.
LOCAL_REQUIREMENTS_FILENAME = "osprey-local-requirements.txt"


def _wheel_base_requirements(wheel_path: str | Path) -> list[str]:
    """Extract the wheel's base (non-extra) ``Requires-Dist`` entries, sorted.

    Parses ``*.dist-info/METADATA`` inside the wheel and returns every
    ``Requires-Dist`` requirement string EXCEPT those gated behind an extra
    (environment marker referencing ``extra``). Non-extra markers such as
    ``python_version`` are kept verbatim so pip evaluates them in-container.

    The result is fully deterministic for identical wheels — sorted, with
    whitespace normalized — because the manifest built from it is COPY'd into
    a BuildKit-content-hashed layer: byte-identical input keeps the deps-layer
    cache warm across deploys.

    :param wheel_path: Path to the built wheel file
    :type wheel_path: str or pathlib.Path
    :return: Sorted requirement strings (extras excluded)
    :rtype: list[str]
    :raises ValueError: If the wheel contains no ``*.dist-info/METADATA``
    """
    import zipfile
    from email.parser import Parser

    from packaging.requirements import Requirement

    with zipfile.ZipFile(wheel_path) as whl:
        metadata_names = sorted(
            name for name in whl.namelist() if name.endswith(".dist-info/METADATA")
        )
        if not metadata_names:
            raise ValueError(f"no *.dist-info/METADATA found in wheel {wheel_path}")
        metadata = Parser().parsestr(whl.read(metadata_names[0]).decode("utf-8"))

    requirements = []
    for entry in metadata.get_all("Requires-Dist") or []:
        req = Requirement(entry)
        if req.marker is not None and "extra" in str(req.marker):
            continue  # extra-gated dep: stays out of the base install set
        # Collapse header-folding whitespace so the written line is stable
        # and single-line; the requirement text itself stays verbatim.
        requirements.append(" ".join(entry.split()))
    return sorted(requirements)


def _is_workspace_requirement(line: str) -> bool:
    """Whether requirement *line* names a workspace-local distribution.

    Such requirements are excluded from the manifest: the distribution is
    staged as a wheel in the same build context and installed by the
    Dockerfiles' wheel layer, so a PyPI requirement for it would fail until
    (and unless) a satisfying release exists there. Names compare in their
    PEP 503 canonical form, as the resolver compares them.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    members = {canonicalize_name(member) for member in WORKSPACE_MEMBERS}
    return canonicalize_name(Requirement(line).name) in members


def _describe_glob(member_root: Path) -> str | None:
    """The ``git describe --match`` glob *member_root*'s version is read from, if it names one."""
    import shlex
    import tomllib

    try:
        with (member_root / "pyproject.toml").open("rb") as handle:
            options = tomllib.load(handle)["tool"]["hatch"]["version"]["raw-options"]
        # setuptools-scm's `scm.git.describe_command` table (osprey-connectors),
        # else the flat `git_describe_command` key the other members still use.
        command = options.get("scm", {}).get("git", {}).get("describe_command")
        if command is None:
            command = options["git_describe_command"]
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        return argv[argv.index("--match") + 1]
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None


def _require_members_satisfy_framework(
    framework_wheel: Path, member_wheels: dict[str, tuple[Path, Path]]
) -> None:
    """Refuse a member wheel whose version the framework wheel's own requirement refuses.

    Each member versions from git tags (``osprey-connectors`` from its own
    ``osprey-connectors-v*`` tags), so a checkout missing them — a clone that
    never fetched tags, a fork or mirror that synced only the framework's
    ``v*`` tags — builds the member as ``0.0.postN``. The wheel layer would
    then pass over the staged wheel for PyPI and fail late, inside the image
    build; this names the cause before anything is staged. A pre-release
    satisfies: a dev wheel descending from a beta tag is one.

    :param framework_wheel: The built framework wheel
    :type framework_wheel: pathlib.Path
    :param member_wheels: Each member's distribution name mapped to its
        checkout root and built wheel
    :type member_wheels: dict[str, tuple[pathlib.Path, pathlib.Path]]
    :raises DevModeUnavailableError: If a member wheel's version falls outside
        the framework's requirement on it
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name, parse_wheel_filename

    built = {canonicalize_name(member): entry for member, entry in member_wheels.items()}
    for line in _wheel_base_requirements(framework_wheel):
        requirement = Requirement(line)
        entry = built.get(canonicalize_name(requirement.name))
        if entry is None:
            continue
        member_root, wheel = entry
        version = parse_wheel_filename(wheel.name)[1]
        if requirement.specifier.contains(version, prereleases=True):
            continue
        glob = _describe_glob(member_root)
        source = f"the newest git tag matching {glob}" if glob else "its git tags"
        raise DevModeUnavailableError(
            f"the {requirement.name} wheel built from the checkout is {version}, "
            f"which the framework's requirement '{line}' refuses",
            f"{requirement.name} takes its version from {source} reachable from "
            "the checkout; with none it builds as 0.0.postN.\n"
            "Fetch the tags:  git fetch --tags\n"
            "A fork or mirror must sync the osprey-connectors-v* tags as well as "
            "the framework's v* tags.",
        )


def _write_local_requirements_manifest(cached_wheels: list[Path], out_dir: str) -> None:
    """Write ``osprey-local-requirements.txt`` next to the staged dev wheels.

    Content contract (shared with the service Dockerfiles, which COPY the file
    and pip-install it in their toolchain-equipped deps layer): one requirement
    per line, sorted, trailing newline — byte-identical for identical wheels.
    The union of every staged wheel's base requirements, minus requirements on
    workspace-local distributions (see :func:`_is_workspace_requirement`), which the
    wheel layer satisfies from the wheels staged beside this manifest.

    :param cached_wheels: The cached local wheels the manifest derives from
    :type cached_wheels: list[pathlib.Path]
    :param out_dir: Build context directory the wheels were staged into
    :type out_dir: str
    """
    manifest_path = os.path.join(out_dir, LOCAL_REQUIREMENTS_FILENAME)
    requirements: set[str] = set()
    for wheel in cached_wheels:
        requirements.update(_wheel_base_requirements(wheel))
    lines = sorted(line for line in requirements if not _is_workspace_requirement(line))
    content = "".join(f"{line}\n" for line in lines)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


def _staged_dev_artifact_paths(context_dir: str) -> set[Path]:
    """All dev-staging artifacts (wheels + requirements manifest) in a context.

    Callers that stage into a long-lived directory (project-image and render
    builds) snapshot this before/after staging so their ``finally`` cleanup
    removes exactly what staging added — wheel AND manifest — and nothing that
    was already there.

    :param context_dir: Build context directory
    :type context_dir: str
    :return: Paths of staged dev artifacts currently present
    :rtype: set[pathlib.Path]
    """
    root = Path(context_dir)
    artifacts = set(root.glob("*.whl"))
    manifest = root / LOCAL_REQUIREMENTS_FILENAME
    if manifest.exists():
        artifacts.add(manifest)
    return artifacts


# Per-process memo for the dev-wheel build: one deploy stages the SAME wheel
# into many build contexts (every service context, the project image root, each
# render project), and `rebuild_deployment` even renders the compose files
# twice — so without a memo `python -m build` (an isolated-env build taking tens
# of seconds) runs once per staging call. Key: resolved osprey source root.
# Value: path to the cached built wheel (in a memo-owned temp dir), or None when
# the build failed (failures are memoized too — retrying an identical build in
# the same process would just fail identically, slowly). Callers are sequential,
# so no locking is needed.
_wheel_build_cache: dict = {}
_wheel_cache_dir = None
# Whether _reset_wheel_build_cache is already registered with atexit. Set the
# first time a cache dir is created so a real deploy process cannot leak the
# temp dir (and its wheel copy); one registration covers the whole process
# because the reset hook is idempotent and re-reads the current cache dir.
_wheel_cache_cleanup_registered = False


def _reset_wheel_build_cache():
    """Clear the per-process dev-wheel build memo and its cached wheel dir.

    Explicit reset hook for tests (and any long-lived process that needs a
    fresh build, e.g. after editing framework source mid-process). Also
    registered with ``atexit`` when the cache dir is first created, so normal
    process exit removes the temp dir. Idempotent — safe to call repeatedly
    (including once explicitly and once again at exit).
    """
    global _wheel_cache_dir
    _wheel_build_cache.clear()
    if _wheel_cache_dir is not None:
        shutil.rmtree(_wheel_cache_dir, ignore_errors=True)
    _wheel_cache_dir = None


def _build_dev_wheel_cached(osprey_source_root):
    """Build the local osprey wheel at most once per process; return its cached path.

    :param osprey_source_root: Root of the editable osprey checkout (contains
        ``pyproject.toml``)
    :type osprey_source_root: Path
    :return: Path to the cached wheel file, or ``None`` if the build failed
    :rtype: pathlib.Path or None
    """
    global _wheel_cache_dir, _wheel_cache_cleanup_registered

    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    key = str(Path(osprey_source_root).resolve())
    if key in _wheel_build_cache:
        return _wheel_build_cache[key]

    cached_wheel = None
    # Build the wheel package from local source. Not announced first: the build
    # is memoized to one shot per process, and the step below carries the fact
    # with its outcome attached.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use sys.executable, NOT bare "python3": in a non-activated venv,
        # PATH "python3" resolves to the system/pyenv interpreter (which
        # lacks the 'build' package), not the venv running osprey. Bare
        # python3 made --dev silently fall back to the PyPI release, booting
        # containers with stale osprey that lacks unreleased modules.
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", tmpdir],
            cwd=osprey_source_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # A failed build is the most dangerous of the --dev failure modes:
            # if the local checkout is broken, continuing would deploy the
            # RELEASED code under a flag that says "run my local code", so the
            # very changes being tested are invisible. Raise rather than warn,
            # and do not memoize — the next call should retry a fixed checkout.
            if "No module named build" in result.stderr:
                raise DevModeUnavailableError(
                    "the 'build' package is not installed in the environment running osprey",
                    "Install it with:  uv sync --extra dev\n        or with:  uv pip install build",
                )
            raise DevModeUnavailableError(
                f"building a wheel from the local osprey checkout at {osprey_source_root} failed",
                f"Fix the build error below, then re-run:\n\n{result.stderr.strip()}",
            )
        # Find the built wheel and move it to the memo-owned cache dir
        # (tmpdir is deleted on exit; the cache must outlive it so later
        # staging calls can copy the same build instead of rebuilding).
        wheel_files = list(Path(tmpdir).glob("*.whl"))
        if not wheel_files:
            raise DevModeUnavailableError(
                "the wheel build reported success but produced no .whl file",
                f"Check the build backend configured in {osprey_source_root / 'pyproject.toml'}.",
            )
        if _wheel_cache_dir is None:
            _wheel_cache_dir = tempfile.mkdtemp(prefix="osprey-dev-wheel-cache-")
            if not _wheel_cache_cleanup_registered:
                atexit.register(_reset_wheel_build_cache)
                _wheel_cache_cleanup_registered = True
        cached_wheel = Path(_wheel_cache_dir) / wheel_files[0].name
        shutil.copy2(wheel_files[0], cached_wheel)
        # Named per distribution: this builds the framework wheel and
        # every workspace member wheel in one deploy, and several steps
        # wearing one label read as a stutter.
        distribution = cached_wheel.name.split("-", 1)[0].replace("_", "-")
        report_step(f"built the {distribution} wheel from the local checkout")

    _wheel_build_cache[key] = cached_wheel
    return cached_wheel


def _copy_local_framework_for_override(out_dir):
    """Build and copy local osprey wheel to container build directory for development mode.

    This function builds a wheel package from the local osprey source and copies it
    to the container build directory. This approach is cleaner and more reliable than
    copying source files, as it properly handles package structure and avoids namespace
    collisions with other packages.

    The wheel is built using the standard Python build process and can be installed
    in containers to override the PyPI version during development and testing.
    The build itself runs at most once per process (see
    :func:`_build_dev_wheel_cached`); each call copies the cached wheel into its
    own ``out_dir`` and writes ``osprey-local-requirements.txt`` (the wheel's
    own base dependency list — see :func:`_write_local_requirements_manifest`)
    next to it. Success (True) means BOTH landed; a manifest failure removes
    the staged wheel and reports failure so the ``OSPREY_DEV`` gating never
    fires for a half-staged context.

    Note: This function only works when osprey is installed in editable/development mode
    (e.g., `pip install -e .`). If osprey is installed from PyPI or via regular
    `pip install .`, the source files are not available and containers will use
    the installed PyPI version.

    :param out_dir: Container build output directory
    :type out_dir: str
    :return: True if osprey wheel was successfully built and copied, False otherwise
    :rtype: bool
    """
    try:
        # Try to import osprey to get its location
        from pathlib import Path

        import osprey

        # Get the osprey module path
        osprey_module_path = Path(osprey.__file__).parent

        # Check if osprey is installed from source (editable mode) vs from site-packages
        # If installed from site-packages, we can't build a wheel from the source
        osprey_path_str = str(osprey_module_path)
        if "site-packages" in osprey_path_str or "dist-packages" in osprey_path_str:
            raise DevModeUnavailableError(
                f"osprey is installed from PyPI, not as an editable checkout ({osprey_path_str})",
                "--dev builds a wheel from a local osprey source tree, which a "
                "released install does not have.\n"
                "Reinstall editable: uv pip install -e <path-to-osprey-checkout>",
            )

        # Get the osprey source root (go up from src/osprey to root)
        osprey_source_root = osprey_module_path.parent.parent

        # Verify this looks like a valid osprey source directory
        pyproject_path = osprey_source_root / "pyproject.toml"
        if not pyproject_path.exists():
            raise DevModeUnavailableError(
                f"no pyproject.toml found at the osprey source root ({osprey_source_root})",
                "--dev needs a complete source checkout to build a wheel from.\n"
                "Reinstall editable from a full checkout: "
                "uv pip install -e <path-to-osprey-checkout>",
            )

        cached_wheel = _build_dev_wheel_cached(osprey_source_root)

        # The framework wheel requires its workspace members, which pip
        # resolves from PyPI only once a satisfying release exists there. Build
        # each from the same checkout and stage it beside the framework wheel:
        # the Dockerfiles' wheel layer installs every staged wheel in one pip
        # call, so the requirements are satisfied locally.
        cached_wheels = [cached_wheel]
        member_wheels: dict[str, tuple[Path, Path]] = {}
        for member in WORKSPACE_MEMBERS:
            member_root = osprey_source_root / "packages" / member
            if not (member_root / "pyproject.toml").exists():
                raise DevModeUnavailableError(
                    f"the {member} workspace member is missing from the checkout ({member_root})",
                    "--dev stages every workspace member wheel beside the "
                    "framework wheel because the framework requires them.\n"
                    f"Restore packages/{member} in the checkout.",
                )
            member_wheel = _build_dev_wheel_cached(member_root)
            member_wheels[member] = (member_root, member_wheel)
            cached_wheels.append(member_wheel)
        _require_members_satisfy_framework(cached_wheel, member_wheels)

        # Copy the cached wheels to this call's output directory
        dest_wheels = []
        for wheel in cached_wheels:
            dest = os.path.join(out_dir, wheel.name)
            shutil.copy2(wheel, dest)
            dest_wheels.append(dest)

        # Stage the wheels' own base dependency list next to them so the
        # Dockerfiles' toolchain-equipped deps layer can install any deps the
        # released PyPI pin lacks. A context holding wheels without their
        # manifest is half-staged — the OSPREY_DEV success signal must not
        # fire for it, so a manifest failure removes the wheels and fails
        # staging outright (fail-closed, matching the wheel-build path).
        try:
            _write_local_requirements_manifest(cached_wheels, out_dir)
        except Exception as manifest_error:
            for dest in dest_wheels:
                with contextlib.suppress(OSError):
                    os.remove(dest)
            raise DevModeUnavailableError(
                f"could not write {LOCAL_REQUIREMENTS_FILENAME} beside the staged "
                f"wheels ({manifest_error})",
                "A context holding a wheel without its requirements manifest is "
                "half-staged and would build against the released pin's "
                "dependencies.\nCheck that the build context directory is writable.",
            ) from manifest_error

        # Fires once per build context -- the deployment, each render, each
        # image copy -- so as a step it would repeat one fact N times. The step
        # on the build above states it once.
        logger.debug(f"Copied osprey wheels: {', '.join(w.name for w in cached_wheels)}")

        return True

    except DevModeUnavailableError:
        # The whole point of this error: --dev could not be honored, so the
        # deploy must stop rather than quietly fall back to the PyPI release.
        # Must be re-raised ahead of the broad handlers below.
        raise
    except ImportError as e:
        raise DevModeUnavailableError(
            "the osprey package could not be imported from the local environment",
            "Reinstall osprey: uv pip install -e <path-to-osprey-checkout>",
        ) from e
    except Exception as e:
        # Any other staging failure is the same hazard wearing a different
        # exception type: continuing would deploy released osprey under --dev.
        raise DevModeUnavailableError(
            f"staging the local osprey wheel failed: {e}",
            "Re-run once the error above is resolved.",
        ) from e


# ── Deployment packages ──────────────────────────────────────────────────────
# A deployment's own Python packages — each `apps/<name>/` holding a
# `pyproject.toml`, and each local-path entry of the profile's `dependencies`
# (osprey.apps.rules.deployment_packages) — are built to wheels by `osprey build`
# into one directory of the render, which every image context carries. The
# build venv installs each wheel by path; the recorded dependency list and the
# rendered Dockerfile name each by its distribution, bound to its wheel (the record
# through a relative `[tool.uv.sources]` path, the image's deps layer through a
# direct reference into that directory), so no index can serve the name. All
# install on the framework's own resolve, so the framework's `==` requirements
# bind what a package may pull in.

#: The directory, in the render and at the root of every image context, holding
#: the built wheels and their manifest. A subdirectory, never loose wheels at the
#: context root: the Dockerfile's wheel layer reinstalls every root ``*.whl`` as
#: osprey itself.
DEPLOYMENT_PACKAGES_DIRNAME = "osprey-deployment-packages"

#: The manifest listing the built wheels, beside them.
DEPLOYMENT_PACKAGES_MANIFEST = "manifest.json"

#: The entry-point group a measurement kind is registered under.
FACILITY_KINDS_GROUP = "osprey.facility.kinds"

#: The framework's own distribution name.
FRAMEWORK_DISTRIBUTION = "osprey-framework"

# A wheel build that fails exits nonzero; this cap bounds only one that hangs.
_WHEEL_BUILD_TIMEOUT_S = 600


@dataclass(frozen=True)
class DeploymentPackage:
    """One built deployment package.

    Attributes:
        name: The distribution's canonical name.
        version: Its version, as the built wheel spells it.
        wheel: The built wheel.
        replaces: The profile ``dependencies`` entries this wheel stands for, every
            spelling of its path; empty for an application package no entry names.
    """

    name: str
    version: str
    wheel: Path
    replaces: tuple[str, ...] = ()

    @property
    def requirement(self) -> str:
        """The spelling the recorded list and the Dockerfile carry: name and version."""
        return f"{self.name}=={self.version}"

    @property
    def install_requirement(self) -> str:
        """The spelling the build venv installs: the name, bound to the wheel."""
        return f"{self.name} @ {self.wheel.resolve().as_uri()}"


def build_deployment_packages(
    root: Path, dependencies: Sequence[str], dest: Path
) -> tuple[DeploymentPackage, ...]:
    """Build every deployment package of *root* into *dest*, with its manifest.

    Args:
        root: The deployment root.
        dependencies: The profile's ``dependencies`` entries.
        dest: The deployment-packages directory; replaced whole, and not created
            when the deployment has no package.

    Returns:
        The packages, in :func:`osprey.apps.rules.deployment_packages` order.

    Raises:
        BuildProfileError: A local-path entry holds no package to build, or a
            package's wheel build failed.
    """
    from osprey.apps.rules import deployment_packages, local_path
    from osprey_connectors.errors import BuildProfileError

    # Each local-path entry stands for the package at its path, an application
    # package included, so the entry is spelled as that package in place and no
    # distribution is named twice. An entry with no package to build could not
    # be substituted: verbatim, it would reach the Dockerfile and break the image.
    replaced: dict[Path, list[str]] = {}
    unbuildable: list[str] = []
    for raw in dependencies:
        directory = local_path(raw, root)
        if directory is None:
            continue
        pyproject = (directory / "pyproject.toml").resolve()
        if pyproject.is_file():
            replaced.setdefault(pyproject, []).append(raw)
        else:
            unbuildable.append(raw)
    if unbuildable:
        raise BuildProfileError(
            f"dependencies {', '.join(map(repr, unbuildable))}: a local-path entry must be a "
            "directory holding a pyproject.toml, which osprey build turns into a wheel for "
            "the project image.\n"
            "        Point it at the package's directory, or name an index requirement."
        )

    sources = deployment_packages(root, list(dependencies))
    if dest.exists():
        shutil.rmtree(dest)
    if not sources:
        return ()
    dest.mkdir(parents=True)

    from packaging.utils import parse_wheel_filename

    packages: list[DeploymentPackage] = []
    for pyproject in sources:
        wheel = _build_wheel(pyproject.parent, dest)
        name, version, _build, _tags = parse_wheel_filename(wheel.name)
        packages.append(
            DeploymentPackage(
                name=str(name),
                version=str(version),
                wheel=wheel,
                replaces=tuple(replaced.get(pyproject, ())),
            )
        )
    _write_deployment_packages_manifest(packages, dest)
    report_step(f"built {len(packages)} deployment package wheel(s)")
    return tuple(packages)


def _build_wheel(source: Path, out_dir: Path) -> Path:
    """Build *source*'s wheel into *out_dir*; return it.

    uv when it is on hand (as the build venv's installer is), else pip. The build
    sets ``SOURCE_DATE_EPOCH`` unless the operator has, so an unchanged source gives
    a byte-identical wheel and leaves the image's COPY and deps layers cached.
    """
    import sys
    import tempfile

    from osprey_connectors.errors import BuildProfileError

    uv = os.environ.get("UV") or shutil.which("uv")
    env = dict(os.environ)
    env.setdefault("SOURCE_DATE_EPOCH", "315532800")  # 1980-01-01, the earliest zip time
    with tempfile.TemporaryDirectory() as scratch:
        if uv:
            cmd = [uv, "build", "--wheel", "--quiet", "--out-dir", scratch, str(source)]
        else:
            cmd = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--quiet"]
            cmd += ["--wheel-dir", scratch, str(source)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_WHEEL_BUILD_TIMEOUT_S, env=env
            )
        except subprocess.TimeoutExpired as e:
            raise BuildProfileError(
                f"building the wheel of the deployment package {_shown(source)} did not "
                f"finish within {_WHEEL_BUILD_TIMEOUT_S // 60} minutes"
            ) from e
        wheels = sorted(Path(scratch).glob("*.whl"))
        if result.returncode != 0 or len(wheels) != 1:
            output = (result.stdout + result.stderr).strip()
            raise BuildProfileError(
                f"building the wheel of the deployment package {_shown(source)} failed:\n"
                f"{output or 'the build produced no single wheel'}"
            )
        built = out_dir / wheels[0].name
        shutil.copy2(wheels[0], built)
    return built


def _shown(source: Path) -> str:
    """*source* as an author names it: ``apps/<name>`` for an application."""
    parts = source.parts
    if len(parts) >= 2 and parts[-2] == "apps":
        return f"apps/{parts[-1]}"
    return str(source)


def _write_deployment_packages_manifest(packages: Sequence[DeploymentPackage], dest: Path) -> None:
    """Write the manifest: each package's name, version and wheel file, in order."""
    import json

    manifest = {
        "packages": [
            {"name": p.name, "version": p.version, "wheel": p.wheel.name} for p in packages
        ]
    }
    (dest / DEPLOYMENT_PACKAGES_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def refuse_unshipped_kinds(requirements: Iterable[str]) -> None:
    """Refuse a registered measurement kind the project image would not install.

    Every :data:`FACILITY_KINDS_GROUP` entry point is mapped to the distribution
    declaring it. That distribution has to be one the image installs: the
    framework itself, or a distribution *requirements* names (an index
    requirement, or a built deployment package by its name).

    Args:
        requirements: The dependency list the image installs, beside the framework.

    Raises:
        BuildProfileError: Naming each kind whose package is not installed.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    from osprey_connectors.errors import BuildProfileError

    shipped = {canonicalize_name(FRAMEWORK_DISTRIBUTION)}
    for raw in requirements:
        try:
            shipped.add(canonicalize_name(Requirement(raw).name))
        except InvalidRequirement:
            continue
    refused = []
    for kind in entry_points(group=FACILITY_KINDS_GROUP):
        dist = getattr(kind, "dist", None)
        dist_name = getattr(dist, "name", None)
        if dist_name is None or canonicalize_name(dist_name) not in shipped:
            refused.append(f"{kind.name!r} (package {dist_name or 'unknown'})")
    if refused:
        raise BuildProfileError(
            f"measurement kind {', '.join(refused)}: the kind's package is not among the "
            "packages the project image installs.\n"
            "        Add the package to the profile's dependencies, as an index requirement "
            "or a local path."
        )
