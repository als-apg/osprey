"""Dev-wheel artifact helpers: build, per-process cache, staging copy, and manifest.

A ``--dev`` deploy installs a locally-built osprey wheel into container build
contexts instead of the released PyPI pin. This module owns that wheel's whole
lifecycle: building it at most once per process
(:func:`_build_dev_wheel_cached`), caching the build across the many contexts a
single deploy stages (:data:`_wheel_build_cache`), copying it into a context
alongside its base-requirements manifest, the checkout's ``uv.lock`` exported
as constraints, and — for an image that installs an extra — that extra's
requirements (:func:`_copy_local_framework_for_override`,
:func:`_write_local_requirements_manifest`,
:func:`_write_extra_requirements_manifest`), and enumerating what staging added
so callers can clean up exactly that (:func:`_staged_dev_artifact_paths`).

The per-context *decision* to stage — gated on ``--dev`` and the presence of a
``Dockerfile`` — stays in :mod:`osprey.deployment.compose_generator`
(``_stage_dev_wheel_for_context``), where it is wired into the compose render
flow; that module re-exports the copy and cache-reset helpers below so existing
import and monkeypatch call sites keep resolving against it.
"""

import atexit
import contextlib
import os
import shutil
from pathlib import Path

from osprey.cli.phase_reporter import report_step
from osprey.deployment.errors import DevModeUnavailableError
from osprey.deployment.lock_constraints import (
    LOCK_CONSTRAINTS_FILENAME,
    LockExportError,
    export_lock_constraints,
    find_uv,
    reset_export_cache,
)
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

    # The images a --dev build makes install third-party packages at the
    # versions the checkout's uv.lock holds (see osprey.deployment
    # .lock_constraints). Whether the lock is current is a subprocess away and
    # is checked at staging; that there is one, and a uv to read it, is not.
    if not (source_root / "uv.lock").exists():
        raise DevModeUnavailableError(
            f"no uv.lock found at the osprey source root ({source_root})",
            "--dev installs third-party packages at the versions the checkout's "
            "uv.lock holds.\nCreate the lock with:  uv lock",
        )
    if find_uv() is None:
        raise DevModeUnavailableError(
            "uv is not installed, and --dev exports the checkout's uv.lock with it",
            "Install uv (https://docs.astral.sh/uv/), or set UV to its path.",
        )


# Staged next to every dev wheel (see ``_copy_local_framework_for_override``
# below): the wheel's own base dependency list, which the Dockerfiles' deps
# layer installs with the build toolchain still present. Without it the deps
# layer primes only the RELEASED PyPI pin's dependencies, and any base dep the
# local wheel adds that the release lacks (e.g. a native sdist like
# accelerator-toolbox) would have to compile in the later toolchain-less
# wheel-install layer — and fail.
LOCAL_REQUIREMENTS_FILENAME = "osprey-local-requirements.txt"

# Staged beside the wheels only for an image that installs one of the
# framework's extras (``_copy_local_framework_for_override``'s ``extras``): that
# extra's third-party requirements, which the image's extras layer installs
# under the lock constraints. The base manifest above leaves them out on
# purpose — it is the same file in every context, so the deps layer stays one
# layer every service image shares — and on the lock-driven path no layer may
# fetch the framework from the index to learn them.
EXTRA_REQUIREMENTS_FILENAME = "osprey-extra-requirements.txt"


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


def _strip_extra_clause(marker: str, extra: str) -> str | None:
    """*marker* with its ``extra == "<extra>"`` term removed, or ``None`` if it was the whole marker.

    The two shapes a wheel's metadata spells an extra-gated requirement in
    (``extra == "x"`` alone, or ``(<marker>) and extra == "x"``, which is how
    hatchling writes one) are the only ones taken apart. Any other shape is
    refused rather than guessed at: a wrongly unwrapped marker would install a
    requirement on a platform its extra excludes, or drop one it needs.

    :raises ValueError: If the marker names *extra* in any other shape
    """
    import re

    clause = f'extra == "{extra}"'
    if marker == clause:
        return None
    match = re.fullmatch(r"\((?P<rest>.+)\) and " + re.escape(clause), marker)
    if match and "extra" not in match.group("rest"):
        return match.group("rest")
    raise ValueError(f"cannot separate extra {extra!r} from the marker {marker!r}")


def _wheel_extra_requirements(wheel_path: str | Path, extras: tuple[str, ...]) -> list[str]:
    """The third-party requirements *extras* add to the wheel's base set, sorted.

    Each ``Requires-Dist`` gated on one of *extras* is returned with that gate
    removed and any other marker kept (pip and uv evaluate it in-container);
    requirements on workspace distributions are dropped, as they are from the
    base manifest. Deterministic for identical wheels, for the same reason
    :func:`_wheel_base_requirements` is.

    :raises ValueError: If the wheel has no metadata, declares none of
        *extras*, or spells a gate this cannot take apart
    """
    import zipfile
    from email.parser import Parser

    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    with zipfile.ZipFile(wheel_path) as whl:
        metadata_names = sorted(
            name for name in whl.namelist() if name.endswith(".dist-info/METADATA")
        )
        if not metadata_names:
            raise ValueError(f"no *.dist-info/METADATA found in wheel {wheel_path}")
        metadata = Parser().parsestr(whl.read(metadata_names[0]).decode("utf-8"))

    declared = {canonicalize_name(e) for e in metadata.get_all("Provides-Extra") or []}
    requirements: set[str] = set()
    for extra in extras:
        if canonicalize_name(extra) not in declared:
            raise ValueError(f"wheel {Path(wheel_path).name} declares no extra {extra!r}")
        for entry in metadata.get_all("Requires-Dist") or []:
            req = Requirement(entry)
            if req.marker is None or f'extra == "{extra}"' not in str(req.marker):
                continue
            rest = _strip_extra_clause(str(req.marker), extra)
            req.marker = Marker(rest) if rest is not None else None
            line = str(req)
            if not _is_workspace_requirement(line):
                requirements.add(line)
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


def _write_extra_requirements_manifest(
    primary_wheel: Path, extras: tuple[str, ...], out_dir: str
) -> None:
    """Write ``osprey-extra-requirements.txt``: *extras*' third-party requirements.

    Same content contract as the base manifest (one requirement per line,
    sorted, trailing newline). See :data:`EXTRA_REQUIREMENTS_FILENAME`.
    """
    lines = _wheel_extra_requirements(primary_wheel, extras)
    with open(
        os.path.join(out_dir, EXTRA_REQUIREMENTS_FILENAME), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write("".join(f"{line}\n" for line in lines))


def _write_lock_constraints(osprey_source_root: Path, out_dir: str) -> None:
    """Write the checkout's ``uv.lock``, exported as constraints, into *out_dir*.

    The export runs once per process (see
    :func:`osprey.deployment.lock_constraints.export_lock_constraints`), so
    every context a deploy stages receives the same bytes — which is what keeps
    the shared deps layer shared.
    """
    content = export_lock_constraints(osprey_source_root)
    with open(
        os.path.join(out_dir, LOCK_CONSTRAINTS_FILENAME), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write(content)


def _staged_dev_artifact_paths(context_dir: str) -> set[Path]:
    """All dev-staging artifacts (wheels, manifests, lock constraints) in a context.

    Callers that stage into a long-lived directory (project-image and render
    builds) snapshot this before/after staging so their ``finally`` cleanup
    removes exactly what staging added — wheels, manifests AND the lock
    constraints — and nothing that was already there. The constraints file
    matters most: its presence alone switches a Dockerfile onto the
    lock-driven install, so one left behind would turn a later release build
    into a lock-pinned one.

    :param context_dir: Build context directory
    :type context_dir: str
    :return: Paths of staged dev artifacts currently present
    :rtype: set[pathlib.Path]
    """
    root = Path(context_dir)
    artifacts = set(root.glob("*.whl"))
    for name in (
        LOCAL_REQUIREMENTS_FILENAME,
        EXTRA_REQUIREMENTS_FILENAME,
        LOCK_CONSTRAINTS_FILENAME,
    ):
        staged = root / name
        if staged.exists():
            artifacts.add(staged)
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
    reset_export_cache()
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


def _members_to_stage(primary_wheel: Path, member: str | None) -> tuple[str, ...]:
    """The workspace members whose wheels are staged beside *primary_wheel*.

    Every member for the framework wheel (*member* ``None``). For a member's
    own image (*member* a ``packages/<name>`` path), only the members that
    wheel's base requirements name, so the image is never handed a wheel that
    needs the framework it does not install.
    """
    if member is None:
        return WORKSPACE_MEMBERS
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    required = {
        canonicalize_name(Requirement(line).name)
        for line in _wheel_base_requirements(primary_wheel)
    }
    own = canonicalize_name(Path(member).name)
    return tuple(
        name
        for name in WORKSPACE_MEMBERS
        if canonicalize_name(name) in required and canonicalize_name(name) != own
    )


def _copy_local_framework_for_override(out_dir, member=None, extras=()):
    """Build and copy local osprey wheel to container build directory for development mode.

    ``member`` names a workspace package (e.g. ``packages/osprey-optimizer``)
    whose wheel the context installs in place of the framework's: a service
    image built on that package alone is handed that wheel and the connectors
    wheel it requires, and never the framework's.

    ``extras`` names the primary wheel's extras the context's image installs
    in an extras layer of its own (the gchat bridge's ``gchat``, the virtual
    accelerator's ``virtual-accelerator``); their third-party requirements are
    staged as ``osprey-extra-requirements.txt``.

    Every context also receives ``osprey-lock-constraints.txt``, the
    checkout's ``uv.lock`` exported as constraints (see
    :mod:`osprey.deployment.lock_constraints`): its presence is what puts the
    Dockerfile on the lock-driven install, where every third-party package is
    installed at the lock's version and nothing is fetched as osprey-framework
    from the index.

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
    next to it. Success (True) means everything landed; a failure to write any
    of the files beside the wheel removes what was staged and reports failure,
    so the ``OSPREY_DEV`` gating never fires for a half-staged context.

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

        primary_root = osprey_source_root / member if member else osprey_source_root
        if member and not (primary_root / "pyproject.toml").exists():
            raise DevModeUnavailableError(
                f"the {member} workspace member is missing from the checkout ({primary_root})",
                f"--dev builds the {member} wheel for the service image that installs it.\n"
                f"Restore {member} in the checkout.",
            )
        cached_wheel = _build_dev_wheel_cached(primary_root)

        # The primary wheel requires workspace members, which pip resolves
        # from PyPI only once a satisfying release exists there. Build each
        # from the same checkout and stage it beside the primary wheel: the
        # Dockerfiles' wheel layer installs every staged wheel in one pip call,
        # so the requirements are satisfied locally. The framework stages every
        # member; a member's own image stages only the members its wheel
        # requires (the optimizer: the connectors), never the framework's.
        cached_wheels = [cached_wheel]
        member_wheels: dict[str, tuple[Path, Path]] = {}
        for staged in _members_to_stage(cached_wheel, member):
            member_root = osprey_source_root / "packages" / staged
            if not (member_root / "pyproject.toml").exists():
                raise DevModeUnavailableError(
                    f"the {staged} workspace member is missing from the checkout ({member_root})",
                    "--dev stages every workspace member wheel beside the "
                    "wheel that requires it.\n"
                    f"Restore packages/{staged} in the checkout.",
                )
            member_wheel = _build_dev_wheel_cached(member_root)
            member_wheels[staged] = (member_root, member_wheel)
            cached_wheels.append(member_wheel)
        _require_members_satisfy_framework(cached_wheel, member_wheels)

        # Copy the cached wheels to this call's output directory
        dest_wheels = []
        for wheel in cached_wheels:
            dest = os.path.join(out_dir, wheel.name)
            shutil.copy2(wheel, dest)
            dest_wheels.append(dest)

        # Stage the wheels' own base dependency list next to them so the
        # Dockerfiles' toolchain-equipped deps layer can install the wheels'
        # dependencies, then the lock constraints that pin every one of them
        # to the checkout's uv.lock, then the extras' requirements for an
        # image that installs an extra. A context holding wheels without all
        # of these is half-staged — the OSPREY_DEV success signal must not
        # fire for it, so any failure removes what was staged and fails
        # staging outright (fail-closed, matching the wheel-build path).
        staged_beside = [
            os.path.join(out_dir, name)
            for name in (
                LOCAL_REQUIREMENTS_FILENAME,
                LOCK_CONSTRAINTS_FILENAME,
                EXTRA_REQUIREMENTS_FILENAME,
            )
        ]

        def _unstage() -> None:
            for dest in [*dest_wheels, *staged_beside]:
                with contextlib.suppress(OSError):
                    os.remove(dest)

        try:
            _write_local_requirements_manifest(cached_wheels, out_dir)
        except Exception as manifest_error:
            _unstage()
            raise DevModeUnavailableError(
                f"could not write {LOCAL_REQUIREMENTS_FILENAME} beside the staged "
                f"wheels ({manifest_error})",
                "A context holding a wheel without its requirements manifest is "
                "half-staged and would build against the released pin's "
                "dependencies.\nCheck that the build context directory is writable.",
            ) from manifest_error
        try:
            _write_lock_constraints(osprey_source_root, out_dir)
        except LockExportError as lock_error:
            _unstage()
            raise DevModeUnavailableError(lock_error.reason, lock_error.remedy) from lock_error
        except OSError as lock_error:
            _unstage()
            raise DevModeUnavailableError(
                f"could not write {LOCK_CONSTRAINTS_FILENAME} beside the staged "
                f"wheels ({lock_error})",
                "Check that the build context directory is writable.",
            ) from lock_error
        if extras:
            try:
                _write_extra_requirements_manifest(cached_wheel, tuple(extras), out_dir)
            except Exception as extras_error:
                _unstage()
                raise DevModeUnavailableError(
                    f"could not write {EXTRA_REQUIREMENTS_FILENAME} for the "
                    f"{', '.join(extras)} extra beside the staged wheels ({extras_error})",
                    "The image installs that extra's requirements from this file; "
                    "without it the extras layer has nothing to install.",
                ) from extras_error

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
