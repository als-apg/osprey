"""Dev-wheel artifact helpers: build, per-process cache, staging copy, and manifest.

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
"""

import atexit
import os
import shutil
from pathlib import Path

from osprey.utils.logger import get_logger

logger = get_logger("deployment.compose")

# Staged next to every dev wheel (see ``_copy_local_framework_for_override``
# below): the wheel's own base dependency list, which the Dockerfiles' deps
# layer installs with the build toolchain still present. Without it the deps
# layer primes only the RELEASED PyPI pin's dependencies, and any base dep the
# local wheel adds that the release lacks (e.g. a native sdist like softioc)
# would have to compile in the later toolchain-less wheel-install layer — and
# fail.
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


def _write_local_requirements_manifest(cached_wheel: Path, out_dir: str) -> None:
    """Write ``osprey-local-requirements.txt`` next to the staged dev wheel.

    Content contract (shared with the service Dockerfiles, which COPY the file
    and pip-install it in their toolchain-equipped deps layer): one requirement
    per line, sorted, trailing newline — byte-identical for identical wheels.

    :param cached_wheel: The cached local wheel the manifest derives from
    :type cached_wheel: pathlib.Path
    :param out_dir: Build context directory the wheel was staged into
    :type out_dir: str
    """
    manifest_path = os.path.join(out_dir, LOCAL_REQUIREMENTS_FILENAME)
    content = "".join(f"{line}\n" for line in _wheel_base_requirements(cached_wheel))
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


def _staged_dev_artifact_paths(context_dir: str) -> set[Path]:
    """All dev-staging artifacts (wheels + requirements manifest) in a context.

    Callers that stage into a long-lived directory (project-image and persona
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
# persona project), and `rebuild_deployment` even renders the compose files
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
    # Build the wheel package from local source
    logger.info("Building osprey wheel from local source...")
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
            # Check for missing 'build' package
            if "No module named build" in result.stderr:
                logger.warning(
                    "The 'build' package is required for --dev mode. Install with: "
                    r"uv pip install build or pip install build"
                )
            else:
                logger.warning(f"Failed to build osprey wheel: {result.stderr}")
        else:
            # Find the built wheel and move it to the memo-owned cache dir
            # (tmpdir is deleted on exit; the cache must outlive it so later
            # staging calls can copy the same build instead of rebuilding).
            wheel_files = list(Path(tmpdir).glob("*.whl"))
            if not wheel_files:
                logger.warning("No wheel file found after build")
            else:
                if _wheel_cache_dir is None:
                    _wheel_cache_dir = tempfile.mkdtemp(prefix="osprey-dev-wheel-cache-")
                    if not _wheel_cache_cleanup_registered:
                        atexit.register(_reset_wheel_build_cache)
                        _wheel_cache_cleanup_registered = True
                cached_wheel = Path(_wheel_cache_dir) / wheel_files[0].name
                shutil.copy2(wheel_files[0], cached_wheel)
                logger.success(f"Built osprey wheel: {wheel_files[0].name}")

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
            logger.warning(
                "Osprey is installed from PyPI, not in editable mode. "
                "The --dev flag requires an editable install to build a local wheel. "
                "To use --dev, reinstall osprey with: uv pip install -e <path> or pip install -e <path>"
            )
            return False

        # Get the osprey source root (go up from src/osprey to root)
        osprey_source_root = osprey_module_path.parent.parent

        # Verify this looks like a valid osprey source directory
        pyproject_path = osprey_source_root / "pyproject.toml"
        if not pyproject_path.exists():
            logger.warning(
                f"No pyproject.toml found at {osprey_source_root}, cannot build wheel from source"
            )
            return False

        cached_wheel = _build_dev_wheel_cached(osprey_source_root)
        if cached_wheel is None:
            return False

        # Copy the cached wheel to this call's output directory
        dest_wheel = os.path.join(out_dir, cached_wheel.name)
        shutil.copy2(cached_wheel, dest_wheel)

        # Stage the wheel's own base dependency list next to it so the
        # Dockerfiles' toolchain-equipped deps layer can install any deps the
        # released PyPI pin lacks. A context holding a wheel without its
        # manifest is half-staged — the OSPREY_DEV success signal must not
        # fire for it, so a manifest failure removes the wheel and fails
        # staging outright (fail-closed, matching the wheel-build path).
        try:
            _write_local_requirements_manifest(cached_wheel, out_dir)
        except Exception as manifest_error:
            try:
                os.remove(dest_wheel)
            except OSError:
                pass
            logger.warning(
                f"Failed to write {LOCAL_REQUIREMENTS_FILENAME} for dev override: {manifest_error}"
            )
            return False

        logger.success(f"Copied osprey wheel: {cached_wheel.name}")

        return True

    except ImportError:
        logger.warning("Osprey not found in local environment, containers will use PyPI version")
        return False
    except Exception as e:
        logger.warning(f"Failed to build osprey wheel for dev override: {e}")
        return False
