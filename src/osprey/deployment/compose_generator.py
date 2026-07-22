"""Compose file generation and build directory setup.

Handles Jinja2 template rendering, service discovery, build directory
management, and Docker Compose file creation for container deployments.
"""

import atexit
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.deployment.facility_config import normalize_facility_config
from osprey.deployment.runtime_helper import get_runtime_command, runtime_env
from osprey.deployment.wheel_build import (
    LOCAL_REQUIREMENTS_FILENAME,
    _write_local_requirements_manifest,
)
from osprey.utils.config import ConfigBuilder
from osprey.utils.log_filter import quiet_logger
from osprey.utils.logger import get_logger

logger = get_logger("deployment.compose")

SERVICES_DIR = "services"
SRC_DIR = "src"
OUT_SRC_DIR = "repo_src"

TEMPLATE_FILENAME = "docker-compose.yml.j2"
COMPOSE_FILE_NAME = "docker-compose.yml"


def find_service_config(config, service_name):
    """Locate service configuration and template path for deployment.

    Services are declared in the project config's top-level ``services:`` block,
    keyed by service name — the same short name that appears in
    ``deployed_services``. This function looks the name up in that block and
    returns both the service configuration object and the path to its Docker
    Compose template, so the caller can access service settings and initiate
    template rendering.

    :param config: Configuration containing service definitions
    :type config: dict
    :param service_name: Service identifier as it appears under ``services:``
        (and in ``deployed_services``)
    :type service_name: str
    :return: Tuple containing service configuration and template path, or
        ``(None, None)`` if the name is not declared under ``services:``
    :rtype: tuple[dict, str] or tuple[None, None]

    Examples:
        Service discovery::

            >>> config = {'services': {'openobserve': {'path': 'services/openobserve'}}}
            >>> service_config, template_path = find_service_config(config, 'openobserve')
            >>> print(template_path)  # 'services/openobserve/docker-compose.yml.j2'

    .. seealso::
       :func:`setup_build_dir` : Processes discovered services for deployment
    """
    services = config.get("services", {})
    service_config = services.get(service_name)
    if service_config:
        return service_config, os.path.join(service_config["path"], TEMPLATE_FILENAME)

    return None, None


def _normalize_compose_name(name):
    """Normalize a string into a valid docker-compose project name.

    Mirrors docker-compose's own normalization so the value OSPREY pins as
    ``COMPOSE_PROJECT_NAME`` matches what compose would derive on its own:
    lowercase, keep only ``[a-z0-9_-]`` (dropping everything else, including
    whitespace), then strip leading AND trailing ``_``/``-`` so the result
    begins and ends with a letter or number. Trailing separators must go too:
    the project name is used as an image-tag prefix (``<project>-dispatch:local``),
    and a trailing separator would produce invalid docker references like
    ``proj_-dispatch:local``. An input that is already a valid lowercase name
    passes through byte-unchanged.

    :param name: Raw candidate project name
    :type name: str
    :return: Compose-valid project name (may be empty if no valid characters
        remain)
    :rtype: str
    """
    kept = re.findall(r"[a-z0-9_-]", str(name).lower())
    return "".join(kept).strip("_-")


def resolve_project_name(config):
    """Derive the project name used for labels and the ``<project>:local`` image tag.

    The name is resolved with a fixed priority order so that every consumer
    (container labels here, plus the ``<project>:local`` image tag that
    ``osprey deploy up`` builds for the dispatch worker) agrees on one value:

    1. Root-level ``project_name`` attribute (preferred, explicit)
    2. Last component of the ``project_root`` path (smart fallback)
    3. ``"unnamed-project"`` (final fallback)

    The resolved name is normalized to a valid docker-compose project name via
    :func:`_normalize_compose_name` so it matches the value compose itself would
    use for volume/network namespacing. If normalization empties the candidate
    (no valid characters), the ``"unnamed-project"`` fallback is used.

    :param config: Configuration dictionary
    :type config: dict
    :return: Resolved project name
    :rtype: str
    """
    project_name = config.get("project_name")

    if not project_name:
        # Fallback: Extract from project_root path
        project_root = config.get("project_root", "")
        if project_root:
            project_name = os.path.basename(str(project_root).rstrip("/"))

    project_name = _normalize_compose_name(project_name) if project_name else ""

    if not project_name:
        # Final fallback: Default
        project_name = "unnamed-project"

    return project_name


def resolve_user_volume_names(config, user):
    """Derive the real runtime names of a web terminal user's named volumes.

    The web terminal compose template declares each user's volumes bare
    (``<user>-claude-config``, ``<user>-agent-data``). Compose namespaces bare
    volume names with ``COMPOSE_PROJECT_NAME`` (defaulting to
    ``basename(cwd)`` when unset), which is independent of the project name
    :func:`resolve_project_name` derives for container labels. Left alone,
    those two can diverge — the labels say one project, the actual volumes
    live under another. :func:`runtime_helper.runtime_env` pins
    ``COMPOSE_PROJECT_NAME`` to :func:`resolve_project_name`'s result for
    every runtime invocation, so this function returns the same
    ``<project>_<name>`` form to give volume-targeting code (inspect/rm/
    archive) the real runtime volume names deterministically, without
    shelling out to discover them.

    :param config: Configuration dictionary (``None`` is treated as an empty
        config, resolving to the ``"unnamed-project"`` namespace — symmetric with
        :func:`runtime_helper.runtime_env`)
    :type config: dict
    :param user: Web terminal username the volumes belong to
    :type user: str
    :return: Tuple of ``(claude_config_volume, agent_data_volume)`` runtime names
    :rtype: tuple[str, str]
    """
    project = resolve_project_name(config or {})
    return f"{project}_{user}-claude-config", f"{project}_{user}-agent-data"


def _inject_project_metadata(config):
    """Add project tracking metadata for container labels.

    This function injects deployment metadata into the configuration that will
    be used as Docker labels in the rendered compose files. These labels enable
    tracking which project/agent owns each container.

    The project name is extracted via :func:`resolve_project_name` (explicit
    ``project_name`` > last component of ``project_root`` > ``unnamed-project``).

    It also defaults the dispatch worker's image to ``<project>:local`` — the
    tag ``osprey deploy up`` builds from the project ``Dockerfile`` — unless the
    profile pinned an explicit ``services.dispatch_worker.image``. The
    event-dispatcher is left untouched (it builds ``<project>-dispatch:local``
    via its own compose ``build:`` block; the worker deliberately has none).

    :param config: Configuration dictionary
    :type config: dict
    :return: Configuration with added osprey_labels section
    :rtype: dict
    """
    import datetime

    project_name = resolve_project_name(config)

    # Resolve the running framework version so service Dockerfiles can pin the
    # PyPI install (`pip install osprey-framework==<version>`) for production
    # builds. Dev builds install a locally-built wheel instead (see --dev).
    try:
        from osprey import __version__ as osprey_version
    except Exception:
        osprey_version = ""

    # Create enhanced config with label metadata
    config_with_labels = config.copy()
    config_with_labels["osprey_labels"] = {
        "project_name": project_name,
        "project_root": config.get("project_root", os.getcwd()),
        "deployed_at": datetime.datetime.now().isoformat(),
    }
    config_with_labels["osprey_version"] = osprey_version

    # Whether a project ``.env`` exists in the deploy CWD. The dispatch worker
    # mounts it read-only so ``inject_provider_env`` can read provider auth at
    # startup; gating the mount on existence avoids docker auto-creating a stray
    # empty ``.env`` directory when none is present.
    config_with_labels["osprey_env_present"] = os.path.exists(".env")

    # Default the dispatch worker's image to the project image that
    # ``osprey deploy up`` builds (``<project>:local``). The worker compose
    # template renders ``${OSPREY_WORKER_IMAGE:-{{ services.dispatch_worker.image
    # | default(osprey_labels.project_name ~ ':local') }}}`` — the template-level
    # fallback now matches this same project image, so this setdefault and the
    # template agree even when it cannot fire (e.g. a null ``dispatch_worker:``
    # key). ``setdefault`` honors a profile that pinned its own image. Copy the
    # nested dicts we touch so the shared input config is never mutated (the
    # top-level copy() above is shallow).
    services = config_with_labels.get("services")
    if isinstance(services, dict) and isinstance(services.get("dispatch_worker"), dict):
        services = dict(services)
        worker_cfg = dict(services["dispatch_worker"])
        worker_cfg.setdefault("image", f"{project_name}:local")
        services["dispatch_worker"] = worker_cfg
        config_with_labels["services"] = services

    return config_with_labels


def render_template(template_path, config, out_dir):
    """Render Jinja2 template with configuration context to output directory.

    This function processes Jinja2 templates using the configuration
    as context, generating concrete configuration files for container deployment.
    The system supports multiple template types including Docker Compose files
    and Jupyter kernel configurations, with intelligent output filename detection.

    Template rendering uses the complete configuration dictionary as Jinja2 context,
    enabling templates to access any configuration value including environment
    variables, service settings, and application-specific parameters. Environment
    variables can be referenced directly in templates using ${VAR_NAME} syntax
    for deployment-specific configurations like proxy settings. The output
    directory is created automatically if it doesn't exist.

    :param template_path: Path to the Jinja2 template file to render
    :type template_path: str
    :param config: Configuration dictionary to use as template context
    :type config: dict
    :param out_dir: Output directory for the rendered file
    :type out_dir: str
    :return: Full path to the rendered output file
    :rtype: str

    Examples:
        Docker Compose template rendering::

            >>> config = {'database': {'host': 'localhost', 'port': 5432}}
            >>> output_path = render_template(
            ...     'services/mongo/docker-compose.yml.j2',
            ...     config,
            ...     'build/services/mongo'
            ... )
            >>> print(output_path)  # 'build/services/mongo/docker-compose.yml'

        Jupyter kernel template rendering::

            >>> config = {'project_root': '/home/user/project'}
            >>> output_path = render_template(
            ...     'services/jupyter/python3-epics/kernel.json.j2',
            ...     config,
            ...     'build/services/jupyter/python3-epics'
            ... )
            >>> print(output_path)  # 'build/services/jupyter/python3-epics/kernel.json'

    .. note::
       The function automatically determines output filenames based on template
       naming conventions: .j2 extension is removed, and specific patterns
       like docker-compose.yml.j2 and kernel.json.j2 are recognized.

    .. seealso::
       :func:`setup_build_dir` : Uses this function for service template processing
       :func:`render_kernel_templates` : Batch processing of kernel templates
    """
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(template_path)

    # Inject project metadata for container labels
    config_dict = _inject_project_metadata(config)
    rendered_content = template.render(config_dict)

    # Determine output filename based on template type
    if template_path.endswith("docker-compose.yml.j2"):
        output_filename = COMPOSE_FILE_NAME
    elif template_path.endswith("kernel.json.j2"):
        output_filename = "kernel.json"
    else:
        # Generic fallback: remove .j2 extension
        output_filename = os.path.basename(template_path)[:-3]

    output_filepath = os.path.join(out_dir, output_filename)
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filepath, "w") as f:
        f.write(rendered_content)
    return output_filepath


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


def _stage_dev_wheel_for_context(out_dir, dev_mode):
    """Stage the local dev wheel into a service build context; report success.

    The boolean return is the success signal :func:`setup_build_dir` (and its
    incremental sibling) key the rendered ``OSPREY_DEV`` build arg on: only a
    context that actually received a wheel may relax the Dockerfile's pinned
    ``osprey-framework`` install (fail-closed — a failed build/staging keeps
    the fail-loud pin instead of silently installing the latest published
    release under a flag that means "run my local code").

    Only build contexts that contain a ``Dockerfile`` get the wheel: pure-image
    services (postgresql, openobserve) never install it, and a stray wheel in
    their context is dead weight that also churns the build-context hash.

    :param out_dir: The service's build context directory (already populated
        with the service's files, including any ``Dockerfile``)
    :type out_dir: str
    :param dev_mode: Whether ``--dev`` was passed
    :type dev_mode: bool
    :return: True iff a wheel was successfully staged into ``out_dir``
    :rtype: bool
    """
    if not dev_mode:
        logger.info("Production mode: Containers will install osprey from PyPI")
        return False
    if not os.path.isfile(os.path.join(out_dir, "Dockerfile")):
        logger.info(
            f"Development mode: build context {out_dir} has no Dockerfile, "
            "so no osprey wheel was staged for it"
        )
        return False
    if _copy_local_framework_for_override(out_dir):
        logger.key_info("Development mode: Osprey override prepared")
        return True
    logger.warning(
        "Development mode requested but osprey wheel staging failed; "
        "OSPREY_DEV will not be set for this build, so the pinned PyPI "
        "install applies"
    )
    return False


def render_kernel_templates(source_dir, config, out_dir):
    """Process all Jupyter kernel templates in a service directory.

    This function provides batch processing for Jupyter kernel configuration
    templates, automatically discovering all kernel.json.j2 files within a
    service directory and rendering them with the current configuration context.
    This is particularly useful for Jupyter services that provide multiple
    kernel environments with different configurations.

    The function recursively searches the source directory for kernel template
    files and processes each one, maintaining the relative directory structure
    in the output. This ensures that kernel configurations are placed in the
    correct locations for Jupyter to discover them.

    :param source_dir: Source directory to search for kernel templates
    :type source_dir: str
    :param config: Configuration dictionary for template rendering
    :type config: dict
    :param out_dir: Base output directory for rendered kernel files
    :type out_dir: str

    Examples:
        Kernel template processing for Jupyter service::

            >>> # Source structure:
            >>> # services/jupyter/
            >>> #   ├── python3-epics-readonly/kernel.json.j2
            >>> #   └── python3-epics-write/kernel.json.j2
            >>>
            >>> render_kernel_templates(
            ...     'services/jupyter',
            ...     {'project_root': '/home/user/project'},
            ...     'build/services/jupyter'
            ... )
            >>> # Output structure:
            >>> # build/services/jupyter/
            >>> #   ├── python3-epics-readonly/kernel.json
            >>> #   └── python3-epics-write/kernel.json

    .. note::
       This function is typically called automatically by setup_build_dir when
       a service configuration includes 'render_kernel_templates: true'.

    .. seealso::
       :func:`render_template` : Core template rendering used by this function
       :func:`setup_build_dir` : Calls this function for kernel template processing
    """
    kernel_templates = []

    # Look for kernel.json.j2 files in subdirectories
    for root, _dirs, files in os.walk(source_dir):
        for file in files:
            if file == "kernel.json.j2":
                template_path = os.path.relpath(os.path.join(root, file), os.getcwd())
                kernel_templates.append(template_path)

    # Render each kernel template
    for template_path in kernel_templates:
        # Calculate relative output directory
        rel_template_dir = os.path.dirname(os.path.relpath(template_path, source_dir))
        kernel_out_dir = (
            os.path.join(out_dir, rel_template_dir) if rel_template_dir != "." else out_dir
        )

        render_template(template_path, config, kernel_out_dir)
        logger.info(f"Rendered kernel template: {template_path} -> {kernel_out_dir}/kernel.json")


def _ensure_agent_data_structure(config):
    """Ensure _agent_data directory and subdirectories exist before container deployment.

    This function creates the agent data directory structure based on the configuration
    to prevent Docker/Podman mount failures when containers try to mount non-existent
    directories. It creates both the main agent_data_dir and all configured subdirectories.

    :param config: Configuration dictionary containing file_paths settings
    :type config: dict
    """
    # Get file paths configuration
    file_paths = config.get("file_paths", {})
    project_root = config.get("project_root", ".")
    agent_data_dir = file_paths.get("agent_data_dir", "_agent_data")

    # Create main agent data directory
    agent_data_path = Path(project_root) / agent_data_dir
    agent_data_path.mkdir(parents=True, exist_ok=True)

    # Create all configured subdirectories
    subdirs = [
        "executed_python_scripts_dir",
        "execution_plans_dir",
        "user_memory_dir",
        "registry_exports_dir",
        "prompts_dir",
    ]

    for subdir_key in subdirs:
        if subdir_key in file_paths:
            subdir_name = file_paths[subdir_key]
            subdir_path = agent_data_path / subdir_name
            subdir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created agent data subdirectory: {subdir_path}")

    logger.debug(f"Ensured agent data structure exists at: {agent_data_path}")


def setup_build_dir(template_path, config, container_cfg, dev_mode=False):
    """Create complete build environment for service deployment.

    This function orchestrates the complete build directory setup process for
    a service, including template rendering, source code copying, configuration
    flattening, and additional directory management. It creates a self-contained
    build environment that contains everything needed for container deployment.

    The build process follows these steps:
    1. Create clean build directory for the service
    2. Render the Docker Compose template with configuration context
    3. Copy service-specific files (excluding templates)
    4. Copy source code if requested (copy_src: true)
    5. Copy additional directories as specified
    6. Create flattened configuration file for container use
    7. Process kernel templates if specified

    Source code copying includes intelligent handling of requirements files,
    automatically copying global requirements.txt to the container source
    directory to ensure dependency management works correctly in containers.

    :param template_path: Path to the service's Docker Compose template
    :type template_path: str
    :param config: Complete configuration dictionary for template rendering
    :type config: dict
    :param container_cfg: Service-specific configuration settings
    :type container_cfg: dict
    :param dev_mode: Development mode - copy local framework to containers
    :type dev_mode: bool
    :return: Path to the rendered Docker Compose file
    :rtype: str

    Examples:
        Basic service build directory setup::

            >>> container_cfg = {
            ...     'copy_src': True,
            ...     'additional_dirs': ['docs', 'scripts'],
            ...     'render_kernel_templates': False
            ... }
            >>> compose_path = setup_build_dir(
            ...     'services/osprey/jupyter/docker-compose.yml.j2',
            ...     config,
            ...     container_cfg
            ... )
            >>> print(compose_path)  # 'build/services/osprey/jupyter/docker-compose.yml'

        Advanced service with custom directory mapping::

            >>> container_cfg = {
            ...     'copy_src': True,
            ...     'additional_dirs': [
            ...         'docs',  # Simple directory copy
            ...         {'src': 'external_data', 'dst': 'data'}  # Custom mapping
            ...     ],
            ...     'render_kernel_templates': True
            ... }
            >>> compose_path = setup_build_dir(template_path, config, container_cfg)

    .. note::
       The function automatically handles build directory cleanup, removing
       existing directories to ensure clean builds. Global requirements.txt
       is automatically copied to container source directories when present.

    .. warning::
       This function performs destructive operations on build directories.
       Ensure build_dir is properly configured to avoid data loss.

    .. seealso::
       :func:`render_template` : Template rendering used by this function
       :func:`render_kernel_templates` : Kernel template processing
       :class:`configs.config.ConfigBuilder` : Configuration flattening
    """
    # Create the build directory for this service
    source_dir = os.path.relpath(os.path.dirname(template_path), os.getcwd())

    # Extract service name from the path for container path resolution
    # e.g., "services/jupyter" -> "jupyter", "src/osprey/templates/services/pipelines" -> "pipelines"
    os.path.basename(source_dir)

    # Clear the directory if it exists
    build_dir = config.get("build_dir", "./build")
    out_dir = os.path.join(build_dir, source_dir)
    if os.path.exists(out_dir):
        try:
            shutil.rmtree(out_dir)
        except OSError as e:
            if (
                "Device or resource busy" in str(e) or "nfs" in str(e).lower() or e.errno == 39
            ):  # Directory not empty
                logger.warning(f"Directory in use, attempting incremental update for {out_dir}")
                import time

                time.sleep(1)
                try:
                    shutil.rmtree(out_dir)
                except OSError:
                    logger.warning(f"Could not remove {out_dir}, using incremental update approach")
                    # Use incremental update instead of full rebuild
                    return _incremental_setup_build_dir(
                        template_path, config, container_cfg, out_dir, dev_mode
                    )
            else:
                raise
    os.makedirs(out_dir, exist_ok=True)

    # Copy the service's own files (except templates) and stage the dev wheel
    # BEFORE rendering the compose template: the render context's dev_mode flag
    # is what service templates turn into the OSPREY_DEV=1 build arg, and that
    # pin-relaxing arg must only be emitted for a context that actually
    # received a wheel (see _stage_dev_wheel_for_context).
    wheel_staged = False
    if source_dir != SERVICES_DIR:  # ignore the top level dir
        # Deep copy everything in source directory except templates
        for file in os.listdir(source_dir):
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(out_dir, file)
            # Skip template files (both docker-compose and kernel templates)
            if file != TEMPLATE_FILENAME and not file.endswith(".j2"):
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

        # Copy local osprey for development override (only in dev mode).
        # This will override the PyPI osprey after standard installation.
        wheel_staged = _stage_dev_wheel_for_context(out_dir, dev_mode)

    # Create the docker compose file from the template. The render context
    # carries dev_mode so service templates can emit dev-only build args
    # (e.g. OSPREY_DEV) exactly when `osprey deploy up --dev` runs AND the
    # local wheel actually landed in this context — never when staging failed
    # (fail-closed: the Dockerfile then keeps its fail-loud pinned install).
    render_config = {**config, "dev_mode": dev_mode and wheel_staged}
    compose_filepath = render_template(template_path, render_config, out_dir)

    if source_dir != SERVICES_DIR:  # ignore the top level dir
        # Copy the source directory
        if container_cfg.get("copy_src", False):
            shutil.copytree(SRC_DIR, os.path.join(out_dir, OUT_SRC_DIR))

            # Copy global requirements.txt to repo_src if it exists
            # This handles consolidated requirements files
            global_requirements = "requirements.txt"
            if os.path.exists(global_requirements):
                repo_src_requirements = os.path.join(out_dir, OUT_SRC_DIR, "requirements.txt")
                shutil.copy2(global_requirements, repo_src_requirements)
                logger.debug(f"Copied global requirements.txt to {repo_src_requirements}")

            # Copy project's pyproject.toml to repo_src
            # Note: This is the user's project pyproject.toml, not framework's
            global_pyproject = "pyproject.toml"
            if os.path.exists(global_pyproject):
                repo_src_pyproject = os.path.join(out_dir, OUT_SRC_DIR, "pyproject_user.toml")
                shutil.copy2(global_pyproject, repo_src_pyproject)
                logger.debug(f"Copied user pyproject.toml to {repo_src_pyproject}")

        # Copy additional directories if specified in service configuration
        additional_dirs = container_cfg.get("additional_dirs", [])
        if additional_dirs:
            for dir_spec in additional_dirs:
                if isinstance(dir_spec, str):
                    # Simple string: copy directory with same name
                    src_dir = dir_spec
                    dst_dir = os.path.join(out_dir, dir_spec)
                elif isinstance(dir_spec, dict):
                    # Dictionary: allows custom source -> destination mapping
                    src_dir = dir_spec.get("src")
                    dst_dir = os.path.join(out_dir, dir_spec.get("dst", src_dir))
                else:
                    continue

                if src_dir and os.path.exists(src_dir):
                    # Handle both files and directories
                    if os.path.isfile(src_dir):
                        # For files, create parent directory and copy file
                        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
                        shutil.copy2(src_dir, dst_dir)
                        logger.debug(f"Copied file {src_dir} to {dst_dir}")
                    elif os.path.isdir(src_dir):
                        # For directories, use copytree
                        shutil.copytree(src_dir, dst_dir)
                        logger.debug(f"Copied directory {src_dir} to {dst_dir}")
                elif src_dir:
                    logger.warning(f"Path {src_dir} does not exist, skipping")

        # Ensure _agent_data directory structure exists before container deployment
        # This prevents mount failures when containers try to mount non-existent directories
        _ensure_agent_data_structure(config)

        # Create flattened configuration file for container
        # This merges all imports and creates a complete config without import directives
        # SECURITY: Use unexpanded config to prevent API keys from being written to disk
        # The container will expand ${VAR} placeholders at runtime from environment variables
        try:
            with quiet_logger(["registry", "CONFIG"]):
                global_config = ConfigBuilder()
                flattened_config = normalize_facility_config(
                    global_config.get_unexpanded_config()
                )  # Preserves ${VAR} placeholders - secrets not written to disk

            # Adjust paths for container environment
            # In containers, src/ is copied to repo_src/, so config paths must be updated

            def adjust_src_paths_recursive(obj):
                """Recursively adjust all src/ paths in config for container environment.

                When deploying to containers, the deployment system copies src/ -> repo_src/.
                Any config values that are paths starting with 'src/' must be updated to
                'repo_src/' to work correctly in the container environment.

                Args:
                    obj: Config dictionary or list to process
                """
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str):
                            if value.startswith("src/"):
                                obj[key] = f"repo_src/{value[4:]}"
                                logger.debug(f"Container path adjustment: {value} -> {obj[key]}")
                            elif value.startswith("./src/"):
                                obj[key] = f"./repo_src/{value[6:]}"
                                logger.debug(f"Container path adjustment: {value} -> {obj[key]}")
                        elif isinstance(value, (dict, list)):
                            adjust_src_paths_recursive(value)
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            adjust_src_paths_recursive(item)

            # Recursively adjust all src/ paths in the config
            adjust_src_paths_recursive(flattened_config)

            # Drop the host interpreter path: execution.python_env_path is the
            # build machine's venv (e.g. /Users/.../.venv/bin/python3), which does
            # not exist in the container. Claude Code MCP-server generation prefers
            # python_env_path over sys.executable, so leaving it baked the host
            # interpreter into the container's .mcp.json — every MCP server then
            # failed to launch. Removing it makes generation fall back to the
            # container's own sys.executable (/usr/local/bin/python).
            exec_cfg = flattened_config.get("execution")
            if isinstance(exec_cfg, dict) and "python_env_path" in exec_cfg:
                removed = exec_cfg.pop("python_env_path")
                logger.debug(f"Dropped host python_env_path for container config: {removed}")

            # Handle claude_config_path: copy the file and adjust path
            # The config explicitly specifies which file to use, so we copy exactly that
            # and update the reference to match where we put it
            claude_generators = (
                flattened_config.get("execution", {}).get("generators", {}).get("claude_code", {})
            )
            claude_config_path = claude_generators.get("claude_config_path")
            if claude_config_path and os.path.exists(claude_config_path):
                # Copy the config file to build directory
                filename = os.path.basename(claude_config_path)
                dst_path = os.path.join(out_dir, filename)
                shutil.copy2(claude_config_path, dst_path)
                logger.debug(f"Copied {claude_config_path} to {dst_path}")

                claude_generators["claude_config_path"] = filename

            config_yml_dst = os.path.join(out_dir, "config.yml")
            with open(config_yml_dst, "w") as f:
                yaml.dump(flattened_config, f, default_flow_style=False, sort_keys=False)
            logger.debug(f"Created flattened config.yml at {config_yml_dst}")
        except Exception as e:
            logger.warning(f"Failed to create flattened config: {e}")
            # Fallback to copying original config, normalized first. The primary
            # deploy entry (`prepare_compose_files`) already rejects a removed
            # `gitlab:` block up front, so this degraded path only ever sees a
            # canonical config. A config.yml the normalizer/YAML loader can't
            # parse falls back further, to a verbatim copy, so this degraded
            # path never blocks the build.
            config_yml_src = "config.yml"
            if os.path.exists(config_yml_src):
                config_yml_dst = os.path.join(out_dir, "config.yml")
                try:
                    with open(config_yml_src, encoding="utf-8") as src_f:
                        fallback_config = normalize_facility_config(yaml.safe_load(src_f) or {})
                    with open(config_yml_dst, "w") as dst_f:
                        yaml.dump(fallback_config, dst_f, default_flow_style=False, sort_keys=False)
                    logger.debug(f"Copied normalized original config.yml to {config_yml_dst}")
                except Exception as fallback_exc:
                    logger.warning(
                        f"Could not normalize fallback config.yml ({fallback_exc}); "
                        "copying it verbatim instead"
                    )
                    shutil.copy2(config_yml_src, config_yml_dst)
                    logger.debug(f"Copied original config.yml to {config_yml_dst}")

        # Render kernel templates if specified in service configuration
        if container_cfg.get("render_kernel_templates", False):
            logger.info(f"Processing kernel templates for {source_dir}")
            render_kernel_templates(source_dir, config, out_dir)

    return compose_filepath


def _incremental_setup_build_dir(template_path, config, service_config, out_dir, dev_mode=False):
    """Setup build directory using incremental updates when full cleanup fails.

    This fallback function handles cases where the build directory cannot be
    completely removed due to file locks or NFS issues. It updates files
    incrementally instead of doing a full rebuild.

    Args:
        template_path (str): Path to the docker-compose template file
        config (dict): Configuration dictionary
        service_config (dict): Service-specific configuration
        out_dir (str): Output directory path that couldn't be cleaned

    Returns:
        str: Path to the rendered docker-compose.yml file
    """
    source_dir = os.path.relpath(os.path.dirname(template_path), os.getcwd())

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Copy/update files from source directory (skip if top-level services dir),
    # then stage the dev wheel — BEFORE rendering, exactly as in
    # setup_build_dir: the rendered OSPREY_DEV build arg must reflect whether a
    # wheel actually landed in this context.
    wheel_staged = False
    if source_dir != SERVICES_DIR:
        for file in os.listdir(source_dir):
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(out_dir, file)

            # Skip template files
            if file != TEMPLATE_FILENAME and not file.endswith(".j2"):
                try:
                    if os.path.isdir(src_path):
                        # For directories, use copytree with dirs_exist_ok (Python 3.8+)
                        if (
                            hasattr(shutil, "copytree")
                            and "dirs_exist_ok" in shutil.copytree.__code__.co_varnames
                        ):
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        else:
                            # Fallback for older Python versions
                            if not os.path.exists(dst_path):
                                shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                except (OSError, shutil.Error) as e:
                    logger.warning(f"Could not update {dst_path}: {e}")

        wheel_staged = _stage_dev_wheel_for_context(out_dir, dev_mode)

    # Create/update the docker compose file from the template (dev_mode gated
    # on staging success, exactly as in setup_build_dir).
    compose_filepath = render_template(
        template_path, {**config, "dev_mode": dev_mode and wheel_staged}, out_dir
    )

    # Handle source directory copying if needed
    if service_config.get("copy_src", False):
        src_dst_path = os.path.join(out_dir, OUT_SRC_DIR)
        try:
            if (
                hasattr(shutil, "copytree")
                and "dirs_exist_ok" in shutil.copytree.__code__.co_varnames
            ):
                shutil.copytree(SRC_DIR, src_dst_path, dirs_exist_ok=True)
            else:
                if not os.path.exists(src_dst_path):
                    shutil.copytree(SRC_DIR, src_dst_path)
        except (OSError, shutil.Error) as e:
            logger.warning(f"Could not update source directory {src_dst_path}: {e}")

    return compose_filepath


def find_existing_compose_files(config, deployed_services, quiet=False):
    """Find existing compose files without rebuilding directories.

    This function locates existing docker-compose.yml files in the build directory
    for the specified services without triggering any rebuild operations.

    Args:
        config (dict): Configuration dictionary containing build_dir
        deployed_services (list): List of service names to find compose files for
        quiet (bool): If True, suppress warning messages about missing files

    Returns:
        list: List of paths to existing compose files

    Example:
        compose_files = find_existing_compose_files(config, ['osprey.jupyter'])
        # Returns: ['./build/services/docker-compose.yml',
        #          './build/services/osprey/jupyter/docker-compose.yml']
    """
    compose_files = []
    build_dir = config.get("build_dir", "./build")

    # Add top-level compose file if it exists
    top_compose = os.path.join(build_dir, SERVICES_DIR, "docker-compose.yml")
    if os.path.exists(top_compose):
        compose_files.append(top_compose)

    # Add service-specific compose files
    for service_name in deployed_services:
        service_config, template_path = find_service_config(config, service_name)
        if template_path:
            # Construct expected compose file path
            source_dir = os.path.relpath(os.path.dirname(template_path), os.getcwd())
            compose_path = os.path.join(build_dir, source_dir, "docker-compose.yml")
            if os.path.exists(compose_path):
                compose_files.append(compose_path)
            elif not quiet:
                logger.warning(
                    f"Compose file not found for service '{service_name}' at {compose_path}"
                )

    return compose_files


def clean_deployment(compose_files, config=None):
    """Clean up containers, images, volumes, and networks for a fresh deployment.

    This function provides comprehensive cleanup capabilities for container
    deployments, removing containers, images, volumes, and networks to enable
    fresh rebuilds. It's particularly useful when configuration changes require
    complete environment reconstruction.

    :param compose_files: List of Docker Compose file paths for the deployment
    :type compose_files: list[str]
    :param config: Optional configuration dictionary for runtime detection
    :type config: dict, optional
    """
    logger.key_info("Cleaning up deployment...")

    # Pin COMPOSE_PROJECT_NAME so `down --volumes` targets THIS deploy's project
    # and volumes; unpinned it derives the shared "services" project and would
    # destroy a sibling deploy's data volumes.
    run_env = runtime_env(config, os.environ.copy())

    # Stop and remove containers, networks, volumes
    cmd_down = get_runtime_command(config)
    for compose_file in compose_files:
        cmd_down.extend(("-f", compose_file))
    cmd_down.extend(["--env-file", ".env", "down", "--volumes", "--remove-orphans"])

    logger.info(f"Running: {' '.join(cmd_down)}")
    subprocess.run(cmd_down, env=run_env)

    # Remove images built by the compose files
    cmd_rmi = get_runtime_command(config)
    for compose_file in compose_files:
        cmd_rmi.extend(("-f", compose_file))
    cmd_rmi.extend(["--env-file", ".env", "down", "--rmi", "all"])

    logger.info(f"Running: {' '.join(cmd_rmi)}")
    subprocess.run(cmd_rmi, env=run_env)

    logger.success("Cleanup completed")


def prepare_compose_files(config_path, dev_mode=False, expose_network=False):
    """Prepare compose files from configuration.

    Loads configuration and generates all necessary compose files for deployment.

    :param config_path: Path to the configuration file
    :type config_path: str
    :param dev_mode: Development mode - copy local framework to containers
    :type dev_mode: bool
    :param expose_network: Expose services to all network interfaces (0.0.0.0)
    :type expose_network: bool
    :return: Tuple of (config dict, list of compose file paths)
    :rtype: tuple[dict, list[str]]
    :raises RuntimeError: If configuration loading fails
    """
    try:
        with quiet_logger(["registry", "CONFIG"]):
            config = ConfigBuilder(config_path)
            config = normalize_facility_config(config.raw_config)
    except Exception as e:
        raise RuntimeError(f"Could not load config file {config_path}: {e}") from e

    # Handle network exposure setting
    # Default to localhost-only binding for security (Issue #126)
    if "deployment" not in config:
        config["deployment"] = {}
    if expose_network:
        config["deployment"]["bind_address"] = "0.0.0.0"
        logger.warning(
            "Network exposure enabled: services will bind to 0.0.0.0 (all interfaces). "
            "Ensure proper authentication is configured!"
        )
    elif "bind_address" not in config.get("deployment", {}):
        config["deployment"]["bind_address"] = "127.0.0.1"
        logger.info("Services will bind to localhost only (127.0.0.1) for security")

    # Get deployed services list
    deployed_services = config.get("deployed_services", [])
    if deployed_services:
        deployed_service_names = [str(service) for service in deployed_services]
        logger.info(f"Deployed services: {', '.join(deployed_service_names)}")
    else:
        logger.warning("No deployed_services list found, no services will be processed")
        deployed_service_names = []

    compose_files = []

    # Create the top level compose file. Skipped when no services are deployed:
    # an attached project (deploy_services: false) scaffolds no services/
    # templates at all, so rendering would throw — and there is nothing for the
    # top-level file to describe. deploy_up's empty-services guard (or its
    # web-terminals branch, which brings its own compose file) handles the rest.
    if deployed_service_names:
        top_template = os.path.join(SERVICES_DIR, TEMPLATE_FILENAME)
        build_dir = config.get("build_dir", "./build")
        out_dir = os.path.join(build_dir, SERVICES_DIR)
        top_template = render_template(top_template, config, out_dir)
        compose_files.append(top_template)

    # Create the service build directory for deployed services only
    for service_name in deployed_service_names:
        service_config, template_path = find_service_config(config, service_name)
        if service_config and template_path:
            if not os.path.isfile(template_path):
                raise RuntimeError(
                    f"Template file {template_path} not found for service '{service_name}'"
                )

            out = setup_build_dir(template_path, config, service_config, dev_mode)
            compose_files.append(out)
        else:
            raise RuntimeError(f"Service '{service_name}' not found in configuration")

    return config, compose_files
