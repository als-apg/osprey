"""Container Management and Service Orchestration System.

This module provides comprehensive container orchestration capabilities for the
deployment framework, handling service discovery, template rendering, build
directory management, and Podman Compose integration. The system supports
hierarchical service configurations with framework and application-specific
services that can be independently deployed and managed.

The container manager implements a sophisticated template processing pipeline
that converts Jinja2 templates into Docker Compose files, copies necessary
source code and configuration files, and orchestrates multi-service deployments
through Podman Compose with proper networking and dependency management.

Key Features:
    - Hierarchical service discovery (osprey.service, applications.app.service)
    - Jinja2 template rendering with configuration context
    - Intelligent build directory management with selective file copying
    - Environment variable expansion and configuration flattening
    - Podman Compose orchestration with multi-file support
    - Kernel template processing for Jupyter notebook environments

Architecture:
    The system supports two service categories:

    1. Framework Services: Core infrastructure services like databases,
       web interfaces, and development tools (jupyter, open-webui, pipelines)

    2. Application Services: Domain-specific services tied to particular
       applications (als_assistant.mongo, als_assistant.pv_finder)

Examples:
    Basic service deployment::

        $ python container_manager.py config.yml up -d
        # Deploys all services listed in deployed_services configuration

    Service discovery patterns::

        # Framework service (short name)
        deployed_services: ["jupyter", "pipelines"]

        # Framework service (full path)
        deployed_services: ["osprey.jupyter", "osprey.pipelines"]

        # Application service (full path required)
        deployed_services: ["applications.als_assistant.mongo"]

    Template rendering workflow::

        1. Load configuration with imports and merging
        2. Discover services listed in deployed_services
        3. Process Jinja2 templates with configuration context
        4. Copy source code and additional directories as specified
        5. Flatten configuration files for container consumption
        6. Execute Podman Compose with generated files

.. seealso::
   :mod:`deployment.loader` : Configuration loading system used by this module
   :class:`configs.config.ConfigBuilder` : Configuration management
   :func:`find_service_config` : Service discovery implementation
   :func:`render_template` : Template processing engine
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.deployment.runtime_helper import (
    get_ps_command,
    get_runtime_command,
    verify_runtime_is_running,
)
from osprey.utils.config import ConfigBuilder
from osprey.utils.log_filter import quiet_logger
from osprey.utils.logger import get_logger

# Initialize component logger for deployment operations
logger = get_logger("deployment")

SERVICES_DIR = "services"
SRC_DIR = "src"
OUT_SRC_DIR = "repo_src"

TEMPLATE_FILENAME = "docker-compose.yml.j2"
COMPOSE_FILE_NAME = "docker-compose.yml"


def find_service_config(config, service_name):
    """Locate service configuration and template path for deployment.

    This function implements the service discovery logic for the container
    management system, supporting both hierarchical service naming (full paths)
    and legacy short names for backward compatibility. The system searches
    through framework services and application-specific services to find
    the requested service configuration.

    Service naming supports three patterns:
    1. Framework services: "osprey.service_name" or just "service_name"
    2. Application services: "applications.app_name.service_name"
    3. Legacy services: "service_name" (deprecated, for backward compatibility)

    The function returns both the service configuration object and the path
    to the Docker Compose template, enabling the caller to access service
    settings and initiate template rendering.

    :param config: Configuration containing service definitions
    :type config: dict
    :param service_name: Service identifier (short name or full dotted path)
    :type service_name: str
    :return: Tuple containing service configuration and template path,
        or (None, None) if service not found
    :rtype: tuple[dict, str] or tuple[None, None]

    Examples:
        Framework service discovery::

            >>> config = {'osprey': {'services': {'jupyter': {'path': 'services/osprey/jupyter'}}}}
            >>> service_config, template_path = find_service_config(config, 'osprey.jupyter')
            >>> print(template_path)  # 'services/osprey/jupyter/docker-compose.yml.j2'

        Application service discovery::

            >>> config = {'applications': {'als_assistant': {'services': {'mongo': {'path': 'services/applications/als_assistant/mongo'}}}}}
            >>> service_config, template_path = find_service_config(config, 'applications.als_assistant.mongo')
            >>> print(template_path)  # 'services/applications/als_assistant/mongo/docker-compose.yml.j2'

        Legacy service discovery::

            >>> config = {'services': {'legacy_service': {'path': 'services/legacy'}}}
            >>> service_config, template_path = find_service_config(config, 'legacy_service')
            >>> print(template_path)  # 'services/legacy/docker-compose.yml.j2'

    .. note::
       Legacy service support (services.* configuration) is deprecated and
       will be removed in future versions. Use osprey.* or applications.*
       naming patterns for new services.

    .. seealso::
       :func:`get_templates` : Uses this function to build template lists
       :func:`setup_build_dir` : Processes discovered services for deployment
    """
    # Handle full path notation (osprey.jupyter, applications.als_assistant.mongo)
    if "." in service_name:
        parts = service_name.split(".")

        if parts[0] == "osprey" and len(parts) == 2:
            # osprey.service_name
            framework_services = config.get("osprey", {}).get("services", {})
            service_config = framework_services.get(parts[1])
            if service_config:
                return service_config, os.path.join(service_config["path"], TEMPLATE_FILENAME)

        elif parts[0] == "applications" and len(parts) == 3:
            # applications.app_name.service_name
            app_name, service_name_short = parts[1], parts[2]
            applications = config.get("applications", {})
            app_config = applications.get(app_name, {})
            app_services = app_config.get("services", {})
            service_config = app_services.get(service_name_short)
            if service_config:
                return service_config, os.path.join(service_config["path"], TEMPLATE_FILENAME)

    # Handle short names - check legacy services first for backward compatibility
    # TODO: remove this once we have migrated all services to the new config structure
    legacy_services = config.get("services", {})
    service_config = legacy_services.get(service_name)
    if service_config:
        return service_config, os.path.join(service_config["path"], TEMPLATE_FILENAME)

    return None, None


def get_templates(config):
    """Collect template paths for all deployed services in the configuration.

    This function builds a comprehensive list of Docker Compose template paths
    based on the services specified in the deployed_services configuration.
    It processes both the root services template and individual service templates,
    providing the complete set of templates needed for deployment.

    The function always includes the root services template (services/docker-compose.yml.j2)
    which defines the shared network configuration and other global service settings.
    Individual service templates are then discovered through the service discovery
    system and added to the template list.

    :param config: Configuration containing deployed_services list
    :type config: dict
    :return: List of template file paths for processing
    :rtype: list[str]
    :raises Warning: Prints warning if deployed_services is not configured

    Examples:
        Template collection for mixed services::

            >>> config = {
            ...     'deployed_services': ['osprey.jupyter', 'applications.als_assistant.mongo'],
            ...     'osprey': {'services': {'jupyter': {'path': 'services/osprey/jupyter'}}},
            ...     'applications': {'als_assistant': {'services': {'mongo': {'path': 'services/applications/als_assistant/mongo'}}}}
            ... }
            >>> templates = get_templates(config)
            >>> print(templates)
            ['services/docker-compose.yml.j2',
             'services/osprey/jupyter/docker-compose.yml.j2',
             'services/applications/als_assistant/mongo/docker-compose.yml.j2']

    .. warning::
       If deployed_services is not configured or empty, only the root services
       template will be returned, which may not provide functional services.

    .. seealso::
       :func:`find_service_config` : Service discovery used by this function
       :func:`render_template` : Processes the templates returned by this function
    """
    templates = []

    # Add the services root template
    templates.append(os.path.join(SERVICES_DIR, TEMPLATE_FILENAME))

    # Get deployed services list
    deployed_services = config.get("deployed_services", [])
    if deployed_services:
        deployed_service_names = [str(service) for service in deployed_services]
    else:
        logger.warning("No deployed_services list found, no service templates will be processed")
        return templates

    # Add templates for deployed services
    for service_name in deployed_service_names:
        service_config, template_path = find_service_config(config, service_name)
        if template_path:
            templates.append(template_path)
        else:
            logger.warning(f"Service '{service_name}' not found in configuration")

    return templates


def _inject_project_metadata(config):
    """Add project tracking metadata for container labels.

    This function injects deployment metadata into the configuration that will
    be used as Docker labels in the rendered compose files. These labels enable
    tracking which project/agent owns each container.

    The project name is extracted with the following priority:
    1. Root-level 'project_name' attribute (preferred, explicit)
    2. Last component of 'project_root' path (smart fallback)
    3. Default to 'unnamed-project'

    :param config: Configuration dictionary
    :type config: dict
    :return: Configuration with added osprey_labels section
    :rtype: dict
    """
    import datetime

    # Extract project name with priority order
    project_name = config.get("project_name")

    if not project_name:
        # Fallback: Extract from project_root path
        project_root = config.get("project_root", "")
        if project_root:
            project_name = os.path.basename(project_root.rstrip("/"))

    if not project_name:
        # Final fallback: Default
        project_name = "unnamed-project"

    # Create enhanced config with label metadata
    config_with_labels = config.copy()
    config_with_labels["osprey_labels"] = {
        "project_name": project_name,
        "project_root": config.get("project_root", os.getcwd()),
        "deployed_at": datetime.datetime.now().isoformat(),
    }

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


def _copy_local_framework_for_override(out_dir):
    """Build and copy local osprey wheel to container build directory for development mode.

    This function builds a wheel package from the local osprey source and copies it
    to the container build directory. This approach is cleaner and more reliable than
    copying source files, as it properly handles package structure and avoids namespace
    collisions with other packages.

    The wheel is built using the standard Python build process and can be installed
    in containers to override the PyPI version during development and testing.

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
        import subprocess
        import tempfile
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

        # Build the wheel package from local source
        logger.info("Building osprey wheel from local source...")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["python3", "-m", "build", "--wheel", "--outdir", tmpdir],
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
                return False

            # Find the built wheel
            wheel_files = list(Path(tmpdir).glob("*.whl"))
            if not wheel_files:
                logger.warning("No wheel file found after build")
                return False

            wheel_file = wheel_files[0]

            # Copy wheel to output directory
            dest_wheel = os.path.join(out_dir, wheel_file.name)
            shutil.copy2(wheel_file, dest_wheel)
            logger.success(f"Built and copied osprey wheel: {wheel_file.name}")

            return True

    except ImportError:
        logger.warning("Osprey not found in local environment, containers will use PyPI version")
        return False
    except Exception as e:
        logger.warning(f"Failed to build osprey wheel for dev override: {e}")
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
        "checkpoints",
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

    # Create the docker compose file from the template
    compose_filepath = render_template(template_path, config, out_dir)

    # Copy the contents of the services directory, except the template
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

        # Copy local osprey for development override (only in dev mode)
        # This will override the PyPI osprey after standard installation
        # Moved outside copy_src block so all services can use dev mode
        if dev_mode:
            osprey_copied = _copy_local_framework_for_override(out_dir)
            if osprey_copied:
                logger.key_info("Development mode: Osprey override prepared")
            else:
                logger.warning("Development mode requested but osprey override failed, using PyPI")
        else:
            logger.info("Production mode: Containers will install osprey from PyPI")

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
                flattened_config = (
                    global_config.get_unexpanded_config()
                )  # Preserves ${VAR} placeholders - secrets not written to disk

            # Adjust paths for container environment
            # In containers, src/ is copied to repo_src/, so config paths must be updated
            # For pipelines service: working directory is /app but files are mounted at /pipelines
            # For other services: working directory matches mount point

            def adjust_src_paths_recursive(obj, is_pipelines):
                """Recursively adjust all src/ paths in config for container environment.

                When deploying to containers, the deployment system copies src/ → repo_src/.
                Any config values that are paths starting with 'src/' must be updated to
                'repo_src/' (or '/pipelines/repo_src/' for pipelines service) to work correctly
                in the container environment.

                Args:
                    obj: Config dictionary or list to process
                    is_pipelines: Whether this is the pipelines service (needs absolute paths)
                """
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str):
                            # Only adjust paths that clearly start with src/ directory reference
                            # This is safe because 'src/' at start is always a path to source files
                            if value.startswith("src/"):
                                if is_pipelines:
                                    # Pipelines: absolute path since working dir (/app) != mount point (/pipelines)
                                    obj[key] = (
                                        f"/pipelines/repo_src/{value[4:]}"  # Remove 'src/' prefix
                                    )
                                    logger.debug(
                                        f"Container path adjustment: {value} -> {obj[key]}"
                                    )
                                else:
                                    # Other services: relative path since working dir == mount point
                                    obj[key] = f"repo_src/{value[4:]}"
                                    logger.debug(
                                        f"Container path adjustment: {value} -> {obj[key]}"
                                    )
                            elif value.startswith("./src/"):
                                if is_pipelines:
                                    obj[key] = (
                                        f"/pipelines/repo_src/{value[6:]}"  # Remove './src/' prefix
                                    )
                                    logger.debug(
                                        f"Container path adjustment: {value} -> {obj[key]}"
                                    )
                                else:
                                    obj[key] = f"./repo_src/{value[6:]}"
                                    logger.debug(
                                        f"Container path adjustment: {value} -> {obj[key]}"
                                    )
                        elif isinstance(value, (dict, list)):
                            adjust_src_paths_recursive(value, is_pipelines)
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            adjust_src_paths_recursive(item, is_pipelines)

            # Determine if this is a pipelines service
            is_pipelines_service = "pipelines" in source_dir

            # Recursively adjust all src/ paths in the config
            adjust_src_paths_recursive(flattened_config, is_pipelines_service)

            # Handle claude_config_path: copy the file and adjust path for pipelines
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

                # Update path in config: pipelines needs absolute path, others use filename
                if is_pipelines_service:
                    claude_generators["claude_config_path"] = f"/pipelines/{filename}"
                    logger.debug(f"Updated claude_config_path for pipelines: /pipelines/{filename}")
                else:
                    claude_generators["claude_config_path"] = filename

            config_yml_dst = os.path.join(out_dir, "config.yml")
            with open(config_yml_dst, "w") as f:
                yaml.dump(flattened_config, f, default_flow_style=False, sort_keys=False)
            logger.debug(f"Created flattened config.yml at {config_yml_dst}")
        except Exception as e:
            logger.warning(f"Failed to create flattened config: {e}")
            # Fallback to copying original config
            config_yml_src = "config.yml"
            if os.path.exists(config_yml_src):
                config_yml_dst = os.path.join(out_dir, "config.yml")
                shutil.copy2(config_yml_src, config_yml_dst)
                logger.debug(f"Copied original config.yml to {config_yml_dst}")

        # Render kernel templates if specified in service configuration
        if container_cfg.get("render_kernel_templates", False):
            logger.info(f"Processing kernel templates for {source_dir}")
            render_kernel_templates(source_dir, config, out_dir)

    return compose_filepath


def parse_args():
    """Parse command-line arguments for container management operations.

    This function defines and processes the command-line interface for the
    container management system, supporting configuration file specification,
    deployment commands (up/down), and operational flags like detached mode.

    The argument parser enforces logical constraints, such as requiring the
    'up' command when using detached mode, and provides clear error messages
    for invalid argument combinations.

    :return: Parsed command-line arguments
    :rtype: argparse.Namespace
    :raises SystemExit: If invalid argument combinations are provided

    Command-line Interface:
        python container_manager.py CONFIG [COMMAND] [OPTIONS]

        Positional Arguments:
            CONFIG: Path to the configuration file (required)
            COMMAND: Deployment command - 'up' or 'down' (optional)

        Options:
            -d, --detached: Run in detached mode (only with 'up' or 'rebuild')
            --dev: Development mode - use local osprey package instead of PyPI

    Examples:
        Generate compose files only::

            $ python container_manager.py config.yml
            # Creates build directory and compose files without deployment

        Deploy services in foreground::

            $ python container_manager.py config.yml up
            # Deploys services and shows output (uses PyPI framework)

        Deploy services in background::

            $ python container_manager.py config.yml up -d
            # Deploys services in detached mode (uses PyPI framework)

        Deploy with local osprey for development::

            $ python container_manager.py config.yml up --dev
            # Deploys services using local osprey package for testing

        Deploy with local osprey in background::

            $ python container_manager.py config.yml up -d --dev
            # Deploys services in detached mode with local osprey

        Stop services::

            $ python container_manager.py config.yml down
            # Stops and removes deployed services

        Clean deployment (remove images/volumes)::

            $ python container_manager.py config.yml clean
            # Removes containers, images, volumes, and networks

        Rebuild from scratch with local framework::

            $ python container_manager.py config.yml rebuild -d --dev
            # Clean + rebuild + start in detached mode with local framework

    .. seealso::
       :func:`main execution block` : Uses parsed arguments for deployment operations
    """
    parser = argparse.ArgumentParser(description="Run podman compose with config file.")

    # Mandatory config path
    parser.add_argument("config", help="Path to the config file")

    # Optional command
    parser.add_argument(
        "command",
        nargs="?",
        choices=["up", "down", "clean", "rebuild"],
        help="Command to run: 'up' (start), 'down' (stop), 'clean' (remove images/volumes), 'rebuild' (clean + up). If not provided, just generate compose files",
    )

    # Optional -d / --detached flag
    parser.add_argument(
        "-d", "--detached", action="store_true", help="Run in detached mode. Only valid with 'up'."
    )

    # Optional --dev flag for local osprey development
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode: copy local osprey package to containers instead of using PyPI version. Use this when testing local osprey changes.",
    )

    args = parser.parse_args()

    # Validation
    if args.detached and args.command not in ["up", "rebuild"]:
        parser.error("The -d/--detached flag is only allowed with 'up' or 'rebuild'.")

    return args


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

    # Create/update the docker compose file from the template
    compose_filepath = render_template(template_path, config, out_dir)

    # Copy/update files from source directory (skip if top-level services dir)
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

    # Stop and remove containers, networks, volumes
    cmd_down = get_runtime_command(config)
    for compose_file in compose_files:
        cmd_down.extend(("-f", compose_file))
    cmd_down.extend(["--env-file", ".env", "down", "--volumes", "--remove-orphans"])

    logger.info(f"Running: {' '.join(cmd_down)}")
    subprocess.run(cmd_down)

    # Remove images built by the compose files
    cmd_rmi = get_runtime_command(config)
    for compose_file in compose_files:
        cmd_rmi.extend(("-f", compose_file))
    cmd_rmi.extend(["--env-file", ".env", "down", "--rmi", "all"])

    logger.info(f"Running: {' '.join(cmd_rmi)}")
    subprocess.run(cmd_rmi)

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
            config = config.raw_config
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

    # Create the top level compose file
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


def deploy_up(config_path, detached=False, dev_mode=False, expose_network=False):
    """Start services using container runtime (Docker or Podman).

    :param config_path: Path to the configuration file
    :type config_path: str
    :param detached: Run in detached mode
    :type detached: bool
    :param dev_mode: Development mode for local framework testing
    :type dev_mode: bool
    :param expose_network: Expose services to all network interfaces (0.0.0.0)
    :type expose_network: bool
    """
    config, compose_files = prepare_compose_files(config_path, dev_mode, expose_network)

    # Verify container runtime is actually running
    is_running, error_msg = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(error_msg)

    # Set up environment for containers
    env = os.environ.copy()
    if dev_mode:
        env["DEV_MODE"] = "true"
        logger.key_info("Development mode: DEV_MODE environment variable set for containers")

    cmd = get_runtime_command(config)
    for compose_file in compose_files:
        cmd.extend(("-f", compose_file))

    # Only add --env-file if .env exists, otherwise let docker-compose use defaults
    from pathlib import Path

    env_file = Path(".env")
    if env_file.exists():
        cmd.extend(["--env-file", ".env"])
    else:
        logger.warning(
            "No .env file found - services will start with default/empty environment variables"
        )
        logger.info("To configure API keys: cp .env.example .env && edit .env")

    cmd.append("up")
    if detached:
        cmd.append("-d")

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    os.execvpe(cmd[0], cmd, env)


def deploy_down(config_path, dev_mode=False):
    """Stop services using container runtime (Docker or Podman).

    :param config_path: Path to the configuration file
    :type config_path: str
    """
    try:
        with quiet_logger(["registry", "CONFIG"]):
            config = ConfigBuilder(config_path)
            config = config.raw_config
    except Exception as e:
        raise RuntimeError(f"Could not load config file {config_path}: {e}") from e

    deployed_services = config.get("deployed_services", [])
    deployed_service_names = (
        [str(service) for service in deployed_services] if deployed_services else []
    )

    # Try to use existing compose files (suppress warnings for status check)
    compose_files = find_existing_compose_files(config, deployed_service_names, quiet=True)

    # If no existing compose files found, rebuild them
    if not compose_files:
        logger.info("No existing compose files found, rebuilding...")
        _, compose_files = prepare_compose_files(config_path, dev_mode)
    else:
        logger.info("Using existing compose files for 'down' operation:")
        for f in compose_files:
            logger.info(f"  - {f}")

    cmd = get_runtime_command(config)
    for compose_file in compose_files:
        cmd.extend(("-f", compose_file))

    # Only add --env-file if .env exists
    from pathlib import Path

    env_file = Path(".env")
    if env_file.exists():
        cmd.extend(["--env-file", ".env"])

    cmd.append("down")

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


def deploy_restart(config_path, detached=False, expose_network=False):
    """Restart services using container runtime (Docker or Podman).

    :param config_path: Path to the configuration file
    :type config_path: str
    :param detached: Run in detached mode
    :type detached: bool
    :param expose_network: Expose services to all network interfaces (0.0.0.0)
    :type expose_network: bool
    """
    config, compose_files = prepare_compose_files(config_path, expose_network=expose_network)

    # Verify container runtime is actually running
    is_running, error_msg = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(error_msg)

    cmd = get_runtime_command(config)
    for compose_file in compose_files:
        cmd.extend(("-f", compose_file))
    cmd.extend(["--env-file", ".env", "restart"])

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    subprocess.run(cmd)

    # If detached mode requested, detach after restart
    if detached:
        logger.info("Services restarted. Running in detached mode.")


def show_status(config_path):
    """Show detailed status of services with formatted output.

    Uses direct container runtime ps to show actual container state, independent of compose files.
    Displays containers for this project separately from other running containers.

    :param config_path: Path to the configuration file
    :type config_path: str
    """
    try:
        import json

        from rich.table import Table

        from osprey.cli.styles import Styles, console

        with quiet_logger(["registry", "CONFIG"]):
            config = ConfigBuilder(config_path)
            config = config.raw_config
    except Exception as e:
        raise RuntimeError(f"Could not load config file {config_path}: {e}") from e

    # Get deployed services and current project name
    deployed_services = config.get("deployed_services", [])
    deployed_service_names = (
        [str(service) for service in deployed_services] if deployed_services else []
    )

    # Determine current project name (same logic as _inject_project_metadata)
    current_project = config.get("project_name")
    if not current_project:
        project_root = config.get("project_root", "")
        if project_root:
            current_project = os.path.basename(project_root.rstrip("/"))
    if not current_project:
        current_project = "unnamed-project"

    # Get all containers using direct runtime ps (not compose-dependent)
    try:
        result = subprocess.run(
            get_ps_command(config, all_containers=True), capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            console.print("\n[red]Error: Could not query container status[/red]")
            console.print(f"[dim]Command failed with return code {result.returncode}[/dim]\n")
            return

        # Parse newline-separated JSON objects (Docker format) or JSON array (Podman format)
        all_containers = []
        if result.stdout.strip():
            try:
                # Try parsing as JSON array first (Podman format)
                all_containers = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Fall back to newline-separated JSON objects (Docker format)
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        all_containers.append(json.loads(line))

    except subprocess.TimeoutExpired:
        console.print("\n[red]Error: Container query timed out[/red]\n")
        return
    except json.JSONDecodeError as e:
        console.print("\n[red]Error: Could not parse container data[/red]")
        console.print(f"[dim]{e}[/dim]\n")
        return
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]\n")
        return

    # Separate containers into project and non-project
    project_containers = []
    other_containers = []

    for container in all_containers:
        # Extract project label
        labels = container.get("Labels", {})
        container_project = "unknown"

        if isinstance(labels, dict):
            container_project = labels.get("osprey.project.name", "unknown")
        elif isinstance(labels, str):
            for label in labels.split(","):
                if "=" in label:
                    key, value = label.split("=", 1)
                    if key.strip() == "osprey.project.name":
                        container_project = value.strip()
                        break

        # Check if container belongs to this project
        belongs_to_project = container_project == current_project

        # Also check if container name matches any deployed service (for backward compatibility)
        names = container.get("Names", [])
        if isinstance(names, list):
            names_str = " ".join(str(n) for n in names).lower()
        else:
            names_str = str(names).lower()

        matches_service = any(
            service.split(".")[-1].lower() in names_str for service in deployed_service_names
        )

        if belongs_to_project or matches_service:
            project_containers.append(container)
        else:
            # Only include containers with osprey labels in "other"
            if container_project != "unknown":
                other_containers.append(container)

    # Helper functions for status display
    def _create_status_table():
        """Create a status table with consistent styling."""
        table = Table(show_header=True, header_style=Styles.BOLD_PRIMARY)
        table.add_column("Service", style=Styles.ACCENT, no_wrap=True)
        table.add_column("Project", style=Styles.SUCCESS, no_wrap=True)
        table.add_column("Status", style=Styles.PRIMARY)
        table.add_column("Ports", style=Styles.INFO)
        table.add_column("Image", style=Styles.DIM)
        return table

    def _add_container_to_table(table, container):
        """Add a container as a row in the status table."""
        # Extract container name
        names = container.get("Names", [])
        if isinstance(names, list) and names:
            container_name = names[0]
        else:
            container_name = str(names) if names else "unknown"

        # Extract project label
        labels = container.get("Labels", {})
        project_name = "unknown"
        if isinstance(labels, dict):
            project_name = labels.get("osprey.project.name", "unknown")
        elif isinstance(labels, str):
            for label in labels.split(","):
                if "=" in label:
                    key, value = label.split("=", 1)
                    if key.strip() == "osprey.project.name":
                        project_name = value.strip()
                        break

        # Truncate long project names
        if len(project_name) > 12:
            project_name = project_name[:9] + "..."

        # Format status
        state = container.get("State", "unknown")
        if state == "running":
            status = f"[{Styles.SUCCESS}]● Running[/{Styles.SUCCESS}]"
        elif state == "exited":
            status = f"[{Styles.ERROR}]● Stopped[/{Styles.ERROR}]"
        elif state == "restarting":
            status = f"[{Styles.WARNING}]● Restarting[/{Styles.WARNING}]"
        else:
            status = f"[{Styles.DIM}]● {state}[/{Styles.DIM}]"

        # Format ports
        ports_raw = container.get("Ports", [])
        port_list = []
        if ports_raw:
            for port in ports_raw:
                if isinstance(port, dict):
                    # Handle different port format variations
                    # podman ps format: host_port, container_port
                    # compose ps format: PublishedPort, TargetPort
                    published = (
                        port.get("host_port")
                        or port.get("PublishedPort")
                        or port.get("published", "")
                    )
                    target = (
                        port.get("container_port")
                        or port.get("TargetPort")
                        or port.get("target", "")
                    )
                    if published and target:
                        port_list.append(f"{published}→{target}")
        ports = ", ".join(port_list) if port_list else "-"

        # Get image
        image = container.get("Image", "unknown")
        if len(image) > 40:
            image = "..." + image[-37:]

        table.add_row(container_name, project_name, status, ports, image)

    # Display project containers
    console.print("\n[bold]Service Status:[/bold]")

    if project_containers:
        table = _create_status_table()
        for container in project_containers:
            _add_container_to_table(table, container)
        console.print(table)
    else:
        console.print(
            f"\n[warning]ℹ️  No services running for project '{current_project}'[/warning]"
        )
        if deployed_service_names:
            console.print(f"[dim]Configured services: {', '.join(deployed_service_names)}[/dim]")
        console.print("\n[info]Start services with:[/info]")
        console.print("  • [command]osprey deploy up[/command]")

    # Display other osprey containers
    if other_containers:
        console.print("\n[bold]Other Osprey Containers:[/bold]")
        other_table = _create_status_table()
        for container in other_containers:
            _add_container_to_table(other_table, container)
        console.print(other_table)

    console.print()


def rebuild_deployment(config_path, detached=False, dev_mode=False, expose_network=False):
    """Rebuild deployment from scratch (clean + up).

    :param config_path: Path to the configuration file
    :type config_path: str
    :param detached: Run in detached mode
    :type detached: bool
    :param dev_mode: Development mode for local framework testing
    :type dev_mode: bool
    :param expose_network: Expose services to all network interfaces (0.0.0.0)
    :type expose_network: bool
    """
    config, compose_files = prepare_compose_files(config_path, dev_mode, expose_network)

    # Verify container runtime is actually running (for the rebuild phase)
    is_running, error_msg = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(error_msg)

    # Clean first
    clean_deployment(compose_files, config)

    # Set up environment for containers
    env = os.environ.copy()
    if dev_mode:
        env["DEV_MODE"] = "true"
        logger.key_info("Development mode: DEV_MODE environment variable set for containers")

    # Then start up
    cmd = get_runtime_command(config)
    for compose_file in compose_files:
        cmd.extend(("-f", compose_file))

    # Only add --env-file if .env exists
    from pathlib import Path

    env_file = Path(".env")
    if env_file.exists():
        cmd.extend(["--env-file", ".env"])
    else:
        logger.warning(
            "No .env file found - services will start with default/empty environment variables"
        )
        logger.info("To configure API keys: cp .env.example .env && edit .env")

    cmd.extend(["up", "--build"])
    if detached:
        cmd.append("-d")

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    """Main execution block for container management operations.

    This section orchestrates the complete deployment workflow:
    1. Parse command-line arguments
    2. Load and validate configuration
    3. Discover and process services
    4. Generate build directories and compose files
    5. Execute Podman Compose commands if requested

    The execution block handles errors gracefully, providing clear feedback
    for configuration issues, missing services, or deployment failures.
    Exit codes indicate success (0) or various failure conditions (1).

    Workflow:
        1. Configuration Loading: Use ConfigBuilder to load and merge
           configuration files with proper error handling
        2. Service Discovery: Process deployed_services list to identify
           active services for deployment
        3. Template Processing: Generate build directories for root services
           and each deployed service
        4. Container Orchestration: Execute Podman Compose with generated
           files and environment configuration

    Examples:
        Successful deployment workflow::

            $ python container_manager.py config.yml up -d
            Deployed services: osprey.jupyter, applications.als_assistant.mongo
            Generated compose files:
             - build/services/docker-compose.yml
             - build/services/osprey/jupyter/docker-compose.yml
             - build/services/applications/als_assistant/mongo/docker-compose.yml
            Running command:
                podman compose -f build/services/docker-compose.yml \
                               -f build/services/osprey/jupyter/docker-compose.yml \
                               -f build/services/applications/als_assistant/mongo/docker-compose.yml \
                               --env-file .env up -d

    .. seealso::
       :func:`parse_args` : Command-line argument processing
       :class:`configs.config.ConfigBuilder` : Configuration management
       :func:`find_service_config` : Service discovery implementation
    """
    args = parse_args()

    try:
        if args.command == "up":
            deploy_up(args.config, detached=args.detached, dev_mode=args.dev)
        elif args.command == "down":
            deploy_down(args.config, dev_mode=args.dev)
        elif args.command == "clean":
            # For clean, we need to prepare files first
            _, compose_files = prepare_compose_files(args.config, dev_mode=args.dev)
            clean_deployment(compose_files)
        elif args.command == "rebuild":
            rebuild_deployment(args.config, detached=args.detached, dev_mode=args.dev)
        else:
            # No command specified - just generate compose files
            _, compose_files = prepare_compose_files(args.config, dev_mode=args.dev)
            logger.success("Generated compose files:")
            for compose_file in compose_files:
                logger.info(f" - {compose_file}")
    except RuntimeError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
